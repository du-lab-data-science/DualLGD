import os
import sys
import pathlib
import warnings
import logging

import torch
torch.cuda.empty_cache()
# Suppress DDP stream mismatch warning - this is a known PyTorch issue with DDP that doesn't cause actual problems
torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
try:
    torch.set_float32_matmul_precision('high')
    logging.info("Enabled float32 matmul precision - medium")
except:
    logging.info("Could not enable float32 matmul precision - medium")
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils
from src.diffusion_model_fp2mol import FP2MolDenoisingDiffusion
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.progress import DetailedProgressLogger
from src.safe_csv_logger import SafeCSVLogger


warnings.filterwarnings("ignore", category=PossibleUserWarning)


def _is_discrete_model(cfg):
    """Backward-compatible check for diffusion type.

    Older configs used `model.type`, while current FP2Mol configs do not define it.
    """
    model_type = cfg.model.get("type", "discrete")
    return model_type == "discrete"


def _load_from_trusted_checkpoint(module_cls, checkpoint_path, **kwargs):
    """Load local trusted checkpoints across PyTorch/Lightning versions."""
    try:
        return module_cls.load_from_checkpoint(checkpoint_path, weights_only=False, **kwargs)
    except TypeError:
        # Older Lightning versions do not expose `weights_only`.
        return module_cls.load_from_checkpoint(checkpoint_path, **kwargs)


def _torch_load_trusted(path, map_location="cpu"):
    """Load local trusted checkpoint files across PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch versions do not expose `weights_only`.
        return torch.load(path, map_location=map_location)


def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only
    val_samples_to_generate = cfg.general.val_samples_to_generate
    test_samples_to_generate = cfg.general.test_samples_to_generate
    if _is_discrete_model(cfg):
        model = _load_from_trusted_checkpoint(FP2MolDenoisingDiffusion, resume, **model_kwargs)
    else:
        raise NotImplementedError("Only discrete diffusion models are supported for FP2Mol dataset currently")
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg.general.val_samples_to_generate = val_samples_to_generate
    cfg.general.test_samples_to_generate = test_samples_to_generate
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]

    resume_path = os.path.join(root_dir, cfg.general.resume)

    if _is_discrete_model(cfg):
        model = _load_from_trusted_checkpoint(FP2MolDenoisingDiffusion, resume_path, **model_kwargs)
    else:
        raise NotImplementedError("Only discrete diffusion models are supported for FP2Mol dataset currently")
    new_cfg = model.cfg

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)

    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '_resume'
    return new_cfg, model

def load_decoder_from_lightning_ckpt(model, ckpt_path):
    """ Load a model from a PyTorch Lightning checkpoint. """
    state_dict = _torch_load_trusted(ckpt_path, map_location='cpu')["state_dict"]
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            k = k[6:]
            cleaned_state_dict[k] = v

    model.model.load_state_dict(cleaned_state_dict, strict=True)
    logging.info(f"Loaded model from: '{ckpt_path}'")

def freeze_weights(model, cfg):
    if cfg.general.finetune_strategy == 'freeze_transformer_layers':
        for param in model.model.tf_layers.parameters():
            param.requires_grad = False
    else:
        raise NotImplementedError("Unknown finetuning strategy")


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    logger = logging.getLogger("msms_main")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    path = os.path.join("msms_main.log")
    fh = logging.FileHandler(path)
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    logging.info(cfg)

    utils._resolve_dataset_paths(cfg)

    dataset_config = cfg["dataset"]

    if dataset_config["name"] != "fp2mol":
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))
    
    from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
    from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
    from diffusion.extra_features_molecular import ExtraMolecularFeatures
    from analysis.visualization import MolecularVisualization

    from datasets import fp2mol_dataset
        
    datamodule = fp2mol_dataset.FP2MolDataModule(cfg)
    logging.info("Dataset loaded")
    train_batches = len(datamodule.train_dataloader())
    val_batches = len(datamodule.val_dataloader())
    test_batches = len(datamodule.test_dataloader())
    train_samples = len(datamodule.train_dataset)
    val_samples = len(datamodule.val_dataset)
    test_samples = len(datamodule.test_dataset)
    logging.info(f"Train: {train_samples} samples ({train_batches} batches), Val: {val_samples} samples ({val_batches} batches), Test: {test_samples} samples ({test_batches} batches)")
    dataset_infos = fp2mol_dataset.FP2Mol_infos(datamodule, cfg, recompute_statistics=False)

    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    if cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features, domain_features=domain_features)

    logging.info("Dataset infos:", dataset_infos.output_dims)
    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)

    visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                    'visualization_tools': visualization_tools, 'extra_features': extra_features, 'domain_features': domain_features}

    # Get Hydra output directory before any potential os.chdir()
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir

    if cfg.general.test_only:
        # When testing, previous configuration is fully loaded
        cfg, _ = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        # Temporarily change to old directory (may be needed for compatibility)
        old_dir = cfg.general.resume.split('checkpoints')[0]
        os.chdir(old_dir)
        # Change back to Hydra's output directory for saving new results
        os.chdir(output_dir)

    try:
        os.makedirs('preds/')
    except OSError:
        pass
    try:
        os.makedirs('models/')
    except OSError:
        pass
    try:
        os.makedirs('logs/')
    except OSError:
        pass

    try:
        os.makedirs('logs/' + cfg.general.name)
    except OSError:
        pass

    model = FP2MolDenoisingDiffusion(cfg=cfg, **model_kwargs)
    
    try:
        pretrained_ckpt = cfg.general.get("pretrained", None)
        if pretrained_ckpt is not None:
            if pretrained_ckpt.endswith('.ckpt'):
                load_decoder_from_lightning_ckpt(model, pretrained_ckpt)
            else:
                raise NotImplementedError("Only PyTorch Lightning checkpoints currently supported!")
    except Exception as e:
        print("Could not load pretrained model:", e)
            
    try:
        finetune_strategy = cfg.general.get("finetune_strategy", None)
        if finetune_strategy is not None:
            freeze_weights(model, cfg)
    except Exception as e:
        print("Could not freeze weights:", e)
        
    callbacks = []
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    if bool(getattr(cfg.general, 'text_progress_log', True)):
        callbacks.append(
            DetailedProgressLogger(
                train_log_every_n_steps=int(getattr(cfg.general, "text_log_every_n_steps", 20)),
                val_log_every_n_batches=int(getattr(cfg.general, "text_log_val_every_n_batches", 1)),
                test_log_every_n_batches=int(getattr(cfg.general, "text_log_test_every_n_batches", 1)),
            )
        )
    if cfg.train.progress_bar:
        callbacks.append(TQDMProgressBar(refresh_rate=getattr(cfg.train, "progress_bar_refresh_rate", 10), leave=True))
    if cfg.train.save_model:
        # Use Hydra output directory to ensure checkpoints are saved in the correct location
        # even when os.chdir() is called during resume
        hydra_cfg = HydraConfig.get()
        output_dir = hydra_cfg.runtime.output_dir
        checkpoint_dir = os.path.join(output_dir, f"checkpoints/{cfg.general.name}")
        checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, # best (top-5) checkpoints
                                              filename='{epoch}',
                                              monitor='val/NLL',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=checkpoint_dir, filename='last', every_n_epochs=1) # most recent checkpoint
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

    if cfg.train.ema_decay > 0: # TODO: Implement EMA for FP2Mol
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    name = cfg.general.name
    if name == 'debug':
        logging.warning("Run is called 'debug' -- it will run with fast_dev_run. ")

    loggers = [
        SafeCSVLogger(save_dir=f"logs/{name}", name=name),
        WandbLogger(name=name, save_dir=f"logs/{name}", project=cfg.general.wandb_name, log_model=False, config=utils.cfg_to_dict(cfg))
    ]

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    if use_gpu and cfg.general.gpus > 1:
        # DualLGD has optional branches that can leave params unused in a step.
        # Use DDP with unused-parameter detection for multi-GPU training.
        strategy = "ddp_find_unused_parameters_true"
    else:
        # Single GPU / CPU does not need DDP.
        strategy = "auto"

    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      strategy=strategy,
                      accelerator='gpu' if use_gpu else 'cpu',
                      devices=cfg.general.gpus if use_gpu else 1,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=bool(cfg.train.progress_bar),
                      callbacks=callbacks,
                      log_every_n_steps=500 if name != 'debug' else 1,
                      logger=loggers)

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume, weights_only=False)
        if cfg.general.name not in ['debug', 'test']:
            trainer.test(model, datamodule=datamodule)
    else:
        # Start by evaluating test_only_path
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only, weights_only=False)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            logging.info("Directory:", directory)
            files_list = os.listdir(directory)
            for file in files_list:
                if '.ckpt' in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    logging.info("Loading checkpoint", ckpt_path)
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)


if __name__ == '__main__':
    main()

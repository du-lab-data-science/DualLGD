import os
import sys
import pathlib
import warnings
import logging
import gc
import re
import shutil
import torch
torch.cuda.empty_cache()
# Suppress DDP stream mismatch warning - this is a known PyTorch issue with DDP that doesn't cause actual problems
torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
try:
    torch.set_float32_matmul_precision('medium')
    logging.info("Enabled float32 matmul precision - medium")
except:
    logging.info("Could not enable float32 matmul precision - medium")
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from datetime import timedelta
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils
from src.diffusion_model_spec2mol import Spec2MolDenoisingDiffusion
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
from src.analysis.visualization import MolecularVisualization
from src.datasets import spec2mol_dataset
from src.progress import DetailedProgressLogger
from src.safe_csv_logger import SafeCSVLogger


warnings.filterwarnings("ignore", category=PossibleUserWarning)

# TODO: refactor how configs are resumed (need old cfg.model and cfg.train but probably not general)
def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only
    val_samples_to_generate = cfg.general.val_samples_to_generate
    test_samples_to_generate = cfg.general.test_samples_to_generate
    gpus = cfg.general.gpus

    model = Spec2MolDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)

    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg.general.val_samples_to_generate = val_samples_to_generate
    cfg.general.test_samples_to_generate = test_samples_to_generate
    cfg.general.gpus = gpus
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]

    resume_path = os.path.join(root_dir, cfg.general.resume)

    model = Spec2MolDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    
    new_cfg = model.cfg

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)

    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '_resume'
    return new_cfg, model

def apply_encoder_finetuning(model, strategy):    
    if strategy is None:
        pass
    elif strategy == 'freeze':
        for param in model.encoder.parameters():
            param.requires_grad = False
    elif strategy == 'ft-unfold':
        for param in model.encoder.named_parameters():
            layer = param[0].split('.')[1]
            if layer != '2':
                param[1].requires_grad = False
    elif strategy == 'freeze-unfold':
        for param in model.encoder.named_parameters():
            layer = param[0].split('.')[1]
            if layer == '2':
                param[1].requires_grad = False
    elif strategy == 'ft-transformer':
        for param in model.encoder.named_parameters():
            layer = param[0].split('.')[1]
            if layer != '0':
                param[1].requires_grad = False
    elif strategy == 'freeze-transformer':
        for param in model.encoder.named_parameters():
            layer = param[0].split('.')[1]
            if layer == '0':
                param[1].requires_grad = False
    else:
        raise NotImplementedError(f'Unknown Finetune Strategy: {strategy}')
    
def apply_decoder_finetuning(model, strategy):
    if strategy is None:
        pass
    elif strategy == 'freeze':
        for param in model.decoder.parameters():
            param.requires_grad = False
    elif strategy == 'ft-input':
        for p in model.decoder.named_parameters():
            layer_name = p[0].split('.')[0]
            if layer_name not in ['mlp_in_X', 'mlp_in_E', 'mlp_in_y']:
                p[1].requires_grad = False
    elif strategy == 'freeze-input':
        for p in model.decoder.named_parameters():
            layer_name = p[0].split('.')[0]
            if layer_name in ['mlp_in_X', 'mlp_in_E', 'mlp_in_y']:
                p[1].requires_grad = False
    elif strategy == 'ft-transformer':
        for param in model.decoder.parameters():
            param.requires_grad = False
        for param in model.decoder.tf_layers.parameters():
            param.requires_grad = True
    elif strategy == 'freeze-transformer':
        for param in model.decoder.tf_layers.parameters():
            param.requires_grad = False
    elif strategy == 'ft-output':
        for p in model.decoder.named_parameters():
            layer_name = p[0].split('.')[0]
            if layer_name not in ['mlp_out_X', 'mlp_out_E', 'mlp_out_y']:
                p[1].requires_grad = False
    else:
        raise NotImplementedError(f'Unknown Finetune Strategy: {strategy}')

def load_weights(model, path):
    """
    Loads only the weights from a checkpoint file into the model without loading the full Lightning module.
    
    Args:
        model: The model to load weights into
        path: Path to the checkpoint file
        
    Returns:
        The model with loaded weights
    """
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    # Filter out keys that don't match the model (for partial loading)
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
    
    # Load the weights
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    logging.info(f"Loaded weights from {path}")
    logging.info(f"Missing keys: {missing_keys}")
    logging.info(f"Unexpected keys: {unexpected_keys}")
    
    # Clear CUDA cache after loading weights
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    return model


def _rename_prediction_file_for_current_run(filename: str, run_name: str) -> str:
    rank_match = re.match(r"^.+(_rank_\d+_(?:pred|true)_\d+\.pkl)$", filename)
    if rank_match:
        return f"{run_name}{rank_match.group(1)}"

    single_match = re.match(r"^.+(_(?:pred|true)_\d+\.pkl)$", filename)
    if single_match:
        return f"{run_name}{single_match.group(1)}"

    return filename


def _copy_resume_predictions(cfg: DictConfig) -> None:
    resume_preds_dir = cfg.general.get("resume_preds_dir")
    if not resume_preds_dir:
        return

    if bool(getattr(cfg.general, "evaluate_all_checkpoints", False)):
        raise ValueError("general.resume_preds_dir is not supported together with general.evaluate_all_checkpoints=true")

    source_dir = hydra.utils.to_absolute_path(str(resume_preds_dir))
    target_dir = os.path.abspath("preds")

    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Resume prediction directory does not exist: {source_dir}")

    if os.path.samefile(source_dir, target_dir):
        logging.info("Resume prediction directory is the current preds directory; skipping copy.")
        return

    copied = 0
    skipped = 0
    renamed = 0

    for entry in sorted(os.listdir(source_dir)):
        source_path = os.path.join(source_dir, entry)
        if not os.path.isfile(source_path) or not entry.endswith(".pkl"):
            continue

        target_name = _rename_prediction_file_for_current_run(entry, cfg.general.name)
        target_path = os.path.join(target_dir, target_name)
        if os.path.exists(target_path):
            skipped += 1
            continue

        shutil.copy2(source_path, target_path)
        copied += 1
        if target_name != entry:
            renamed += 1

    logging.info(
        "Prepared resume predictions from %s into %s: copied=%d, renamed=%d, skipped_existing=%d",
        source_dir,
        target_dir,
        copied,
        renamed,
        skipped,
    )

@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    import pytorch_lightning
    # Patch torch.load globally to inject pytorch-lightning_version if missing
    orig_torch_load = torch.load
    def patched_load(*args, **kwargs):
        load_kwargs = dict(kwargs)
        load_kwargs["weights_only"] = False
        try:
            checkpoint = orig_torch_load(*args, **load_kwargs)
        except TypeError:
            load_kwargs.pop("weights_only", None)
            checkpoint = orig_torch_load(*args, **load_kwargs)
        if isinstance(checkpoint, dict) and "pytorch-lightning_version" not in checkpoint:
            checkpoint["pytorch-lightning_version"] = pytorch_lightning.__version__
        return checkpoint
    torch.load = patched_load

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

    if dataset_config["name"] not in ("canopus", "msg", "nist", 'mona'):
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    datamodule = spec2mol_dataset.Spec2MolDataModule(cfg) # TODO: Add hyper for n_bits
    train_batches = len(datamodule.train_dataloader())
    dataset_infos = spec2mol_dataset.Spec2MolDatasetInfos(datamodule, cfg)

    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    if cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features, domain_features=domain_features)

    logging.info("Dataset infos:", dataset_infos.output_dims)
    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)

    # We do not evaluate novelty during training
    visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

    model_kwargs = {'cfg': cfg, 'dataset_infos': dataset_infos, 'train_metrics': train_metrics, 'visualization_tools': visualization_tools,
                    'extra_features': extra_features, 'domain_features': domain_features}

    if cfg.general.test_only:
        # When testing, previous configuration is fully loaded
        cfg, _ = get_resume(cfg, model_kwargs)
        # os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        # os.chdir(cfg.general.resume.split('checkpoints')[0])

    resume_mode = str(getattr(cfg.general, 'resume_mode', 'state')).lower()
    if resume_mode not in {'state', 'weights'}:
        raise ValueError(f"Unknown general.resume_mode='{resume_mode}'. Expected one of: state, weights")
    use_resume_weights_only = bool(cfg.general.resume is not None and resume_mode == 'weights')
    resume_override_optim = bool(getattr(cfg.general, 'resume_override_optim', False))
    if use_resume_weights_only and resume_override_optim:
        logging.warning("general.resume_override_optim is ignored when general.resume_mode=weights")

    os.makedirs('preds/', exist_ok=True)
    os.makedirs('logs/', exist_ok=True)
    os.makedirs('logs/' + cfg.general.name, exist_ok=True)
    _copy_resume_predictions(cfg)

    model_kwargs['cfg'] = cfg
    model = Spec2MolDenoisingDiffusion(**model_kwargs)

    callbacks = []
    callbacks.append(LearningRateMonitor(logging_interval='step'))

    if bool(getattr(cfg.general, 'text_progress_log', True)):
        callbacks.append(
            DetailedProgressLogger(
                train_log_every_n_steps=int(getattr(cfg.general, 'text_log_every_n_steps', 20)),
                val_log_every_n_batches=int(getattr(cfg.general, 'text_log_val_every_n_batches', 1)),
                test_log_every_n_batches=int(getattr(cfg.general, 'text_log_test_every_n_batches', 1)),
            )
        )

    if cfg.train.progress_bar:
        progress_bar_refresh_rate = int(getattr(cfg.train, "progress_bar_refresh_rate", 10))
        if train_batches > 0 and progress_bar_refresh_rate >= train_batches:
            progress_bar_refresh_rate = max(1, train_batches // 5)
        callbacks.append(TQDMProgressBar(refresh_rate=progress_bar_refresh_rate, leave=True))
    if cfg.train.save_model: # TODO: More advanced checkpointing
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

    if cfg.train.ema_decay > 0:
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
    log_every_n_steps = int(getattr(cfg.general, 'log_every_steps', 50))
    if train_batches > 0:
        log_every_n_steps = max(1, min(log_every_n_steps, train_batches))

    # Use a long NCCL timeout for DDP to avoid collective timeouts when
    # per-batch wall-clock varies widely across ranks (e.g. test-time sampling).
    strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(hours=3))

    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      strategy=strategy,
                      accelerator='gpu' if use_gpu else 'cpu',
                      devices=cfg.general.gpus if use_gpu else 1,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=bool(cfg.train.progress_bar),
                      callbacks=callbacks,
                      log_every_n_steps=log_every_n_steps if name != 'debug' else 1,
                      logger=loggers)

    apply_encoder_finetuning(model, cfg.general.encoder_finetune_strategy)
    apply_decoder_finetuning(model, cfg.general.decoder_finetune_strategy)

    load_weights_path = cfg.general.get("load_weights", None)
    if use_resume_weights_only and load_weights_path is None:
        load_weights_path = cfg.general.resume
        logging.info(f"Using weights-only resume from checkpoint: {load_weights_path}")
    elif use_resume_weights_only and load_weights_path is not None:
        logging.info(
            f"general.resume_mode=weights with explicit general.load_weights={load_weights_path}; "
            "optimizer/scheduler state will not be restored from general.resume"
        )

    if load_weights_path is not None:
        logging.info(f"Loading weights from {load_weights_path}")
        model = load_weights(model, load_weights_path)

    if not cfg.general.test_only:
        fit_ckpt_path = None if use_resume_weights_only else cfg.general.resume
        trainer.fit(model, datamodule=datamodule, ckpt_path=fit_ckpt_path, weights_only=False)
        
        # Clear CUDA cache after training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        if cfg.general.name not in ['debug', 'test']:
            trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.checkpoint_strategy, weights_only=False)
        
        # Use this to evaluate without doing training
        # trainer.test(model, datamodule=datamodule)
    else:
        # Start by evaluating test_only_path
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only, weights_only=False)
        
        # Clear CUDA cache after test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
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
                    
                    # Clear CUDA cache after each checkpoint evaluation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()

    torch.load = orig_torch_load  # Restore original torch.load


if __name__ == '__main__':
    main()

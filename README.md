# DualLGD

This is the implementation for **Unlocking High-Fidelity Molecular Generation from Mass Spectra via Dual-Stream Line Graph Diffusion**.

## Environment Installation

We recommend Python 3.10. Install the required dependencies listed in `pyproject.toml` with your preferred Python environment manager.

```bash
git clone <repo-url>
cd DualLGD
pip install -e .
```

## Datasets

Download and preprocess data with:

```bash
bash data_processing/01_download_canopus_data.sh
bash data_processing/02_download_msg_data.sh
bash data_processing/00_download_fp2mol_data.sh
bash data_processing/03_preprocess_fp2mol.sh
```

## Running the Code

Hydra configuration files are under `configs/`.

Decoder pretraining:

```bash
python src/fp2mol_main.py dataset=fp2mol
```

End-to-end finetuning:

```bash
python src/spec2mol_main.py dataset=canopus
python src/spec2mol_main.py dataset=msg
```

## Pretrained Checkpoints

We provide end-to-end finetuned models and pretrained encoder/decoder weights. The checkpoints can be downloaded from Zenodo:

https://zenodo.org/records/20281201

## Metrics and MCES Solver

Use `compute_metrics.py` to evaluate generated molecules. MCES computation is optional and enabled with `--mces`.

```bash
python compute_metrics.py --pred_dir <pred_dir> --pred_prefix <prefix> --output_dir <output_dir>
python compute_metrics.py --pred_dir <pred_dir> --pred_prefix <prefix> --output_dir <output_dir> --mces
```

MCES uses PuLP to call a MILP solver. The default `--mces_solver auto` is recommended; it will use an available solver such as HiGHS. The `highspy` dependency is included for HiGHS support. Only set `--mces_solver GUROBI` if Gurobi is installed and licensed in your environment. If Gurobi is not available, use the default auto mode or specify another available solver:

```bash
python compute_metrics.py --pred_dir <pred_dir> --pred_prefix <prefix> --output_dir <output_dir> --mces --mces_solver auto
python compute_metrics.py --pred_dir <pred_dir> --pred_prefix <prefix> --output_dir <output_dir> --mces --mces_solver HiGHS
```

## License

This project is released under the MIT License. See `LICENSE.txt` for details.

## Contact

For questions, please contact:

**Xujun Che**  
Email: xche@charlotte.edu

## Citation

If you use this code, please cite:

```bibtex
@article{che2026unlocking,
  title={Unlocking High-Fidelity Molecular Generation from Mass Spectra via Dual-Stream Line Graph Diffusion},
  author={Che, Xujun and Du, Xiuxia and Xu, Depeng},
  journal={arXiv preprint arXiv:2605.07048},
  year={2026}
}
```

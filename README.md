# VASR-Net

Source code for the VASR-Net 3-D segmentation network and its nnU-Net
inference entry point. Dataset files and model weights are intentionally not
included.

## Install

```bash
python -m pip install -r vasrnet_release/requirements.txt
```

Run commands from this directory so the local architecture package is on
`PYTHONPATH`.

## Inference

Use a trained nnU-Net model folder containing `plans.json`, `dataset.json`,
and the selected fold:

```bash
python -m vasrnet_release.predict \
  --model /path/to/model_folder \
  --input /path/to/imagesTs \
  --output /path/to/predictions \
  --folds 0 \
  --checkpoint checkpoint_best.pth
```

Input files should follow the nnU-Net naming convention, for example
`CASE_0000.nii.gz`. The shell wrapper is:

```bash
bash vasrnet_release/scripts/predict.sh \
  /path/to/model_folder /path/to/imagesTs /path/to/predictions
```

Use `--device cpu` for CPU inference, `--disable-tta` to disable mirroring,
and `--save-probabilities` to export probability maps.

The main class is defined in
`dynamic_network_architectures_compare/architectures/vasr_net.py`.

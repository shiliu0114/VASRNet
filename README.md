# VASR-Net

## Install

```bash
python -m pip install -r vasrnet_release/requirements.txt
```

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

The data from the First Affiliated Hospital of Nanchang University in the paper cannot be disclosed due to privacy concerns. However, we have prepared a subset of this data that can be used for testing. To obtain this data, please contact the author of the paper.
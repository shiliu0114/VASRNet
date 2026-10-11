"""Batch inference entry point using a trained model folder."""

from __future__ import annotations

import argparse
import multiprocessing

import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="VASR-Net/SRKD batch inference")
    parser.add_argument("--model", required=True, help="Training output folder containing plans.json and fold folders")
    parser.add_argument("--input", required=True, help="Input folder; files use CASE_0000.nii.gz naming")
    parser.add_argument("--output", required=True, help="Output segmentation folder")
    parser.add_argument("--folds", nargs="+", default=["0"], help="Fold identifiers, for example: 0 1 2 3 4")
    parser.add_argument("--checkpoint", default="checkpoint_best.pth")
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--step-size", type=float, default=0.5)
    parser.add_argument("--npp", type=int, default=3, help="Preprocessing workers")
    parser.add_argument("--nps", type=int, default=3, help="Segmentation export workers")
    parser.add_argument("--disable-tta", action="store_true")
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--continue-prediction", action="store_true")
    args = parser.parse_args()

    folds = tuple(x if x == "all" else int(x) for x in args.folds)
    device = torch.device(args.device)
    if args.device == "cpu":
        torch.set_num_threads(multiprocessing.cpu_count())
    elif args.device == "cuda":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    predictor = nnUNetPredictor(
        tile_step_size=args.step_size,
        use_gaussian=True,
        use_mirroring=not args.disable_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(args.model, folds, args.checkpoint)
    predictor.predict_from_files(
        args.input,
        args.output,
        save_probabilities=args.save_probabilities,
        overwrite=not args.continue_prediction,
        num_processes_preprocessing=args.npp,
        num_processes_segmentation_export=args.nps,
    )


if __name__ == "__main__":
    main()

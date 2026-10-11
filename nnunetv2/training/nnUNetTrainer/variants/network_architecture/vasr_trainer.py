"""Training implementations for VASR-Net and SRKD.

These classes retain the data loading, augmentation, validation and checkpoint
format of the surrounding framework.  Public launch scripts use the paper
names ``VASRNetTrainer`` and ``SRKDTrainer``.
"""

from __future__ import annotations

import os
import pydoc
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import autocast, nn

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.network_architecture.vasr_net import VASRNet
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import (
    nnUNetTrainer_onlyMirror01_1500ep,
)
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import dummy_context


def _main_output(output):
    """Return the highest-resolution logits from a network output."""
    if isinstance(output, dict):
        output = output["seg"]
    return output[0] if isinstance(output, (list, tuple)) else output


def _presence_target(target: torch.Tensor, classes: int) -> torch.Tensor:
    target = target[:, 0].long()
    return torch.stack([(target == c).flatten(1).any(1) for c in range(1, classes + 1)], dim=1).float()


def _soft_presence(logits: torch.Tensor) -> torch.Tensor:
    probability = torch.softmax(logits, dim=1)[:, 1:]
    # Spatial maximum retains small structures and remains differentiable at
    # the winning voxel for every class.
    return probability.flatten(2).amax(dim=2)


def _soft_relation(logits: torch.Tensor, max_voxels: int = 4096) -> torch.Tensor:
    probability = torch.softmax(logits, dim=1)[:, 1:]
    spatial = probability.shape[2:]
    if int(np.prod(spatial)) > max_voxels:
        scale = (max_voxels / float(np.prod(spatial))) ** (1.0 / len(spatial))
        size = tuple(max(1, int(round(i * scale))) for i in spatial)
        mode = "trilinear" if probability.ndim == 5 else "bilinear"
        probability = F.interpolate(probability, size=size, mode=mode, align_corners=False)
    vectors = F.normalize(probability.flatten(2), p=2, dim=2, eps=1e-6)
    relation = torch.bmm(vectors, vectors.transpose(1, 2)).clamp_min(0)
    return relation / relation.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def _boundary_strength(probability: torch.Tensor) -> torch.Tensor:
    """Finite-difference boundary magnitude for 2-D or 3-D probability maps."""
    magnitude = torch.zeros_like(probability)
    for axis in range(2, probability.ndim):
        left = [slice(None)] * probability.ndim
        right = [slice(None)] * probability.ndim
        left[axis] = slice(1, None)
        right[axis] = slice(None, -1)
        diff = (probability[tuple(left)] - probability[tuple(right)]).abs()
        magnitude[tuple(left)] += diff
        magnitude[tuple(right)] += diff
    return magnitude / max(1, probability.ndim - 2)


class VASRNetTrainer(nnUNetTrainer_onlyMirror01_1500ep):
    """End-to-end VASR-Net trainer with presence and relation supervision."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = int(os.environ.get("VASR_EPOCHS", self.num_epochs))
        self.lambda_presence = float(os.environ.get("VASR_LAMBDA_PRESENCE", "0.05"))
        self.lambda_relation = float(os.environ.get("VASR_LAMBDA_RELATION", "0.02"))

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        kwargs = dict(arch_init_kwargs)
        for key in arch_init_kwargs_req_import:
            if kwargs.get(key) is not None:
                kwargs[key] = pydoc.locate(kwargs[key])
        network = VASRNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=enable_deep_supervision,
            **kwargs,
        )
        network.apply(network.initialize)
        network.reset_vasr_parameters()
        return network

    def set_deep_supervision_enabled(self, enabled: bool):
        module = self.network.module if self.is_ddp else self.network
        if hasattr(module, "_orig_mod"):
            module = module._orig_mod
        module.deep_supervision = enabled
        module.backbone.decoder.deep_supervision = enabled

    def _losses(self, output: dict, target):
        segmentation = output["seg"]
        seg_loss = self.loss(segmentation, target)
        target_main = target[0] if isinstance(target, list) else target
        presence_target = _presence_target(target_main, output["presence"].shape[1])
        presence_loss = F.binary_cross_entropy(output["presence"].clamp(1e-6, 1 - 1e-6), presence_target)
        relation_target = presence_target[:, :, None] * presence_target[:, None, :]
        relation_target = relation_target / relation_target.sum(-1, keepdim=True).clamp_min(1.0)
        relation_loss = F.smooth_l1_loss(output["relation"], relation_target)
        total = seg_loss + self.lambda_presence * presence_loss + self.lambda_relation * relation_loss
        return total, seg_loss, presence_loss, relation_loss

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = [x.to(self.device, non_blocking=True) for x in target] if isinstance(target, list) else target.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data, return_aux=True)
            total, seg, presence, relation = self._losses(output, target)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(total).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": total.detach().cpu().numpy(), "seg_loss": seg.detach().cpu().numpy(),
                "presence_loss": presence.detach().cpu().numpy(), "relation_loss": relation.detach().cpu().numpy()}


class SRKDTrainer(nnUNetTrainer_onlyMirror01_1500ep):
    """Reliability-weighted distillation into the unchanged Student-M graph.

    Teacher-only tensors are derived during training.  The checkpoint contains
    only ``self.network`` (the student), so inference cost and architecture are
    identical to the supervised Student-M model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = int(os.environ.get("SRKD_EPOCHS", "500"))
        self.temperature = float(os.environ.get("SRKD_TEMPERATURE", "2.0"))
        self.lambda_output = float(os.environ.get("SRKD_LAMBDA_OUTPUT", "0.5"))
        self.lambda_presence = float(os.environ.get("SRKD_LAMBDA_PRESENCE", "0.05"))
        self.lambda_relation = float(os.environ.get("SRKD_LAMBDA_RELATION", "0.05"))
        self.lambda_boundary = float(os.environ.get("SRKD_LAMBDA_BOUNDARY", "0.1"))
        self.teacher_checkpoint = os.environ.get("SRKD_TEACHER_CHECKPOINT", "")
        self.teacher: nn.Module | None = None

    def initialize(self):
        super().initialize()
        self.teacher = self._load_teacher()
        if self.teacher is not None:
            self.teacher = self.teacher.to(self.device).eval()
            self.teacher.requires_grad_(False)

    def _load_teacher(self) -> nn.Module:
        if not self.teacher_checkpoint:
            # Validation and inference only need the student checkpoint.
            return None
        if not os.path.isfile(self.teacher_checkpoint):
            raise FileNotFoundError(f"Teacher checkpoint not found: {self.teacher_checkpoint}")
        checkpoint = torch.load(self.teacher_checkpoint, map_location="cpu", weights_only=False)
        trainer_name = checkpoint.get("trainer_name")
        if trainer_name != "VASRNetTrainer":
            raise ValueError(f"SRKD requires a VASRNetTrainer checkpoint, got {trainer_name!r}")
        init_args = checkpoint["init_args"]
        trainer_class = recursive_find_python_class(
            os.path.dirname(__file__), trainer_name,
            "nnunetv2.training.nnUNetTrainer.variants.network_architecture",
        )
        teacher_trainer = trainer_class(**init_args)
        teacher_trainer.initialize()
        teacher_trainer.network.load_state_dict(checkpoint["network_weights"])
        return teacher_trainer.network

    def _distillation_losses(self, student_output, teacher_output, target):
        student_logits = _main_output(student_output)
        teacher_logits = _main_output(teacher_output)
        if student_logits.shape[2:] != teacher_logits.shape[2:]:
            mode = "trilinear" if student_logits.ndim == 5 else "bilinear"
            teacher_logits = F.interpolate(teacher_logits, student_logits.shape[2:], mode=mode, align_corners=False)
        target_main = target[0] if isinstance(target, list) else target
        target_labels = target_main[:, 0].long()

        temperature = self.temperature
        teacher_probability = torch.softmax(teacher_logits / temperature, dim=1)
        student_log_probability = torch.log_softmax(student_logits / temperature, dim=1)
        confidence, teacher_label = torch.softmax(teacher_logits, dim=1).max(dim=1)
        agreement = teacher_label.eq(target_labels).float()
        reliability = (confidence * (0.25 + 0.75 * agreement)).detach()
        voxel_kl = F.kl_div(student_log_probability, teacher_probability, reduction="none").sum(1)
        output_loss = (voxel_kl * reliability).sum() / reliability.sum().clamp_min(1.0)
        output_loss = output_loss * temperature * temperature

        student_presence = _soft_presence(student_logits)
        teacher_presence = teacher_output.get("presence") if isinstance(teacher_output, dict) else None
        if teacher_presence is None:
            teacher_presence = _soft_presence(teacher_logits)
        presence_loss = F.binary_cross_entropy(
            student_presence.clamp(1e-6, 1 - 1e-6),
            teacher_presence.detach().clamp(1e-6, 1 - 1e-6),
        )

        student_relation = _soft_relation(student_logits)
        teacher_relation = teacher_output.get("relation") if isinstance(teacher_output, dict) else None
        if teacher_relation is None:
            teacher_relation = _soft_relation(teacher_logits)
        relation_loss = F.smooth_l1_loss(student_relation, teacher_relation.detach())

        student_fg = torch.softmax(student_logits, dim=1)[:, 1:].sum(1, keepdim=True)
        teacher_fg = torch.softmax(teacher_logits, dim=1)[:, 1:].sum(1, keepdim=True)
        student_boundary = _boundary_strength(student_fg)
        teacher_boundary = _boundary_strength(teacher_fg)
        uncertainty = 1.0 - confidence[:, None]
        boundary_weight = (1.0 + uncertainty + teacher_boundary).detach()
        boundary_loss = ((student_boundary - teacher_boundary.detach()).abs() * boundary_weight).mean()
        return output_loss, presence_loss, relation_loss, boundary_loss

    def train_step(self, batch: dict) -> dict:
        if self.teacher is None:
            raise RuntimeError("Training SRKD requires SRKD_TEACHER_CHECKPOINT")
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = [x.to(self.device, non_blocking=True) for x in target] if isinstance(target, list) else target.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            with torch.no_grad():
                teacher_output = self.teacher(data, return_aux=True)
            student_output = self.network(data)
            supervised = self.loss(student_output, target)
            output_kd, presence, relation, boundary = self._distillation_losses(student_output, teacher_output, target)
            total = (supervised + self.lambda_output * output_kd + self.lambda_presence * presence
                     + self.lambda_relation * relation + self.lambda_boundary * boundary)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(total).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": total.detach().cpu().numpy(), "supervised_loss": supervised.detach().cpu().numpy(),
                "output_kd": output_kd.detach().cpu().numpy(), "presence_kd": presence.detach().cpu().numpy(),
                "relation_kd": relation.detach().cpu().numpy(), "boundary_kd": boundary.detach().cpu().numpy()}


__all__ = ["VASRNetTrainer", "SRKDTrainer"]

"""Variable Anatomical Set Reasoning network for 3-D segmentation.

This module implements the inference graph described in the VASR-Net
manuscript.  It retains a residual encoder--decoder backbone and adds:

* foreground-class and anatomical-group tokens;
* voxel-to-token cross-attention;
* a per-class presence head and learned null states;
* two set-relation Transformer blocks; and
* entropy-gated token-to-voxel logit feedback.

The class follows the constructor convention used by
``dynamic_network_architectures`` so that it can be instantiated from an
nnU-Net-style plans file.  ``num_classes`` includes background, whereas the
token bank represents foreground classes only.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.initialization.weight_init import (
    InitWeights_He,
    init_last_bn_before_add_to_0,
)


class _ResidualEncoderUNet(nn.Module):
    """Minimal canonical ResEnc-UNet without comparison-repository extras."""

    def __init__(
        self,
        input_channels,
        n_stages,
        features_per_stage,
        conv_op,
        kernel_sizes,
        strides,
        n_blocks_per_stage,
        num_classes,
        n_conv_per_stage_decoder,
        conv_bias,
        norm_op,
        norm_op_kwargs,
        dropout_op,
        dropout_op_kwargs,
        nonlin,
        nonlin_kwargs,
        deep_supervision,
        block,
        bottleneck_channels,
        stem_channels,
    ):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        self.encoder = ResidualEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_blocks_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            block,
            bottleneck_channels,
            return_skips=True,
            disable_default_stem=False,
            stem_channels=stem_channels,
        )
        self.decoder = UNetDecoder(
            self.encoder,
            num_classes,
            n_conv_per_stage_decoder,
            deep_supervision,
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def compute_conv_feature_map_size(self, input_size):
        if len(input_size) != convert_conv_op_to_dim(self.encoder.conv_op):
            raise ValueError("input_size must contain spatial dimensions only")
        return self.encoder.compute_conv_feature_map_size(
            input_size
        ) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


class VASRNet(nn.Module):
    """Residual encoder--decoder with variable anatomical set reasoning.

    Parameters specific to VASR are intentionally explicit.  The manuscript
    configuration uses 42 foreground classes, four group tokens, a token width
    of 256, two relation blocks, and eight attention heads.  By default,
    reasoning operates on decoder stage 2 counted from the bottleneck; for the
    seven-stage ResEnc-L configuration this is the 192-channel intermediate
    decoder grid used by the manuscript configuration.
    """

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        dropout_op: Optional[Type[_DropoutNd]] = None,
        dropout_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
        deep_supervision: bool = False,
        block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
        bottleneck_channels: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
        stem_channels: Optional[int] = None,
        num_group_tokens: int = 4,
        token_dim: int = 256,
        token_heads: int = 8,
        relation_depth: int = 2,
        relation_mlp_ratio: int = 4,
        relation_dropout: float = 0.1,
        relation_dim: int = 64,
        reasoning_decoder_stage: int = 2,
    ) -> None:
        super().__init__()
        if conv_op is not nn.Conv3d:
            raise ValueError("VASRNet currently implements the 3-D CBCT configuration only")
        if num_classes < 2:
            raise ValueError("num_classes must include background and at least one foreground class")
        if token_dim % token_heads != 0:
            raise ValueError("token_dim must be divisible by token_heads")

        self.num_classes = int(num_classes)
        self.num_foreground_classes = self.num_classes - 1
        self.num_group_tokens = int(num_group_tokens)
        self.token_dim = int(token_dim)
        self.deep_supervision = bool(deep_supervision)

        self.backbone = _ResidualEncoderUNet(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
            block=block,
            bottleneck_channels=bottleneck_channels,
            stem_channels=stem_channels,
        )

        decoder = self.backbone.decoder
        if not 0 <= reasoning_decoder_stage < len(decoder.stages):
            raise ValueError(
                f"reasoning_decoder_stage must be in [0, {len(decoder.stages) - 1}]"
            )
        self.reasoning_decoder_stage = int(reasoning_decoder_stage)
        reasoning_channels = decoder.encoder.output_channels[-(reasoning_decoder_stage + 2)]
        self.reasoning_channels = int(reasoning_channels)

        self.class_tokens = nn.Parameter(
            torch.empty(self.num_foreground_classes, self.token_dim)
        )
        self.group_tokens = nn.Parameter(torch.empty(self.num_group_tokens, self.token_dim))
        self.null_states = nn.Parameter(
            torch.empty(self.num_foreground_classes, self.token_dim)
        )

        self.voxel_key_value = nn.Linear(self.reasoning_channels, self.token_dim)
        self.voxel_to_token = nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=token_heads,
            dropout=relation_dropout,
            batch_first=True,
        )
        self.presence_head = nn.Linear(self.token_dim, 1)

        relation_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=token_heads,
            dim_feedforward=self.token_dim * relation_mlp_ratio,
            dropout=relation_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.relation_blocks = nn.TransformerEncoder(
            relation_layer,
            num_layers=relation_depth,
            enable_nested_tensor=False,
        )
        self.relation_projection = nn.Linear(self.token_dim, relation_dim, bias=False)

        self.feedback_voxel_projection = nn.Linear(self.reasoning_channels, self.token_dim)
        self.feedback_token_projection = nn.Linear(self.token_dim, self.token_dim)
        self.feedback_head = nn.Sequential(
            nn.Linear(self.num_foreground_classes + self.token_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.num_classes),
        )
        self.gamma = nn.Parameter(torch.zeros(()))

        self.reset_vasr_parameters()

    def reset_vasr_parameters(self) -> None:
        nn.init.trunc_normal_(self.class_tokens, std=0.02)
        nn.init.trunc_normal_(self.group_tokens, std=0.02)
        nn.init.trunc_normal_(self.null_states, std=0.02)
        nn.init.zeros_(self.feedback_head[-1].weight)
        nn.init.zeros_(self.feedback_head[-1].bias)
        nn.init.zeros_(self.gamma)

    def _decode_with_reasoning_feature(self, skips):
        decoder = self.backbone.decoder
        low_resolution = skips[-1]
        segmentation_outputs = []
        reasoning_feature = None

        for stage_index in range(len(decoder.stages)):
            x = decoder.transpconvs[stage_index](low_resolution)
            x = torch.cat((x, skips[-(stage_index + 2)]), dim=1)
            x = decoder.stages[stage_index](x)

            if stage_index == self.reasoning_decoder_stage:
                reasoning_feature = x

            if decoder.deep_supervision:
                segmentation_outputs.append(decoder.seg_layers[stage_index](x))
            elif stage_index == len(decoder.stages) - 1:
                segmentation_outputs.append(decoder.seg_layers[-1](x))
            low_resolution = x

        if reasoning_feature is None:
            raise RuntimeError("The configured reasoning decoder stage was not executed")

        segmentation_outputs = segmentation_outputs[::-1]
        if decoder.deep_supervision:
            return segmentation_outputs, reasoning_feature
        return segmentation_outputs[0], reasoning_feature

    def _reason(self, feature: torch.Tensor, baseline_logits: torch.Tensor):
        batch_size = feature.shape[0]
        reasoning_shape = feature.shape[2:]
        voxel_feature = feature.flatten(2).transpose(1, 2)
        voxel_tokens = self.voxel_key_value(voxel_feature)

        initial_tokens = torch.cat((self.class_tokens, self.group_tokens), dim=0)
        initial_tokens = initial_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        aggregated_tokens, _ = self.voxel_to_token(
            query=initial_tokens,
            key=voxel_tokens,
            value=voxel_tokens,
            need_weights=False,
        )
        observed_tokens = initial_tokens + aggregated_tokens

        observed_classes = observed_tokens[:, : self.num_foreground_classes]
        presence = torch.sigmoid(self.presence_head(observed_classes))
        null_states = self.null_states.unsqueeze(0).expand(batch_size, -1, -1)
        calibrated_classes = presence * observed_classes + (1.0 - presence) * null_states
        calibrated_tokens = torch.cat(
            (calibrated_classes, observed_tokens[:, self.num_foreground_classes :]), dim=1
        )

        related_tokens = self.relation_blocks(calibrated_tokens)
        related_classes = related_tokens[:, : self.num_foreground_classes]
        relation_embedding = self.relation_projection(related_classes)
        relation = torch.softmax(
            torch.bmm(relation_embedding, relation_embedding.transpose(1, 2))
            / math.sqrt(relation_embedding.shape[-1]),
            dim=-1,
        )

        feedback_voxels = self.feedback_voxel_projection(voxel_feature)
        feedback_tokens = self.feedback_token_projection(related_classes)
        compatibility = torch.bmm(feedback_voxels, feedback_tokens.transpose(1, 2))
        compatibility = compatibility / math.sqrt(self.token_dim)
        attended_context = torch.bmm(torch.softmax(compatibility, dim=-1), related_classes)
        delta_logits = self.feedback_head(torch.cat((compatibility, attended_context), dim=-1))
        delta_logits = delta_logits.transpose(1, 2).reshape(
            batch_size, self.num_classes, *reasoning_shape
        )
        delta_logits = F.interpolate(
            delta_logits,
            size=baseline_logits.shape[2:],
            mode="trilinear",
            align_corners=False,
        )

        probability = torch.softmax(baseline_logits, dim=1)
        entropy = -(probability * torch.log(probability.clamp_min(1e-8))).sum(dim=1, keepdim=True)
        entropy = entropy / math.log(self.num_classes)
        # Subclasses can replace the generic entropy gate with a more specific
        # interface gate while retaining the VASR token reasoning branch.
        if getattr(self, "use_legacy_entropy_gate", True):
            refined_logits = baseline_logits + torch.tanh(self.gamma) * entropy * delta_logits
        else:
            refined_logits = baseline_logits

        auxiliary = {
            "presence": presence.squeeze(-1),
            "relation": relation,
            "delta_logits": delta_logits,
            "tokens": related_classes,
        }
        return refined_logits, auxiliary

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        skips = self.backbone.encoder(x)
        segmentation, reasoning_feature = self._decode_with_reasoning_feature(skips)

        if isinstance(segmentation, list):
            baseline_logits = segmentation[0]
        else:
            baseline_logits = segmentation
        refined_logits, auxiliary = self._reason(reasoning_feature, baseline_logits)

        if isinstance(segmentation, list):
            segmentation = [refined_logits, *segmentation[1:]]
        else:
            segmentation = refined_logits

        if return_aux:
            return {"seg": segmentation, **auxiliary}
        return segmentation

    def compute_conv_feature_map_size(self, input_size):
        """Return the backbone estimate plus the main VASR token tensors."""
        baseline = self.backbone.compute_conv_feature_map_size(input_size)
        spatial = np.asarray(input_size, dtype=np.int64)
        encoder_strides = self.backbone.encoder.strides
        target_encoder_stage = len(encoder_strides) - self.reasoning_decoder_stage - 2
        for stage_index in range(target_encoder_stage + 1):
            spatial = spatial // np.asarray(encoder_strides[stage_index], dtype=np.int64)
        voxels = int(np.prod(spatial, dtype=np.int64))
        token_count = self.num_foreground_classes + self.num_group_tokens
        auxiliary = (
            2 * voxels * self.token_dim
            + voxels * self.num_foreground_classes
            + token_count * self.token_dim
            + self.num_foreground_classes * self.num_foreground_classes
        )
        return baseline + np.int64(auxiliary)

    @staticmethod
    def initialize(module: nn.Module) -> None:
        """Compatibility hook for nnU-Net architecture construction."""
        _ResidualEncoderUNet.initialize(module)


__all__ = ["VASRNet"]

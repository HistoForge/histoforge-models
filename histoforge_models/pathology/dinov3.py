"""Fixed DINOv3 adapter architecture used by HistoForge runners.

The checkpoint contract is intentionally narrow: DINOv3 ViT-L/16, a
256-channel LayerNorm spatial-prior module, four bidirectional CTI
stages, and a stride-4 W2 readout. The only task-specific choice is the final
flow or HoVer output head.
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

try:
    import timm
except ImportError as error:
    raise ImportError(
        "The DINOv3 backbone requires timm. Install it with "
        "`pip install 'timm>=1.0.18'`."
    ) from error


DINOV3_MODEL = "vit_large_patch16_dinov3.lvd1689m"


class LayerNorm2d(nn.Module):
    """Per-channel LayerNorm over channels-first image tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        mean = value.mean(1, keepdim=True)
        variance = (value - mean).pow(2).mean(1, keepdim=True)
        normalized = (value - mean) / torch.sqrt(variance + self.eps)
        return normalized * self.weight[:, None, None] + self.bias[:, None, None]


def compute_reference_points(
    spatial_shapes: List[Tuple[int, int]],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return normalized reference points for every multi-scale token."""

    references = []
    for height, width in spatial_shapes:
        grid_y = (
            torch.linspace(
                0.5,
                height - 0.5,
                int(height),
                device=device,
                dtype=dtype,
            )
            / height
        )
        grid_x = (
            torch.linspace(
                0.5,
                width - 0.5,
                int(width),
                device=device,
                dtype=dtype,
            )
            / width
        )
        reference_y, reference_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
        references.append(
            torch.stack([reference_x.flatten(), reference_y.flatten()], dim=-1)
        )
    all_references = torch.cat(references, dim=0)
    level_count = len(spatial_shapes)
    return all_references.unsqueeze(0).unsqueeze(2).expand(-1, -1, level_count, -1)


class MSDeformableAttention(nn.Module):
    """Multi-scale deformable attention implemented with ``grid_sample``."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_levels: int = 3,
        n_points: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        self.value_proj = nn.Linear(d_model, d_model)
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value_list: List[torch.Tensor],
        spatial_shapes: List[Tuple[int, int]],
    ) -> torch.Tensor:
        batch_size, query_count, channels = query.shape
        head_count = self.n_heads
        level_count = self.n_levels
        point_count = self.n_points
        head_dim = self.head_dim

        offsets = self.sampling_offsets(query).view(
            batch_size,
            query_count,
            head_count,
            level_count,
            point_count,
            2,
        )
        weights = self.attention_weights(query).view(
            batch_size,
            query_count,
            head_count,
            level_count * point_count,
        )
        weights = F.softmax(weights, dim=-1).view(
            batch_size,
            query_count,
            head_count,
            level_count,
            point_count,
        )

        output = query.new_zeros(batch_size, query_count, head_count, head_dim)
        for level in range(level_count):
            height, width = spatial_shapes[level]
            values = value_list[level].permute(0, 2, 3, 1)
            values = self.value_proj(values)
            values = (
                values.view(
                    batch_size,
                    height,
                    width,
                    head_count,
                    head_dim,
                )
                .permute(0, 3, 4, 1, 2)
                .reshape(batch_size * head_count, head_dim, height, width)
            )

            reference = reference_points[:, :, level, :]
            normalizer = query.new_tensor([width, height])
            locations = (
                reference.unsqueeze(2).unsqueeze(3)
                + offsets[:, :, :, level] / normalizer
            )
            grid = (
                (2.0 * locations - 1.0)
                .permute(0, 2, 1, 3, 4)
                .reshape(
                    batch_size * head_count,
                    query_count,
                    point_count,
                    2,
                )
            )

            sampled = F.grid_sample(
                values,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled = sampled.view(
                batch_size,
                head_count,
                head_dim,
                query_count,
                point_count,
            ).permute(0, 3, 1, 4, 2)

            level_weights = weights[:, :, :, level, :].unsqueeze(-1)
            output = output + (level_weights * sampled).sum(dim=3)

        return self.output_proj(output.reshape(batch_size, query_count, channels))


class SPM(nn.Module):
    """Stride-4/8/16 features with the retained five-level checkpoint schema."""

    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 64,
        out_channels: int = 256,
    ):
        super().__init__()
        # Published checkpoints and optimizer groups contain this unused branch.
        # Keep its construction order (including RNG consumption), but do not
        # calculate a full-resolution feature that no supported head consumes.
        self.stem_s1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            LayerNorm2d(32),
            nn.GELU(),
        )
        self.stem_s2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                7,
                stride=2,
                padding=3,
                bias=False,
            ),
            LayerNorm2d(mid_channels),
            nn.GELU(),
        )
        self.stage1_s4 = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> List[torch.Tensor]:
        stride_2 = self.stem_s2(value)
        stride_4 = self.stage1_s4(stride_2)
        stride_8 = self.stage2(stride_4)
        stride_16 = self.stage3(stride_8)
        return [stride_4, stride_8, stride_16]


class CTIModule(nn.Module):
    """One bidirectional CNN-transformer interaction stage."""

    def __init__(
        self,
        d_model: int = 256,
        vit_dim: int = 1024,
        n_heads: int = 8,
        n_levels: int = 3,
        n_points: int = 4,
        ffn_dim: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.msda = MSDeformableAttention(d_model, n_heads, n_levels, n_points)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.alpha = nn.Parameter(torch.zeros(1))
        self.vit_to_adapter = nn.Linear(vit_dim, d_model)
        self.adapter_to_vit = nn.Linear(d_model, vit_dim)

    def forward(
        self,
        vit_tokens: torch.Tensor,
        cnn_features: List[torch.Tensor],
        grid_height: int,
        grid_width: int,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        batch_size = vit_tokens.shape[0]

        vit_projection = self.vit_to_adapter(vit_tokens)
        vit_image = vit_projection.view(
            batch_size, grid_height, grid_width, self.d_model
        ).permute(0, 3, 1, 2)
        cnn_features = list(cnn_features)
        cnn_features[2] = cnn_features[2] + vit_image

        spatial_shapes = [
            (feature.shape[2], feature.shape[3]) for feature in cnn_features
        ]
        query_parts = [
            feature.permute(0, 2, 3, 1).reshape(batch_size, -1, self.d_model)
            for feature in cnn_features
        ]
        query = torch.cat(query_parts, dim=1)

        reference_points = compute_reference_points(
            spatial_shapes, vit_tokens.device, vit_tokens.dtype
        ).expand(batch_size, -1, -1, -1)
        query = query + self.msda(
            self.norm1(query),
            reference_points,
            cnn_features,
            spatial_shapes,
        )
        query = query + self.ffn(self.norm2(query))

        updated_features = []
        offset = 0
        for height, width in spatial_shapes:
            token_count = height * width
            updated_features.append(
                query[:, offset : offset + token_count]
                .view(
                    batch_size,
                    height,
                    width,
                    self.d_model,
                )
                .permute(0, 3, 1, 2)
            )
            offset += token_count

        stride_16 = (
            updated_features[2]
            .permute(0, 2, 3, 1)
            .reshape(batch_size, -1, self.d_model)
        )
        updated_tokens = vit_tokens + self.alpha * self.adapter_to_vit(stride_16)
        return updated_tokens, updated_features


_HEAD_TYPE_CODE = {"flow": 0, "hovernet": 1}
_HEAD_TYPE_NAME = {value: name for name, value in _HEAD_TYPE_CODE.items()}
_HEAD_TYPE_NOUT = {"flow": 3, "hovernet": 4}


class DinoV3AdapterBackbone(nn.Module):
    """The sole DINOv3 adapter architecture supported by HistoForge."""

    STAGE_BOUNDARIES = (0, 6, 12, 18, 24)
    ADAPTER_DIM = 256
    ATTENTION_HEADS = 8
    ATTENTION_POINTS = 4
    FFN_DIM = 1024
    READOUT_STRIDE = 4

    def __init__(
        self,
        *,
        head_type: str,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if head_type not in _HEAD_TYPE_CODE:
            raise ValueError(
                f"head_type={head_type!r} must be one of {list(_HEAD_TYPE_CODE)}"
            )

        self.head_type = head_type
        self.register_buffer(
            "head_type_code",
            torch.tensor(_HEAD_TYPE_CODE[head_type], dtype=torch.long),
        )

        self.backbone = timm.create_model(
            DINOV3_MODEL,
            pretrained=False,
            num_classes=0,
        )
        if self.backbone.num_prefix_tokens != 5:
            raise RuntimeError(
                "DINOv3 ViT-L/16 must expose one CLS and four register tokens"
            )
        self.num_prefix = self.backbone.num_prefix_tokens
        self._grad_checkpointing = False

        self.register_buffer(
            "adapter_dim_buf",
            torch.tensor(self.ADAPTER_DIM, dtype=torch.long),
        )
        self.spm = SPM(3, 64, self.ADAPTER_DIM)
        self.cti_modules = nn.ModuleList(
            [
                CTIModule(
                    d_model=self.ADAPTER_DIM,
                    vit_dim=1024,
                    n_heads=self.ATTENTION_HEADS,
                    n_levels=3,
                    n_points=self.ATTENTION_POINTS,
                    ffn_dim=self.FFN_DIM,
                )
                for _ in range(4)
            ]
        )

        # ``ps`` is the retained Cellpose network interface name. It describes
        # the W2 readout stride, not the DINOv3 backbone's 16-pixel patch size.
        self.ps = self.READOUT_STRIDE
        self.nclasses = 0
        self.neck = nn.Sequential(
            nn.Conv2d(self.ADAPTER_DIM, 256, kernel_size=1, bias=False),
            LayerNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(256),
        )
        self.reset_segmentation_head(head_type)

        diameter = torch.tensor([30.0])
        self.diam_labels = nn.Parameter(diameter.clone(), requires_grad=False)
        self.diam_mean = nn.Parameter(diameter, requires_grad=False)
        self._dtype = dtype
        if dtype != torch.float32:
            self.to(dtype)

    def reset_segmentation_head(self, head_type: str) -> None:
        """Initialize a fresh task readout without changing encoder/adapter weights.

        Construction and explicit standalone task transfer share this tensor layout.
        Checkpoint loading never calls it to repair incomplete model state.
        """
        if head_type not in _HEAD_TYPE_CODE:
            raise ValueError(f"unsupported head_type={head_type!r}")
        reference = self.neck[0].weight
        self.head_type = head_type
        self.head_type_code.fill_(_HEAD_TYPE_CODE[head_type])
        self.nout = _HEAD_TYPE_NOUT[head_type]
        self.out = nn.Conv2d(256, self.nout * self.ps**2, kernel_size=1).to(
            device=reference.device, dtype=reference.dtype
        )
        self.W2 = nn.Parameter(
            torch.eye(
                self.nout * self.ps**2,
                device=reference.device,
                dtype=reference.dtype,
            ).reshape(self.nout * self.ps**2, self.nout, self.ps, self.ps),
            requires_grad=False,
        )

    def _readout(self, features: torch.Tensor) -> torch.Tensor:
        segmentation = self.out(features)
        segmentation = F.conv_transpose2d(
            segmentation,
            self.W2,
            stride=self.ps,
            padding=0,
        )
        if self.nclasses <= 0:
            return segmentation
        classification = self.out_class(features)
        classification = F.conv_transpose2d(
            classification,
            self.W3,
            stride=self.ps,
            padding=0,
        )
        return torch.cat((classification, segmentation), dim=1)

    def forward(self, value: torch.Tensor):
        batch_size = value.shape[0]
        cnn_features = self.spm(value)

        backbone = self.backbone
        patch_embedding = backbone.patch_embed(value)
        grid_height, grid_width = (
            patch_embedding.shape[1],
            patch_embedding.shape[2],
        )
        tokens, rope = backbone._pos_embed(patch_embedding)
        tokens = backbone.norm_pre(tokens)

        for stage_index in range(4):
            start = self.STAGE_BOUNDARIES[stage_index]
            end = self.STAGE_BOUNDARIES[stage_index + 1]
            for block_index in range(start, end):
                block = backbone.blocks[block_index]
                if self._grad_checkpointing and not torch.jit.is_scripting():
                    tokens = torch_checkpoint(
                        block,
                        tokens,
                        rope=rope,
                        use_reentrant=False,
                    )
                else:
                    tokens = block(tokens, rope=rope)

            prefix_tokens = tokens[:, : self.num_prefix, :]
            patch_tokens = tokens[:, self.num_prefix :, :]
            patch_tokens, cnn_features = self.cti_modules[stage_index](
                patch_tokens,
                cnn_features,
                grid_height,
                grid_width,
            )
            tokens = torch.cat([prefix_tokens, patch_tokens], dim=1)

        features = self.neck(cnn_features[0])
        output = self._readout(features)
        style = torch.zeros((batch_size, 256), device=value.device)
        return output, style

    def gradient_checkpointing_enable(self) -> None:
        self._grad_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self._grad_checkpointing = False

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @dtype.setter
    def dtype(self, value: torch.dtype) -> None:
        if self._dtype != value:
            self.to(value)
            self._dtype = value

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

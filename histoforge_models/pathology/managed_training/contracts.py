"""Scientific types for managed flow/HoVer losses and objective combination."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from torch import Tensor

ClassificationObjective = Literal[
    "cross_entropy",
    "tversky",
    "cross_entropy+tversky",
]
FlowLossWeighting = Literal["pixel", "equal_instance"]
HeadType = Literal["flow", "hovernet"]


@dataclass(frozen=True)
class ManagedLossResult:
    """Differentiable sums over equally weighted fields in one microbatch.

    The task owns within-field reduction and combination into ``total_sum``.
    ``field_count`` counts actual fields, not pixels or nominal batch capacity.
    Diagnostics may overlap; the optimizer must not sum ``component_sums``.
    """

    total_sum: Tensor
    task_sums: Mapping[str, Tensor]
    component_sums: Mapping[str, Tensor]
    field_count: int

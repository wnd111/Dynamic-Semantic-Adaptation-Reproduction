from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Protocol

import torch
from torch import Tensor, nn

from .anchor import AnchorApproximator, AnchorBank, ApproximationDecision
from .feedback import EndToEndFeedback
from .precision import PrecisionPath, PrecisionScheduler, precision_score
from .pruning import MaskDecision, MaskSelector, ProgressiveRefresh
from .signals import ppr_divergence


class FullBlock(Protocol):
    def __call__(
        self,
        hidden_state: Tensor,
        selected_key_indices: Tensor,
        precision: PrecisionPath,
    ) -> Tensor: ...


@dataclass(frozen=True)
class LayerContext:
    layer_idx: int
    hidden_state: Tensor
    kv_length: int
    importance: Tensor
    complexity_score: float
    confidence: float
    uncertainty: float
    ambiguity: float
    audit_error: float | None = None
    precision_changes: int = 0
    force_full: bool = False
    stored_complexity: float | None = None


@dataclass(frozen=True)
class LayerTrace:
    layer_idx: int
    path: Literal["approximate", "full"]
    reason: str
    similarity: float
    drift: float
    corrected: bool
    complexity_score: float
    mask_mode: str
    selected_keys: int
    total_keys: int
    precision: str
    chain_depth: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LayerResult:
    layer_idx: int
    hidden_state: Tensor
    trace: LayerTrace
    full_computed: bool


@dataclass(frozen=True)
class LayerPlan:
    layer_idx: int
    path: Literal["approximate", "full"]
    input_hidden_state: Tensor
    approximated_hidden_state: Tensor | None
    mask: MaskDecision
    precision: PrecisionPath
    reason: str
    trace: LayerTrace

    def execute(self, full_block: FullBlock) -> LayerResult:
        if self.path == "approximate":
            if self.approximated_hidden_state is None:
                raise RuntimeError("approximate plan is missing an approximated hidden state")
            output = self.approximated_hidden_state
            full_computed = False
        else:
            output = full_block(self.input_hidden_state, self.mask.indices, self.precision)
            if not isinstance(output, Tensor):
                raise TypeError("full_block must return a torch.Tensor")
            full_computed = True
        return LayerResult(
            layer_idx=self.layer_idx,
            hidden_state=output,
            trace=self.trace,
            full_computed=full_computed,
        )


@dataclass
class RuntimeState:
    step: int = 0
    audit_count: int = 0
    force_full_next_step: bool = False
    force_full_current_step: bool = False
    path_changes: int = 0
    approximation_count: int = 0
    full_count: int = 0
    last_precision_by_layer: dict[int, PrecisionPath] = field(default_factory=dict)
    output_error: float = 0.3


class RuntimeController(nn.Module):
    """Algorithm 1: one ordered policy for approximation, pruning, and precision."""

    def __init__(
        self,
        *,
        num_layers: int,
        approximator: AnchorApproximator,
        mask_selector: MaskSelector,
        feedback: EndToEndFeedback,
        refresh: ProgressiveRefresh,
        e2e_interval: int = 100,
    ) -> None:
        super().__init__()
        if num_layers < 1 or e2e_interval < 1:
            raise ValueError("num_layers and e2e_interval must be positive")
        self.num_layers = num_layers
        self.approximator = approximator
        self.mask_selector = mask_selector
        self.feedback = feedback
        self.refresh = refresh
        self.e2e_interval = e2e_interval
        self.anchor_bank = AnchorBank(num_layers)
        self.state = RuntimeState()
        # Explicit switches make the paper's component ablations reproducible
        # without replacing the controller or silently changing other policy.
        self.approximation_enabled = True
        self.pruning_enabled = True
        self.precision_override: PrecisionPath | None = None

    def begin_step(self) -> None:
        self.state.force_full_current_step = self.state.force_full_next_step
        self.state.force_full_next_step = False

    def _record_precision(self, layer_idx: int, precision: PrecisionPath) -> None:
        previous = self.state.last_precision_by_layer.get(layer_idx)
        if previous is not None and previous is not precision:
            self.state.path_changes += 1
        self.state.last_precision_by_layer[layer_idx] = precision

    def _trace(
        self,
        context: LayerContext,
        *,
        path: Literal["approximate", "full"],
        reason: str,
        mask: MaskDecision,
        precision: PrecisionPath,
        approximation: ApproximationDecision | None,
    ) -> LayerTrace:
        return LayerTrace(
            layer_idx=context.layer_idx,
            path=path,
            reason=reason,
            similarity=approximation.similarity if approximation else 0.0,
            drift=approximation.drift if approximation else 0.0,
            corrected=approximation.corrected if approximation else False,
            complexity_score=float(context.complexity_score),
            mask_mode=mask.mode,
            selected_keys=int(mask.indices.numel()),
            total_keys=mask.total_keys,
            precision=precision.value,
            chain_depth=approximation.chain_depth if approximation else 0,
        )

    def enter_layer(self, context: LayerContext) -> LayerPlan:
        if not 0 <= context.layer_idx < self.num_layers:
            raise ValueError("layer_idx outside controller range")
        if context.kv_length < 1:
            raise ValueError("kv_length must be positive")

        complexity_fallback = (
            context.stored_complexity is not None
            and self.feedback.fallback_requested(
                stored_complexity=float(context.stored_complexity),
                current_complexity=float(context.complexity_score),
            )
        )
        common_fallback = (
            context.force_full or self.state.force_full_current_step or complexity_fallback
        )
        if common_fallback:
            mask = MaskDecision.full(
                context.kv_length,
                device=context.importance.device,
                reason="common_fallback",
            )
            precision = PrecisionPath.FP32_ACC
            self._record_precision(context.layer_idx, precision)
            trace = self._trace(
                context,
                path="full",
                reason="common_fallback",
                mask=mask,
                precision=precision,
                approximation=None,
            )
            return LayerPlan(
                layer_idx=context.layer_idx,
                path="full",
                input_hidden_state=context.hidden_state,
                approximated_hidden_state=None,
                mask=mask,
                precision=precision,
                reason="common_fallback",
                trace=trace,
            )

        if self.approximation_enabled:
            approximation = self.approximator.try_approximate(
                context.layer_idx,
                context.hidden_state,
                self.anchor_bank,
            )
        else:
            approximation = ApproximationDecision(
                accepted=False,
                hidden_state=context.hidden_state,
                use_full_layer=True,
                similarity=0.0,
                drift=0.0,
                corrected=False,
                chain_depth=0,
                anchor_layer=None,
                reason="approximation_disabled",
            )
        if approximation.accepted:
            mask = MaskDecision.full(
                context.kv_length,
                device=context.importance.device,
                reason="layer_approximated",
            )
            precision = PrecisionPath.FP16
            self._record_precision(context.layer_idx, precision)
            self.state.approximation_count += 1
            trace = self._trace(
                context,
                path="approximate",
                reason=approximation.reason,
                mask=mask,
                precision=precision,
                approximation=approximation,
            )
            return LayerPlan(
                layer_idx=context.layer_idx,
                path="approximate",
                input_hidden_state=context.hidden_state,
                approximated_hidden_state=approximation.hidden_state,
                mask=mask,
                precision=precision,
                reason=approximation.reason,
                trace=trace,
            )

        if self.pruning_enabled:
            mask = self.mask_selector.select(
                torch.tensor(context.complexity_score),
                context.importance,
                context.kv_length,
            )
            mask = self.refresh.update(
                step=self.state.step,
                audit_error=context.audit_error,
                decision=mask,
                precision_changes=context.precision_changes,
            )
        else:
            mask = MaskDecision.full(
                context.kv_length,
                device=context.importance.device,
                reason="pruning_disabled",
            )
        score = precision_score(
            uncertainty=torch.tensor(context.uncertainty),
            ambiguity=torch.tensor(context.ambiguity),
            confidence=torch.tensor(context.confidence),
        )
        effective = self.feedback.effective_thresholds(context.complexity_score)
        precision = self.precision_override or PrecisionScheduler(effective).select(score)
        self._record_precision(context.layer_idx, precision)
        self.state.full_count += 1
        trace = self._trace(
            context,
            path="full",
            reason=approximation.reason,
            mask=mask,
            precision=precision,
            approximation=approximation,
        )
        return LayerPlan(
            layer_idx=context.layer_idx,
            path="full",
            input_hidden_state=context.hidden_state,
            approximated_hidden_state=None,
            mask=mask,
            precision=precision,
            reason=approximation.reason,
            trace=trace,
        )

    def finish_layer(self, result: LayerResult, kv: object) -> None:
        if result.full_computed:
            self.anchor_bank.stage_full(
                layer_idx=result.layer_idx,
                hidden_state=result.hidden_state,
                kv=kv,
                chain_depth=0,
            )

    def finish_step(
        self,
        controller_distribution: Tensor | None = None,
        full_distribution: Tensor | None = None,
        complexity: float = 0.5,
    ) -> None:
        if (controller_distribution is None) != (full_distribution is None):
            raise ValueError("controller and full distributions must be provided together")
        if controller_distribution is not None and full_distribution is not None:
            updated = ppr_divergence(
                controller_distribution,
                full_distribution,
                previous=self.state.output_error,
            )
            self.state.output_error = float(updated.mean().detach().cpu())
            self.feedback.observe(error=self.state.output_error, complexity=complexity)
            if self.feedback.fallback_requested(complexity, complexity):
                self.state.force_full_next_step = True

        self.anchor_bank.release_after_step()
        self.state.step += 1
        if self.state.step % self.e2e_interval == 0:
            self.state.audit_count += 1
            self.state.force_full_next_step = True
        self.state.force_full_current_step = False
        self.state.path_changes = 0

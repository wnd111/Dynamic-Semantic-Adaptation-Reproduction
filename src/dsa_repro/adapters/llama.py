from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import replace

import torch
import transformers
from torch import Tensor
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from ..anchor import AnchorApproximator
from ..config import ExperimentConfig
from ..controller import LayerContext, RuntimeController, RuntimeState
from ..feedback import EndToEndFeedback
from ..generation import GenerationRecord
from ..precision import PrecisionPath, fake_quantize
from ..pruning import MaskSelector, ProgressiveRefresh
from ..training import AuxiliaryModules

SignalProvider = Callable[[int, Tensor], tuple[float, float, float, float]]


class DSALlamaAdapter:
    """Single-request LLaMA execution with real DSA layer skipping and resident K/V."""

    def __init__(
        self,
        *,
        model: LlamaForCausalLM,
        controller: RuntimeController,
        backend: str,
        result_kind: str,
        signal_provider: SignalProvider | None = None,
        recency_weight: float = 0.3,
    ) -> None:
        self.model = model.requires_grad_(False).eval()
        self.controller = controller.eval()
        self.backend = backend
        self.result_kind = result_kind
        self.signal_provider = signal_provider or self._neutral_signals
        self.recency_weight = float(recency_weight)
        self._importance: dict[int, Tensor] = {}

    @staticmethod
    def _neutral_signals(layer_idx: int, hidden: Tensor) -> tuple[float, float, float, float]:
        del layer_idx
        dispersion_tensor = torch.nan_to_num(hidden.float(), nan=0.0, posinf=1.0, neginf=-1.0)
        dispersion = float(dispersion_tensor.std(unbiased=False).detach().cpu())
        complexity = float(torch.sigmoid(torch.tensor(dispersion - 1.0)).item())
        return complexity, 0.5, 0.5, 0.0

    @staticmethod
    def _make_controller(
        model: LlamaForCausalLM,
        experiment: ExperimentConfig | None = None,
    ) -> RuntimeController:
        config = model.config
        gate_threshold = experiment.anchor.gate_threshold if experiment else 0.85
        drift_threshold = experiment.anchor.drift_threshold if experiment else 0.15
        max_chain = experiment.anchor.max_chain if experiment else 3
        projection_dim = experiment.anchor.projection_dim if experiment else 128
        approximator = AnchorApproximator(
            hidden_size=config.hidden_size,
            projection_dim=min(projection_dim, config.hidden_size),
            gate_threshold=gate_threshold,
            drift_threshold=drift_threshold,
            max_chain=max_chain,
        )
        mask_selector = (
            MaskSelector(
                high_threshold=experiment.pruning.high_threshold,
                low_threshold=experiment.pruning.low_threshold,
                topk_fraction=experiment.pruning.topk_fraction,
                window_fraction=experiment.pruning.window_fraction,
            )
            if experiment
            else MaskSelector.default()
        )
        feedback = (
            EndToEndFeedback(
                thresholds=experiment.precision.thresholds,
                target_band=experiment.feedback.target_band,
                momentum=experiment.feedback.momentum,
                step_size=experiment.feedback.step_size,
                complexity_margin=experiment.feedback.complexity_margin,
            )
            if experiment
            else EndToEndFeedback.default()
        )
        controller = RuntimeController(
            num_layers=config.num_hidden_layers,
            approximator=approximator,
            mask_selector=mask_selector,
            feedback=feedback,
            refresh=ProgressiveRefresh(
                interval=experiment.pruning.refresh_interval if experiment else 20,
                precision_change_limit=(
                    experiment.pruning.refresh_precision_changes if experiment else 4
                ),
                error_sq_limit=experiment.pruning.refresh_error_sq if experiment else 0.1,
            ),
            e2e_interval=experiment.feedback.e2e_interval if experiment else 100,
        )
        if experiment:
            controller.state.output_error = experiment.feedback.default_eout
        return controller

    @classmethod
    def from_random_tiny(
        cls,
        *,
        seed: int,
        num_hidden_layers: int = 2,
    ) -> DSALlamaAdapter:
        torch.manual_seed(seed)
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            attention_dropout=0.0,
        )
        config._attn_implementation = "eager"
        model = LlamaForCausalLM(config)
        return cls(
            model=model,
            controller=cls._make_controller(model),
            backend="tiny_reference",
            result_kind="synthetic_smoke",
        )

    @classmethod
    def from_pretrained(
        cls,
        config: ExperimentConfig,
        token: str | None,
    ) -> DSALlamaAdapter:
        if not token:
            raise RuntimeError("HF_TOKEN is required for meta-llama/Llama-2-7b-hf")
        dtype = torch.float16 if config.model.dtype == "float16" else torch.bfloat16
        kwargs: dict[str, object] = {
            "token": token,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",
        }
        model = LlamaForCausalLM.from_pretrained(config.model.model_id, **kwargs)
        if torch.cuda.is_available():
            model = model.to("cuda")
        return cls(
            model=model,
            controller=cls._make_controller(model, config),
            backend=config.model.backend,
            result_kind="measured",
            recency_weight=config.pruning.recency_weight,
        )

    def install_auxiliary(self, auxiliary: AuxiliaryModules) -> None:
        """Bind a trained auxiliary checkpoint to every runtime decision path."""
        parameter = next(self.model.parameters())
        auxiliary = auxiliary.to(device=parameter.device, dtype=parameter.dtype).eval()
        approximator = self.controller.approximator
        approximator.projector = auxiliary.projection
        approximator.mapper = auxiliary.residual_mapper
        approximator.corrector = auxiliary.residual_corrector
        approximator.drift_detector = auxiliary.drift_detector
        self.auxiliary = auxiliary

        def trained_signals(
            layer_idx: int,
            hidden: Tensor,
        ) -> tuple[float, float, float, float]:
            del layer_idx
            with torch.inference_mode():
                complexity = auxiliary.complexity_predictor(hidden).mean()
                predicted = auxiliary.complexity_predictor.predict_signals(hidden)
                confidence, uncertainty = auxiliary.confidence_estimator(hidden)
            return (
                float(complexity.detach().float().cpu()),
                float(confidence.mean().detach().float().cpu()),
                float(uncertainty.mean().detach().float().cpu()),
                float(predicted.information_gain.mean().detach().float().cpu()),
            )

        self.signal_provider = trained_signals

    def reset_runtime(self) -> None:
        """Reset per-request policy history between variants or benchmark trials."""
        self.controller.anchor_bank.clear()
        self.controller.state = RuntimeState()
        self._importance.clear()

    @staticmethod
    def apply_selected_key_mask(base_mask: Tensor, selected: Tensor) -> Tensor:
        if base_mask.ndim != 4:
            raise ValueError("base_mask must have shape [batch, 1, query, key]")
        key_count = base_mask.shape[-1]
        if selected.ndim != 1 or torch.any(selected < 0) or torch.any(selected >= key_count):
            raise ValueError("selected key indices are outside the attention mask")
        allowed = torch.zeros(key_count, dtype=torch.bool, device=base_mask.device)
        allowed[selected.long()] = True
        masked = base_mask.clone()
        query_count = base_mask.shape[-2]
        for query_index in range(query_count):
            row_allowed = allowed.clone()
            finite = torch.isfinite(base_mask[0, 0, query_index])
            finite_indices = torch.nonzero(finite, as_tuple=False).flatten()
            if finite_indices.numel() == 0:
                raise ValueError("base attention mask contains a query with no causal key")
            row_allowed[finite_indices[-1]] = True
            masked[:, :, query_index, ~row_allowed] = float("-inf")
        return masked

    @staticmethod
    def _causal_mask(
        batch_size: int,
        query_length: int,
        past_length: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        key_length = past_length + query_length
        query_positions = torch.arange(
            past_length,
            past_length + query_length,
            device=device,
        ).unsqueeze(-1)
        key_positions = torch.arange(key_length, device=device).unsqueeze(0)
        blocked = key_positions > query_positions
        mask = torch.zeros(query_length, key_length, dtype=dtype, device=device)
        mask.masked_fill_(blocked, float("-inf"))
        return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

    @staticmethod
    def _apply_precision(hidden: Tensor, path: PrecisionPath) -> Tensor:
        if path is PrecisionPath.INT8:
            return fake_quantize(hidden, bits=8).dequantize(hidden.dtype)
        if path is PrecisionPath.INT4:
            return fake_quantize(hidden, bits=4).dequantize(hidden.dtype)
        return hidden

    @staticmethod
    def _rotary(
        attention: torch.nn.Module,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        position_ids: Tensor,
        total_length: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rotary_signature = inspect.signature(attention.rotary_emb.forward)
        if "position_ids" in rotary_signature.parameters:
            cos, sin = attention.rotary_emb(value, position_ids)
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
        else:
            cos, sin = attention.rotary_emb(value, seq_len=total_length)
            query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)
        return query, key, cos, sin

    def _synthesize_kv(
        self,
        *,
        layer_idx: int,
        hidden_state: Tensor,
        cache: DynamicCache,
        position_ids: Tensor,
        cache_position: Tensor,
    ) -> None:
        attention = self.model.model.layers[layer_idx].self_attn
        batch, sequence, _ = hidden_state.shape
        query = (
            attention.q_proj(hidden_state)
            .view(batch, sequence, attention.num_heads, attention.head_dim)
            .transpose(1, 2)
        )
        key = (
            attention.k_proj(hidden_state)
            .view(batch, sequence, attention.num_key_value_heads, attention.head_dim)
            .transpose(1, 2)
        )
        value = (
            attention.v_proj(hidden_state)
            .view(batch, sequence, attention.num_key_value_heads, attention.head_dim)
            .transpose(1, 2)
        )
        total_length = cache.get_seq_length(layer_idx) + sequence
        _, key, cos, sin = self._rotary(
            attention,
            query,
            key,
            value,
            position_ids,
            total_length,
        )
        cache.update(
            key,
            value,
            layer_idx,
            {"sin": sin, "cos": cos, "cache_position": cache_position},
        )

    def _position_embeddings(
        self, hidden: Tensor, position_ids: Tensor
    ) -> tuple[Tensor, Tensor] | None:
        rotary = getattr(self.model.model, "rotary_emb", None)
        if rotary is None:
            return None
        return rotary(hidden, position_ids)

    def _forward_step(
        self, input_ids: Tensor, cache: DynamicCache
    ) -> tuple[Tensor, list[dict[str, object]]]:
        core = self.model.model
        hidden = core.embed_tokens(input_ids)
        batch, query_length = input_ids.shape
        past_length = cache.get_seq_length(0)
        cache_position = torch.arange(
            past_length,
            past_length + query_length,
            device=input_ids.device,
        )
        position_ids = cache_position.unsqueeze(0).expand(batch, -1)
        base_mask = self._causal_mask(
            batch,
            query_length,
            past_length,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        position_embeddings = self._position_embeddings(hidden, position_ids)
        traces: list[dict[str, object]] = []
        self.controller.begin_step()

        for layer_idx, layer in enumerate(core.layers):
            key_length = past_length + query_length
            mean_attention = self._importance.get(layer_idx)
            if mean_attention is None or mean_attention.numel() != key_length:
                mean_attention = torch.full(
                    (key_length,),
                    1.0 / key_length,
                    device=hidden.device,
                    dtype=torch.float32,
                )
            importance = MaskSelector.importance(
                mean_attention,
                recency_weight=self.recency_weight,
            )
            complexity, confidence, uncertainty, ambiguity = self.signal_provider(layer_idx, hidden)
            context = LayerContext(
                layer_idx=layer_idx,
                hidden_state=hidden,
                kv_length=key_length,
                importance=importance,
                complexity_score=complexity,
                confidence=confidence,
                uncertainty=uncertainty,
                ambiguity=ambiguity,
                precision_changes=self.controller.state.path_changes,
            )
            plan = self.controller.enter_layer(context)
            observed_attention: Tensor | None = None

            def full_block(
                block_hidden: Tensor,
                selected_keys: Tensor,
                precision: PrecisionPath,
                target_layer: torch.nn.Module = layer,
            ) -> Tensor:
                nonlocal observed_attention
                block_hidden = self._apply_precision(block_hidden, precision)
                layer_mask = self.apply_selected_key_mask(base_mask, selected_keys)
                kwargs: dict[str, object] = {
                    "hidden_states": block_hidden,
                    "attention_mask": layer_mask,
                    "position_ids": position_ids,
                    "past_key_value": cache,
                    "output_attentions": True,
                    "use_cache": True,
                    "cache_position": cache_position,
                }
                if position_embeddings is not None:
                    kwargs["position_embeddings"] = position_embeddings
                outputs = target_layer(**kwargs)
                if len(outputs) > 1 and isinstance(outputs[1], Tensor):
                    observed_attention = outputs[1]
                return outputs[0]

            result = plan.execute(full_block)
            if plan.path == "approximate":
                self._synthesize_kv(
                    layer_idx=layer_idx,
                    hidden_state=result.hidden_state,
                    cache=cache,
                    position_ids=position_ids,
                    cache_position=cache_position,
                )
            if observed_attention is not None:
                self._importance[layer_idx] = (
                    observed_attention[:, :, -1, :].float().mean(dim=(0, 1)).detach()
                )
            anchor_result = replace(
                result,
                hidden_state=result.hidden_state[:, -1:, :],
            )
            self.controller.finish_layer(anchor_result, kv=cache)
            hidden = result.hidden_state
            traces.append(result.trace.to_dict())

        hidden = core.norm(hidden)
        logits = self.model.lm_head(hidden)
        self.controller.finish_step()
        return logits, traces

    @torch.inference_mode()
    def generate_ids(self, prompt_ids: list[int], max_new_tokens: int) -> GenerationRecord:
        if not prompt_ids:
            raise ValueError("prompt_ids cannot be empty")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        device = next(self.model.parameters()).device
        current = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        cache = DynamicCache()
        generated: list[int] = []
        traces: list[dict[str, object]] = []
        for _ in range(max_new_tokens):
            logits, step_traces = self._forward_step(current, cache)
            next_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            generated.append(next_token)
            traces.extend(step_traces)
            current = torch.tensor([[next_token]], dtype=torch.long, device=device)
        return GenerationRecord(
            token_ids=generated,
            traces=traces,
            backend=self.backend,
            result_kind=self.result_kind,
            transformers_version=transformers.__version__,
        )

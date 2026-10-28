# 
# Copyright (c) 2026 Manish. All rights reserved.
# 
# This work is licensed under the terms of the MIT license.  
# For a copy, see <https://opensource.org/licenses/MIT>.
# 

"""
Transformer Attention Surgery and Layer-Wise Distillation Module.

Provides automated cross-architecture attention bypassing to replace quadratic
softmax attention layers with linear associative sub-quadratic attention,
paired with a Two-Stage Distillation Trainer (Stage 1: Layer-wise hidden-state
alignment, Stage 2: End-to-end soft logit distillation).

Features:
- Grouped-Query Attention (GQA) & Multi-Query Attention (MQA) support
- Drop-in tuple contract matching for Hugging Face Transformers (Whisper, Llama, Mistral, Bert)
- Padding mask state protection (attention_mask)
- Bidirectional context scanning for non-causal encoder blocks
- Persistent hook management, mixed precision (AMP), and gradient clipping in distillation
"""

import math
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from subq_audio.autograd import SubQLinearAttention


class SubQAttentionOutput(tuple):
    """
    Drop-in Attention Output.
    Behaves simultaneously as:
    1. A tuple: supports `out, _ = res` (2-tuple) or `out, _, past = res` (when return_tuple_arity == 3).
    2. A Tensor delegate: supports `x + res`, `res + x`, `.shape`, `.dtype`, `.device`, `.view()`, etc.
    """
    def __new__(
        cls,
        output: torch.Tensor,
        attn_weights: Optional[torch.Tensor] = None,
        past_state: Optional[torch.Tensor] = None,
        return_tuple_arity: int = 2,
    ):
        if return_tuple_arity == 3:
            obj = super().__new__(cls, (output, attn_weights, past_state))
        else:
            obj = super().__new__(cls, (output, attn_weights))
        obj.past_state = past_state
        return obj

    @property
    def output(self) -> torch.Tensor:
        return self[0]

    def __add__(self, other):
        return self[0] + other

    def __radd__(self, other):
        return other + self[0]

    def __sub__(self, other):
        return self[0] - other

    def __rsub__(self, other):
        return other - self[0]

    def __mul__(self, other):
        return self[0] * other

    def __rmul__(self, other):
        return other * self[0]

    def __truediv__(self, other):
        return self[0] / other

    def __rtruediv__(self, other):
        return other / self[0]

    def __getattr__(self, name):
        return getattr(self[0], name)


class SubQBypassAttention(nn.Module):
    """
    Adapter wrapper that houses SubQLinearAttention with a signature and return contract
    strictly compatible with Hugging Face Transformer attention blocks (Llama, Whisper, Bert, etc.).
    """

    def __init__(self, core_attn: SubQLinearAttention, return_tuple_arity: int = 2):
        super().__init__()
        self.core = core_attn
        self.hidden_size = core_attn.hidden_size
        self.num_heads = core_attn.num_heads
        self.num_key_value_heads = core_attn.num_key_value_heads
        self.head_dim = core_attn.head_dim
        self.is_causal = core_attn.is_causal
        self.return_tuple_arity = return_tuple_arity

        # Expose sub-modules for direct attribute access
        self.q_proj = core_attn.q_proj
        self.k_proj = core_attn.k_proj
        self.v_proj = core_attn.v_proj
        self.out_proj = core_attn.out_proj
        self.gate_proj = core_attn.gate_proj
        self.raw_decay = core_attn.raw_decay

    @property
    def decay_factor(self) -> torch.Tensor:
        return self.core.decay_factor

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> SubQAttentionOutput:
        """
        Drop-in forward pass compatible with Hugging Face Attention call sites.
        Returns SubQAttentionOutput satisfying:
        - 2-tuple unpacking (`hidden_states, _ = self.self_attn(...)`)
        - 3-tuple unpacking when return_tuple_arity == 3 or past_key_value requested
        - Direct Tensor arithmetic operations (`hidden_states = residual + attn_out`)
        """
        past_state = past_key_value if isinstance(past_key_value, torch.Tensor) else None

        output, next_state = self.core(
            hidden_states=hidden_states,
            past_state=past_state,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )

        ret_past = next_state if (self.return_tuple_arity == 3 or (use_cache and next_state is not None)) else None
        return SubQAttentionOutput(
            output,
            None,
            ret_past,
            return_tuple_arity=self.return_tuple_arity,
        )


def _find_linear_submodule(module: nn.Module, candidate_names: Tuple[str, ...]) -> Optional[Tuple[str, nn.Linear]]:
    """Helper to locate linear projections matching standard naming schemes."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name.lower() in candidate_names:
            return name, child
    return None


def bypass_transformer_with_subq(
    model: nn.Module,
    target_select_suffixes: Tuple[str, ...] = ("self_attn", "attention", "attn"),
    decay_init: float = 0.99,
    use_data_dependent_gating: bool = True,
    copy_weights: bool = True,
    verbose: bool = False,
) -> Tuple[nn.Module, List[Dict[str, Any]]]:
    """
    Performs cross-architecture surgery on a PyTorch Transformer model, replacing
    softmax self-attention layers with Sub-Quadratic Linear Attention modules.

    Supports:
    - Grouped-Query Attention (GQA / MQA) with exact projection weight transfer
    - Causal decoders and bidirectional encoders (WhisperEncoder, Bert)
    - Drop-in tuple contracts matching Hugging Face layers without unpacking crashes
    - Zero-initialized dynamic gates for smooth warm-start identity initialization

    Args:
        model: PyTorch Transformer model (Hugging Face or custom).
        target_select_suffixes: Module name suffixes indicating attention layers.
        decay_init: Initial value for the base temporal decay factor in (0, 1).
        use_data_dependent_gating: Whether to enable dynamic x_t projection for decay.
        copy_weights: Whether to copy teacher projection weights in-place.
        verbose: If True, prints telemetry for each modified module.

    Returns:
        Tuple of (modified_model, list_of_replaced_module_metadata).
    """
    replaced_info: List[Dict[str, Any]] = []

    q_candidates = ("q_proj", "query", "q", "in_proj_q")
    k_candidates = ("k_proj", "key", "k", "in_proj_k")
    v_candidates = ("v_proj", "value", "v", "in_proj_v")
    o_candidates = ("out_proj", "o_proj", "dense", "proj", "c_proj")

    def _recursive_replace(current_module: nn.Module, prefix: str = ""):
        for child_name, child_module in list(current_module.named_children()):
            full_path = f"{prefix}.{child_name}" if prefix else child_name

            # Skip if already a subq attention module
            if isinstance(child_module, (SubQLinearAttention, SubQBypassAttention)):
                continue

            is_target_suffix = any(
                child_name.lower().endswith(suffix.lower())
                for suffix in target_select_suffixes
            )

            q_found = _find_linear_submodule(child_module, q_candidates)
            k_found = _find_linear_submodule(child_module, k_candidates)
            v_found = _find_linear_submodule(child_module, v_candidates)
            o_found = _find_linear_submodule(child_module, o_candidates)

            # Match if module name matches target suffixes (or if no suffixes given, any module with 4 projections)
            matches_criteria = is_target_suffix if target_select_suffixes else True

            # If matching criteria is satisfied and all 4 linear projections are found, perform surgery
            if matches_criteria and q_found and k_found and v_found and o_found:

                q_name, orig_q = q_found
                k_name, orig_k = k_found
                v_name, orig_v = v_found
                o_name, orig_o = o_found

                hidden_size = orig_q.in_features
                q_out = orig_q.out_features
                k_out = orig_k.out_features
                device = orig_q.weight.device
                dtype = orig_q.weight.dtype

                # Infer number of query attention heads
                num_heads = getattr(child_module, "num_heads", None) or \
                            getattr(child_module, "num_attention_heads", None) or \
                            getattr(child_module, "n_heads", None)
                if num_heads is None and hasattr(model, "config"):
                    num_heads = getattr(model.config, "num_attention_heads", None) or \
                                getattr(model.config, "encoder_attention_heads", None)
                if num_heads is None:
                    num_heads = 1

                if hidden_size % num_heads != 0:
                    head_dim = q_out // num_heads
                else:
                    head_dim = hidden_size // num_heads

                # Infer GQA / MQA key-value heads
                num_key_value_heads = getattr(child_module, "num_key_value_heads", None) or \
                                      getattr(child_module, "num_key_value_groups", None)
                if num_key_value_heads is None and hasattr(model, "config"):
                    num_key_value_heads = getattr(model.config, "num_key_value_heads", None)
                if num_key_value_heads is None:
                    num_key_value_heads = k_out // head_dim if head_dim > 0 else num_heads

                if num_key_value_heads <= 0 or num_heads % num_key_value_heads != 0:
                    num_key_value_heads = num_heads

                # Infer causality (decoder vs encoder)
                is_causal = getattr(child_module, "is_causal", None)
                if is_causal is None:
                    is_decoder = getattr(child_module, "is_decoder", None)
                    if is_decoder is False:
                        is_causal = False
                    else:
                        full_path_lower = full_path.lower()
                        class_name_lower = child_module.__class__.__name__.lower()
                        if "encoder" in full_path_lower or "bert" in full_path_lower or "encoder" in class_name_lower:
                            is_causal = False
                        else:
                            is_causal = True

                has_bias = (orig_q.bias is not None)

                # Instantiate SubQ attention core with matching GQA dimensions
                core_attn = SubQLinearAttention(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_key_value_heads=num_key_value_heads,
                    decay_init=decay_init,
                    bias=has_bias,
                    use_data_dependent_gating=use_data_dependent_gating,
                    is_causal=is_causal,
                ).to(device=device, dtype=dtype)

                if copy_weights:
                    with torch.no_grad():
                        core_attn.q_proj.weight.copy_(orig_q.weight)
                        core_attn.k_proj.weight.copy_(orig_k.weight)
                        core_attn.v_proj.weight.copy_(orig_v.weight)
                        core_attn.out_proj.weight.copy_(orig_o.weight)

                        if has_bias:
                            if orig_q.bias is not None and core_attn.q_proj.bias is not None:
                                core_attn.q_proj.bias.copy_(orig_q.bias)
                            if orig_k.bias is not None and core_attn.k_proj.bias is not None:
                                core_attn.k_proj.bias.copy_(orig_k.bias)
                            if orig_v.bias is not None and core_attn.v_proj.bias is not None:
                                core_attn.v_proj.bias.copy_(orig_v.bias)
                            if orig_o.bias is not None and core_attn.out_proj.bias is not None:
                                core_attn.out_proj.bias.copy_(orig_o.bias)

                        if core_attn.gate_proj is not None:
                            core_attn.gate_proj.weight.zero_()
                            if core_attn.gate_proj.bias is not None:
                                core_attn.gate_proj.bias.zero_()

                # Wrap with HF drop-in adapter (2-tuple contract)
                replacement = SubQBypassAttention(core_attn, return_tuple_arity=2)
                setattr(current_module, child_name, replacement)

                info = {
                    "path": full_path,
                    "hidden_size": hidden_size,
                    "num_heads": num_heads,
                    "num_key_value_heads": num_key_value_heads,
                    "is_causal": is_causal,
                    "has_bias": has_bias,
                }
                replaced_info.append(info)

                if verbose:
                    print(
                        f"[Surgery] Replaced {full_path} -> SubQLinearAttention "
                        f"(H={num_heads}, KV_H={num_key_value_heads}, D={hidden_size}, causal={is_causal})"
                    )

            else:
                _recursive_replace(child_module, full_path)

    _recursive_replace(model)
    return model, replaced_info


class LayerWiseDistillationTrainer:
    """
    Two-Stage Distillation Trainer for converting standard Transformers to
    Sub-Quadratic Linear Attention models.

    Stage 1: Layer-wise hidden-state alignment loss (MSE + Cosine Distance).
             The student backbone is frozen, optimizing only the SubQ attention parameters.
    Stage 2: End-to-end soft logit distillation with Kullback-Leibler (KL) divergence
             and optional task cross-entropy.

    Features:
    - Persistent forward hook registration with context-managed lifecycle cleanup
    - Mixed precision (Automatic Mixed Precision - AMP) integration
    - Gradient clipping for numerical stability
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        lambda_cos: float = 0.5,
        temperature: float = 2.0,
        alpha_kd: float = 0.5,
        use_amp: bool = True,
        max_grad_norm: float = 1.0,
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.lambda_cos = lambda_cos
        self.temperature = temperature
        self.alpha_kd = alpha_kd
        self.use_amp = use_amp
        self.max_grad_norm = max_grad_norm

        self._teacher_outputs: List[torch.Tensor] = []
        self._student_outputs: List[torch.Tensor] = []
        self._hook_handles: List[Any] = []
        self._register_persistent_hooks()

    def _get_subq_modules(self) -> List[Tuple[str, nn.Module]]:
        """Finds all top-level SubQBypassAttention or unwrapped SubQLinearAttention modules in the student."""
        subq_modules = []
        for name, mod in self.student_model.named_modules():
            if isinstance(mod, SubQBypassAttention):
                subq_modules.append((name, mod))
            elif isinstance(mod, SubQLinearAttention):
                parent_parts = name.split(".")[:-1]
                parent_name = ".".join(parent_parts)
                parent_mod = self.student_model.get_submodule(parent_name) if parent_name else None
                if not isinstance(parent_mod, SubQBypassAttention):
                    subq_modules.append((name, mod))
        return subq_modules

    def _register_persistent_hooks(self) -> None:
        """Registers persistent forward hooks once on teacher and student attention blocks."""
        self.remove_hooks()
        self._teacher_outputs.clear()
        self._student_outputs.clear()

        def make_hook(target_list: List[torch.Tensor]):
            def hook(module: nn.Module, input: Any, output: Any):
                out_tensor = output[0] if isinstance(output, tuple) else output
                target_list.append(out_tensor)
            return hook

        student_subq = self._get_subq_modules()
        for name, student_mod in student_subq:
            self._hook_handles.append(student_mod.register_forward_hook(make_hook(self._student_outputs)))
            if hasattr(self.teacher_model, "get_submodule"):
                try:
                    teacher_mod = self.teacher_model.get_submodule(name)
                    self._hook_handles.append(teacher_mod.register_forward_hook(make_hook(self._teacher_outputs)))
                except (AttributeError, KeyError):
                    pass

    def remove_hooks(self) -> None:
        """Removes all registered persistent forward hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def __del__(self):
        self.remove_hooks()

    def _freeze_non_attention_student_parameters(self) -> List[nn.Parameter]:
        """Freezes all student parameters except SubQ linear attention parameters."""
        subq_modules = dict(self._get_subq_modules())
        trainable_params: List[nn.Parameter] = []

        for name, param in self.student_model.named_parameters():
            is_subq_param = any(
                name.startswith(f"{mod_name}.") or name == mod_name
                for mod_name in subq_modules
            )
            param.requires_grad = is_subq_param
            if is_subq_param:
                trainable_params.append(param)

        return trainable_params

    def _unfreeze_all_student_parameters(self) -> List[nn.Parameter]:
        """Unfreezes all student parameters for Stage 2 end-to-end training."""
        trainable_params = []
        for param in self.student_model.parameters():
            param.requires_grad = True
            trainable_params.append(param)
        return trainable_params

    def train_stage1_step(
        self,
        batch_inputs: Union[torch.Tensor, Dict[str, Any]],
        optimizer: torch.optim.Optimizer,
        scaler: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Executes a single Stage 1 layer-wise alignment training step.
        Utilizes persistent hooks, AMP, and gradient clipping.
        """
        self.teacher_model.eval()
        self.student_model.train()

        self._teacher_outputs.clear()
        self._student_outputs.clear()

        first_param = next(self.student_model.parameters())
        device_type = "cuda" if first_param.is_cuda else "cpu"
        amp_enabled = self.use_amp and (device_type == "cuda")

        with torch.no_grad():
            with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
                if isinstance(batch_inputs, dict):
                    self.teacher_model(**batch_inputs)
                else:
                    self.teacher_model(batch_inputs)

        with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
            if isinstance(batch_inputs, dict):
                self.student_model(**batch_inputs)
            else:
                self.student_model(batch_inputs)

            total_mse = torch.tensor(0.0, device=first_param.device)
            total_cos = torch.tensor(0.0, device=first_param.device)
            num_layers = min(len(self._teacher_outputs), len(self._student_outputs))

            for l in range(num_layers):
                t_out = self._teacher_outputs[l]
                s_out = self._student_outputs[l]

                mse = F.mse_loss(s_out, t_out)
                cos_sim = F.cosine_similarity(s_out, t_out, dim=-1).mean()
                cos_loss = 1.0 - cos_sim

                total_mse = total_mse + mse
                total_cos = total_cos + cos_loss

            total_loss = (total_mse + self.lambda_cos * total_cos) / max(num_layers, 1)

        optimizer.zero_grad()
        if scaler is not None and amp_enabled:
            scaler.scale(total_loss).backward()
            if self.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.student_model.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.student_model.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )
            optimizer.step()

        return {
            "loss": float(total_loss.item()),
            "mse_loss": float(total_mse.item() / max(num_layers, 1)),
            "cos_loss": float(total_cos.item() / max(num_layers, 1)),
            "layers_aligned": num_layers,
        }

    def train_stage2_step(
        self,
        batch_inputs: Union[torch.Tensor, Dict[str, Any]],
        optimizer: torch.optim.Optimizer,
        labels: Optional[torch.Tensor] = None,
        scaler: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Executes a single Stage 2 end-to-end soft logit distillation training step
        using KL divergence, optional task CrossEntropy, AMP, and gradient clipping.
        """
        self.teacher_model.eval()
        self.student_model.train()

        first_param = next(self.student_model.parameters())
        device_type = "cuda" if first_param.is_cuda else "cpu"
        amp_enabled = self.use_amp and (device_type == "cuda")

        with torch.no_grad():
            with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
                if isinstance(batch_inputs, dict):
                    t_out = self.teacher_model(**batch_inputs)
                else:
                    t_out = self.teacher_model(batch_inputs)

        t_logits = t_out.logits if hasattr(t_out, "logits") else (t_out[0] if isinstance(t_out, tuple) else t_out)

        with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
            if isinstance(batch_inputs, dict):
                s_out = self.student_model(**batch_inputs)
            else:
                s_out = self.student_model(batch_inputs)

            s_logits = s_out.logits if hasattr(s_out, "logits") else (s_out[0] if isinstance(s_out, tuple) else s_out)

            T = self.temperature
            p_teacher = F.softmax(t_logits / T, dim=-1)
            log_p_student = F.log_softmax(s_logits / T, dim=-1)
            kd_loss = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (T * T)

            if labels is not None:
                ce_loss = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)), labels.view(-1))
                total_loss = self.alpha_kd * ce_loss + (1.0 - self.alpha_kd) * kd_loss
            else:
                ce_loss = torch.tensor(0.0, device=s_logits.device)
                total_loss = kd_loss

        optimizer.zero_grad()
        if scaler is not None and amp_enabled:
            scaler.scale(total_loss).backward()
            if self.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.student_model.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.student_model.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )
            optimizer.step()

        return {
            "loss": float(total_loss.item()),
            "kd_loss": float(kd_loss.item()),
            "ce_loss": float(ce_loss.item()),
        }

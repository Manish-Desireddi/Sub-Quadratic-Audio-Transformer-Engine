# 
# Copyright (c) 2026 Manish. All rights reserved.
# 
# This work is licensed under the terms of the MIT license.  
# For a copy, see <https://opensource.org/licenses/MIT>.
# 

import pytest
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from subq_audio.autograd import chunkwise_flash_linear_attention, SubQLinearAttention
from subq_audio.model import SubQAudioConfig, SubQAudioForCausalLM
from subq_audio.surgery import bypass_transformer_with_subq, LayerWiseDistillationTrainer, SubQBypassAttention


def sequential_linear_attention_reference(q, k, v, decay_w):
    """
    Reference sequential token-by-token linear attention.
    Computes exact recurrent state updates without chunking.
    """
    B, H, L, D = q.shape
    phi_q = torch.where(q > 0.0, q + 1.0, torch.exp(q))
    phi_k = torch.where(k > 0.0, k + 1.0, torch.exp(k))

    if decay_w.dim() == 1:
        log_gamma = -F.softplus(-decay_w).view(1, H, 1).expand(B, H, L)
    elif decay_w.dim() == 2:
        log_gamma = -F.softplus(-decay_w).unsqueeze(-1).expand(B, H, L)
    else:
        log_gamma = -F.softplus(-decay_w)

    gamma = torch.exp(log_gamma)

    out = torch.zeros((B, H, L, D), dtype=q.dtype, device=q.device)
    S = torch.zeros((B, H, D, D), dtype=q.dtype, device=q.device)

    for t in range(L):
        g_t = gamma[:, :, t].unsqueeze(-1).unsqueeze(-1)
        k_t = phi_k[:, :, t].unsqueeze(-1)  # [B, H, D, 1]
        v_t = v[:, :, t].unsqueeze(-2)      # [B, H, 1, D]
        S = g_t * S + torch.matmul(k_t, v_t)

        q_t = phi_q[:, :, t].unsqueeze(-2)  # [B, H, 1, D]
        o_t = torch.matmul(q_t, S).squeeze(-2)
        out[:, :, t] = o_t

    return out, S


class StandardAttentionBlock(nn.Module):
    """Mock standard quadratic attention block for surgery testing."""
    def __init__(self, hidden_size=64, num_heads=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, L, _ = hidden_states.shape
        Q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, self.hidden_size)
        return self.out_proj(out)


class MockTransformerModel(nn.Module):
    """Mock multi-layer transformer for surgery and distillation testing."""
    def __init__(self, vocab_size=100, hidden_size=64, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": StandardAttentionBlock(hidden_size=hidden_size, num_heads=4),
                "mlp": nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 2),
                    nn.GELU(),
                    nn.Linear(hidden_size * 2, hidden_size),
                ),
            })
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            attn_out = layer["self_attn"](x)
            x = x + attn_out
            x = x + layer["mlp"](x)
        return self.lm_head(x)


# ==============================================================================
# TESTS
# ==============================================================================

def test_chunkwise_parallel_attention_numerical_equivalence():
    """Verify that chunkwise flash linear attention exactly matches sequential reference."""
    torch.manual_seed(42)
    B, H, L, D = 2, 4, 137, 32  # Non-power-of-2 length crossing multiple chunk boundaries
    chunk_size = 32

    q = torch.randn(B, H, L, D, dtype=torch.float32)
    k = torch.randn(B, H, L, D, dtype=torch.float32)
    v = torch.randn(B, H, L, D, dtype=torch.float32)
    decay_w = torch.full((H,), 2.5, dtype=torch.float32)

    # 1. Run sequential reference
    out_ref, state_ref = sequential_linear_attention_reference(q, k, v, decay_w)

    # 2. Run chunkwise parallel implementation
    out_chunk, state_chunk = chunkwise_flash_linear_attention(
        q, k, v, decay_w, chunk_size=chunk_size
    )

    # 3. Assert outputs match within float32 tolerance
    max_out_diff = (out_ref - out_chunk).abs().max().item()
    max_state_diff = (state_ref - state_chunk).abs().max().item()

    assert max_out_diff < 2e-4, f"Output mismatch: {max_out_diff}"
    assert max_state_diff < 2e-4, f"Final state mismatch: {max_state_diff}"


def test_chunkwise_parallel_attention_data_dependent_gating():
    """Verify chunkwise attention with full 3D dynamic decay tensor [B, H, L]."""
    torch.manual_seed(123)
    B, H, L, D = 2, 2, 80, 16
    chunk_size = 32

    q = torch.randn(B, H, L, D, dtype=torch.float32)
    k = torch.randn(B, H, L, D, dtype=torch.float32)
    v = torch.randn(B, H, L, D, dtype=torch.float32)
    decay_w = torch.randn(B, H, L, dtype=torch.float32)

    out_ref, state_ref = sequential_linear_attention_reference(q, k, v, decay_w)
    out_chunk, state_chunk = chunkwise_flash_linear_attention(
        q, k, v, decay_w, chunk_size=chunk_size
    )

    max_out_diff = (out_ref - out_chunk).abs().max().item()
    assert max_out_diff < 2e-4, f"Dynamic gating mismatch: {max_out_diff}"


def test_stateful_recurrent_generation_identical_tokens():
    """
    Verify that O(1) autoregressive generation with past_key_values cache
    produces 100% bitwise identical sequences as un-cached full sequence passes.
    """
    torch.manual_seed(999)
    config = SubQAudioConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        use_cache=True,
    )
    model = SubQAudioForCausalLM(config)
    model.eval()

    prompt_ids = torch.randint(0, 100, (2, 8))

    # Generate with cache
    with torch.no_grad():
        out_cached = model.generate(
            prompt_ids.clone(),
            max_new_tokens=8,
            use_cache=True,
            do_sample=False,
        )

    # Generate without cache
    with torch.no_grad():
        out_uncached = model.generate(
            prompt_ids.clone(),
            max_new_tokens=8,
            use_cache=False,
            do_sample=False,
        )

    # Check that generation fed length 1 at step > 0
    prepared = model.prepare_inputs_for_generation(
        out_cached,
        past_key_values=((torch.zeros(2, 2, 32, 32),),)
    )
    assert prepared["input_ids"].shape[1] == 1, "prepare_inputs_for_generation did not slice input_ids to length 1"

    # Both sequences must be 100% bitwise identical
    assert torch.equal(out_cached, out_uncached), "Cached vs Un-cached generation token mismatch!"


def test_transformer_attention_surgery():
    """Verify bypass_transformer_with_subq weight copying and architecture modification."""
    torch.manual_seed(42)
    model = MockTransformerModel(vocab_size=100, hidden_size=64, num_layers=2)

    # Record initial weights of teacher's layer 0
    orig_q_w = model.layers[0]["self_attn"].q_proj.weight.clone()
    orig_q_b = model.layers[0]["self_attn"].q_proj.bias.clone()
    orig_o_w = model.layers[0]["self_attn"].out_proj.weight.clone()

    # Perform surgery
    bypassed_model, replaced_info = bypass_transformer_with_subq(
        model,
        target_select_suffixes=("self_attn",),
        decay_init=0.99,
        use_data_dependent_gating=True,
        copy_weights=True,
    )

    assert len(replaced_info) == 2, f"Expected 2 replaced layers, got {len(replaced_info)}"

    # Check replacement class
    assert isinstance(bypassed_model.layers[0]["self_attn"], SubQBypassAttention)

    # Check copied weights
    new_attn = bypassed_model.layers[0]["self_attn"]
    assert torch.allclose(new_attn.q_proj.weight, orig_q_w)
    assert torch.allclose(new_attn.q_proj.bias, orig_q_b)
    assert torch.allclose(new_attn.out_proj.weight, orig_o_w)

    # Check dynamic gate zero initialization
    assert torch.count_nonzero(new_attn.gate_proj.weight) == 0
    assert torch.count_nonzero(new_attn.gate_proj.bias) == 0

    # Test forward pass through bypassed model
    dummy_input = torch.randint(0, 100, (2, 16))
    output = bypassed_model(dummy_input)
    assert output.shape == (2, 16, 100)


def test_layerwise_distillation_trainer_two_stages():
    """Verify LayerWiseDistillationTrainer Stage 1 (alignment) and Stage 2 (KL-div)."""
    torch.manual_seed(777)
    teacher = MockTransformerModel(vocab_size=100, hidden_size=64, num_layers=2)
    teacher.eval()

    # Clone for student and apply surgery
    import copy
    student = copy.deepcopy(teacher)
    bypass_transformer_with_subq(student, target_select_suffixes=("self_attn",))

    trainer = LayerWiseDistillationTrainer(
        teacher_model=teacher,
        student_model=student,
        lambda_cos=0.5,
        temperature=2.0,
        alpha_kd=0.5,
    )

    # --- Stage 1 Testing ---
    trainable_params_s1 = trainer._freeze_non_attention_student_parameters()
    assert len(trainable_params_s1) > 0
    optimizer_s1 = torch.optim.AdamW(trainable_params_s1, lr=1e-3)

    dummy_input = torch.randint(0, 100, (4, 16))

    loss_history_s1 = []
    for _ in range(5):
        metrics_s1 = trainer.train_stage1_step(dummy_input, optimizer_s1)
        loss_history_s1.append(metrics_s1["loss"])

    assert metrics_s1["layers_aligned"] == 2
    # Loss should decrease monotonically or finish significantly lower than step 0
    assert loss_history_s1[-1] < loss_history_s1[0], f"Stage 1 loss did not decrease: {loss_history_s1}"

    # --- Stage 2 Testing ---
    trainable_params_s2 = trainer._unfreeze_all_student_parameters()
    optimizer_s2 = torch.optim.AdamW(trainable_params_s2, lr=1e-4)

    dummy_labels = torch.randint(0, 100, (4, 16))
    loss_history_s2 = []
    for _ in range(5):
        metrics_s2 = trainer.train_stage2_step(dummy_input, optimizer_s2, labels=dummy_labels)
        loss_history_s2.append(metrics_s2["loss"])

    assert loss_history_s2[-1] < loss_history_s2[0], f"Stage 2 loss did not decrease: {loss_history_s2}"


def test_whisper_surgery_bidirectional_and_generation():
    """Verify WhisperForConditionalGeneration surgery: bidirectional encoder, causal decoder, and generation."""
    pytest.importorskip("transformers")
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    cfg = WhisperConfig(
        vocab_size=128,
        d_model=64,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
        max_source_positions=1500,
    )
    whisper_model = WhisperForConditionalGeneration(cfg)

    # Perform surgery on all self_attn modules
    bypassed_whisper, replaced = bypass_transformer_with_subq(
        whisper_model,
        target_select_suffixes=("self_attn",),
        copy_weights=True,
    )
    assert len(replaced) == 4, f"Expected 4 replaced self_attn layers, got {len(replaced)}"

    # Check encoder self_attn is bidirectional (is_causal == False)
    for layer in bypassed_whisper.model.encoder.layers:
        assert not layer.self_attn.is_causal, "Encoder self_attn should be non-causal (bidirectional)"

    # Check decoder self_attn is causal (is_causal == True)
    for layer in bypassed_whisper.model.decoder.layers:
        assert layer.self_attn.is_causal, "Decoder self_attn should be causal"

    # Forward pass with real mel features shape (B, n_mels=80, T=3000)
    input_features = torch.randn(1, 80, 3000)
    decoder_input_ids = torch.tensor([[1, 10, 11]])
    out = bypassed_whisper(input_features=input_features, decoder_input_ids=decoder_input_ids)
    assert out.logits.shape == (1, 3, 128), f"Unexpected logits shape: {out.logits.shape}"

    # Verify autoregressive generation produces tokens without tuple unpacking crash
    gen_tokens = bypassed_whisper.generate(input_features, max_new_tokens=4)
    assert gen_tokens.shape[0] == 1
    assert gen_tokens.shape[1] >= 1


def test_llama_gqa_surgery_and_padding_invariance():
    """Verify Llama GQA weight copying, forward pass, and padding mask state protection."""
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    llama_model = LlamaForCausalLM(cfg)

    # Record teacher projection weights
    orig_k_weight = llama_model.model.layers[0].self_attn.k_proj.weight.clone()
    orig_v_weight = llama_model.model.layers[0].self_attn.v_proj.weight.clone()

    bypassed_llama, replaced = bypass_transformer_with_subq(
        llama_model,
        target_select_suffixes=("self_attn",),
        copy_weights=True,
    )
    assert len(replaced) == 2

    # Verify GQA dimensions and exact weight transfer
    subq_attn = bypassed_llama.model.layers[0].self_attn
    assert subq_attn.num_heads == 4
    assert subq_attn.num_key_value_heads == 2
    assert subq_attn.core.num_key_value_groups == 2
    assert torch.allclose(subq_attn.k_proj.weight, orig_k_weight)
    assert torch.allclose(subq_attn.v_proj.weight, orig_v_weight)

    # Verify padding mask protection at attention layer level
    attn_layer = subq_attn.core
    attn_layer.eval()

    x_unpadded = torch.randn(1, 4, 64)
    mask_unpadded = torch.ones(1, 4)
    out_unpadded, _ = attn_layer(x_unpadded, attention_mask=mask_unpadded)

    pad_noise = torch.randn(1, 6, 64)
    x_padded = torch.cat([x_unpadded, pad_noise], dim=1)
    mask_padded = torch.cat([torch.ones(1, 4), torch.zeros(1, 6)], dim=1)
    out_padded, _ = attn_layer(x_padded, attention_mask=mask_padded)

    # Tokens in unpadded span [0..3] must be bitwise invariant to pad tokens
    max_diff = (out_unpadded - out_padded[:, :4]).abs().max().item()
    assert max_diff < 1e-6, f"Padding corrupted past context state: max_diff={max_diff}"

    # Forward pass and generation on full bypassed model
    input_ids = torch.tensor([[1, 25, 30]])
    res = bypassed_llama(input_ids=input_ids)
    assert res.logits.shape == (1, 3, 128)

    gen = bypassed_llama.generate(input_ids, max_new_tokens=3)
    assert gen.shape == (1, 6)


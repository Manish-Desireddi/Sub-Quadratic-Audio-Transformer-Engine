# 
# Copyright (c) 2026 Manish. All rights reserved.
# 
# This work is licensed under the terms of the GNU GPLv3 license.  
# For a copy, see <https://www.gnu.org/licenses/>.
# 

import pytest
import os
import shutil
import tempfile
import torch

from subq_audio.model import (
    SubQAudioConfig,
    SubQAudioModel,
    SubQAudioForCausalLM,
)


@pytest.fixture
def small_config():
    return SubQAudioConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        decay_init=0.99,
        max_position_embeddings=256,
    )


# ==============================================================================
# 1. Config Serialization Tests
# ==============================================================================

def test_config_serialization(small_config):
    with tempfile.TemporaryDirectory() as tmp_dir:
        small_config.save_pretrained(tmp_dir)
        loaded_config = SubQAudioConfig.from_pretrained(tmp_dir)

        assert loaded_config.hidden_size == small_config.hidden_size
        assert loaded_config.num_hidden_layers == small_config.num_hidden_layers
        assert loaded_config.num_attention_heads == small_config.num_attention_heads
        assert loaded_config.decay_init == small_config.decay_init
        assert loaded_config.model_type == "subq_audio"


# ==============================================================================
# 2. Model Instantiation & Parameter Shapes
# ==============================================================================

def test_model_instantiation_and_shapes(small_config):
    model = SubQAudioForCausalLM(small_config)
    assert len(model.model.layers) == small_config.num_hidden_layers
    assert model.model.embed_tokens.weight.shape == (small_config.vocab_size, small_config.hidden_size)
    assert model.lm_head.weight.shape == (small_config.vocab_size, small_config.hidden_size)

    for layer in model.model.layers:
        assert layer.self_attn.raw_decay.shape == (small_config.num_attention_heads,)
        assert layer.self_attn.q_proj.weight.shape == (small_config.hidden_size, small_config.hidden_size)
        assert layer.mlp.fc1.weight.shape == (small_config.intermediate_size, small_config.hidden_size)
        assert layer.mlp.fc2.weight.shape == (small_config.hidden_size, small_config.intermediate_size)


# ==============================================================================
# 3. Forward Pass with Loss Calculation
# ==============================================================================

def test_forward_pass_with_loss(small_config):
    model = SubQAudioForCausalLM(small_config)
    B, T = 2, 16
    input_ids = torch.randint(0, small_config.vocab_size, (B, T))
    labels = input_ids.clone()

    # Forward without labels
    out_no_labels = model(input_ids=input_ids)
    assert out_no_labels.loss is None
    assert out_no_labels.logits.shape == (B, T, small_config.vocab_size)

    # Forward with labels
    out_with_labels = model(input_ids=input_ids, labels=labels)
    assert out_with_labels.loss is not None
    assert out_with_labels.loss.item() > 0.0
    assert not torch.isnan(out_with_labels.loss)


# ==============================================================================
# 4. Save and From Pretrained (Weights Roundtrip)
# ==============================================================================

def test_save_and_from_pretrained(small_config):
    torch.manual_seed(42)
    model = SubQAudioForCausalLM(small_config)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save_pretrained(tmp_dir)
        loaded_model = SubQAudioForCausalLM.from_pretrained(tmp_dir)

        B, T = 2, 8
        input_ids = torch.randint(0, small_config.vocab_size, (B, T))

        model.eval()
        loaded_model.eval()

        with torch.no_grad():
            out1 = model(input_ids=input_ids).logits
            out2 = loaded_model(input_ids=input_ids).logits

        assert torch.allclose(out1, out2, atol=1e-6), "Loaded model logits do not match original model"


# ==============================================================================
# 5. Autoregressive Generation
# ==============================================================================

def test_generate_autoregressive(small_config):
    model = SubQAudioForCausalLM(small_config)
    model.eval()

    prompt = torch.tensor([[1, 10, 20]], dtype=torch.long)
    max_new_tokens = 6

    generated = model.generate(
        input_ids=prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=small_config.pad_token_id,
        eos_token_id=small_config.eos_token_id,
    )

    assert generated.shape == (1, prompt.shape[1] + max_new_tokens)
    assert torch.equal(generated[:, :prompt.shape[1]], prompt)

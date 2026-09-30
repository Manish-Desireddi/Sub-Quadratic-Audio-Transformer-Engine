# 
# Copyright (c) 2026 Manish. All rights reserved.
# 
# This work is licensed under the terms of the GNU GPLv3 license.  
# For a copy, see <https://www.gnu.org/licenses/>.
# 

import pytest
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from subq_audio.model import (
    SubQAudioConfig,
    SubQAudioModel,
    SubQAudioForCausalLM,
)


@pytest.fixture
def tiny_config():
    return SubQAudioConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        decay_init=0.95,
        max_position_embeddings=128,
    )


# ==============================================================================
# 1. Differentiable Log-Probabilities
# ==============================================================================

def test_differentiable_log_probabilities(tiny_config):
    torch.manual_seed(42)
    model = SubQAudioForCausalLM(tiny_config)
    
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 8))
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # [2, 8, vocab_size]

    log_probs = F.log_softmax(logits, dim=-1)
    target_tokens = input_ids[:, 1:].unsqueeze(-1)
    token_log_probs = log_probs[:, :-1, :].gather(dim=-1, index=target_tokens).squeeze(-1)

    loss = -token_log_probs.mean()
    loss.backward()

    # Verify all layers have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"
            assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf gradient"


# ==============================================================================
# 2. Direct Preference Optimization (DPO) Training Step
# ==============================================================================

def test_dpo_loss_backward_and_step(tiny_config):
    torch.manual_seed(42)
    beta = 0.1
    policy_model = SubQAudioForCausalLM(tiny_config)
    ref_model = SubQAudioForCausalLM(tiny_config)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-3)

    # Inputs: Prompt (length 4) + Chosen/Rejected completion (length 4)
    prompt_chosen = torch.randint(0, tiny_config.vocab_size, (2, 8))
    prompt_rejected = torch.randint(0, tiny_config.vocab_size, (2, 8))

    def get_batch_logps(model, input_ids):
        logits = model(input_ids=input_ids).logits
        log_probs = F.log_softmax(logits, dim=-1)
        targets = input_ids[:, 1:].unsqueeze(-1)
        token_logps = log_probs[:, :-1, :].gather(dim=-1, index=targets).squeeze(-1)
        # Sum over completion tokens (last 4 tokens)
        return token_logps[:, -4:].sum(dim=-1)

    # Compute policy and reference logps
    policy_chosen_logps = get_batch_logps(policy_model, prompt_chosen)
    policy_rejected_logps = get_batch_logps(policy_model, prompt_rejected)

    with torch.no_grad():
        ref_chosen_logps = get_batch_logps(ref_model, prompt_chosen)
        ref_rejected_logps = get_batch_logps(ref_model, prompt_rejected)

    # DPO Loss calculation
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits_dpo = pi_logratios - ref_logratios
    dpo_loss = -F.logsigmoid(beta * logits_dpo).mean()

    initial_loss_val = dpo_loss.item()
    optimizer.zero_grad()
    dpo_loss.backward()
    optimizer.step()

    # Re-evaluate DPO loss
    new_policy_chosen_logps = get_batch_logps(policy_model, prompt_chosen)
    new_policy_rejected_logps = get_batch_logps(policy_model, prompt_rejected)
    new_pi_logratios = new_policy_chosen_logps - new_policy_rejected_logps
    new_logits_dpo = new_pi_logratios - ref_logratios
    new_dpo_loss = -F.logsigmoid(beta * new_logits_dpo).mean()

    assert not math.isnan(initial_loss_val)
    assert not math.isinf(initial_loss_val)
    assert new_dpo_loss.item() < initial_loss_val, "DPO loss should decrease after optimization step"


# ==============================================================================
# 3. Group Relative Policy Optimization (GRPO) Step
# ==============================================================================

def test_grpo_group_rollout_and_advantage_backward(tiny_config):
    torch.manual_seed(42)
    model = SubQAudioForCausalLM(tiny_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    group_size = 4
    seq_len = 8
    # 4 completions sampled for the same prompt
    group_inputs = torch.randint(0, tiny_config.vocab_size, (group_size, seq_len))
    
    # Reward scores for the group
    rewards = torch.tensor([2.0, 1.0, -0.5, -1.0], dtype=torch.float32)
    # Standardize advantages
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    outputs = model(input_ids=group_inputs)
    log_probs = F.log_softmax(outputs.logits, dim=-1)
    targets = group_inputs[:, 1:].unsqueeze(-1)
    token_logps = log_probs[:, :-1, :].gather(dim=-1, index=targets).squeeze(-1).sum(dim=-1)

    # Simulated old policy log-probs (detached)
    old_logps = token_logps.detach() - 0.05

    # Importance sampling ratios
    ratios = torch.exp(token_logps - old_logps)
    clip_eps = 0.2
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    grpo_loss = -torch.min(surr1, surr2).mean()

    optimizer.zero_grad()
    grpo_loss.backward()
    optimizer.step()

    assert not math.isnan(grpo_loss.item())
    assert not math.isinf(grpo_loss.item())


# ==============================================================================
# 4. Value Head / Critic Gradient Flow (PPO / Reward Modeling)
# ==============================================================================

def test_value_head_gradient_flow(tiny_config):
    torch.manual_seed(42)
    
    class SubQWithValueHead(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.model = SubQAudioModel(config)
            self.v_head = nn.Linear(config.hidden_size, 1, bias=False)

        def forward(self, input_ids):
            out = self.model(input_ids=input_ids)
            last_hidden = out.last_hidden_state
            values = self.v_head(last_hidden).squeeze(-1)
            return values

    critic = SubQWithValueHead(tiny_config)
    optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-3)

    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 8))
    target_values = torch.randn(2, 8)

    predicted_values = critic(input_ids)
    loss = F.mse_loss(predicted_values, target_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert critic.v_head.weight.grad is not None
    assert not torch.isnan(critic.v_head.weight.grad).any()


# ==============================================================================
# 5. Policy Stability Across Decay Bounds
# ==============================================================================

@pytest.mark.parametrize("decay_init", [0.001, 0.5, 0.999])
def test_policy_gradient_stability_across_decay_bounds(decay_init):
    torch.manual_seed(42)
    cfg = SubQAudioConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=64,
        decay_init=decay_init,
    )
    model = SubQAudioForCausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 6))
    
    out = model(input_ids=input_ids, labels=input_ids)
    loss = out.loss
    loss.backward()

    assert not torch.isnan(loss).any()
    assert not torch.isinf(loss).any()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert not torch.isnan(p.grad).any(), f"NaN in {name} grad with decay_init={decay_init}"
            assert not torch.isinf(p.grad).any(), f"Inf in {name} grad with decay_init={decay_init}"

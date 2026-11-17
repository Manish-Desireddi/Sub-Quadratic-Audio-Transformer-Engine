#
# Attention-mask normalisation:
# 4D key-padding masks must not be silently dropped.
#

import pytest
import torch

from subq_audio.autograd import _normalize_attention_mask, chunkwise_flash_linear_attention

B, T = 2, 6
KEYS_VALID = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)


def _valid(mask4d):
    return _normalize_attention_mask(mask4d, torch.float32, T).flatten(1).tolist()


def test_4d_key_padding_mask_marks_padded_tokens_invalid():
    mask = KEYS_VALID[:, None, None, :].expand(B, 1, T, T)
    assert _valid(mask) == [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]]


def test_4d_causal_plus_key_padding_mask():
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    mask = causal[None, None] & KEYS_VALID[:, None, None, :]
    assert _valid(mask) == [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]]


def test_4d_pure_causal_mask_keeps_all_tokens():
    causal = torch.tril(torch.ones(B, 1, T, T, dtype=torch.bool))
    assert _valid(causal) == [[1] * T] * B


def test_4d_query_padding_rows_are_invalid():
    mask = KEYS_VALID[:, None, :, None].expand(B, 1, T, T)
    assert _valid(mask) == [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]]


def test_4d_non_square_mask_is_rejected():
    with pytest.raises(ValueError):
        _normalize_attention_mask(torch.ones(B, 1, T, T + 2, dtype=torch.bool), torch.float32, T)


def test_4d_key_padding_equals_2d_mask_output():
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, 2, T, 4) for _ in range(3))
    w = torch.tensor(0.5)
    out2d, _ = chunkwise_flash_linear_attention(q, k, v, w, attention_mask=KEYS_VALID)
    mask4d = KEYS_VALID[:, None, None, :].expand(B, 1, T, T)
    out4d, _ = chunkwise_flash_linear_attention(q, k, v, w, attention_mask=mask4d)
    assert torch.allclose(out2d, out4d)

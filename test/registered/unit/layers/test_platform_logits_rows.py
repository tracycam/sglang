"""Prepared token rows must select the same logits rows for extend and decode."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import logits_processor
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class PlatformLogitsRowsTest(unittest.TestCase):
    @staticmethod
    def batch(**updates):
        fields = dict(
            return_logprob=False,
            capture_hidden_mode=None,
            next_token_logits_buffer=None,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            is_prefill_only=False,
            global_num_tokens_gpu=None,
            dp_local_start_pos=None,
            dp_local_num_tokens=None,
            global_dp_buffer_len=None,
            global_num_tokens_for_logprob_cpu=None,
            global_num_tokens_for_logprob_gpu=None,
            mm_input_embeds=None,
        )
        fields.update(updates)
        return SimpleNamespace(**fields)

    def test_prepared_rows_do_not_require_mode_specific_metadata(self):
        indices = torch.tensor([2, 4])
        with patch.object(logits_processor.current_platform, "get_last_token_indices", return_value=indices):
            metadata = logits_processor.LogitsMetadata.from_forward_batch(self.batch())
            hidden = torch.arange(15).reshape(5, 3)
            result = logits_processor.LogitsProcessor._get_pruned_states(
                None, hidden, hidden + 100, [hidden + 200], metadata
            )
            self.assertTrue(torch.equal(result[0], hidden[indices]))
            self.assertTrue(torch.equal(result[1], (hidden + 100)[indices]))
            self.assertTrue(torch.equal(result[2][0], (hidden + 200)[indices]))
            self.assertIsNone(result[3])
            self.assertIsNone(result[4])
            with self.assertRaisesRegex(ValueError, "input logprobs"):
                logits_processor.LogitsMetadata.from_forward_batch(self.batch(return_logprob=True))

    def test_default_extend_still_computes_last_rows(self):
        batch = self.batch(
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=torch.tensor([3, 2]),
            extend_seq_lens_cpu=[3, 2],
            extend_logprob_start_lens_cpu=None,
            extend_input_logprob_token_ids_gpu=None,
        )
        with patch.object(logits_processor.current_platform, "get_last_token_indices", return_value=None):
            metadata = logits_processor.LogitsMetadata.from_forward_batch(batch)
            hidden = torch.arange(15).reshape(5, 3)
            result = logits_processor.LogitsProcessor._get_pruned_states(
                None, hidden, None, None, metadata
            )
            self.assertTrue(torch.equal(result[0], hidden[torch.tensor([2, 4])]))


if __name__ == "__main__":
    unittest.main()

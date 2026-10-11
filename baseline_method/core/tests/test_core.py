"""Pure-function checks; no model, GPU, or API access is needed."""

import json
import tempfile
import unittest
from pathlib import Path

from encoding import encode
from metrics import score
from trace_format import parse_trace


class FakeTokenizer:
    def apply_chat_template(self, *_args, **_kwargs):
        return "USER<think>\n"

    def encode(self, text, **_kwargs):
        return [ord(char) for char in text]

    def convert_tokens_to_ids(self, token):
        assert token == "<|im_end|>"
        return 999


class CoreTests(unittest.TestCase):
    def test_encoding_masks_prompt_only(self):
        ids, labels = encode(
            FakeTokenizer(), "question", "reason", "answer", "Qwen_35_9B"
        )
        prompt_length = len("USER<think>\n")
        self.assertEqual(labels[:prompt_length], [-100] * prompt_length)
        self.assertEqual(labels[prompt_length:], ids[prompt_length:])
        self.assertEqual(ids[-1], 999)

    def test_trace_parser_rejects_extra_text_and_bad_skeleton(self):
        reason = "Detailed explanation " * 6
        self.assertEqual(
            parse_trace("NEU", f"<reason>{reason}</reason>")[0], reason.strip()
        )
        with self.assertRaises(ValueError):
            parse_trace("NEU", f"prefix<reason>{reason}</reason>")
        with self.assertRaises(ValueError):
            parse_trace(
                "SSR", f"<skeleton>2. [PLAN] Check</skeleton><reason>{reason}</reason>"
            )

    def test_metrics_require_completed_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            self.assertIsNone(score(path, "ifeval"))
            path.write_text(
                json.dumps(
                    {
                        "status": "incomplete",
                        "metrics": {"prompt_strict": {"score": 12}},
                    }
                )
            )
            self.assertIsNone(score(path, "ifeval"))
            path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "metrics": {"prompt_strict": {"score": 82.5}},
                    }
                )
            )
            self.assertEqual(score(path, "ifeval"), 82.5)


if __name__ == "__main__":
    unittest.main()

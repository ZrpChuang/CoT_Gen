import unittest

from formats import candidate, matched_random


class FormatTests(unittest.TestCase):
    def test_choice_threshold_and_identical_body(self):
        self.assertEqual(candidate(2e-6), "paired_steps")
        self.assertEqual(candidate(1e-6), "paired_plain")
        self.assertEqual(candidate(1, identical=True), "paired_plain")
        with self.assertRaises(ValueError):
            candidate(float("nan"))

    def test_random_keeps_each_bin_quota_and_is_order_independent(self):
        selected = {
            "a": "paired_steps",
            "b": "paired_plain",
            "c": "paired_steps",
            "d": "paired_plain",
            "e": "paired_plain",
        }
        bins = {
            "a": ("code", 0),
            "b": ("code", 0),
            "c": ("math", 2),
            "d": ("math", 2),
            "e": ("code", 0),
        }
        result = matched_random(selected, bins, identical=("e",))
        self.assertEqual(result["e"], "paired_plain")
        self.assertEqual(
            result,
            matched_random(
                dict(reversed(list(selected.items()))), bins, identical=("e",)
            ),
        )
        for group in set(bins.values()):
            ids = [rid for rid in bins if bins[rid] == group]
            self.assertEqual(
                sum(result[rid] == "paired_steps" for rid in ids),
                sum(selected[rid] == "paired_steps" for rid in ids),
            )

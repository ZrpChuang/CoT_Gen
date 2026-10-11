"""Deterministic body-NLL choice and quota-matched random format assignment."""

import hashlib
import math
from collections import defaultdict

SEED = 20261004


def candidate(delta, identical=False):
    if (
        isinstance(delta, bool)
        or not isinstance(delta, (float, int))
        or not math.isfinite(delta)
    ):
        raise ValueError("Body NLL delta must be a finite number")
    return "paired_steps" if not identical and delta > 1e-6 else "paired_plain"


def matched_random(selected, bins, identical=()):
    """Keep each (stratum, length-quintile) steps quota and identical-body fallback."""
    if set(selected) != set(bins):
        raise ValueError("Selected IDs and bin IDs must match")
    identical = set(identical)
    if not identical <= set(selected):
        raise ValueError("Identical-body IDs must belong to the selection")
    if any(
        value not in ("paired_plain", "paired_steps") for value in selected.values()
    ):
        raise ValueError("Unknown format")
    if any(selected[rid] != "paired_plain" for rid in identical):
        raise ValueError("Identical bodies must use plain format")
    groups = defaultdict(list)
    for rid, group in bins.items():
        groups[tuple(group)].append(rid)
    choices = {rid: "paired_plain" for rid in identical}
    for (stratum, quintile), ids in sorted(groups.items()):
        quota = sum(selected[rid] == "paired_steps" for rid in ids)
        eligible = [rid for rid in ids if rid not in identical]
        order = sorted(
            eligible,
            key=lambda rid: (
                hashlib.sha256(
                    f"matched-random-format-v1:{SEED}:{stratum}:{quintile}:{rid}".encode()
                ).hexdigest(),
                rid,
            ),
        )
        steps = set(order[:quota])
        choices.update(
            {
                rid: "paired_steps" if rid in steps else "paired_plain"
                for rid in eligible
            }
        )
    return choices

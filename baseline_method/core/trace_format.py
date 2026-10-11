"""Validate teacher-generated synthetic CoT before adding it to training data."""

from __future__ import annotations

import re

from settings import METHODS

_CONTROL_TOKENS = ("<think>", "</think>", "<|im_end|>", "<|im_start|>")
_SKELETON_LINE = re.compile(
    r"(\d+)\.\s+\[(PLAN|RETR|INFR|EVAL|SUMM|BTRK|RFLX|BRCH)\]\s+(.+)"
)


def parse_trace(method: str, raw: str) -> tuple[str, str | None, str]:
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")

    pattern = (
        r"\s*<skeleton>(.*?)</skeleton>\s*<reason>(.*?)</reason>\s*"
        if method == "SSR"
        else r"\s*<reason>(.*?)</reason>\s*"
    )
    match = re.fullmatch(pattern, raw, flags=re.DOTALL)
    if match is None:
        raise ValueError("Required complete blocks missing, or text outside blocks")

    skeleton = match.group(1).strip() if method == "SSR" else None
    reason = (match.group(2) if method == "SSR" else match.group(1)).strip()
    if len(reason) < 80:
        raise ValueError("Reasoning block too short for a usable trace")
    if any(token in reason for token in _CONTROL_TOKENS):
        raise ValueError("Unexpected native chat control token")

    if skeleton is not None:
        lines = [line.strip() for line in skeleton.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Empty skeleton")
        for number, line in enumerate(lines, start=1):
            parsed = _SKELETON_LINE.fullmatch(line)
            if (
                parsed is None
                or int(parsed.group(1)) != number
                or len(parsed.group(3).split()) >= 20
            ):
                raise ValueError("Invalid skeleton numbering, tag, or length")
        cot = f"<skeleton>\n{skeleton}\n</skeleton>\n\n{reason}"
    else:
        cot = reason
    return cot, skeleton, reason

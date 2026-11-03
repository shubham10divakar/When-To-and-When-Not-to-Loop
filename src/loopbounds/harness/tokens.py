"""Deterministic token estimate used by the task generator and the mock clients.

Real runs always use the provider's reported usage; this counter only controls
padding sizes and the mocks' synthetic usage. 1 token := 4 characters (ceil).
"""

from __future__ import annotations

import json
import math

CHARS_PER_TOKEN = 4


def count_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def count_json_tokens(obj) -> int:
    return count_tokens(json.dumps(obj, sort_keys=True, separators=(",", ":")))

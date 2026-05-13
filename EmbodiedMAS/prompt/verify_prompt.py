#!/usr/bin/env python3
"""
Smoke-test for modular prompt assembly.

Run from EmbodiedMAS (this file lives under EmbodiedMAS/prompt/):
  python3 prompt/verify_prompt.py

Or from this directory:
  python3 verify_prompt.py
"""

from __future__ import annotations

import os
import sys

# Ensure imports work when executed as a script
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from prompt.prompt_assembler import assemble_prompt  # noqa: E402

_ENV = {
    "num_agents": 3,
    "num_fire_agents": 2,
    "num_rescue_agents": 1,
    "num_civilians": 4,
    "num_fires": "unknown",
    "other_info": "",
}

def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _word_count(text: str) -> int:
    """Whitespace-separated token count (approximate words)."""
    return len(text.split())


def main() -> None:
    p = assemble_prompt("fire", "DISTRIBUTED", _ENV, obs="VLM")
    _assert("extinguish_fire" in p, "fire role should include extinguish")

    env_lw = {**_ENV, "other_info": "Test limited-water note."}
    plw = assemble_prompt("limited_water_fire", "DISTRIBUTED", env_lw)
    _assert("extinguish_fire" in plw, "limited_water_fire inherits fire content")
    _assert("Test limited-water note." in plw, "other_info should appear in P2 for limited_water_fire")

    p2 = assemble_prompt("rescue", "HYB_SUP", _ENV)

    p3 = assemble_prompt("super", "CENTRAL", _ENV)

    # Super + HYB_SUP activates Super_FEEDBACK; Super + HYB_TEA activates Super_DECISION (see P3_Role_Definition)
    sup_fb = assemble_prompt("super", "HYB_SUP", _ENV)
    sup_dc = assemble_prompt("super", "HYB_TEA", _ENV)
    _assert("provide feedback" in sup_fb, "Super_FEEDBACK block expected for super+HYB_SUP")
    _assert("provide feedback" not in sup_dc, "Super_FEEDBACK block must not appear for super+HYB_TEA")

    print("verify_prompt: OK (6 scenarios)")
    print(f"  fire/DISTRIBUTED           length={len(p)} words={_word_count(p)}")
    print(p)
    print(f"  limited_water_fire/DIST    length={len(plw)} words={_word_count(plw)}")
    print(f"  rescue/HYB_SUP     length={len(p2)} words={_word_count(p2)}")
    print(f"  super/CENTRAL      length={len(p3)} words={_word_count(p3)}")
    print(f"  super/HYB_SUP      length={len(sup_fb)} words={_word_count(sup_fb)}")
    print(f"  super/HYB_TEA      length={len(sup_dc)} words={_word_count(sup_dc)}")
    # print(sup_fb)


if __name__ == "__main__":
    main()

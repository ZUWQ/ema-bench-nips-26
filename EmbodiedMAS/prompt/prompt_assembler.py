"""
Assemble system prompts from P0–P10 modules with tag-conditioned blocks.

Tag syntax:
  <<[TAG]>>{ body }
  <<[TAG_A], [TAG_B]>>{ body }   # keep if any listed tag is active

Super-only extra tags (added to active_tags when conditions match):
  Super_FEEDBACK  — role super and coordination HYB_SUP
  Super_DECISION  — role super and coordination HYB_TEA
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

_MODULES_DIR = Path(__file__).parent

Role = Literal["fire", "rescue", "super", "limited_water_fire"]
Coordination = Literal["DISTRIBUTED", "HYB_SUP", "HYB_TEA", "CENTRAL"]

# Only Language-base Agent
_MODULE_ORDER: list[str] = [
    "P0_Global_Identity",
    "P1_Task_Description",
    "P2_Environment_Assumptions",
    "P3_Role_Definition",
    "P4_Coordination_Framework",
    "P5_Decision_Guidance",
    "P6_Shared_Summary_Requirement",
    "P7_Observation_Space",
    "P8_Action_Space",
    "P9_Execution_Format",
    "P10_Basic_Rules",
]

# Only Language-base Agent with Super Agent
_MODULE_ORDER_SUPER: list[str] = [
    "P0S_Global_Identity",
    "P1_Task_Description",
    "P2_Environment_Assumptions",
    "P3_Role_Definition",
    "P4_Coordination_Framework",
    "P5S_Decision_Guidance",
    "P6S_Shared_Summary_Requirement",
    "P7S_Observation_Space",
    "P8_Action_Space",
    "P9S_Execution_Format",
    "P10_Basic_Rules",
]

# Vision-Language Model Agent
_MODULE_ORDER_VL: list[str] = [
    "P0_Global_Identity",
    "P1_Task_Description",
    "P2_Environment_Assumptions",
    "P3_Role_Definition",
    "P4_Coordination_Framework",
    "P5_Decision_Guidance",
    "P6_Shared_Summary_Requirement",
    "P7VL_Observation_Space",
    "P8_Action_Space",
    "P9_Execution_Format",
    "P10_Basic_Rules",
]

# Vision-Language Model Agent with Super Agent
_MODULE_ORDER_VL_SUPER: list[str] = [
    "P0S_Global_Identity",
    "P1_Task_Description",
    "P2_Environment_Assumptions",
    "P3_Role_Definition",
    "P4_Coordination_Framework",
    "P5S_Decision_Guidance",
    "P6S_Shared_Summary_Requirement",
    "P7VLS_Observation_Space",
    "P8_Action_Space",
    "P9S_Execution_Format",
    "P10_Basic_Rules",
]

# VLM Agent
_MODULE_ORDER_VLM: list[str] = [
    "P0_Global_Identity",
    "P1_Task_Description",
    "P2_Environment_Assumptions",
    "P3_Role_Definition",
    "P4_Coordination_Framework",
    "P5_Decision_Guidance",
    "P6_Shared_Summary_Requirement",
    "P7VLM_Observation_Space",
    "P8VLM_Action_Space",
    "P9_Execution_Format",
    "P10VLM_Basic_Rules",
]

# VLM Agent with Super Agent
_MODULE_ORDER_VLM_SUPER: list[str] = [
    "P0S_Global_Identity",
    "P1_Task_Description",
    "P2_Environment_Assumptions",
    "P3_Role_Definition",
    "P4_Coordination_Framework",
    "P5S_Decision_Guidance",
    "P6S_Shared_Summary_Requirement",
    "P7VLMS_Observation_Space",
    "P8VLM_Action_Space",
    "P9S_Execution_Format",
    "P10VLM_Basic_Rules",
]

def _load_module(name: str) -> str:
    path = _MODULES_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")


def _fill_slots(text: str, slots: dict[str, str]) -> str:
    """Replace {SLOT_NAME} placeholders. Missing keys become empty string."""
    result = text
    for key, value in slots.items():
        result = result.replace(f"{{{key}}}", value or "")
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _parse_tag_list(header_inner: str) -> list[str]:
    header_inner = header_inner.strip()
    if not header_inner:
        return []
    if "[" not in header_inner:
        return [header_inner]
    tags: list[str] = []
    for part in header_inner.split(","):
        part = part.strip()
        if part.startswith("["):
            part = part[1:]
        if part.endswith("]"):
            part = part[:-1]
        tags.append(part.strip())
    return tags


def _strip_tag_blocks(text: str, active_tags: set[str]) -> str:
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("<<[", i):
            end_header = text.find("]>>", i + 3)
            if end_header == -1:
                out.append(text[i])
                i += 1
                continue
            header_inner = text[i + 3 : end_header]
            j = end_header + 3
            if j >= n or text[j] != "{":
                out.append(text[i : end_header + 3])
                i = end_header + 3
                continue
            tag_list = _parse_tag_list(header_inner)
            keep = any(t in active_tags for t in tag_list)
            depth = 1
            k = j + 1
            while k < n and depth > 0:
                c = text[k]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                k += 1
            body = text[j + 1 : k - 1]
            if keep:
                out.append(body)
            i = k
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _build_active_tags(role: Role, coordination: Coordination) -> set[str]:
    # limited_water_fire keeps fire's Firefighting_Agent tag and adds Limited_Water_Fire_Agent (e.g. P2 extra row).
    role_tags: set[str] = set()
    if role == "fire":
        role_tags.add("Firefighting_Agent")
    elif role == "limited_water_fire":
        role_tags.add("Firefighting_Agent")
        role_tags.add("Limited_Water_Fire_Agent")
    elif role == "rescue":
        role_tags.add("Rescue_Agent")
    elif role == "super":
        role_tags.add("Super_Agent")
    coord_tags: dict[Coordination, set[str]] = {
        "DISTRIBUTED": {"DISTRIBUTED_DECISION"},
        "HYB_SUP": {"Super_FEEDBACK"},
        "HYB_TEA": {"Super_DECISION"},
        "CENTRAL": {"CENTRALIZED_DECISION"},
    }
    tags = role_tags | coord_tags[coordination]
    if role == "super" and coordination == "HYB_SUP":
        tags.add("Super_FEEDBACK")
    if role == "super" and coordination == "HYB_TEA":
        tags.add("Super_DECISION")
    if role == "super" and coordination == "CENTRAL":
        tags.add("Super_CENTRAL")
    return tags


# String slots in P2-style templates; numeric slots come from the caller's environment — missing keys are not forced to defaults.
_ENV_SLOT_DEFAULTS: dict[str, str] = {
    "other_info": "",
}


def assemble_prompt(
    role: Role,
    coordination: Coordination,
    environment: dict[str, Any],
    obs: Literal["VL", "VLM", None] = None,
) -> str:
    """
    Assemble the full system prompt.

    Args:
        role: Agent role (fire / rescue / super / limited_water_fire).
        coordination: Coordination framework (DISTRIBUTED, HYB_SUP, HYB_TEA, CENTRAL).
        environment: Values for placeholders such as num_agents, num_fire_agents,
            num_rescue_agents, num_civilians, num_fires (numeric-ish), and other_info (str).
        obs: Observation space (VL, VLM, None).
    """
    active = _build_active_tags(role, coordination)
    slots: dict[str, str] = dict(_ENV_SLOT_DEFAULTS)
    slots.update({k: str(v) for k, v in (environment or {}).items()})

    if obs == "VLM":
        module_order = _MODULE_ORDER_VLM_SUPER if role == "super" else _MODULE_ORDER_VLM
    elif obs == "VL":
        module_order = _MODULE_ORDER_VL_SUPER if role == "super" else _MODULE_ORDER_VL
    else:
        module_order = _MODULE_ORDER_SUPER if role == "super" else _MODULE_ORDER

    parts: list[str] = []
    for mod_name in module_order:
        raw = _load_module(mod_name)
        stripped = _strip_tag_blocks(raw, active)
        filled = _fill_slots(stripped, slots)
        parts.append(filled)

    return "\n\n---\n\n".join(parts)

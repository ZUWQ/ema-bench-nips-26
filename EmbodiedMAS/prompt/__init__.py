"""Modular prompts (P0–P10) with tag-based assembly."""

from .prompt_assembler import (
    Coordination,
    Role,
    assemble_prompt,
)

__all__ = ["assemble_prompt", "assemble_prompt_vlmagent", "Coordination", "Role"]

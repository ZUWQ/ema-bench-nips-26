from __future__ import annotations

from pathlib import Path
from typing import Optional

import tongsim as ts
from tongsim.core.world_context import WorldContext

if __package__ is None or __package__ == "":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))


# Blueprints used for spawning actors (aligned with examples/demo_mas.py)
COIN_BLUEPRINT = "/Game/Developer/DemoCoin/BP_DemoCoin.BP_DemoCoin_C"


class EnvironmentWrapper:
    """
    Thin wrapper around TongSim SDK to expose common environment operations
    needed by higher-level agents.

    Responsibilities:
    - Reset level via `ts.UnaryAPI.reset_level`
    - Query environment state via `ts.UnaryAPI.query_info`
    - Spawn coins via `ts.UnaryAPI.spawn_actor`
    
    Note: Agent spawning and visual perception functionality has been moved to BaseAgent.
    """

    def __init__(self, context: WorldContext, log_dir: Optional[Path] = None):
        self._context = context
        self._conn = context.conn
        # Base directory for saving camera observations (if needed by other components)
        self._log_dir = log_dir or Path(__file__).resolve().parents[1] / "logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)

    # -------------
    # Core controls
    # -------------
    async def reset_level(self) -> bool:
        return await ts.UnaryAPI.reset_level(self._conn)

    async def query_info(self) -> list[dict]:
        return await ts.UnaryAPI.query_info(self._conn)

    async def spawn_coin(
        self,
        *,
        location: Optional[ts.Vector3] = None,
        name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        timeout: float = 5.0,
    ) -> Optional[dict]:
        """Spawn a demo coin actor and return its server-side info dict.

        Args:
            location: Spawn location; defaults to a sensible demo position.
            name: Optional actor name.
            tags: Optional list of gameplay tags.
            timeout: RPC timeout seconds.
        """
        spawn_location = location or ts.Vector3(1500, -2800, 890)
        return await ts.UnaryAPI.spawn_actor(
            self._conn,
            blueprint=COIN_BLUEPRINT,
            transform=ts.Transform(location=spawn_location),
            name=name,
            tags=tags,
            timeout=timeout,
        )

    # ---------------
    # Utility helpers
    # ---------------
    @property
    def context(self) -> WorldContext:
        return self._context

    async def close(self) -> None:
        """Clean up all resources."""
        # Environment wrapper cleanup (no agent-specific resources to clean)
        pass

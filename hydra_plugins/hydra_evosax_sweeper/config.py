from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from hydra.core.config_store import ConfigStore


@dataclass
class EvosaxSweeperConf:
    _target_: str = "hydra_plugins.hydra_evosax_sweeper.evosax_sweeper.EvosaxSweeper"
    strategy: str = "CMA_ES"
    popsize: int = 32
    num_generations: int = 50
    direction: str = "maximize"
    seed: int = 0
    strategy_kwargs: Dict[str, Any] = field(default_factory=dict)
    params: Optional[Dict[str, str]] = None


ConfigStore.instance().store(
    group="hydra/sweeper",
    name="evosax",
    node=EvosaxSweeperConf,
    provider="evosax_sweeper",
)

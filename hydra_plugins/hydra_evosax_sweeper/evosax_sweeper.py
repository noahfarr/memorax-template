from typing import Any, Dict, List, Optional

from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig


class EvosaxSweeper(Sweeper):
    def __init__(
        self,
        strategy: str,
        popsize: int,
        num_generations: int,
        direction: str,
        seed: int,
        strategy_kwargs: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> None:
        from ._impl import EvosaxSweeperImpl

        self.sweeper = EvosaxSweeperImpl(
            strategy=strategy,
            popsize=popsize,
            num_generations=num_generations,
            direction=direction,
            seed=seed,
            strategy_kwargs=strategy_kwargs,
            params=params,
        )

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.sweeper.setup(
            hydra_context=hydra_context,
            task_function=task_function,
            config=config,
        )

    def sweep(self, arguments: List[str]) -> Any:
        return self.sweeper.sweep(arguments)

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.override_parser.types import (
    ChoiceSweep,
    IntervalSweep,
    RangeSweep,
    Transformer,
)
from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


class ParameterSpec:
    """A single parameter in the evolutionary search space."""

    def __init__(self, name: str, spec_type: str, **kwargs):
        self.name = name
        self.spec_type = spec_type
        self.kwargs = kwargs

    @property
    def num_dims(self) -> int:
        if self.spec_type == "categorical":
            return len(self.kwargs["choices"])
        return 1

    def decode(self, values: np.ndarray) -> Any:
        if self.spec_type == "float":
            low, high = self.kwargs["low"], self.kwargs["high"]
            t = float(_sigmoid(values[0]))
            if self.kwargs.get("log", False):
                return float(
                    np.exp(np.log(low) + (np.log(high) - np.log(low)) * t)
                )
            return float(low + (high - low) * t)
        elif self.spec_type == "int":
            low, high = self.kwargs["low"], self.kwargs["high"]
            t = float(_sigmoid(values[0]))
            if self.kwargs.get("log", False):
                return int(
                    np.clip(
                        np.round(
                            np.exp(
                                np.log(low) + (np.log(high) - np.log(low)) * t
                            )
                        ),
                        low,
                        high,
                    )
                )
            return int(np.clip(np.round(low + (high - low) * t), low, high))
        elif self.spec_type == "categorical":
            return self.kwargs["choices"][int(np.argmax(values))]
        raise ValueError(f"Unknown spec type: {self.spec_type}")


def create_params_from_overrides(
    arguments: List[str],
) -> Tuple[List[ParameterSpec], Dict[str, str]]:
    parser = OverridesParser.create()
    parsed = parser.parse_overrides(arguments)

    specs = []
    fixed_params = {}

    for override in parsed:
        param_name = override.get_key_element()

        if not override.is_sweep_override():
            fixed_params[param_name] = override.get_value_element_as_str()
            continue

        value = override.value()

        if override.is_interval_sweep():
            assert isinstance(value, IntervalSweep)
            assert value.start is not None
            assert value.end is not None
            is_log = "log" in value.tags
            if isinstance(value.start, int) and isinstance(value.end, int):
                specs.append(
                    ParameterSpec(
                        name=param_name,
                        spec_type="int",
                        low=int(value.start),
                        high=int(value.end),
                        log=is_log,
                    )
                )
            else:
                specs.append(
                    ParameterSpec(
                        name=param_name,
                        spec_type="float",
                        low=float(value.start),
                        high=float(value.end),
                        log=is_log,
                    )
                )

        elif override.is_range_sweep():
            assert isinstance(value, RangeSweep)
            assert value.start is not None
            assert value.stop is not None
            step = value.step or 1
            if value.shuffle or step != 1:
                choices = list(
                    override.sweep_iterator(transformer=Transformer.encode)
                )
                specs.append(
                    ParameterSpec(
                        name=param_name,
                        spec_type="categorical",
                        choices=choices,
                    )
                )
            else:
                specs.append(
                    ParameterSpec(
                        name=param_name,
                        spec_type="int",
                        low=int(value.start),
                        high=int(value.stop - step),
                    )
                )

        elif override.is_choice_sweep():
            assert isinstance(value, ChoiceSweep)
            choices = list(
                override.sweep_iterator(transformer=Transformer.encode)
            )
            specs.append(
                ParameterSpec(
                    name=param_name,
                    spec_type="categorical",
                    choices=choices,
                )
            )
        else:
            raise NotImplementedError(
                f"{override} is not supported by the evosax sweeper."
            )

    return specs, fixed_params


class EvosaxSweeperImpl(Sweeper):
    def __init__(
        self,
        strategy: str = "CMA_ES",
        popsize: int = 32,
        num_generations: int = 50,
        direction: str = "maximize",
        seed: int = 0,
        strategy_kwargs: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.strategy_name = strategy
        self.popsize = popsize
        self.num_generations = num_generations
        self.direction = direction
        self.seed = seed
        self.strategy_kwargs = strategy_kwargs or {}
        self.params = params
        self.job_idx: int = 0

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.job_idx = 0
        self.config = config
        self.hydra_context = hydra_context
        self.launcher = Plugins.instance().instantiate_launcher(
            config=config,
            hydra_context=hydra_context,
            task_function=task_function,
        )
        self.sweep_dir = config.hydra.sweep.dir

    def _parse_sweeper_params_config(self) -> List[str]:
        if self.params is None:
            return []
        return [f"{k!s}={v}" for k, v in self.params.items()]

    def sweep(self, arguments: List[str]) -> Any:
        import evosax.algorithms
        import jax
        import jax.numpy as jnp
        from evosax.algorithms.distribution_based.base import (
            DistributionBasedAlgorithm,
        )
        from evosax.algorithms.population_based.base import (
            PopulationBasedAlgorithm,
        )

        assert self.config is not None
        assert self.launcher is not None

        params_conf = self._parse_sweeper_params_config()
        params_conf.extend(arguments)
        specs, fixed_params = create_params_from_overrides(params_conf)
        num_dims = sum(s.num_dims for s in specs)

        if num_dims == 0:
            raise ValueError(
                "No sweep parameters found. "
                "Use interval(), range(), or choice() syntax."
            )

        log.info(
            f"EvosaxSweeper: strategy={self.strategy_name}, "
            f"popsize={self.popsize}, generations={self.num_generations}, "
            f"dims={num_dims}, direction={self.direction}"
        )
        for spec in specs:
            log.info(f"  {spec.name}: {spec.spec_type} ({spec.kwargs})")

        strategy_cls = getattr(evosax.algorithms, self.strategy_name)
        strategy = strategy_cls(
            population_size=self.popsize,
            solution=jnp.zeros(num_dims),
            **self.strategy_kwargs,
        )
        es_params = strategy.default_params
        key = jax.random.key(self.seed)

        if isinstance(strategy, DistributionBasedAlgorithm):
            key, init_key = jax.random.split(key)
            state = strategy.init(init_key, jnp.zeros(num_dims), es_params)
        elif isinstance(strategy, PopulationBasedAlgorithm):
            key, init_key, pop_key = jax.random.split(key, 3)
            init_pop = jax.random.normal(pop_key, (self.popsize, num_dims))
            init_fitness = jnp.full(self.popsize, jnp.inf)
            state = strategy.init(init_key, init_pop, init_fitness, es_params)
        else:
            raise ValueError(
                f"Unknown strategy family for {self.strategy_name}"
            )

        best_val = float("-inf") if self.direction == "maximize" else float("inf")
        best_params = None

        for gen in range(self.num_generations):
            key, ask_key = jax.random.split(key)
            x, state = strategy.ask(ask_key, state, es_params)
            x_np = np.array(x)

            # Decode population into override sequences
            batch_overrides: List[Tuple[str, ...]] = []
            for i in range(self.popsize):
                overrides = []
                offset = 0
                for spec in specs:
                    dims = spec.num_dims
                    val = spec.decode(x_np[i, offset : offset + dims])
                    overrides.append(f"{spec.name}={val}")
                    offset += dims
                for name, val in fixed_params.items():
                    overrides.append(f"{name}={val}")
                batch_overrides.append(tuple(overrides))

            returns = self.launcher.launch(
                batch_overrides, initial_job_idx=self.job_idx
            )
            self.job_idx += len(returns)

            # Collect fitness values
            fitness = np.array(
                [
                    float(r.return_value)
                    if r.return_value is not None
                    else float("nan")
                    for r in returns
                ]
            )

            nan_mask = np.isnan(fitness)
            if nan_mask.any():
                log.warning(
                    f"Generation {gen + 1}: "
                    f"{nan_mask.sum()}/{self.popsize} runs failed"
                )
                worst = (
                    float("-inf")
                    if self.direction == "maximize"
                    else float("inf")
                )
                fitness = np.where(nan_mask, worst, fitness)

            # evosax minimizes by convention
            es_fitness = jnp.array(
                -fitness if self.direction == "maximize" else fitness
            )
            key, tell_key = jax.random.split(key)
            state, _metrics = strategy.tell(
                tell_key, x, es_fitness, state, es_params
            )

            # Track best
            if self.direction == "maximize":
                gen_best_idx = int(np.argmax(fitness))
            else:
                gen_best_idx = int(np.argmin(fitness))
            gen_best_val = float(fitness[gen_best_idx])

            is_better = (
                gen_best_val > best_val
                if self.direction == "maximize"
                else gen_best_val < best_val
            )
            if is_better:
                best_val = gen_best_val
                best_params = {}
                offset = 0
                for spec in specs:
                    dims = spec.num_dims
                    best_params[spec.name] = spec.decode(
                        x_np[gen_best_idx, offset : offset + dims]
                    )
                    offset += dims

            log.info(
                f"Generation {gen + 1}/{self.num_generations}: "
                f"gen_best={gen_best_val:.4f}, overall_best={best_val:.4f}"
            )

        log.info(f"Best parameters: {best_params}")
        log.info(f"Best value: {best_val:.4f}")

        results_to_serialize = {
            "name": "evosax",
            "best_params": best_params,
            "best_value": best_val,
            "strategy": self.strategy_name,
            "num_generations": self.num_generations,
            "popsize": self.popsize,
        }
        OmegaConf.save(
            OmegaConf.create(results_to_serialize),
            f"{self.sweep_dir}/optimization_results.yaml",
        )

        return best_val

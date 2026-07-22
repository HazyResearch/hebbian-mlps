"""GPU-aware grid and binary search orchestration.

The scheduler assigns jobs round-robin across visible GPUs and runs each job in
an isolated spawned process. Experiment-specific aggregation remains in the
search config, which keeps this module independent of model code.
"""

from __future__ import annotations

import asyncio
import itertools
import multiprocessing as mp
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import pandas as pd

from hebbian.config import pydraclass


class ConfigLike(Protocol):
    def finalize(self) -> None: ...


class Job(ABC):
    @abstractmethod
    def run(self) -> Any: ...

    @abstractmethod
    def get_out_file(self) -> str | None: ...


@dataclass
class GPUJobResult:
    success: bool
    error: str | None
    gpu_id: int
    out_file: str | None
    job: Job
    result: Any | None


@pydraclass
class _BaseSearchConfig:
    base_dir: str | None = None
    sweep_props: dict[str, list[Any]] | None = None
    base_experiment_config: ConfigLike | None = None

    def _get_experiment_config_and_base_dir(self, **prop_values: Any) -> tuple[ConfigLike, str]:
        config, base_dir = self.get_experiment_config_and_base_dir(**prop_values)
        config.finalize()
        return config, base_dir

    def get_experiment_config_and_base_dir(self, **prop_values: Any) -> tuple[ConfigLike, str]:
        raise NotImplementedError

    def run_experiment_config(self, config: ConfigLike) -> Any:
        raise NotImplementedError

    def agg_results(self, results: list[GPUJobResult]) -> Any:
        raise NotImplementedError


@pydraclass
class BinarySearchConfig(_BaseSearchConfig):
    prop: str | None = None
    range: tuple[float, float] | None = None
    precision: float | None = None
    success_direction_lower: bool = True

    def agg_results(self, results: list[GPUJobResult]) -> tuple[bool, Any]:
        raise NotImplementedError


@pydraclass
class GridSearchConfig(_BaseSearchConfig):
    pass


class _SearchJob(Job):
    def __init__(
        self,
        config: ConfigLike,
        base_dir: str,
        run_experiment: Callable[[ConfigLike], Any],
    ) -> None:
        self.config = config
        self.base_dir = base_dir
        self.run_experiment = run_experiment

    def run(self) -> Any:
        return self.run_experiment(self.config)

    def get_out_file(self) -> str:
        if not self.base_dir:
            raise ValueError("Search jobs require a non-empty base_dir")
        return str(Path(self.base_dir) / "experiment_output.log")


class BinarySearchJob(_SearchJob):
    pass


class GridSearchJob(_SearchJob):
    pass


def get_jobs(
    properties: dict[str, list[Any]] | None,
    config: _BaseSearchConfig,
    job_type: type[_SearchJob],
) -> list[_SearchJob]:
    names = list(properties or {})
    values = [(properties or {})[name] for name in names]
    combinations = itertools.product(*values) if values else [()]
    jobs: list[_SearchJob] = []
    for combination in combinations:
        props = dict(zip(names, combination))
        experiment_config, base_dir = config._get_experiment_config_and_base_dir(**props)
        jobs.append(job_type(experiment_config, base_dir, config.run_experiment_config))
    return jobs


def get_jobs_for_mid(config: BinarySearchConfig, midpoint: float) -> list[_SearchJob]:
    properties = dict(config.sweep_props or {})
    if config.prop is None:
        raise ValueError("BinarySearchConfig.prop must be set")
    properties[config.prop] = [midpoint]
    return get_jobs(properties, config, BinarySearchJob)


def get_available_gpu_ids() -> list[int]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [device.strip() for device in visible.split(",") if device.strip()]
        try:
            return [int(device) for device in devices]
        except ValueError as exc:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES must contain numeric GPU IDs for the sweep runner"
            ) from exc
    import torch

    return list(range(torch.cuda.device_count()))


def _execute_job(job: Job, gpu_id: int) -> GPUJobResult:
    out_file = job.get_out_file()
    try:
        if out_file is None:
            result = job.run()
        else:
            output = Path(out_file)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8") as stream:
                print(f"Running on GPU {gpu_id}", file=stream, flush=True)
                with redirect_stdout(stream), redirect_stderr(stream):
                    result = job.run()
        return GPUJobResult(True, None, gpu_id, out_file, job, result)
    except Exception as exc:
        return GPUJobResult(False, str(exc), gpu_id, out_file, job, None)


def _gpu_worker(job: Job, gpu_id: int) -> GPUJobResult:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return _execute_job(job, gpu_id)


def _run_in_spawned_process(job: Job, gpu_id: int) -> GPUJobResult:
    context = mp.get_context("spawn")
    with context.Pool(processes=1) as pool:
        return pool.apply(_gpu_worker, (job, gpu_id))


class GPUScheduler:
    """Schedule isolated jobs over the GPUs visible to the current process."""

    def __init__(
        self,
        max_gpus: int | None = None,
        simultaneous_jobs_per_gpu: int | None = 1,
    ) -> None:
        gpu_ids = get_available_gpu_ids()
        if max_gpus is not None:
            gpu_ids = gpu_ids[:max_gpus]
        if not gpu_ids:
            raise ValueError("No GPUs are visible; use the experiment's local/CPU runner")
        slots = 1 if simultaneous_jobs_per_gpu is None else simultaneous_jobs_per_gpu
        if slots < 1:
            raise ValueError("simultaneous_jobs_per_gpu must be positive")
        self.gpu_ids = gpu_ids
        self._next_gpu = 0
        self._semaphores = {gpu_id: asyncio.Semaphore(slots) for gpu_id in gpu_ids}
        self._executor = ThreadPoolExecutor(max_workers=len(gpu_ids) * slots)

    async def run_job(self, job: Job) -> GPUJobResult:
        gpu_id = self.gpu_ids[self._next_gpu % len(self.gpu_ids)]
        self._next_gpu += 1
        async with self._semaphores[gpu_id]:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor,
                _run_in_spawned_process,
                job,
                gpu_id,
            )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


async def run_binary_search(
    config: BinarySearchConfig,
    scheduler: GPUScheduler,
) -> tuple[Any, Any]:
    if config.range is None or config.precision is None:
        raise ValueError("Binary search requires range and precision")
    lo, hi = config.range
    achieved_results = None
    failed_results = None
    start = time.time()

    while (hi - lo) >= config.precision:
        midpoint = (lo + hi) / 2
        results = await asyncio.gather(
            *(scheduler.run_job(job) for job in get_jobs_for_mid(config, midpoint))
        )
        success, aggregate = config.agg_results(results)
        if success:
            achieved_results = (midpoint, aggregate)
            if config.success_direction_lower:
                hi = midpoint
            else:
                lo = midpoint
        else:
            failed_results = (midpoint, aggregate)
            if config.success_direction_lower:
                lo = midpoint
            else:
                hi = midpoint

    output_dir = Path(config.base_dir or ".")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    pd.to_pickle(
        {
            "search_range": [lo, hi],
            "precision": config.precision,
            "achieved_results": achieved_results,
            "failed_results": failed_results,
            "total_time": time.time() - start,
            "timestamp": timestamp,
        },
        output_dir / f"binary_search_results_{timestamp}.pkl",
    )
    return achieved_results, failed_results


async def run_grid_search(config: GridSearchConfig, scheduler: GPUScheduler) -> Any:
    start = time.time()
    results = await asyncio.gather(
        *(scheduler.run_job(job) for job in get_jobs(config.sweep_props, config, GridSearchJob))
    )
    aggregate = config.agg_results(results)
    output_dir = Path(config.base_dir or ".")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    pd.to_pickle(
        {
            "results": aggregate,
            "total_time": time.time() - start,
            "timestamp": timestamp,
        },
        output_dir / f"grid_search_results_{timestamp}.pkl",
    )
    return aggregate


async def _run_all_binary(configs: list[BinarySearchConfig], scheduler: GPUScheduler) -> list[Any]:
    return await asyncio.gather(*(run_binary_search(config, scheduler) for config in configs))


async def _run_all_grid(configs: list[GridSearchConfig], scheduler: GPUScheduler) -> list[Any]:
    return await asyncio.gather(*(run_grid_search(config, scheduler) for config in configs))


def run_binary_searches(
    configs: list[BinarySearchConfig],
    max_gpus: int | None = None,
    simultaneous_jobs_per_gpu: int | None = 1,
) -> list[Any]:
    scheduler = GPUScheduler(max_gpus, simultaneous_jobs_per_gpu)
    try:
        return asyncio.run(_run_all_binary(configs, scheduler))
    finally:
        scheduler.shutdown()


def run_grid_searches(
    configs: list[GridSearchConfig],
    max_gpus: int | None = None,
    simultaneous_jobs_per_gpu: int | None = 1,
) -> list[Any]:
    scheduler = GPUScheduler(max_gpus, simultaneous_jobs_per_gpu)
    try:
        return asyncio.run(_run_all_grid(configs, scheduler))
    finally:
        scheduler.shutdown()


__all__ = [
    "BinarySearchConfig",
    "GPUJobResult",
    "GPUScheduler",
    "GridSearchConfig",
    "get_available_gpu_ids",
    "get_jobs_for_mid",
    "run_binary_search",
    "run_binary_searches",
    "run_grid_search",
    "run_grid_searches",
]

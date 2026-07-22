"""Shared runtime for paper reproduction commands."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


class PaperError(RuntimeError):
    """A user-facing paper reproduction error."""


def _resolve_path(value: str | Path, *, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


def default_gpu_count() -> int:
    try:
        import torch

        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


@dataclass(frozen=True)
class PaperContext:
    """Resolved paths and execution options shared by all paper targets."""

    target: str
    repo_root: Path
    python: str
    mode: str
    artifact_root: Path
    result_root: Path
    figure_root: Path
    paper_root: Path | None
    n_gpus: int
    jobs_per_gpu: int
    device: str

    @classmethod
    def from_args(cls, target: str, args) -> "PaperContext":
        repo_root = _resolve_path(
            args.repo_root
            or os.environ.get("REPO_ROOT")
            or Path(__file__).resolve().parents[2]
        )
        mode = args.mode or os.environ.get("MODE", "plot-only")
        if mode not in {"plot-only", "run-and-plot"}:
            raise PaperError(f"mode must be plot-only or run-and-plot; got {mode!r}")

        artifact_root = _resolve_path(
            args.artifact_root
            or os.environ.get("ARTIFACT_ROOT")
            or repo_root / "artifacts/paper",
            base=repo_root,
        )
        result_root = _resolve_path(
            args.result_root
            or os.environ.get("RESULT_ROOT")
            or artifact_root / "results",
            base=repo_root,
        )
        figure_root = _resolve_path(
            args.figure_root
            or os.environ.get("FIGURE_ROOT")
            or artifact_root / "figures",
            base=repo_root,
        )
        paper_value = args.paper_root or os.environ.get("PAPER_ROOT")
        paper_root = (
            _resolve_path(paper_value, base=repo_root) if paper_value else None
        )
        n_gpus = args.n_gpus
        if n_gpus is None:
            n_gpus = int(os.environ.get("N_GPUS", default_gpu_count()))
        jobs_per_gpu = args.jobs_per_gpu
        if jobs_per_gpu is None:
            jobs_per_gpu = int(
                os.environ.get(
                    "SIMULTANEOUS_JOBS_PER_GPU",
                    os.environ.get("JOBS_PER_GPU", "1"),
                )
            )
        if n_gpus < 1 or jobs_per_gpu < 1:
            raise PaperError("n-gpus and jobs-per-gpu must both be positive")

        return cls(
            target=target,
            repo_root=repo_root,
            python=args.python or os.environ.get("PYTHON_BIN", sys.executable),
            mode=mode,
            artifact_root=artifact_root,
            result_root=result_root,
            figure_root=figure_root,
            paper_root=paper_root,
            n_gpus=n_gpus,
            jobs_per_gpu=jobs_per_gpu,
            device=args.device or os.environ.get("DEVICE", "cuda"),
        )

    @property
    def should_run(self) -> bool:
        return self.mode == "run-and-plot"

    @property
    def gpu_ids(self) -> str:
        return ",".join(str(index) for index in range(self.n_gpus))

    def env(self, name: str, default: str | None = None) -> str | None:
        return os.environ.get(name, default)

    def env_bool(self, name: str, default: bool = False) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def output_dir(self, name: str | None = None) -> Path:
        return self.figure_root / (name or self.target)

    def result_dir(self, name: str) -> Path:
        return self.result_root / name

    def path(self, value: str | Path) -> Path:
        """Resolve a target-specific path relative to the repository root."""
        return _resolve_path(value, base=self.repo_root)

    def run(
        self,
        command: Sequence[str | Path],
        *,
        extra_env: Mapping[str, str] | None = None,
    ) -> None:
        argv = [str(value) for value in command]
        print(f"+ {shlex.join(argv)}", file=sys.stderr, flush=True)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        pythonpath = [str(self.repo_root / "src"), str(self.repo_root)]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        if extra_env:
            env.update({key: str(value) for key, value in extra_env.items()})
        subprocess.run(argv, cwd=self.repo_root, env=env, check=True)

    def run_module(
        self,
        module: str,
        *args: str | Path,
        extra_env: Mapping[str, str] | None = None,
    ) -> None:
        self.run([self.python, "-m", module, *args], extra_env=extra_env)

    def require_file(self, path: str | Path, hint: str) -> Path:
        resolved = _resolve_path(path, base=self.repo_root)
        if not resolved.is_file():
            raise PaperError(f"required file not found: {resolved}\n{hint}")
        return resolved

    def require_dir(self, path: str | Path, hint: str) -> Path:
        resolved = _resolve_path(path, base=self.repo_root)
        if not resolved.is_dir():
            raise PaperError(f"required directory not found: {resolved}\n{hint}")
        return resolved

    def copy_asset(self, source: str | Path, paper_relative_path: str) -> None:
        if self.paper_root is None:
            return
        source_path = self.require_file(
            source, "The plotting command produced no asset."
        )
        destination = self.paper_root / paper_relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        print(f"[paper] copied {source_path} -> {destination}", file=sys.stderr)

    def latest(self, directory: str | Path, pattern: str) -> Path | None:
        matches = sorted(self.path(directory).glob(pattern))
        return matches[-1] if matches else None


def quoted_override(name: str, value: str | Path) -> str:
    """Format a string override for the repository's pydra-style CLIs."""
    return f'{name}="{value}"'

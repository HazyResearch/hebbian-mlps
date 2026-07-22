"""Extract Qwen3 decoder-MLP activation pairs for the LLM paper experiments.

For each non-padding token, this records:

* ``x``: the output of ``post_attention_layernorm`` in decoder layer 14, i.e.
  the normalized hidden state passed into that layer's MLP;
* ``y``: the output of that same layer's MLP before it is added back to the
  residual stream.

The default settings reproduce the paper's Qwen3-0.6B layer-14 bundle protocol
up to the requested number of token pairs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from hebbian.expts.llm_embeddings.bundle import FORMAT_VERSION, save_activation_pair


DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B-Base"
DEFAULT_DATASET_NAME = "wikitext"
DEFAULT_DATASET_CONFIG = "wikitext-103-raw-v1"
DEFAULT_DATASET_SPLIT = "train"
DEFAULT_TEXT_COLUMN = "text"
DEFAULT_LAYER_INDEX = 14
DEFAULT_MAX_PAIRS = 500_000


def _torch_dtype(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            f"unsupported dtype {name!r}; expected one of {sorted(table)}"
        ) from exc


def _model_torch_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    return _torch_dtype(name)


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError(f"expected hook output to contain a tensor, got {type(value)!r}")


def resolve_decoder_layers(model: Any) -> Sequence[Any]:
    """Return the decoder layer sequence for HF Qwen-style models."""

    candidates = [
        ("layers", model),
        ("model.layers", getattr(model, "model", None)),
        ("model.model.layers", getattr(getattr(model, "model", None), "model", None)),
    ]
    for _, owner in candidates:
        if owner is not None and hasattr(owner, "layers"):
            layers = getattr(owner, "layers")
            if isinstance(layers, Sequence) or hasattr(layers, "__len__"):
                return layers
    raise ValueError(
        "could not locate decoder layers; expected model.layers or model.model.layers"
    )


class QwenMlpHookCapture:
    """Forward hooks that capture one layer's MLP input and output."""

    def __init__(self, model: Any, layer_index: int):
        layers = resolve_decoder_layers(model)
        if layer_index < 0 or layer_index >= len(layers):
            raise IndexError(
                f"layer_index={layer_index} outside decoder layer range [0, {len(layers) - 1}]"
            )
        self.layer_index = int(layer_index)
        self.layer_count = int(len(layers))
        self.layer = layers[layer_index]
        if not hasattr(self.layer, "post_attention_layernorm"):
            raise ValueError("Qwen layer is missing post_attention_layernorm")
        if not hasattr(self.layer, "mlp"):
            raise ValueError("Qwen layer is missing mlp")
        self.x: torch.Tensor | None = None
        self.y: torch.Tensor | None = None
        self._handles = [
            self.layer.post_attention_layernorm.register_forward_hook(self._capture_x),
            self.layer.mlp.register_forward_hook(self._capture_y),
        ]

    def _capture_x(self, _module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        self.x = _first_tensor(output).detach()

    def _capture_y(self, _module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        self.y = _first_tensor(output).detach()

    def clear(self) -> None:
        self.x = None
        self.y = None

    def pop(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.x is None or self.y is None:
            raise RuntimeError("activation hooks did not fire for both x and y")
        x, y = self.x, self.y
        self.clear()
        return x, y

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def iter_text_file(path: str | Path) -> Iterator[str]:
    """Yield non-empty lines from a local text file."""

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                yield text


def iter_hf_dataset_text(
    *,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    text_column: str,
    streaming: bool,
) -> Iterator[str]:
    """Yield non-empty text examples from a Hugging Face dataset."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face datasets is required for dataset extraction. "
            "Install with `pip install -e .[llm]` or pass --text-file."
        ) from exc

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        streaming=streaming,
    )
    for example in dataset:
        text = str(example.get(text_column, "")).strip()
        if text:
            yield text


def batched(items: Iterable[str], batch_size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _forward_model(model: Any, batch: dict[str, torch.Tensor]) -> None:
    kwargs = dict(batch)
    kwargs["use_cache"] = False
    try:
        model(**kwargs)
    except TypeError:
        kwargs.pop("use_cache", None)
        model(**kwargs)


def _collect_nonpadding_rows(
    x: torch.Tensor,
    y: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    remaining: int,
    save_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 3 or y.ndim != 3 or x.shape != y.shape:
        raise ValueError(
            "captured activations must have matching [batch, seq, d] shapes; "
            f"got x={tuple(x.shape)}, y={tuple(y.shape)}"
        )
    mask = attention_mask.to(device=x.device, dtype=torch.bool)
    x_rows = x[mask]
    y_rows = y[mask]
    if x_rows.shape[0] == 0:
        return (
            torch.empty((0, x.shape[-1]), dtype=save_dtype),
            torch.empty((0, y.shape[-1]), dtype=save_dtype),
        )
    take = min(int(remaining), int(x_rows.shape[0]))
    return (
        x_rows[:take].to(device="cpu", dtype=save_dtype),
        y_rows[:take].to(device="cpu", dtype=save_dtype),
    )


def extract_qwen3_activation_bundle(
    *,
    output_root: str | Path,
    model_id: str = DEFAULT_MODEL_ID,
    layer_index: int = DEFAULT_LAYER_INDEX,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    batch_size: int = 4,
    max_length: int = 512,
    text_file: str | Path | None = None,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_config: str | None = DEFAULT_DATASET_CONFIG,
    dataset_split: str = DEFAULT_DATASET_SPLIT,
    text_column: str = DEFAULT_TEXT_COLUMN,
    streaming: bool = True,
    device: str = "cuda",
    model_dtype: torch.dtype | str = "auto",
    save_dtype: torch.dtype = torch.float32,
    trust_remote_code: bool = False,
    allow_short: bool = False,
    revision: str | None = None,
) -> Path:
    """Extract and save an activation bundle, returning the activation dir."""

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for Qwen extraction; install project deps first"
        ) from exc

    output_root = Path(output_root).expanduser()
    activation_dir = output_root / "activations"
    activation_dir.mkdir(parents=True, exist_ok=True)

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested --device cuda, but torch.cuda.is_available() is false")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        revision=revision,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise RuntimeError("tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "revision": revision,
    }
    try:
        model = AutoModel.from_pretrained(model_id, dtype=model_dtype, **model_kwargs)
    except TypeError:
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=model_dtype,
            **model_kwargs,
        )
    model.eval()
    model.to(device)
    capture = QwenMlpHookCapture(model, int(layer_index))

    if text_file is not None:
        text_iter = iter_text_file(text_file)
        source: dict[str, Any] = {"text_file": str(Path(text_file).expanduser())}
    else:
        text_iter = iter_hf_dataset_text(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=dataset_split,
            text_column=text_column,
            streaming=streaming,
        )
        source = {
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "dataset_split": dataset_split,
            "text_column": text_column,
            "streaming": streaming,
        }

    x_store: torch.Tensor | None = None
    y_store: torch.Tensor | None = None
    filled = 0

    progress = tqdm(total=int(max_pairs), desc="activation pairs", unit="tok")
    try:
        with torch.inference_mode():
            for texts in batched(text_iter, int(batch_size)):
                encoded = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=int(max_length),
                )
                attention_mask = encoded.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(encoded["input_ids"])
                    encoded["attention_mask"] = attention_mask
                batch = {key: value.to(device) for key, value in encoded.items()}
                capture.clear()
                _forward_model(model, batch)
                x_batch, y_batch = capture.pop()
                x_rows, y_rows = _collect_nonpadding_rows(
                    x_batch,
                    y_batch,
                    batch["attention_mask"],
                    remaining=int(max_pairs) - filled,
                    save_dtype=save_dtype,
                )
                if x_rows.numel() == 0:
                    continue
                if x_store is None:
                    d_model = int(x_rows.shape[1])
                    x_store = torch.empty((int(max_pairs), d_model), dtype=save_dtype)
                    y_store = torch.empty((int(max_pairs), d_model), dtype=save_dtype)
                take = int(x_rows.shape[0])
                x_store[filled : filled + take].copy_(x_rows)
                y_store[filled : filled + take].copy_(y_rows)
                filled += take
                progress.update(take)
                if filled >= int(max_pairs):
                    break
    finally:
        progress.close()
        capture.close()

    if x_store is None or y_store is None or filled == 0:
        raise RuntimeError("no activation pairs were captured")
    if filled < int(max_pairs):
        if not allow_short:
            raise RuntimeError(
                f"captured only {filled} pairs, requested {max_pairs}; pass --allow-short "
                "for short inputs"
            )
        x_store = x_store[:filled].clone()
        y_store = y_store[:filled].clone()

    metadata = {
        "format": FORMAT_VERSION,
        "model_id": model_id,
        "revision": revision,
        "layer_index": int(layer_index),
        "layer_count": capture.layer_count,
        "activation_x": "decoder.layers[layer_index].post_attention_layernorm output",
        "activation_y": "decoder.layers[layer_index].mlp output before residual add",
        "num_pairs": int(filled),
        "d_model": int(x_store.shape[1]),
        "max_length": int(max_length),
        "batch_size": int(batch_size),
        "save_dtype": str(save_dtype),
        "model_dtype": str(model_dtype),
        "source": source,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    save_activation_pair(activation_dir, x_store, y_store, metadata=metadata)
    return activation_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract Qwen3 layer-14 MLP activation pairs into x.pt/y.pt."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--layer-index", type=int, default=DEFAULT_LAYER_INDEX)
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN)
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-dtype", default="auto", type=_model_torch_dtype)
    parser.add_argument("--save-dtype", default=torch.float32, type=_torch_dtype)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-short", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    activation_dir = extract_qwen3_activation_bundle(
        output_root=args.output_root,
        model_id=args.model_id,
        revision=args.revision,
        layer_index=args.layer_index,
        max_pairs=args.max_pairs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        text_file=args.text_file,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        text_column=args.text_column,
        streaming=not args.no_streaming,
        device=args.device,
        model_dtype=args.model_dtype,
        save_dtype=args.save_dtype,
        trust_remote_code=args.trust_remote_code,
        allow_short=args.allow_short,
    )
    info = {
        "activation_dir": str(activation_dir),
        "x": str(activation_dir / "x.pt"),
        "y": str(activation_dir / "y.pt"),
        "metadata": str(activation_dir / "metadata.json"),
    }
    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

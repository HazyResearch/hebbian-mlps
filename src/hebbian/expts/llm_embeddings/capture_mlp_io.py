"""Capture Qwen-style decoder-MLP IO pairs with the paper's packed-token protocol.

This implementation preserves the data semantics used by the paper:

* tokenize WikiText/text without special tokens;
* concatenate into a continuous token stream;
* split into fixed-length chunks;
* run full chunks through the model for context;
* save MLP input/output rows for all positions except the final position in
  each chunk, so every saved row has a valid next-token target.

The output is written in the public bundle layout used by the paper scripts:

    <output_root>/activations/x.pt
    <output_root>/activations/y.pt
    <output_root>/activations/metadata.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import platform
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from hebbian.expts.llm_embeddings.bundle import FORMAT_VERSION, save_activation_pair
from hebbian.expts.llm_embeddings.extract_qwen3 import (
    DEFAULT_DATASET_CONFIG,
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_SPLIT,
    DEFAULT_LAYER_INDEX,
    DEFAULT_MAX_PAIRS,
    DEFAULT_MODEL_ID,
    DEFAULT_TEXT_COLUMN,
    _first_tensor,
    _model_torch_dtype,
    _torch_dtype,
    iter_hf_dataset_text,
    iter_text_file,
    resolve_decoder_layers,
)


DEFAULT_SEQ_LENGTH = 1024


class MlpInputOutputCapture:
    """Forward hook that captures the input and output of one decoder MLP."""

    def __init__(self, model: Any, layer_index: int):
        layers = resolve_decoder_layers(model)
        if layer_index < 0 or layer_index >= len(layers):
            raise IndexError(
                f"layer_index={layer_index} outside decoder layer range [0, {len(layers) - 1}]"
            )
        self.layer_index = int(layer_index)
        self.layer_count = int(len(layers))
        self.layer = layers[layer_index]
        if not hasattr(self.layer, "mlp"):
            raise ValueError("decoder layer is missing an mlp module")
        self.x: torch.Tensor | None = None
        self.y: torch.Tensor | None = None
        self._handle = self.layer.mlp.register_forward_hook(self._capture)

    def _capture(self, _module: Any, args: tuple[Any, ...], output: Any) -> None:
        self.x = _first_tensor(args).detach()
        self.y = _first_tensor(output).detach()

    def clear(self) -> None:
        self.x = None
        self.y = None

    def pop(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.x is None or self.y is None:
            raise RuntimeError("MLP IO hook did not fire")
        x, y = self.x, self.y
        self.clear()
        return x, y

    def close(self) -> None:
        self._handle.remove()


def iter_token_id_lists(text_iter: Iterable[str], tokenizer: Any) -> Iterator[list[int]]:
    """Yield non-empty tokenizer ids without adding special tokens."""

    for text in text_iter:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids:
            yield list(ids)


def iter_packed_token_chunks(
    token_id_iter: Iterable[list[int]],
    *,
    seq_length: int,
    max_chunks: int | None = None,
) -> Iterator[torch.Tensor]:
    """Yield contiguous fixed-length token chunks from an ID stream."""

    if int(seq_length) < 2:
        raise ValueError(f"seq_length must be >= 2, got {seq_length}")
    buf: list[int] = []
    n_yielded = 0
    for ids in token_id_iter:
        if not ids:
            continue
        buf.extend(int(v) for v in ids)
        while len(buf) >= int(seq_length):
            yield torch.tensor(buf[: int(seq_length)], dtype=torch.long)
            del buf[: int(seq_length)]
            n_yielded += 1
            if max_chunks is not None and n_yielded >= int(max_chunks):
                return


def batched_chunks(
    chunks: Iterable[torch.Tensor],
    *,
    batch_size: int,
) -> Iterator[list[torch.Tensor]]:
    """Group same-length token chunks into batches."""

    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    batch: list[torch.Tensor] = []
    for chunk in chunks:
        batch.append(chunk)
        if len(batch) == int(batch_size):
            yield batch
            batch = []
    if batch:
        yield batch


def _load_model_and_tokenizer(
    *,
    model_id: str,
    model_dtype: torch.dtype | str,
    device: str,
    trust_remote_code: bool,
    revision: str | None,
) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for Qwen extraction; install project deps first"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        revision=revision,
    )
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "revision": revision,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=model_dtype,
            **model_kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=model_dtype,
            **model_kwargs,
        )
    model.eval()
    model.to(device)
    return model, tokenizer


def _forward_model(model: Any, input_ids: torch.Tensor) -> None:
    try:
        model(input_ids=input_ids, use_cache=False)
    except TypeError:
        model(input_ids=input_ids)


def extract_packed_mlp_io_bundle(
    *,
    output_root: str | Path,
    model_id: str = DEFAULT_MODEL_ID,
    layer_index: int = DEFAULT_LAYER_INDEX,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    seq_length: int = DEFAULT_SEQ_LENGTH,
    batch_size: int = 4,
    text_file: str | Path | None = None,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_config: str | None = DEFAULT_DATASET_CONFIG,
    dataset_split: str = DEFAULT_DATASET_SPLIT,
    text_column: str = DEFAULT_TEXT_COLUMN,
    streaming: bool = False,
    device: str = "cuda",
    model_dtype: torch.dtype | str = "auto",
    save_dtype: torch.dtype = torch.float32,
    trust_remote_code: bool = False,
    allow_short: bool = False,
    revision: str | None = None,
) -> Path:
    """Extract packed-stream MLP IO pairs and return the activation directory."""

    if int(max_pairs) <= 0:
        raise ValueError(f"max_pairs must be positive, got {max_pairs}")
    if int(seq_length) < 2:
        raise ValueError(f"seq_length must be >= 2, got {seq_length}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested --device cuda, but torch.cuda.is_available() is false")

    output_root = Path(output_root).expanduser()
    activation_dir = output_root / "activations"
    activation_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = _load_model_and_tokenizer(
        model_id=model_id,
        model_dtype=model_dtype,
        device=device,
        trust_remote_code=trust_remote_code,
        revision=revision,
    )
    capture = MlpInputOutputCapture(model, int(layer_index))

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

    captured_per_chunk = int(seq_length) - 1
    max_chunks = int(math.ceil(float(max_pairs) / float(captured_per_chunk)))
    chunk_iter = iter_packed_token_chunks(
        iter_token_id_lists(text_iter, tokenizer),
        seq_length=int(seq_length),
        max_chunks=max_chunks,
    )

    x_store: torch.Tensor | None = None
    y_store: torch.Tensor | None = None
    token_store = torch.empty((int(max_pairs),), dtype=torch.int32)
    filled = 0
    chunks_seen = 0

    progress = tqdm(total=int(max_pairs), desc="packed activation pairs", unit="tok")
    try:
        with torch.inference_mode():
            for chunk_batch in batched_chunks(chunk_iter, batch_size=int(batch_size)):
                input_ids_cpu = torch.stack(chunk_batch, dim=0)
                input_ids = input_ids_cpu.to(device)
                capture.clear()
                _forward_model(model, input_ids)
                x_batch, y_batch = capture.pop()
                if x_batch.ndim != 3 or y_batch.ndim != 3 or x_batch.shape != y_batch.shape:
                    raise ValueError(
                        "captured MLP IO tensors must have matching [batch, seq, d] shape; "
                        f"got x={tuple(x_batch.shape)}, y={tuple(y_batch.shape)}"
                    )

                x_rows = x_batch[:, :-1, :].reshape(-1, x_batch.shape[-1])
                y_rows = y_batch[:, :-1, :].reshape(-1, y_batch.shape[-1])
                token_rows = input_ids_cpu[:, :-1].reshape(-1)
                take = min(int(max_pairs) - filled, int(x_rows.shape[0]))
                if take <= 0:
                    break
                if x_store is None:
                    d_model = int(x_rows.shape[1])
                    x_store = torch.empty((int(max_pairs), d_model), dtype=save_dtype)
                    y_store = torch.empty((int(max_pairs), d_model), dtype=save_dtype)
                x_store[filled : filled + take].copy_(x_rows[:take].to("cpu", dtype=save_dtype))
                y_store[filled : filled + take].copy_(y_rows[:take].to("cpu", dtype=save_dtype))
                token_store[filled : filled + take].copy_(token_rows[:take].to(torch.int32))
                filled += take
                chunks_seen += len(chunk_batch)
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
        token_store = token_store[:filled].clone()

    metadata = {
        "format": FORMAT_VERSION,
        "capture_implementation": "packed_stream",
        "source_commit": "d6ad2f7",
        "model_id": model_id,
        "revision": revision,
        "layer_index": int(layer_index),
        "layer_count": capture.layer_count,
        "activation_x": "decoder.layers[layer_index].mlp input",
        "activation_y": "decoder.layers[layer_index].mlp output before residual add",
        "capture_semantics": (
            "tokenize without special tokens, concatenate the text stream, chunk to "
            "fixed seq_length, and drop each chunk's final position"
        ),
        "num_pairs": int(filled),
        "d_model": int(x_store.shape[1]),
        "seq_length": int(seq_length),
        "captured_per_chunk": int(captured_per_chunk),
        "chunks_seen": int(chunks_seen),
        "batch_size": int(batch_size),
        "save_dtype": str(save_dtype),
        "model_dtype": str(model_dtype),
        "source": source,
        "token_ids_file": "token_ids.pt",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    save_activation_pair(activation_dir, x_store, y_store, metadata=metadata)
    torch.save(token_store.cpu(), activation_dir / "token_ids.pt")
    return activation_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--layer-index", type=int, default=DEFAULT_LAYER_INDEX)
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS)
    parser.add_argument("--seq-length", type=int, default=DEFAULT_SEQ_LENGTH)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-dtype", default="auto", type=_model_torch_dtype)
    parser.add_argument("--save-dtype", default=torch.float32, type=_torch_dtype)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-short", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    activation_dir = extract_packed_mlp_io_bundle(
        output_root=args.output_root,
        model_id=args.model_id,
        revision=args.revision,
        layer_index=args.layer_index,
        max_pairs=args.max_pairs,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        text_file=args.text_file,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        text_column=args.text_column,
        streaming=args.streaming,
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
        "token_ids": str(activation_dir / "token_ids.pt"),
    }
    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

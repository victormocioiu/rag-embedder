"""ONNX Runtime inference.

Three things here are load-bearing and easy to get wrong:

1. **Mean pooling, not CLS.** e5 is trained with mean pooling over the attention
   mask. Using CLS produces embeddings that look fine and retrieve badly.
2. **L2 normalisation at write time.** Once unit-length, cosine and negative inner
   product rank identically and IP is cheaper -- which is why the HNSW index uses
   `halfvec_ip_ops` and queries use `<#>`. If norms are not 1.0, that equivalence
   silently breaks.
3. **The feed dict is built from `session.get_inputs()`**, not hardcoded.
   multilingual-e5-small is XLM-R based and has no `token_type_ids`; BERT-based
   models do. Hardcoding either one breaks the other.

Threading: `intra_op_num_threads` is the setting that matters and the default is
usually wrong. Set it to PHYSICAL core count. On the current node that is 1 (the
node has 2 cores and one is left for kubelet + DaemonSets).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

PREFIXES = {"query": "query: ", "passage": "passage: "}


class EmbeddingEngine:
    def __init__(
        self,
        model_path: str,
        intra_op_num_threads: int = 1,
        inter_op_num_threads: int = 1,
        max_length: int = 512,
    ) -> None:
        root = Path(model_path)
        candidates = sorted(root.glob("*.onnx"))
        if not candidates:
            raise FileNotFoundError(f"no .onnx file under {root}")
        # quantized export is usually model_quantized.onnx; prefer it if present
        onnx_file = next((c for c in candidates if "quant" in c.name), candidates[0])

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = intra_op_num_threads
        opts.inter_op_num_threads = inter_op_num_threads
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            str(onnx_file), opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.model_file = onnx_file.name

        tok_path = root / "tokenizer.json"
        if not tok_path.exists():
            raise FileNotFoundError(
                f"{tok_path} missing -- export_model.py must copy tokenizer files "
                "into the quantized output directory"
            )
        self.tokenizer = Tokenizer.from_file(str(tok_path))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding()

        self._ready = False
        self.embed(["warmup"], "passage")   # first call includes lazy allocation
        self._ready = True

    @property
    def is_ready(self) -> bool:
        return self._ready

    def encode(self, texts: list[str], input_type: str) -> dict[str, np.ndarray]:
        try:
            prefix = PREFIXES[input_type]
        except KeyError:
            raise ValueError(
                f"input_type must be one of {sorted(PREFIXES)}, got {input_type!r}"
            ) from None

        encs = self.tokenizer.encode_batch([prefix + t for t in texts])
        ids = np.asarray([e.ids for e in encs], dtype=np.int64)
        mask = np.asarray([e.attention_mask for e in encs], dtype=np.int64)

        feed: dict[str, np.ndarray] = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        return {k: v for k, v in feed.items() if k in self.input_names}

    def embed(self, texts: list[str], input_type: str) -> np.ndarray:
        feed = self.encode(texts, input_type)
        # real (unpadded) length per text -- read by the padding-waste metric
        self.last_token_lengths = feed["attention_mask"].sum(axis=1).tolist()
        last_hidden = self.session.run(None, feed)[0]

        mask = feed["attention_mask"][..., None].astype(last_hidden.dtype)
        pooled = (last_hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)

        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled / np.clip(norms, 1e-12, None)


def token_lengths(engine: EmbeddingEngine, texts: list[str], input_type: str) -> list[int]:
    """Real (unpadded) token length per text. NOTE: with enable_padding() on,
    len(e.ids) is the padded length -- the attention mask holds the truth."""
    prefix = PREFIXES[input_type]
    encs = engine.tokenizer.encode_batch([prefix + t for t in texts])
    return [sum(e.attention_mask) for e in encs]

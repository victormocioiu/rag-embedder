"""HuggingFace `tokenizers` (the Rust implementation).

Releases the GIL and is fast enough to be irrelevant in the profile. The only
interesting logic here is the E5 prefix, and it is deliberately not optional.
"""

PREFIXES = {"query": "query: ", "passage": "passage: "}


def apply_prefix(texts: list[str], input_type: str) -> list[str]:
    """Prepend the E5 prefix. Raises on an unknown input_type rather than
    guessing -- a wrong prefix is worse than a loud failure."""
    try:
        prefix = PREFIXES[input_type]
    except KeyError:
        raise ValueError(f"input_type must be one of {sorted(PREFIXES)}, got {input_type!r}")
    return [prefix + t for t in texts]


class Tokenizer:
    def __init__(self, model_path: str, max_length: int) -> None:
        # TODO(part-1): tokenizers.Tokenizer.from_file, enable truncation/padding
        raise NotImplementedError

    def encode_batch(self, texts: list[str]) -> dict:
        """Returns input_ids / attention_mask as numpy arrays. TODO(part-1)."""
        raise NotImplementedError

from typing import Literal

from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)

    # REQUIRED, no default. E5 is trained with asymmetric prefixes and silently
    # degrades without them; a defaulted kwarg is a kwarg someone forgets.
    input_type: Literal["query", "passage"]


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dimensions: int
    model: str
    # Exposed so callers (and the benchmark harness) can see how effective the
    # batcher was without scraping /metrics.
    batch_size: int
    inference_ms: float

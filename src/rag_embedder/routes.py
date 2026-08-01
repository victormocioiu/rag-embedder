"""HTTP surface. Thin -- all the interesting logic is in batching/engine."""

from fastapi import APIRouter

from rag_embedder.schemas import EmbedRequest, EmbedResponse

router = APIRouter()


@router.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    """TODO(part-1)."""
    raise NotImplementedError


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz() -> dict[str, str]:
    """Must 503 until the model is loaded. TODO(part-1)."""
    raise NotImplementedError

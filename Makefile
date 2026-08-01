.PHONY: sync export export-fp32 run test lint docker

sync:
	uv sync

export:  ## quantized ONNX -- set TARGET from `grep avx512 /proc/cpuinfo` (see README)
	uv sync --group export
	uv run python scripts/export_model.py --target $(or $(TARGET),avx512_vnni) --out onnx-int8

export-fp32:  ## part 1 needs fp32 too, for the comparison
	uv run python scripts/export_model.py --skip-quantize --out onnx-fp32

run:
	uv run uvicorn rag_embedder.main:app --reload --port 8001

test:
	uv run pytest -v

lint:
	uv run ruff check src && uv run mypy src

docker:  ## model comes prebuilt from the rag-embedder-model image
	docker build -t rag-embedder:dev .

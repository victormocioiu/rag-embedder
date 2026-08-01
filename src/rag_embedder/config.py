from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_id: str = "intfloat/multilingual-e5-small"
    model_path: str = "/models/onnx-int8"
    embed_dimensions: int = 384
    max_sequence_length: int = 512

    quantization_target: Literal["avx2", "avx512", "avx512_vnni", "arm64"] = "avx512_vnni"

    batch_max_size: int = 32
    batch_max_wait_ms: int = 10

    # Threads must not exceed the pod's cpu LIMIT (cgroup throttling thrashes:
    # the default-4 on a 1-cpu pod measured 41.7ms/text vs 24.7 at intra=2 with
    # limit 2). See docs/benchmarks-1.2.md before changing.
    intra_op_num_threads: int = 1
    inter_op_num_threads: int = 1
    inference_workers: int = 1


@lru_cache
def get_settings() -> Settings:
    return Settings()

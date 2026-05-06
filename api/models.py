from typing import Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):
    prompt: str
    model: str = "balanced"
    max_tokens: Optional[int] = None
    system: Optional[str] = None


class QueryResponse(BaseModel):
    response: str
    model: str
    duration_ms: int


class ModelWorkerStatus(BaseModel):
    idle: int
    busy: int
    ready: int


class HealthResponse(BaseModel):
    status: str
    workers: dict[str, ModelWorkerStatus]

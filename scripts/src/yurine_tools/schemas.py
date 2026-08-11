"""Pydantic schemas for generated Yurine data and conversion metadata."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class YurineModel(BaseModel):
    """Base schema for immutable generated data with finite numeric values."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)


class ConversionStats(YurineModel):
    """Metadata reported after a successful embedding conversion."""

    records: int = Field(ge=0)
    dimension: int = Field(gt=0)


class EmbeddingRecord(YurineModel):
    """One token and its vector in Yurine's JSON Lines format."""

    token: str = Field(min_length=1)
    embedding: list[float] = Field(min_length=1)


class EmbeddingSource(YurineModel):
    """Location and serialization format of an embedding data file."""

    path: str = Field(min_length=1)
    format: Literal["jsonl"] = "jsonl"


class EmbeddingCostConfig(YurineModel):
    """Version 1 Yurine configuration for cosine embedding edit costs."""

    version: Literal[1] = 1
    type: Literal["embedding"] = "embedding"
    embeddings: EmbeddingSource
    missing_substitution_cost: float = Field(ge=0)
    deletion_cost: float = Field(ge=0)
    insertion_cost: float = Field(ge=0)

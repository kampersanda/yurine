"""Pydantic schemas for generated Yurine data and conversion metadata."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

F32_MAX = 3.4028234663852886e38
Float32 = Annotated[float, Field(ge=-F32_MAX, le=F32_MAX)]


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
    embedding: list[Float32] = Field(min_length=1)


class JawikiPassage(YurineModel):
    """One passage record published by the Wikipedia-Utils releases."""

    id: int = Field(ge=0)
    pageid: int = Field(ge=0)
    revid: int = Field(ge=0)
    title: str
    section: str
    text: str = Field(min_length=1)


class JawikiPassageMetadata(YurineModel):
    """Sidecar record locating one corpus line in Japanese Wikipedia.

    Yurine reports matches by line position, so titles and identifiers are
    kept out of the corpus itself and recovered from this file instead.
    """

    line: int = Field(gt=0)
    id: int = Field(ge=0)
    pageid: int = Field(ge=0)
    revid: int = Field(ge=0)
    title: str
    section: str


class EmbeddingSource(YurineModel):
    """Location and serialization format of an embedding data file."""

    path: str = Field(min_length=1)
    format: Literal["jsonl"] = "jsonl"


class EmbeddingCostConfig(YurineModel):
    """Version 1 Yurine configuration for cosine embedding edit costs."""

    version: Literal[1] = 1
    type: Literal["embedding"] = "embedding"
    embeddings: EmbeddingSource
    missing_substitution_cost: float = Field(ge=0, le=F32_MAX)
    deletion_cost: float = Field(ge=0, le=F32_MAX)
    insertion_cost: float = Field(ge=0, le=F32_MAX)

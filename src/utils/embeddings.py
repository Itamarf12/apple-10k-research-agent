"""Text embeddings for the vector search.

Default: Chroma's built-in local model (all-MiniLM-L6-v2 via ONNX). It is free, needs no API
key, and downloads about 80 MB the first time it is used.

EMBEDDING_BACKEND=offline switches to a small hashing embedder that needs no download. It is
only meant for tests on machines without internet access; search quality is much lower.
"""
import math
import os
import re
import zlib

DIMENSIONS = 512
_default_model = None


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9][a-z0-9.%'’-]*", text.lower())
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]      # unigrams + bigrams


def _hash_embed(text: str) -> list[float]:
    vector = [0.0] * DIMENSIONS
    for token in _tokens(text):
        vector[zlib.crc32(token.encode()) % DIMENSIONS] += 1.0
    vector = [math.log1p(v) for v in vector]                           # dampen repeated words
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def embed(texts: list[str]) -> list[list[float]]:
    """Return one embedding vector per text."""
    global _default_model
    if os.getenv("EMBEDDING_BACKEND") == "offline":
        return [_hash_embed(t) for t in texts]
    if _default_model is None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        _default_model = DefaultEmbeddingFunction()
    return [list(map(float, v)) for v in _default_model(texts)]

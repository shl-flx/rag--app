
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, List
import weaviate
from openai import OpenAI

SearchMode = Literal["vector", "hybrid"]

@dataclass
class RetrievedChunk:
    chunk_id: int
    source: str
    text: str

def retrieve(
    query: str,
    *,
    mode: SearchMode = "hybrid",
    alpha: float = 0.5,
    limit: int = 5,
    collection: str = "DocsChunks",
    embed_model: str = "text-embedding-3-small",
) -> List[RetrievedChunk]:
    oai = OpenAI()

    def embed(text: str) -> list[float]:
        return oai.embeddings.create(model=embed_model, input=text).data[0].embedding

    client = weaviate.connect_to_local(host="localhost", port=8080, grpc_port=50051)
    col = client.collections.get(collection)

    try:
        if mode == "vector":
            res = col.query.near_vector(
                near_vector=embed(query),
                limit=limit,
                return_properties=["chunk_id", "source", "text"],
            )
        else:
            res = col.query.hybrid(
                query=query,
                vector=embed(query),
                alpha=alpha,
                limit=limit,
                return_properties=["chunk_id", "source", "text"],
            )

        out = []
        for obj in res.objects:
            p = obj.properties
            out.append(
                RetrievedChunk(
                    chunk_id=int(p.get("chunk_id", -1)),
                    source=str(p.get("source", "")),
                    text=str(p.get("text", "")),
                )
            )
        return out
    finally:
        client.close()

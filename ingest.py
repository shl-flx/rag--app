import json
import weaviate
from openai import OpenAI
from weaviate.classes.config import Configure, Property, DataType

COLLECTION = "DocsChunks"
CHUNKS_JSONL = "chunks.jsonl"
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 50

oai = OpenAI()


client = weaviate.connect_to_local(
    host="127.0.0.1",
    port=8080,
    grpc_port=50051,
)


if not client.collections.exists(COLLECTION):
    client.collections.create(
        name=COLLECTION,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="chunk_id", data_type=DataType.INT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="text", data_type=DataType.TEXT),
        ],
    )
    print(f"✅ Created collection: {COLLECTION}")

col = client.collections.get(COLLECTION)

def embed(text: str) -> list[float]:
    return oai.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    ).data[0].embedding

buffer = []
count = 0

with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)

        text = row["text"]
        chunk_id = int(row["chunk_id"])
        source = (row.get("metadata") or {}).get("source", "")

        vec = embed(text)

        buffer.append(
            (
                {
                    "text": text,
                    "chunk_id": chunk_id,
                    "source": source,
                },
                vec,
            )
        )

        if len(buffer) >= BATCH_SIZE:
            with col.batch.dynamic() as batch:
                for props, v in buffer:
                    batch.add_object(properties=props, vector=v)
            count += len(buffer)
            print(f"✅ Inserted {count} chunks...")
            buffer.clear()


if buffer:
    with col.batch.dynamic() as batch:
        for props, v in buffer:
            batch.add_object(properties=props, vector=v)
    count += len(buffer)

print(f"✅ Ingest complete: {count} chunks")

client.close()

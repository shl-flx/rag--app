
from openai import OpenAI
from retrieval import retrieve

SYSTEM = """You are a helpful assistant.
Answer ONLY using the provided context. If the answer isn't in context, say you don't know and ask for the missing info.
Cite chunks as [chunk_id]."""

def answer(query: str) -> str:
    chunks = retrieve(query, mode="hybrid", alpha=0.5, limit=6)

    context = "\n\n".join(
        [f"[{c.chunk_id}] {c.text}" for c in chunks]
    )

    oai = OpenAI()
    resp = oai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"QUESTION:\n{query}\n\nCONTEXT:\n{context}"},
        ],
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    q = "What does the playbook say about SFDC opportunities?"
    print(answer(q))

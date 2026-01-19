
import streamlit as st
from rag_answer import answer
from retrieval import retrieve

st.set_page_config(page_title="RAG Chat Demo", layout="wide")
st.title("RAG Chat Demo")
st.caption("Hybrid retrieval (BM25 + vector) → answer grounded in retrieved chunks.")


with st.sidebar:
    st.header("Settings")
    show_sources = st.toggle("Show retrieved sources", value=True)
    alpha = st.slider("Hybrid alpha (0=keyword, 1=vector)", 0.0, 1.0, 0.5, 0.05)
    top_k = st.slider("Top-k chunks", 1, 12, 6, 1)

    if st.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_input = st.chat_input("Ask a question...")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)


    with st.chat_message("assistant"):
        with st.spinner("Retrieving + generating..."):

            chunks = retrieve(user_input, mode="hybrid", alpha=alpha, limit=top_k)

            
            context = "\n\n".join([f"[{c.chunk_id}] {c.text}" for c in chunks])

            from openai import OpenAI
            from rag_answer import SYSTEM

            oai = OpenAI()
            resp = oai.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"QUESTION:\n{user_input}\n\nCONTEXT:\n{context}"},
                ],
            )
            assistant_text = resp.choices[0].message.content

        st.markdown(assistant_text)

        if show_sources:
            with st.expander("Sources (retrieved chunks)"):
                for c in chunks:
                    st.markdown(f"**Chunk [{c.chunk_id}]**  \nSource: `{c.source}`")
                    st.code(c.text[:1200])

    
    st.session_state.messages.append({"role": "assistant", "content": assistant_text})

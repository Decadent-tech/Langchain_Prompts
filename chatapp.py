import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings  # used only for embeddings <optional>
import openai

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Set OPENAI_API_KEY in your .env file")
    raise SystemExit
openai.api_key = OPENAI_API_KEY

# ------- Helpers -------
def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def get_text_chunks(text, chunk_size=2000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def create_and_save_faiss(text_chunks, embeddings_model="text-embedding-3-large"):
    # Use OpenAI embeddings (via langchain wrapper) to create FAISS index locally
    embeddings = OpenAIEmbeddings(model=embeddings_model)
    index = FAISS.from_texts(text_chunks, embedding=embeddings)
    index.save_local("faiss_index")
    return index

def load_faiss_and_embeddings(embeddings_model="text-embedding-3-large"):
    embeddings = OpenAIEmbeddings(model=embeddings_model)
    index = FAISS.load_local("faiss_index", embeddings)
    return index

def build_context_from_docs(docs, max_chars=3000):
    """
    Combine retrieved doc texts into a single context string.
    Trim if too long to fit within prompt budget.
    """
    pieces = []
    total = 0
    for d in docs:
        # `d.page_content` for LangChain Document-like objects, fallback to str(d)
        text = getattr(d, "page_content", None) or getattr(d, "text", None) or str(d)
        if not text:
            continue
        # keep adding until near max_chars
        if total + len(text) > max_chars:
            # add a truncated piece and break
            remaining = max_chars - total
            pieces.append(text[:remaining])
            break
        pieces.append(text)
        total += len(text)
    return "\n\n---\n\n".join(pieces)

def ask_openai_chat_system(context, question, model="gpt-4o-mini", temperature=0.2):
    """
    Send the prompt to OpenAI ChatCompletion and return assistant text.
    Uses the official openai python client.
    """
    system_prompt = (
        "You are a helpful assistant that answers questions strictly using the provided CONTEXT. "
        "If the answer is not contained in the context, reply exactly: "
        "\"Answer not available in the provided context.\""
    )
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nAnswer concisely and include relevant details from the context."

    # ChatCompletion (OpenAI python sdk - supports gpt-4o-mini, gpt-4, gpt-4o etc depending on your access)
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=800,
        n=1,
    )
    return resp["choices"][0]["message"]["content"].strip()

# ------- Streamlit UI -------
def main():
    st.set_page_config(page_title="Multi-PDF QA (OpenAI + FAISS)", page_icon="📚")
    st.title("Multi-PDF QA — OpenAI + FAISS (no fragile imports)")

    with st.sidebar:
        st.header("Upload & Process PDFs")
        uploaded = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
        if st.button("Create Vector Index"):
            if not uploaded:
                st.warning("Upload at least one PDF first.")
            else:
                with st.spinner("Extracting text and creating FAISS index..."):
                    raw = get_pdf_text(uploaded)
                    chunks = get_text_chunks(raw, chunk_size=2000, chunk_overlap=200)
                    create_and_save_faiss(chunks)
                    st.success("FAISS index created and saved as `faiss_index/`")
        st.markdown("---")
        st.write("Ensure `OPENAI_API_KEY` set in .env.")
    st.markdown("---")

    # Question input
    query = st.text_input("Ask a question about the processed PDFs:")
    if st.button("Get Answer") and query:
        try:
            with st.spinner("Searching index and asking OpenAI..."):
                index = load_faiss_and_embeddings()
                # similarity_search returns list of Documents (or text fragments)
                docs = index.similarity_search(query, k=4)
                context = build_context_from_docs(docs, max_chars=3000)
                if not context.strip():
                    st.info("No relevant context found in the index. Re-process PDFs or increase k.")
                else:
                    answer = ask_openai_chat_system(context, query, model="gpt-4o-mini", temperature=0.2)
                    st.markdown("### 🔎 Retrieved context (short):")
                    st.write(context[:2000] + ("... (truncated)" if len(context) > 2000 else ""))
                    st.markdown("### 🤖 Answer:")
                    st.write(answer)
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()

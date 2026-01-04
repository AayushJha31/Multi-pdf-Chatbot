# chatbot.py
# LangChain 1.2.0 — Modern RAG Chatbot (Ollama + FAISS)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

import os

# ----------------------------
# 1. LOAD PDFS
# ----------------------------
pdf_dir = "pdfs"  # put PDFs here
documents = []

for file in os.listdir(pdf_dir):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_dir, file))
        documents.extend(loader.load())

# ----------------------------
# 2. SPLIT TEXT
# ----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)

# ----------------------------
# 3. EMBEDDINGS + VECTOR STORE
# ----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ----------------------------
# 4. LLM (OLLAMA)
# ----------------------------
llm = Ollama(model="mistral")

# ----------------------------
# 5. PROMPT
# ----------------------------
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Answer the question using ONLY the context below.
If you don't know, say "I don't know".

Context:
{context}

Question:
{question}
""")

# ----------------------------
# 6. RAG CHAIN (LCEL)
# ----------------------------
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# ----------------------------
# 7. CHAT LOOP
# ----------------------------
print("📄 Multi-PDF Chatbot (Ollama + LangChain 1.2.0)\n")

while True:
    query = input("🧑 Ask: ")
    if query.lower() in ["exit", "quit"]:
        break

    answer = rag_chain.invoke(query)
    print("\n🤖 Answer:\n", answer, "\n")

#  Multi-PDF Chatbot (RAG using LangChain)

An AI-powered chatbot that allows users to **upload multiple PDFs and ask questions based on their content**.  
Built using **LangChain, embeddings, and vector search**, this project implements a full **Retrieval-Augmented Generation (RAG)** pipeline.

---

## Problem Statement

Reading and extracting information from multiple documents is time-consuming and inefficient.

**Goal:** Build an AI system that can:
- Understand multiple PDFs  
- Retrieve relevant information  
- Answer user queries accurately  

---

##  Solution Overview

This project uses a **RAG (Retrieval-Augmented Generation)** approach:

1. Upload PDFs  
2. Split them into smaller chunks  
3. Convert chunks into embeddings  
4. Store embeddings in a vector database  
5. Retrieve relevant chunks based on user query  
6. Generate answers using an LLM  

---

## 🏗️ System Architecture

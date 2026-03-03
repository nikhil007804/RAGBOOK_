# Agentic RAG Architecture & Tech Stack

This document provides a comprehensive end-to-end breakdown of the technologies used to build this Artificial Intelligence application. It explains exactly what each tool does and why it was chosen to build a production-grade, highly-performant RAG pipeline.

## 1. The Core Infrastructure

### Python 3.12 
* **What it is:** The programming language used for the entire backend.
* **Why we chose it:** Python is the undisputed industry standard for AI, Machine Learning, and Data Engineering, boasting the richest ecosystem of AI libraries (like Pinecone, Gemini, network tools).

### Docker Desktop & Docker Compose
* **What it is:** Containerization technology that packages and runs background software as isolated "virtual machines" (containers) that run identically on any operating system.
* **Why we installed it:** Our high-speed cache (`Redis`) is written in C and is natively designed for Linux servers. Installing and compiling Redis natively on a Windows machine is notoriously difficult and prone to breaking. By having you install Docker Desktop, we were able to download a "Linux container" that already had Redis perfectly installed inside of it. 
* **How it fits:** Simply by running `docker compose up -d`, Docker spins up an isolated mini-server running Redis on port `6379`, completely bypassing all Windows compatibility issues and giving our FastAPI backend instant access to an enterprise-grade caching database. 

## 2. Ingestion & Storage (The "Brain")

### PyMuPDF (`fitz`)
* **What it is:** A highly efficient Python library for manipulating PDF documents.
* **Why we chose it:** We needed to ingest the 129-page "Agentic Design Patterns" textbook. PyMuPDF is much faster than alternatives like PyPDF2, and handles complex document structures (headers, bold text, lists) smoothly, which allows our chunking algorithm to preserve semantic context.

### Pinecone Serverless (Vector Database)
* **What it is:** A managed, cloud-native Vector Database.
* **Why we chose it:** Once we slice the book into 1,200 "chunks", we need to store them. Traditional relational databases (like PostgreSQL) search by *exact keyword match*. Vector databases search by *concept and meaning*. We chose Pinecone because its Serverless tier is totally free, requires zero infrastructure management, and scales infinitely.

## 3. The Embedding & Retrieval Engines

### `multilingual-e5-large` (Dense Embedding Model)
* **What it is:** A machine learning model hosted via Pinecone Inference.
* **Why we chose it:** We originally tried running a local model (via Ollama), but that requires significant local RAM and GPU power. E5-large is a state-of-the-art dense model that converts text chunks into 1024-dimensional arrays of numbers ("dense vectors"). Since we run it through Pinecone's API, it consumes zero local computing power. 

### BM25 (Sparse Embedding Model)
* **What it is:** A classic, highly-proven statistical text search algorithm based on TF-IDF (Term Frequency-Inverse Document Frequency).
* **Why we chose it:** Dense vectors (`e5-large`) are great for *semantic meaning* (e.g., knowing "puppy" and "dog" are similar). But they are terrible at finding exact acronyms or specific code snippets. We run a BM25 sparse model locally to count exact keyword overlaps, giving us the best of both worlds.

### Reciprocal Rank Fusion (RRF)
* **What it is:** An algorithmic technique for combining the results of multiple search algorithms.
* **Why we chose it:** Our system performs a **Hybrid Search**. It gets top results from the Dense search (Pinecone E5) and the Sparse search (BM25). RRF mathematically merges these two distinct ranked lists into one theoretically optimal master list, dramatically raising the quality of our search.

### `bge-reranker-v2-m3` (Cross-Encoder Reranker)
* **What it is:** A secondary machine learning model (also hosted via Pinecone) that acts as a judge.
* **Why we chose it:** Vector search is fast, but imprecise. The Reranker takes the top 20 results found by RRF, and deeply analyzes how well each chunk perfectly answers the user's specific question. It then re-orders them, ensuring the most accurate chunks are placed at the very top before we hand them to the final LLM. 

## 4. The Intelligence Layer

### Google Gemini 2.5 Flash (`google-generativeai`)
* **What it is:** A cutting-edge large language model created by Google DeepMind.
* **Why we chose it:** We needed an LLM to read the top ranked textbook chunks, synthesize an answer, and format it exactly as a JSON response. Gemini 2.5 Flash was chosen because it is incredibly fast, highly capable of understanding long contexts, and has a very generous free tier compared to OpenAI's GPT-4o.

## 5. The Backend Server

### FastAPI & Uvicorn
* **What it is:** `FastAPI` is a modern, high-performance web framework for Python. `Uvicorn` is the ultra-fast ASGI web server that actually runs the FastAPI code.
* **Why we chose it:** We needed an API endpoint (`/query`) to handle requests from the user interface. FastAPI is much faster than traditional frameworks like Flask or Django because it supports asynchronous programming out-of-the-box. It also automatically generates interactive Swagger API documentation.

### Redis (Caching Layer)
* **What it is:** An in-memory, ultra-fast key-value data store.
* **Why we chose it:** API calls to Pinecone and Gemini take 2 to 5 seconds and cost network bandwidth. By injecting a Redis layer into our API, if a user asks a question they have already asked before, the system skips the entire RAG pipeline and returns the saved answer from RAM in ~2 milliseconds. 

## 6. The User Interface

### Streamlit
* **What it is:** An open-source Python framework for rapidly building data science and machine learning web applications.
* **Why we chose it:** Building a UI with React or Vue.js would require a totally separate Javascript codebase and API networking boilerplate. `app.py` allows us to build a beautiful, reactive, fully-styled chat interface using pure Python, natively binding our frontend components to our backend logic without ever leaving the language ecosystem. 

---
### End-to-End Diagram Narrative:
1. **User asks a question** in the Streamlit UI.
2. The UI sends a POST request to the **FastAPI/Uvicorn Backend**.
3. FastAPI checks the **Redis Cache**. If found, return instantly. If not...
4. The question is passed to Pinecone's **E5-large** to generate a Dense Vector, and to our local **BM25** algorithm to generate a Sparse Vector.
5. Both vectors query the **Pinecone Database**, returning two separate lists of matching textbook paragraphs.
6. The lists are merged mathematically using **RRF Fusion**.
7. The highly accurate **BGE-Reranker** double-checks the top 20 paragraphs, and picks the absolute best 5.
8. The question and the 5 paragraphs are bundled into a Prompt and sent to **Google Gemini**.
9. Gemini synthesizes a beautifully formatted JSON answer citing exactly which pages it used.
10. FastAPI saves the answer into **Redis** for the future, and returns it to **Streamlit** to be drawn on the screen!

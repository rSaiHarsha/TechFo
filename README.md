# Flask + LangChain + Qdrant + MongoDB (with Hugging Face embeddings)

This project lets you upload PDF/TXT files, choose which technology collection (e.g., `java`, `python`, `javascript`) to index, create embeddings with **Hugging Face sentence-transformers** (free models), store vectors in Qdrant (one Qdrant collection per technology), and keep collection metadata in MongoDB.

## File structure

```
flask_langchain_qdrant/
├─ app.py
├─ requirements.txt
├─ templates/
│  ├─ layout.html
│  ├─ index.html
│  ├─ upload.html
│  ├─ collections.html
│  └─ view_collection.html
└─ README.md
```

## Setup notes (quick)

1. Run Qdrant (local) e.g. with Docker:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

2. Run MongoDB (local) e.g. using `mongod` or Docker:

```bash
docker run -p 27017:27017 -v mongo-data:/data/db mongo:6
```

3. Install Python deps (see `requirements.txt`).

4. No API keys are required for Hugging Face local models (downloads on first use).

# 🚀 Hybrid RAG Pipeline

A production-grade **Hybrid Retrieval-Augmented Generation (RAG)** pipeline combining **Dense Semantic Retrieval** and **Sparse Lexical Retrieval (BM25)** using **Pinecone Serverless Vector Database**, **LangChain**, and **Google Gemini**.

---

## 🌟 Architecture Overview

Standard RAG typically relies either only on dense vector embeddings (which capture deep semantic meaning but can miss exact keywords, rare terms, or IDs) or only on keyword search (which misses synonyms and conceptual relationships). 

This pipeline implements **Hybrid Search** to achieve the best of both worlds:

```
                                  ┌───────────────────────────┐
                                  │      User Documents       │
                                  │     (AI Knowledge PDFs)   │
                                  └─────────────┬─────────────┘
                                                │
                                    [Text Chunking & Cleaning]
                                                │
                      ┌─────────────────────────┴─────────────────────────┐
                      ▼                                                   ▼
         [Dense Embeddings]                                      [Sparse Embeddings]
    (all-MiniLM-L6-v2, 384-dim)                                  (BM25 Token Frequencies)
                      │                                                   │
                      └─────────────────────────┬─────────────────────────┘
                                                ▼
                               ┌─────────────────────────────────┐
                               │   Pinecone Serverless Index     │
                               │     (metric: dotproduct)        │
                               └────────────────┬────────────────┘
                                                │
User Query ──► [PineconeHybridSearchRetriever] ◄┘
                          │
                   (Ranked Context)
                          ▼
             ┌──────────────────────────┐
             │   Google Gemini LLM      │
             │   (gemini-1.5-flash)     │
             └────────────┬─────────────┘
                          ▼
                    Final Answer
```

- **Dense Retrieval**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) via `langchain-huggingface`.
- **Sparse Retrieval**: `BM25Encoder` via `pinecone-text` fitted on corpus documents and persisted to `bm25_values.json`.
- **Hybrid Fusion**: `PineconeHybridSearchRetriever` blending dense and sparse relevance scores.
- **Generation**: Google Gemini (`gemini-1.5-flash` / `gemini-2.0-flash`) via `langchain-google-genai`.

---

## 📁 Repository Structure

```
.
├── AI knowledge/             # Sample PDF knowledge corpus (Deep Learning, NLP, RAG, etc.)
├── HybridRAG.ipynb           # Main Jupyter Notebook containing the full pipeline
├── bm25_values.json          # Pre-computed BM25 parameter file for sparse search
├── requirements.txt          # Python dependencies
├── .env.example              # Template for required environment variables
├── .gitignore                # Protects secrets (.env) and excludes virtualenv
└── README.md                 # Project documentation
```

---

## 🔑 Prerequisites

You will need API credentials for the following services:

1. **Pinecone**: Sign up at [pinecone.io](https://www.pinecone.io/) and create an API key.
2. **Google Gemini**: Get an API key from [Google AI Studio](https://aistudio.google.com/).
3. **Hugging Face** *(Optional)*: An API token from [huggingface.co](https://huggingface.co/) for downloading models.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and fill in your keys:
```env
PINECONE_API_KEY="your-pinecone-api-key"
HF_TOKEN="your-huggingface-token"
GEMINI_API_KEY="your-google-gemini-api-key"
```

---

## 🚀 Running the Pipeline

Launch Jupyter Notebook or JupyterLab:
```bash
jupyter notebook HybridRAG.ipynb
```

Follow the notebook cells step by step:
1. **Import Libraries & Setup**: Loads environment variables and initializes SDKs.
2. **Pinecone Index Creation**: Creates a serverless index with dotproduct metric and 384 dimensions if it doesn't already exist.
3. **Document Ingestion**: Extracts text from PDFs in `AI knowledge/` and cleans artifacts.
4. **Chunking**: Uses `RecursiveCharacterTextSplitter` with chunk size and overlap suited for technical texts.
5. **Hybrid Indexing**:
   - Generates BM25 sparse vectors and saves parameters to `bm25_values.json`.
   - Embeds text chunks with `all-MiniLM-L6-v2`.
   - Upserts hybrid vectors to Pinecone.
6. **Query & Retrieval**: Runs hybrid search against the index.
7. **Generation**: Prompts Google Gemini with retrieved context to generate grounded, factual responses.

---

## 🛡️ Security Best Practices
- **Never commit `.env` to version control.** The `.gitignore` file is pre-configured to keep your API keys private.
- Always use `.env.example` to share required environment variables with collaborators.

---

## 📄 License
MIT License. Feel free to use and adapt this pipeline for your projects!

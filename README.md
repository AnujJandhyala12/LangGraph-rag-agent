# LangGraph-rag-agent

A complete production level AI system that combines **XGBoost credit risk prediction**, **Retrieval-Augmented Generation (RAG)**, and an **agentic LangGraph pipeline** served via a **FastAPI** backend.

Built using the Lending Club dataset. The system can answer natural language questions related to credit risk, summarize financial reports, and predict loan default probability with SHAP-based explanations.

---

## Architecture

```
User Query
     │
FastAPI Backend (/query or /predict)
     │
LangGraph Agent (conditional routing)
     │
     ├── summarize_node ──> RAG (Pinecone) ──> Claude → Summary
     │
     ├── rag_node ────────> RAG (Pinecone) ──> Claude → Answer
     │
     ├── predict_node ────> XGBoost + SHAP ──> Claude → Risk Report
     │
     └── direct_node ─────> Claude ─ Direct Answer
```
For conditional routing:

- **route_query()** is responsible. It reads the query and returns a string     telling LangGraph which node to go to next.

- **graph.add_conditional_edges()** makes it "conditional" — instead of always going to the same next node, it checks the router function output first, depending on this it makes a decision - summarize, RAG, predict or direct.
---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Claude Haiku (Anthropic) |
| Agent Framework | LangGraph + LangChain |
| Vector Database | Pinecone |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Chunking | Semantic Chunking (LangChain Experimental) |
| Retrieval | MMR (Maximal Marginal Relevance) |
| ML Model | XGBoost Classifier |
| Explainability | SHAP |
| Backend | FastAPI + Uvicorn |
| Data | Lending Club (300MB, ~150k rows) |

---

## Features

- **Agentic routing** — LangGraph automatically routes queries to the right tool based on intent
- **RAG pipeline** — semantic chunking + MMR retrieval from Pinecone for accurate, diverse context
- **Credit risk prediction** — XGBoost trained on 15+ financial features with SHAP explainability
- **LLM explanation** — Claude explains model predictions in plain English
- **Production mindset** — retry logic, in-memory caching, structured logging, error handling
- **FastAPI** — async endpoints with request validation via Pydantic

---

## Project Structure

```
langgraph-rag-agent/
├── app/
│   ├── config.py          # Environment variable loading
│   ├── graph.py           # LangGraph agent definition
│   ├── main.py            # FastAPI endpoints
│   ├── rag.py             # Pinecone + semantic chunking + MMR
│   └── tools.py           # RAG, summarize, predict, direct tools
├── ml/
│   ├── __init__.py
│   ├── train.py           # XGBoost training script
│   └── summarize_data.py  # Generates RAG knowledge base from CSV
├── data/
│   ├── lending_club.csv          # Raw dataset (not in repo)
│   └── credit_risk_report.txt    # Generated RAG knowledge base
├── outputs/               # EDA charts
├── .env.example
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/AnujJandhyala12/LangGraph-rag-agent.git
cd LangGraph-rag-agent
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # For Windows
source venv/bin/activate   # For Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Copy `.env.example` to `.env` and fill in your keys:

```
ANTHROPIC_API_KEY=your_anthropic_key
PINECONE_API_KEY=your_pinecone_key
```

For getting keys, visit:
- Anthropic: [console.anthropic.com](https://console.anthropic.com)
- Pinecone: [app.pinecone.io](https://app.pinecone.io)

### 5. Add dataset

Download the Lending Club dataset from https://www.kaggle.com/datasets/wordsforthewise/lending-club and place it at:
```
data/lending_club.csv
```

### 6. Train the ML model

```bash
python -m ml.train
```

This trains XGBoost on 150k rows (can be modified) and saves:
- `ml/model.pkl`
- `ml/features.pkl`
- `ml/encoders.pkl`

### 7. Generate RAG knowledge base

```bash
python -m ml.summarize_data
```

This generates `data/credit_risk_report.txt` and uploads it to Pinecone.

### 8. Start the server

```bash
uvicorn app.main:app --reload
```

API is live at: [http://127.0.0.1:8000](http://127.0.0.1:8000)
Swagger docs at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## API Endpoints

### `GET /health`
Health check.

```json
{"status": "ok"}
```

---

### `POST /query`
This API endpoint is responsible for Natural language questions about credit risk.

**Request:**
```json
{"query": "What is the default rate by loan grade?"}
```

**Response:**
```json
{
  "query": "What is the default rate by loan grade?",
  "response": "Grade A loans have the lowest default rate at 5.2%, while Grade G loans have the highest at 38.7%..."
}
```

**Example queries:**
```
"What is the overall default rate?"
"Summarize the credit risk report"
"What factors most affect loan default?"
"What is the average DTI for defaulters?"
"Explain the Value at Risk findings"
```

---

### `POST /predict`
This API endpoint is used for credit risk prediction for a loan applicant with SHAP-based explanation.

**Request:**
```json
{
  "loan_amnt": 15000,
  "int_rate": 14.5,
  "annual_inc": 60000,
  "dti": 25.0
}
```

**Response:**
```json
{
  "result": "HIGH RISK (78% default probability). Key risk factors: high DTI of 25 exceeds the safe threshold, interest rate of 14.5% is above average for non-defaulters..."
}
```

---

## Model Performance

| Metric | Score |
|---|---|
| ROC-AUC | ~0.7079 |
| Features used | 15+ (FICO, DTI, grade, income, etc.) |
| Training samples | 150,000 |
| Class imbalance handling | scale_pos_weight |

---

## LangGraph Routing Logic
This shows how each query is routed and depending on the query type selects a node.
```
Query contains "summarize/summarise"          → summarize_node
Query contains "document/report/data"         → rag_node
Query contains "default/risk/loan/credit/dti" → rag_node
Query contains "predict/score"                → predict_node
Everything else                               → direct_node
```

---

## RAG Pipeline

```
credit_risk_report.txt
        │
SemanticChunker (breakpoint_threshold=85th percentile)
        │
HuggingFace Embeddings (all-MiniLM-L6-v2, 384 dims)
        │
Pinecone Index (cosine similarity)
        │
   Query comes in
        │
MMR Retrieval (k=4, fetch_k=20, lambda=0.7 (70% relevance, 30% diversity))
        │
Claude generates answer from retrieved context
```
---

## Author

**Anuj Jandhyala**






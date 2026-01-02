# AI_DMP_RAG — NIH Data Management Plan (DMP) Generator

AI_DMP_RAG generates an NIH-style **Data Management & Sharing Plan (DMP)**.  
It uses a pipeline to **ingest documents**, **build/search an index**, and **draft a DMP** through a **FastAPI** web UI.

---

## Quick Start (Most Common)

1) Create + activate a virtual environment  
2) Install requirements  
3) Add your `.env` (API keys)  
4) Run the app with Uvicorn  
5) Open the browser link  

---

## Repository Structure

```text
AI_DMP_RAG/
│── app.py
│── README.md
│── requirements.txt
│── setup.py
│── .env
│── .gitignore
│
├── config/                 # Configuration files (YAML, etc.)
├── data/                   # Input documents / datasets (raw or processed)
├── model/                  # Saved models / embeddings / checkpoints (if any)
├── logs/                   # Runtime logs (app + pipeline)
├── notebook_DMP_RAG/       # Experiments, notebooks, prototypes
│
├── src/                    # Main application code
│   ├── __init__.py
│   ├── core_pipeline_UI.py # Pipeline logic used by the UI/app
│   └── data_ingestion.py   # Document ingestion + indexing utilities
│
├── prompt/                 # Prompt templates and prompt tools
│   ├── __init__.py
│   └── prompt_library.py
│
├── logger/                 # Custom logger utilities
│   ├── __init__.py
│   └── custom_logger.py
│
├── exception/              # Custom exceptions
│   ├── __init__.py
│   └── custom_exception.py
│
├── utils/                  # Shared helper functions (general utilities)
│   ├── __init__.py
│   └── config_loader.py
│   └── model_loader.py
```

---

## Prerequisites

- Python **3.10+** (recommended)
- (Optional) Git
- A `.env` file for secrets (API keys, endpoints, etc.)

---

## Setup (Local Development)

### Step 1 — Create and activate a virtual environment

**Windows (PowerShell):**
```python
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```python
python -m venv venv
source venv/bin/activate
```

### Step 2 — Install dependencies

```python
pip install -r requirements.txt
```

(Optional but recommended for development / editable installs)
```python
pip install -e .
```

---
### Step 3 — Data Ingestion & Indexing

**What happens:** the app reads your documents in `data/`, splits them into chunks, and builds an index (vector store) for retrieval.

**Main files**
- `src/data_ingestion.py`: loads + cleans + chunks docs, builds the index
- `src/core_pipeline_UI.py`: retrieves relevant chunks and generates the final DMP

**Workflow**
1. Add reference documents to: `data/`
2. Run once to build the index (or enable rebuild)
3. Start the web app and generate DMPs from the UI

**Rebuild the index (if needed)**
- Set `force_rebuild_index=True` in your config/YAML, **or**
- Delete the saved index folder (often `data/index/`) and run again

---

### Run the Web App (FastAPI)

From the project root (where `app.py` is):
```bash
uvicorn app:app --reload
Open in your browser:
- `http://127.0.0.1:8000/`
---

## Setup (Example Commands — Conda)

```python
git clone https://github.com/fairdataihub/dmpchef.git
cd dmpchef
code .

conda create -n dmpchef python=3.10 -y
conda activate dmpchef

python -m pip install --upgrade pip
pip install -r requirements.txt

python setup.py install
# or (recommended for development)
pip install -e .

uvicorn app:app --reload
```

Then open:
- `http://127.0.0.1:8000/`  
and test the app.

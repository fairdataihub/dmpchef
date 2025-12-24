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
│
├── build/                  # Build artifacts (packaging)
├── dist/                   # Distribution artifacts (packaging)
└── DMP_RAG.egg-info/       # Package metadata (created during install/build)
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

## Environment Variables (`.env`)

Create a `.env` file in the project root (same level as `app.py`).

Example:
```python
# LLM Provider / API
OPENAI_API_KEY=your_key_here

# Optional settings
ENV=dev
LOG_LEVEL=INFO
```

Notes:
- Do **not** commit `.env` to Git.
- Make sure `.env` is listed in `.gitignore`.

---

## Configuration (`config/`)

The pipeline typically reads settings from `config/` (example: `config/config.yaml`).

Common items you may configure:
- Data paths (where source docs live)
- Index / vector store settings
- Embedding model settings
- LLM model settings
- Chunking parameters
- Rebuild-index flag (e.g., `force_rebuild_index`)

---

## Data Ingestion & Indexing (How the pipeline works)

### Key modules
- `src/data_ingestion.py`  
  Loads documents, cleans/chunks them, and builds an index/vector store.
- `src/core_pipeline_UI.py`  
  Runs retrieval + prompting + generation to produce the final DMP.

### Typical workflow
1. Put reference documents into `data/`
2. Run the pipeline once (or enable rebuild) to create the index
3. Run the web app and generate DMPs from the UI

### Rebuild the index (if needed)
- Set a config flag like `force_rebuild_index=True` (or in YAML), **or**
- Delete the existing index folder (if you store it under something like `data/index/`)

---

## Run the Web App (FastAPI)

From the project root (where `app.py` is):

```python
uvicorn app:app --reload
```

Open in your browser:
- `http://127.0.0.1:8000/`

If API docs are enabled:
- `http://127.0.0.1:8000/docs`

---

## Logging

- `logger/custom_logger.py` controls logging format and handlers
- Runtime logs are typically written to `logs/`

If logs aren’t showing:
- Check `LOG_LEVEL` in `.env`
- Ensure `logs/` exists and your app has permission to write files

---

## Prompts

Prompt templates/utilities are in:
- `prompt/prompt_library.py`

You can:
- Update the DMP template text
- Add section-by-section prompts
- Enforce NIH structure/format rules (headings, required elements, compliance wording)

---

## Troubleshooting

### App runs but the page doesn’t open
- Copy the URL printed by Uvicorn (example: `http://127.0.0.1:8000`) and paste it into your browser.

### Import errors
- Make sure you run `uvicorn app:app --reload` from the **project root**
- Confirm `src/__init__.py` exists
- Try reinstalling in editable mode:
```python
pip install -e .
```

### Index not found / retrieval results are empty
- Confirm you have documents in `data/`
- Rebuild the index via config or rerun ingestion

### Permission issues (logs or saved files)
- Ensure `logs/` exists
- On Windows, try running your terminal as Administrator

---

## Recommended `.gitignore`

These are commonly ignored:
- `venv/`
- `__pycache__/`
- `.env`
- `build/`
- `dist/`
- `*.egg-info/`
- `logs/` *(optional)*

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

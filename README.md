[![Contributors](https://img.shields.io/github/contributors/fairdataihub/LLM-RAG-demo?style=flat-square&logo=github&logoColor=white&color=2ea44f)](https://github.com/fairdataihub/LLM-RAG-demo/graphs/contributors)
[![Stars](https://img.shields.io/github/stars/fairdataihub/LLM-RAG-demo?style=flat-square&logo=github&logoColor=white&color=f9d949)](https://github.com/fairdataihub/LLM-RAG-demo/stargazers)
[![Issues](https://img.shields.io/github/issues/fairdataihub/LLM-RAG-demo?style=flat-square&logo=github&logoColor=white&color=ff7a00)](https://github.com/fairdataihub/LLM-RAG-demo/issues)
[![License](https://img.shields.io/github/license/fairdataihub/LLM-RAG-demo?style=flat-square&color=1f6feb)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-pending-9e9e9e?style=flat-square)](#how-to-cite)

# DMP-Chef — NIH Data Management Plan (DMP) Generator
DMP-Chef generates an NIH-style **Data Management & Sharing Plan (DMP)** using an end-to-end **Retrieval-Augmented Generation (RAG)** workflow.  
It uses a pipeline to **ingest documents**, **build/search an index**, and **draft a DMP** through a **FastAPI** web UI.  
Learn more: **[DMP-Chef](https://fairdataihub.org/dmp-chef)**.

---

## Standards followed
The overall codebase is organized in alignment with the **[FAIR-BioRS guidelines](https://fair-biors.org/)**. The Python code follows **[PEP 8](https://peps.python.org/pep-0008/)** style conventions (including comments and docstrings). All required dependencies are listed in **[`requirements.txt`](https://github.com/fairdataihub/dmpchef/blob/main/requirements.txt)**.

## Main files
- **[`src/data_ingestion.py`](https://github.com/fairdataihub/dmpchef/blob/main/src/data_ingestion.py)** — Loads, cleans, and chunks documents; builds the vector index.
- **[`src/core_pipeline_UI.py`](https://github.com/fairdataihub/dmpchef/blob/main/src/core_pipeline_UI.py)** — Retrieves relevant chunks and generates the final output.

---

## Repository Structure
```text
dmpchef/
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
│   ├── config_loader.py
│   └── model_loader.py
```
---

## Setup (Local Development)

### Step 1 — Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

(Optional, recommended for development)
```bash
pip install -e .
```

---

### Step 3 — Run the Pipeline (Ingestion + Indexing)
**What happens:** the app reads documents in `data/`, splits them into chunks, and builds an index (vector store) for retrieval.

**Workflow**
1. Add reference documents to: `data/`
2. Run `src/data_ingestion.py` once to build the index (or enable rebuild)

**Rebuild the index (if needed)**
- Set `force_rebuild_index=True` in your config/YAML, **or**
- Delete the saved index folder (often `data/index/`) and run ingestion again

---

### Step 4 — Start the Web App (FastAPI)
Start the server from the project root (where `app.py` is):

```bash
uvicorn app:app --reload
```

Open in your browser:
- `http://127.0.0.1:8000/`

---

## Setup (Example Commands — Conda)
```bash
git clone https://github.com/fairdataihub/dmpchef.git
cd dmpchef
code .

conda create -n dmpchef python=3.10 -y
conda activate dmpchef

python -m pip install --upgrade pip
pip install -r requirements.txt

# optional: install package
pip install -e .

# build index (example)
python src/data_ingestion.py

# start app
uvicorn app:app --reload
```

Then open:
- `http://127.0.0.1:8000/`
---

## License
This work is licensed under the **[MIT License](https://opensource.org/license/mit/)**. See **[LICENSE](https://github.com/fairdataihub/dmpchef/blob/main/LICENSE)** for more information.


---

## Feedback and contribution
Use **[GitHub Issues](https://github.com/fairdataihub/dmpchef/issues)** to submit feedback, report problems, or suggest improvements.  
You can also **fork** the repository and submit a **Pull Request** with your changes.

---

## How to cite
If you use this code, please cite this repository using the **versioned DOI on Zenodo** for the specific release you used (instructions will be added once the Zenodo record is available). For now, you can reference the repository here: **[fairdataihub/dmpchef](https://github.com/fairdataihub/dmpchef)**.

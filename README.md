[![Contributors](https://img.shields.io/github/contributors/fairdataihub/LLM-RAG-demo?style=flat-square&logo=github&logoColor=white&color=2ea44f)](https://github.com/fairdataihub/LLM-RAG-demo/graphs/contributors)
[![Stars](https://img.shields.io/github/stars/fairdataihub/LLM-RAG-demo?style=flat-square&logo=github&logoColor=white&color=f9d949)](https://github.com/fairdataihub/LLM-RAG-demo/stargazers)
[![Issues](https://img.shields.io/github/issues/fairdataihub/LLM-RAG-demo?style=flat-square&logo=github&logoColor=white&color=ff7a00)](https://github.com/fairdataihub/LLM-RAG-demo/issues)
[![License](https://img.shields.io/github/license/fairdataihub/LLM-RAG-demo?style=flat-square&color=1f6feb)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-pending-9e9e9e?style=flat-square)](#how-to-cite)

# DMP Chef
DMP Chef is an open-source (MIT License), Python-based pipeline that draft funder-compliant **Data Management & Sharing Plan (DMPs)** using an end-to-end **Retrieval-Augmented Generation (RAG)** workflow with a **Large Language Model (LLM)**. It provides a pipeline to **ingest documents**, **build/search an index**, and **draft a DMP** through a **FastAPI** web UI. 

This project is part of a broader extension of the DMP Tool platform. The ultimate goal is to integrate the DMP Chef pipeline into the [DMP Tool](https://dmptool.org/) platform, providing researchers with a familiar and convenient user interface that does not require any coding knowledge.

👉 Learn more: **[DMP-Chef](https://fairdataihub.org/dmp-chef)**.

---
## Standards followed
The overall codebase is organized in alignment with the **[FAIR-BioRS guidelines](https://fair-biors.org/)**. All Python code follows **[PEP 8](https://peps.python.org/pep-0008/)** conventions, including consistent formatting, inline comments, and docstrings. Project dependencies are fully captured in **[requirements.txt](https://github.com/fairdataihub/dmpchef/blob/main/requirements.txt)**. We also retain **[dmp-template](https://github.com/fairdataihub/dmpchef/blob/main/data/inputs/dmp-template.md)** as inside the prompt template used by the DMP generation workflow.


## Main files
- **[`src/data_ingestion.py`](https://github.com/fairdataihub/dmpchef/blob/main/src/data_ingestion.py)** — Loads, cleans, and chunks documents; builds the vector index.
- **[`src/core_pipeline_UI.py`](https://github.com/fairdataihub/dmpchef/blob/main/src/core_pipeline_UI.py)** — Core RAG pipeline logic (retrieve → prompt → generate).
- **[`main.py`](https://github.com/fairdataihub/dmpchef/blob/main/main.py)** — Command-line entry point for running the pipeline end-to-end.

---

## Repository Structure
```text
dmpchef/
│── main.py                 # Main script for running the pipeline end-to-end
│── README.md               # Project overview, setup instructions, usage examples, API docs
│── requirements.txt        # Python dependencies for `pip install -r requirements.txt`
│── setup.py                # Optional packaging config (enables `pip install -e .` for editable installs)
│── .env                    # Local environment variables (keys/config) — keep private; DO NOT commit
│── .gitignore              # Git ignore rules (e.g., venv, __pycache__, logs, .env, local data)
│
├── config/                 # App/pipeline configuration
│   ├── __init__.py         # Makes `config` importable as a package
│   ├── config.yaml         # Main settings (models, paths, chunking, retriever params, etc.)
│   └── config_schema.py    # Schema/validation for config (pydantic/dataclasses validation)
│
├── data/                   # Input documents / datasets / outputs
│   ├── inputs/             # User-facing templates + example inputs
│   │   ├── dmp-template.md                 # Markdown prompt template used by the LLM
│   │   ├── nih-dms-plan-template.docx      # NIH blank DOCX template (used to preserve exact Word formatting)
│   │   └── inputs.json                     # hson schema DMPtools
│   ├── pdfs/               # NIH guidance PDFs used for RAG (config.paths.data_pdfs points here)
│   └── outputs/            # Generated artifacts
│       ├── md/             # Generated Markdown DMPs (NIH format)
│       ├── docx/           # Generated DOCX DMPs (NIH Format)
│       └── json/           # Generated JSON outputs (dmptool schema) 
│
├── model/                  # Model-related code + (optionally) persisted artifacts
│   ├── __init__.py         # Makes `model` importable
│   └── models.py           # Model definitions / wrappers (LLM + embeddings config objects, etc.)
│
├── src/                    # Main application source code (core pipeline + reusable modules)
│   ├── __init__.py         # Package marker for `src`
│   ├── core_pipeline_UI.py # Main RAG pipeline logic invoked by the app/UI (retrieve → prompt → generate)
│   └── data_ingestion.py   # Ingestion + preprocessing + indexing utilities (load PDFs, chunk, embed, store)
│
├── prompt/                 # Prompt templates and prompt utilities
│   ├── __init__.py         # Package marker for `prompt`
│   └── prompt_library.py   # Centralized prompt templates (system/user prompts, formatting, guardrails)
│
├── logger/                 # Custom logging utilities
│   ├── __init__.py         # Package marker for `logger`
│   └── custom_logger.py    # Logger setup (formatters, handlers, file/console logging)
│
├── exception/              # Custom exception definitions
│   ├── __init__.py         # Package marker for `exception`
│   └── custom_exception.py # Custom error classes for clearer debugging and error handling
│
├── utils/                  # Shared helpers used across the project
│   ├── __init__.py         # Package marker for `utils`
│   ├── config_loader.py    # Loads/validates configuration (YAML/env), provides defaults
│   ├── model_loader.py     # Loads LLM/embeddings clients and related model settings
│   ├── dmptool_json.py     # Builds dmptool JSON output schema (used by core_pipeline_UI)
│   └── nih_docx_writer.py  # Fills NIH blank DOCX template to preserve exact Word formatting
│
├── notebook_DMP_RAG/       # Notebooks / experiments / prototypes (not production code)
└── venv/                   # Local virtual environment — ignore in git


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
### Step 4 — Run the pipeline

```bash
python app.py
```
---

## Inputs

- Reference documents (e.g., NIH guidance PDFs) used for retrieval  
- User/project metadata (example in `data/inputs/inputs.json`)

## Outputs

- **JSON** (structured)
- **Markdown** (NIH-style narrative)
- **DOCX** (optional; preserves NIH template formatting when enabled)

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
python main.py
```
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

[![Contributors](https://img.shields.io/github/contributors/fairdataihub/LLM-RAG-demo?style=flat-square&logo=github&logoColor=white&color=2ea44f)](https://github.com/fairdataihub/LLM-RAG-demo/graphs/contributors)
[![Stars](https://img.shields.io/github/stars/fairdataihub/LLM-RAG-demo?style=flat-square&logo=github&logoColor=white&color=f9d949)](https://github.com/fairdataihub/LLM-RAG-demo/stargazers)
[![Issues](https://img.shields.io/github/issues/fairdataihub/LLM-RAG-demo?style=flat-square&logo=github&logoColor=white&color=ff7a00)](https://github.com/fairdataihub/LLM-RAG-demo/issues)
[![License](https://img.shields.io/github/license/fairdataihub/LLM-RAG-demo?style=flat-square&color=1f6feb)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-pending-9e9e9e?style=flat-square)](#how-to-cite)

# DMP Chef
DMP Chef is an open-source (MIT License), Python-based pipeline that draft funder-compliant **Data Management & Sharing Plan (DMPs)** using a **Large Language Model (LLM), such as Llama 3.3** 

It supports two modes entirely in Python:
- **RAG**: Retrieves related guidance from an indexed document collection and uses it to ground the draft. In this mode, the pipeline can **ingest documents**, **build and search an index**, and **draft a DMP**.
- **No-RAG**: Generates the draft only from the user’s project inputs (no retrieval).

This project is part of a broader extension of the DMP Tool platform. The ultimate goal is to integrate the DMP Chef pipeline into the [DMP Tool](https://dmptool.org/) platform, providing researchers with a familiar and convenient user interface that does not require any coding knowledge.

👉 Learn more: **[DMP-Chef](https://fairdataihub.org/dmp-chef)**.

---
## Standards followed
The overall codebase is organized in alignment with the **[FAIR-BioRS guidelines](https://fair-biors.org/)**. All Python code follows **[PEP 8](https://peps.python.org/pep-0008/)** conventions, including consistent formatting, inline comments, and docstrings. Project dependencies are fully captured in **[requirements.txt](https://github.com/fairdataihub/dmpchef/blob/main/requirements.txt)**. We also retain **[dmp-template](https://github.com/fairdataihub/dmpchef/blob/main/data/inputs/dmp-template.md)** as inside the prompt template used by the DMP generation workflow.


## Main files for testing
- **[`main.py`](https://github.com/fairdataihub/dmpchef/blob/main/main.py)** — Command-line entry point for running the pipeline end-to-end.
- **[`demo.ipynb`](https://github.com/fairdataihub/dmpchef/blob/main/demo.ipynb)** — Jupyter demo showing.


---

## Repository Structure
```text
dmpchef/
│── main.py                 # CLI entry point (run pipeline end-to-end)
│── README.md               # Project overview + usage
│── requirements.txt        # Python dependencies
│── setup.py                # Packaging (editable installs via pip install -e .)
│── pyproject.toml          # Build system config (wheel builds)
│── MANIFEST.in             # Include non-code files in distributions
│── demo.ipynb              # Notebook demo: import + run generate()
│── LICENSE
│── .gitignore
│── .env                    # Local env vars (do not commit)
│
├── dmpchef/                # Installable Python package (public API)
│   ├── __init__.py         # Exports: generate, draft
│   └── api.py              # Importable API used by notebooks/backends
│
├── config/                 # Configuration
│   ├── config.yaml         # Main settings (models, paths, retriever params)
│   └── config_schema.py    # Pydantic schema for DMPCHEF-Pipeline config
│   └── schema_validate.py  # Validation/schema helpers for input.json 
│
├── data/                   # Local workspace data + artifacts (not guaranteed in wheel)
│   ├── inputs/             # Templates + examples
│   │   ├── nih-dms-plan-template.docx  # NIH blank Word template
│   │   └── input.json                  # Example request file
│   ├── vector_db/              # Vector index artifacts (e.g., FAISS)
|        ├── DMPtools_db/
|        ├── NIH_all_db/
|        └── NIH_sharing_db/
│   ├── data_ingestion/         # Source Pdfs and text from DMPtool+ NIH + NIH_sharing and etc
│   ├── outputs/            # Generated artifacts
│   │   ├── markdown/       # Generated Markdown DMPs
│   │   ├── docx/           # Generated DOCX DMPs (template-preserving)
│   │   ├── json/           # DMPTool-compatible JSON outputs
│   │   └── pdf/            # Optional PDFs converted from DOCX
│   
│
├── src/                    # Core implementation
│   ├── __init__.py
│   ├── core_pipeline.py    # Pipeline logic (RAG/no-RAG)
│   ├── Build_index.py      #Bulid index of vectore db
│   └── NIH_data_ingestion.py # NIH/DMPTool crawl → export PDFs to data/database
│
├── prompt/                 # Prompt templates/utilities
│   └── prompt_library.py
│
├── utils/                  # Shared helpers
│   ├── config_loader.py
│   ├── model_loader.py
│   ├── dmptool_json.py
│   └── nih_docx_writer.py
│   └── download_vector_db.py
│
├── logger/                 # Logging utilities
│   ├── __init__.py
│   └── custom_logger.py
│
├── exception/              # Custom exceptions
│   ├── __init__.py
│   └── custom_exception.py
│
├── notebook_DMP_RAG/       # Notebooks/experiments (non-production)
└── venv/                   # Local virtualenv 



```
## Setup (Local Development)

### Step 1 — Clone the repository
```bash
git clone https://github.com/fairdataihub/dmpchef.git
cd dmpchef
code .
```

### Step 2 — Create and activate a virtual environment

**Windows (cmd):**
```bash
python -m venv venv
venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
# or (recommended for local dev)
pip install -e .
```
### Step 4 — Configure Large Language Models
**Llama 3.3 (via Ollama)**

1.  Install Ollama from:\
    https://ollama.com/

2.  Pull the llama3.3:

``` bash
ollama pull llama3.3:70b 
```
### Step 4 — Run DMP Chef

### Option A — Jupyter demo
Use **[`demo.ipynb`](https://github.com/fairdataihub/dmpchef/blob/main/demo.ipynb)**.

### Option B — CLI: Command-line entry point for running the pipeline end-to-end

Use  **[`main.py`](https://github.com/fairdataihub/dmpchef/blob/main/main.py)** 

---

## Inputs
- **Input.JSON**: A single JSON file (e.g., `data/inputs/input.json`) that tells the pipeline what to generate. 
Before execution, the request is validated against **[Schema.JSON](https://github.com/fairdataihub/dmpchef/blob/main/config/dmpchef_request.schema.json)** using the **[schema_validate](https://github.com/fairdataihub/dmpchef/blob/main/config/schema_validate.py)** validator.


```json
{
  "config": { ... },
  "inputs": { ... }
}
```

### `config` (Execution Settings)

- **config.funding.agency**: Funder key (string; NIH|NSF|OTHER)
- **config.funding.subagency**: sub-agency (string; optional)
- **config.pipeline.rag**: `true` / `false` (boolean flags; If omitted, the pipeline uses the YAML default (`rag.enabled`)).
- **config.pipeline.llm**: LLM settings (boolean flags; e.g., `provider`, `model_name`).
- **config.export**: Output (boolean flags; `md`, `docx`, `pdf`, `dmptool_json`)

### `inputs` 
- **inputs**: A dictionary of user/project fields used to draft the plan include:
  - `research_context`, `data_types`, `data_source`, `human_subjects`, `consent_status`, `data_volume`, etc.

## Outputs 

- **Markdown**: the generated funder-aligned DMP narrative (currently NIH structure).
- **DOCX**: generated using the funder template (NIH template today) to preserve official formatting.
- **PDF**: created by converting the DOCX (platform-dependent; typically works on Windows/macOS with Word).
- **JSON**: a **DMPTool-compatible** JSON file.


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

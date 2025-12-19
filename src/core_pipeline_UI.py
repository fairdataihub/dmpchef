# ===============================================================
# core_pipeline_UI.py — Web-Integrated RAG Core (Fixed + Stable)
# - Fixes FAISS '__fields_set__' load crash by auto-rebuilding index
# - Formats retrieved docs into a STRING (prompt-friendly)
# - Builds vectorstore/retriever/chain ONCE at init (fast for UI)
# ===============================================================

import re
import json
from pathlib import Path
from tqdm import tqdm
import pypandoc
import yaml

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
from langchain_community.llms import Ollama

from utils.model_loader import ModelLoader
from exception.custom_exception import DocumentPortalException
from logger.custom_logger import GLOBAL_LOGGER as log
from prompt.prompt_library import PROMPT_REGISTRY, PromptType


# ===============================================================
# CONFIGURATION MANAGER
# ===============================================================
class ConfigManager:
    """Loads and provides access to YAML configuration."""

    def __init__(self, config_path="config/config.yaml"):
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"❌ Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f) or {}

        self.paths = self.cfg.get("paths", {}) or {}
        self.models = self.cfg.get("models", {}) or {}
        self.rag = self.cfg.get("rag", {}) or {}

        log.info("✅ Config loaded successfully")

    def get_path(self, key: str) -> Path:
        val = self.paths.get(key)
        if not val:
            raise KeyError(f"❌ Missing config.paths.{key} in YAML")
        return Path(val)

    def get_model(self, key: str):
        return self.models.get(key)

    def get_rag_param(self, key: str):
        return self.rag.get(key)


# ===============================================================
# MAIN PIPELINE CLASS
# ===============================================================
class DMPPipeline:
    """End-to-end pipeline for NIH DMP generation via web input."""

    def __init__(self, config_path="config/config.yaml", force_rebuild_index: bool = False):
        try:
            # --- Load configuration ---
            self.config = ConfigManager(config_path)

            self.data_pdfs = self.config.get_path("data_pdfs")
            self.index_dir = self.config.get_path("index_dir")
            self.output_md = self.config.get_path("output_md")
            self.output_docx = self.config.get_path("output_docx")
            self.output_json = Path("data/outputs/json")

            # Template path (can be moved into YAML later)
            self.template_md = Path("data/inputs/dmp-template.md")

            # Create output dirs
            for p in [self.output_md, self.output_docx, self.output_json, self.index_dir]:
                p.mkdir(parents=True, exist_ok=True)

            # --- Load template file ---
            if not self.template_md.exists():
                raise FileNotFoundError(f"❌ DMP template not found: {self.template_md}")
            self.template_text = self.template_md.read_text(encoding="utf-8")

            # --- Load models ---
            self.model_loader = ModelLoader()
            self.embeddings = self.model_loader.load_embeddings()
            self.llm_name = self.model_loader.llm_name
            self.llm = Ollama(model=self.llm_name)

            # --- Load prompt template from registry ---
            self.prompt_template = PROMPT_REGISTRY[PromptType.NIH_DMP.value]

            # --- Build / load index ONCE (UI performance) ---
            self.vectorstore = self._load_or_build_index(force_rebuild=force_rebuild_index)

            # --- Create retriever ONCE ---
            top_k = self.config.get_rag_param("retriever_top_k") or 6
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})

            # --- Build RAG chain ONCE ---
            self.rag_chain = self._build_rag_chain(self.retriever)

            log.info("✅ DMPPipeline initialized (UI-ready, stable FAISS load, formatted context)")

        except Exception as e:
            log.error("❌ Failed to initialize DMPPipeline", error=str(e))
            raise DocumentPortalException("Pipeline initialization error", e)

    # ---------------------------------------------------------------
    def _load_or_build_index(self, force_rebuild: bool = False):
        """
        Load or build FAISS vector index.
        If loading fails (pickle mismatch e.g., '__fields_set__'), rebuild automatically.
        """
        try:
            faiss_path = self.index_dir / "index.faiss"

            if faiss_path.exists() and not force_rebuild:
                try:
                    log.info("📦 Loading existing FAISS index", path=str(faiss_path))
                    return FAISS.load_local(
                        str(self.index_dir),
                        self.embeddings,
                        allow_dangerous_deserialization=True,
                    )
                except Exception as e:
                    log.warning(f"⚠️ Failed to load FAISS index. Rebuilding... Reason: {e}")

            # ---- rebuild path ----
            pdf_files = sorted(self.data_pdfs.glob("*.pdf"))
            if not pdf_files:
                raise FileNotFoundError(f"❌ No PDFs found in: {self.data_pdfs}")

            docs = []
            for pdf in tqdm(pdf_files, desc="📥 Loading PDFs"):
                loader = PyPDFLoader(str(pdf))
                docs.extend(loader.load())

            chunk_size = self.config.get_rag_param("chunk_size") or 800
            chunk_overlap = self.config.get_rag_param("chunk_overlap") or 120

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            chunks = splitter.split_documents(docs)

            log.info("🧱 Building new FAISS index ...", chunks=len(chunks))
            vectorstore = FAISS.from_documents(chunks, self.embeddings)
            vectorstore.save_local(str(self.index_dir))
            log.info("✅ FAISS index built and saved", path=str(self.index_dir))
            return vectorstore

        except Exception as e:
            raise DocumentPortalException("FAISS index error", e)

    # ---------------------------------------------------------------
    def _build_rag_chain(self, retriever):
        """Build RAG chain using the prompt registry (context as STRING)."""
        try:
            def format_docs(docs):
                if not docs:
                    return ""
                formatted = []
                for d in docs:
                    page = d.metadata.get("page", "")
                    src = d.metadata.get("source", "")
                    formatted.append(f"[Page {page}] {src}\n{d.page_content.strip()}")
                return "\n\n".join(formatted)

            rag_chain = (
                RunnableMap(
                    {
                        "context": lambda x: format_docs(retriever.invoke(x["input"])),
                        "question": lambda x: x["input"],
                    }
                )
                | self.prompt_template
                | self.llm
                | StrOutputParser()
            )

            log.info("🔗 RAG chain built successfully", llm=self.llm_name)
            return rag_chain

        except Exception as e:
            raise DocumentPortalException("RAG chain build error", e)

    # ---------------------------------------------------------------
    def generate_dmp(self, title: str, form_inputs: dict):
        """Generate NIH DMP dynamically from user web form input."""
        try:
            title = (title or "").strip()
            if not title:
                raise ValueError("❌ Title is required.")

            # --- Combine user-provided form input ---
            user_elements = [
                f"{key.replace('_',' ').title()}: {val}".strip()
                for key, val in (form_inputs or {}).items()
                if isinstance(val, str) and val.strip()
            ]

            query = (
                f"You are an NIH data steward and grant writer. "
                f"Create a complete NIH Data Management and Sharing Plan (DMSP) "
                f"for the project titled '{title}'.\n\n"
                f"Use retrieved NIH context to help ensure NIH-aligned language.\n\n"
                f"User Inputs:\n{chr(10).join(user_elements)}\n\n"
                f"Use the following NIH DMSP Markdown template. Do not alter section titles:\n"
                f"{self.template_text}"
            )

            # --- Generate ---
            result = self.rag_chain.invoke({"input": query})

            # --- Save outputs ---
            safe_title = re.sub(r'[\\/*?:"<>|]', "_", title).strip()
            md_path = self.output_md / f"{safe_title}.md"
            docx_path = self.output_docx / f"{safe_title}.docx"
            json_path = self.output_json / f"{safe_title}.json"

            md_path.write_text(result, encoding="utf-8")

            # DOCX conversion requires pandoc installed (pypandoc)
            pypandoc.convert_text(result, "docx", format="md", outputfile=str(docx_path))

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "title": title,
                        "form_inputs": form_inputs,
                        "template_used": str(self.template_md),
                        "generated_markdown": result,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            log.info("✅ DMP generated successfully (UI)", title=title)
            return result

        except Exception as e:
            raise DocumentPortalException("DMP generation error", e)

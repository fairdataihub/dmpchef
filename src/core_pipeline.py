# ===============================================================
# core_pipeline.py — RAG Toggle + No extra JSON
# - RAG toggle per call OR via YAML (rag.enabled)
# - NO-RAG mode does NOT load FAISS / retriever / RAG chain
# - Saves: Markdown + NIH-template DOCX + ONLY DMPTool JSON
# ===============================================================

import re
import json
from pathlib import Path
from tqdm import tqdm
import yaml
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
from langchain_community.llms import Ollama

from utils.model_loader import ModelLoader
from exception.custom_exception import DocumentPortalException
from logger.custom_logger import GLOBAL_LOGGER as log
from prompt.prompt_library import PROMPT_REGISTRY, PromptType

from utils.dmptool_json import build_dmptool_json
from utils.nih_docx_writer import build_nih_docx_from_template


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
    """End-to-end pipeline for NIH DMP generation (RAG optional)."""

    def __init__(self, config_path="config/config.yaml", force_rebuild_index: bool = False):
        try:
            self.config = ConfigManager(config_path)
            self.force_rebuild_index = force_rebuild_index

            self.data_pdfs = self.config.get_path("data_pdfs")
            self.index_dir = self.config.get_path("index_dir")
            self.output_md = self.config.get_path("output_md")
            self.output_docx = self.config.get_path("output_docx")
            self.output_json = Path("data/outputs/json")

            # Template path
            self.template_md = Path("data/inputs/dmp-template.md")

            # Create output dirs (index_dir is created, but FAISS is not loaded unless needed)
            for p in [self.output_md, self.output_docx, self.output_json, self.index_dir]:
                p.mkdir(parents=True, exist_ok=True)

            if not self.template_md.exists():
                raise FileNotFoundError(f"❌ DMP template not found: {self.template_md}")
            self.template_text = self.template_md.read_text(encoding="utf-8")

            # Models
            self.model_loader = ModelLoader()
            self.embeddings = self.model_loader.load_embeddings()
            self.llm_name = self.model_loader.llm_name
            self.llm = Ollama(model=self.llm_name)

            # Prompt template
            self.prompt_template = PROMPT_REGISTRY[PromptType.NIH_DMP.value]

            # YAML default (rag.enabled). If missing => True.
            enabled_val = self.config.get_rag_param("enabled")
            self.use_rag_default = True if enabled_val is None else bool(enabled_val)

            # Build NO-RAG chain only (cheap)
            self.no_rag_chain = self._build_no_rag_chain()

            # Lazy RAG fields (built only if use_rag=True)
            self.vectorstore = None
            self.retriever = None
            self.rag_chain = None

            log.info(
                "✅ DMPPipeline initialized",
                llm=self.llm_name,
                rag_default=self.use_rag_default,
            )

        except Exception as e:
            log.error("❌ Failed to initialize DMPPipeline", error=str(e))
            raise DocumentPortalException("Pipeline initialization error", e)

    # ---------------------------------------------------------------
    # ✅ NEW: safe bool parser (so "false" doesn't become True)
    def _to_bool(self, v, default: Optional[bool] = None) -> Optional[bool]:
        if v is None:
            return default
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"true", "1", "yes", "y", "on"}:
                return True
            if s in {"false", "0", "no", "n", "off"}:
                return False
        return default

    # ---------------------------------------------------------------
    def _load_or_build_index(self, force_rebuild: bool = False):
        """
        Load or build FAISS vector index (ONLY when RAG is enabled).
        Robust PDF loading: PyMuPDFLoader primary, PyPDFLoader fallback.
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

            pdf_files = sorted(self.data_pdfs.glob("*.pdf"))
            if not pdf_files:
                raise FileNotFoundError(f"❌ No PDFs found in: {self.data_pdfs}")

            docs = []
            bad_pdfs = []

            for pdf in tqdm(pdf_files, desc="📥 Loading PDFs"):
                try:
                    loader = PyMuPDFLoader(str(pdf))
                    docs.extend(loader.load())
                except Exception as e1:
                    try:
                        loader = PyPDFLoader(str(pdf))
                        docs.extend(loader.load())
                    except Exception as e2:
                        bad_pdfs.append(str(pdf))
                        log.warning(
                            "⚠️ Skipping PDF due to parse error",
                            pdf=str(pdf),
                            pymupdf_error=str(e1),
                            pypdf_error=str(e2),
                        )
                        continue

            if not docs:
                raise RuntimeError(
                    "❌ No documents could be loaded from PDFs. "
                    "All PDFs may be corrupted or unreadable."
                )

            if bad_pdfs:
                log.warning("⚠️ Some PDFs were skipped", skipped=len(bad_pdfs))
                log.warning("First skipped PDF", first=bad_pdfs[0])

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
    def _build_no_rag_chain(self):
        """Build chain without retrieval (context is empty)."""
        try:
            no_rag_chain = (
                RunnableMap(
                    {
                        "context": lambda x: "",
                        "question": lambda x: x["input"],
                    }
                )
                | self.prompt_template
                | self.llm
                | StrOutputParser()
            )

            log.info("🔗 No-RAG chain built successfully", llm=self.llm_name)
            return no_rag_chain

        except Exception as e:
            raise DocumentPortalException("No-RAG chain build error", e)

    # ---------------------------------------------------------------
    def _ensure_rag_ready(self):
        """Lazy-init FAISS + retriever + rag_chain ONLY when needed."""
        if self.rag_chain is not None and self.retriever is not None and self.vectorstore is not None:
            return

        self.vectorstore = self._load_or_build_index(force_rebuild=self.force_rebuild_index)

        top_k = self.config.get_rag_param("retriever_top_k") or 6
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})

        self.rag_chain = self._build_rag_chain(self.retriever)

        log.info("✅ RAG components initialized", llm=self.llm_name, top_k=top_k)

    # ---------------------------------------------------------------
    def _cleanup_title_json(self, safe_title: str):
        """
        Remove any old JSONs for this title except the dmptool JSON.
        Helps avoid duplicates from older runs.
        """
        keep = f"{safe_title}.dmptool.json"
        for p in self.output_json.glob(f"{safe_title}*.json"):
            if p.name != keep:
                try:
                    p.unlink()
                except Exception:
                    pass

    # ---------------------------------------------------------------
    def generate_dmp(self, title: str, form_inputs: dict, use_rag: Optional[bool] = None):
        """
        Generate NIH DMP from inputs.

        use_rag:
          - True  => use retrieval (loads FAISS lazily)
          - False => no retrieval (never touches FAISS)
          - None  => use YAML default rag.enabled
        """
        try:
            title = (title or "").strip()
            if not title:
                raise ValueError("❌ Title is required.")

            # ✅ UPDATED (ONLY THIS LINE): safe boolean handling
            use_rag_final = self._to_bool(use_rag, default=self.use_rag_default)

            log.info(
                "🧭 RAG decision",
                use_rag_input=use_rag,
                rag_default=self.use_rag_default,
                use_rag_final=use_rag_final,
            )

            # choose chain
            if use_rag_final:
                self._ensure_rag_ready()
                chain = self.rag_chain
            else:
                chain = self.no_rag_chain

            # user inputs block
            user_elements = [
                f"{key.replace('_',' ').title()}: {val}".strip()
                for key, val in (form_inputs or {}).items()
                if isinstance(val, str) and val.strip()
            ]

            query = (
                f"You are an NIH data steward and grant writer. "
                f"Create a complete NIH Data Management and Sharing Plan (DMSP) "
                f"for the project titled '{title}'.\n\n"
                f"User Inputs:\n{chr(10).join(user_elements)}\n\n"
                f"Use the following NIH DMSP Markdown template. Do not alter section titles:\n"
                f"{self.template_text}"
            )

            # Only mention retrieval when actually using RAG (keeps behavior clean)
            if use_rag_final:
                query = (
                    f"You are an NIH data steward and grant writer. "
                    f"Create a complete NIH Data Management and Sharing Plan (DMSP) "
                    f"for the project titled '{title}'.\n\n"
                    f"Use retrieved NIH context to help ensure NIH-aligned language.\n\n"
                    f"User Inputs:\n{chr(10).join(user_elements)}\n\n"
                    f"Use the following NIH DMSP Markdown template. Do not alter section titles:\n"
                    f"{self.template_text}"
                )

            # generate
            result = chain.invoke({"input": query})

            # save outputs
            safe_title = re.sub(r'[\\/*?:"<>|]', "_", title).strip()
            md_path = self.output_md / f"{safe_title}.md"
            docx_path = self.output_docx / f"{safe_title}.docx"
            dmptool_json_path = self.output_json / f"{safe_title}.dmptool.json"

            md_path.write_text(result, encoding="utf-8")

            # NIH DOCX (exact template formatting)
            nih_template_docx = Path("data/inputs/nih-dms-plan-template.docx")
            build_nih_docx_from_template(
                template_docx_path=str(nih_template_docx),
                output_docx_path=str(docx_path),
                project_title=title,
                generated_markdown=result,
            )

            dmptool_obj = build_dmptool_json(
                template_title="NIH Data Management and Sharing Plan",
                project_title=title,
                form_inputs=form_inputs,
                generated_markdown=result,
                provenance="dmpchef",
            )

            # prevent duplicates
            self._cleanup_title_json(safe_title)

            with open(dmptool_json_path, "w", encoding="utf-8") as f:
                json.dump(dmptool_obj, f, indent=2, ensure_ascii=False)

            log.info(
                "✅ DMP generated successfully",
                title=title,
                use_rag=use_rag_final,
                md=str(md_path),
                docx=str(docx_path),
                dmptool_json=str(dmptool_json_path),
            )
            return result

        except Exception as e:
            raise DocumentPortalException("DMP generation error", e)

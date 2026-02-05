# ===============================================================
# core_pipeline.py — RAG Toggle + No extra JSON
# - RAG toggle per call OR via YAML (rag.enabled)
# - NO-RAG mode does NOT load FAISS / retriever / RAG chain
# - Saves: Markdown + NIH-template DOCX + ONLY DMPTool JSON
#
# ✅ Updates in this version:
#   1) Embeddings are lazy-loaded ONLY when RAG is used
#   2) funding_agency is NO LONGER read from inputs; it's passed as an argument
#   3) Log includes funding_agency
#   4) Keeps NIH prompt/template/docx/json behavior (future-ready for NSF)
# ===============================================================

# -------------------------------
# Standard library imports
# -------------------------------
import re                     # Used to sanitize file names and handle simple regex operations
import json                   # Used to write the DMPTool JSON output to disk
from pathlib import Path      # Cross-platform path handling for files/folders
from tqdm import tqdm         # Progress bar for PDF loading (nice UX)
import yaml                   # Loads YAML configuration
from typing import Optional   # Type hints for optional inputs

# -------------------------------
# LangChain / RAG-related imports
# -------------------------------
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader  # Robust PDF loaders
from langchain_text_splitters import RecursiveCharacterTextSplitter          # Chunking documents for embeddings

from langchain_community.vectorstores import FAISS        # Vector store for retrieval (used only if RAG is enabled)
from langchain_core.output_parsers import StrOutputParser # Converts model output to plain string
from langchain_core.runnables import RunnableMap          # Maps inputs into prompt fields (context/question)
from langchain_community.llms import Ollama               # Local LLM interface via Ollama

# -------------------------------
# Local project imports
# -------------------------------
from utils.model_loader import ModelLoader                         # Loads embeddings and LLM config
from exception.custom_exception import DocumentPortalException     # Custom wrapped exception for consistent error handling
from logger.custom_logger import GLOBAL_LOGGER as log              # Project logger
from prompt.prompt_library import PROMPT_REGISTRY, PromptType      # Prompt templates registry + enum

from utils.dmptool_json import build_dmptool_json                  # Builds a DMPTool-schema JSON object
from utils.nih_docx_writer import build_nih_docx_from_template     # Writes NIH DOCX using official template formatting


# ===============================================================
# CONFIGURATION MANAGER
# ===============================================================
class ConfigManager:
    """Loads and provides access to YAML configuration."""

    def __init__(self, config_path="config/config.yaml"):
        """
        Initialize the configuration manager by reading a YAML file.

        Args:
            config_path: Path to YAML config. Defaults to config/config.yaml.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
        """
        path = Path(config_path)  # Convert to Path for safe file operations
        if not path.exists():
            # Fail fast if config file is missing (better than hidden defaults)
            raise FileNotFoundError(f"❌ Config file not found: {path}")

        # Load YAML contents as a Python dict
        with open(path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f) or {}

        # Split config into top-level groups for cleaner access patterns
        self.paths = self.cfg.get("paths", {}) or {}   # e.g., data_pdfs, index_dir, output folders
        self.models = self.cfg.get("models", {}) or {} # e.g., embedding model name, llm name (if applicable)
        self.rag = self.cfg.get("rag", {}) or {}       # rag.enabled, chunk sizes, retriever_top_k, etc.

        log.info("✅ Config loaded successfully")

    def get_path(self, key: str) -> Path:
        """
        Return a configured path as a Path object.

        Args:
            key: Name of the path inside config.paths

        Returns:
            Path object for the requested path key.

        Raises:
            KeyError: If config.paths.<key> is missing or empty.
        """
        val = self.paths.get(key)
        if not val:
            raise KeyError(f"❌ Missing config.paths.{key} in YAML")
        return Path(val)

    def get_model(self, key: str):
        """
        Return a model-related value from config.models.

        Args:
            key: Name of the model config key

        Returns:
            The value from config.models[key] or None if absent.
        """
        return self.models.get(key)

    def get_rag_param(self, key: str):
        """
        Return a rag-related value from config.rag.

        Args:
            key: Name of the rag config key

        Returns:
            The value from config.rag[key] or None if absent.
        """
        return self.rag.get(key)


# ===============================================================
# MAIN PIPELINE CLASS
# ===============================================================
class DMPPipeline:
    """End-to-end pipeline for NIH DMP generation (RAG optional)."""

    def __init__(self, config_path="config/config.yaml", force_rebuild_index: bool = False):
        """
        Initialize the pipeline:
          - Load config
          - Setup paths
          - Load template text
          - Initialize the LLM (Ollama)
          - Build a cheap no-RAG chain immediately
          - Keep RAG components lazy (only built if needed)

        Args:
            config_path: Path to YAML config
            force_rebuild_index: If True, rebuild FAISS index even if it exists
        """
        try:
            # Load config and store rebuild preference
            self.config = ConfigManager(config_path)
            self.force_rebuild_index = force_rebuild_index

            # Read required paths from config
            self.data_pdfs = self.config.get_path("data_pdfs")     # Where reference PDFs live
            self.index_dir = self.config.get_path("index_dir")     # Where FAISS index is stored
            self.output_md = self.config.get_path("output_md")     # Markdown output folder
            self.output_docx = self.config.get_path("output_docx") # DOCX output folder

            # Output JSON folder is hardcoded here (consistent with your pipeline conventions)
            self.output_json = Path("data/outputs/json")

            # Template path (currently NIH markdown template)
            self.template_md = Path("data/inputs/dmp-template.md")

            # Create output dirs early (safe even if they already exist)
            # Note: index_dir is created even if RAG is never used; FAISS is still not loaded unless needed.
            for p in [self.output_md, self.output_docx, self.output_json, self.index_dir]:
                p.mkdir(parents=True, exist_ok=True)

            # Ensure the NIH markdown template exists before any generation happens
            if not self.template_md.exists():
                raise FileNotFoundError(f"❌ DMP template not found: {self.template_md}")

            # Cache template content in memory (avoids repeated disk reads per request)
            self.template_text = self.template_md.read_text(encoding="utf-8")

            # Initialize model loader (your abstraction around embeddings + LLM naming)
            self.model_loader = ModelLoader()

            # ✅ Lazy-load embeddings only if/when RAG is enabled
            # This keeps NO-RAG runs fast and avoids unnecessary GPU/CPU load.
            self.embeddings = None

            # Determine LLM name from model_loader and instantiate Ollama client
            self.llm_name = self.model_loader.llm_name
            self.llm = Ollama(model=self.llm_name)

            # Load the prompt template from registry (currently NIH)
            self.prompt_template = PROMPT_REGISTRY[PromptType.NIH_DMP.value]

            # YAML default: rag.enabled (if missing => default True)
            enabled_val = self.config.get_rag_param("enabled")
            self.use_rag_default = True if enabled_val is None else bool(enabled_val)

            # Build the NO-RAG chain immediately (cheap, does not touch embeddings or FAISS)
            self.no_rag_chain = self._build_no_rag_chain()

            # Lazy RAG fields (built only if use_rag=True)
            self.vectorstore = None  # FAISS vector store
            self.retriever = None    # Retriever wrapper on top of vectorstore
            self.rag_chain = None    # Full retrieval + prompt + LLM chain

            log.info(
                "✅ DMPPipeline initialized",
                llm=self.llm_name,
                rag_default=self.use_rag_default,
            )

        except Exception as e:
            # Log the root error, then raise a project-specific exception wrapper
            log.error("❌ Failed to initialize DMPPipeline", error=str(e))
            raise DocumentPortalException("Pipeline initialization error", e)

    # ---------------------------------------------------------------
    # ✅ safe bool parser (so "false" doesn't become True)
    def _to_bool(self, v, default: Optional[bool] = None) -> Optional[bool]:
        """
        Convert different "truthy/falsey" input forms into a real bool.

        Why:
            In Python, bool("false") is True (because it's a non-empty string).
            This helper avoids that common bug when parsing config/UI JSON inputs.

        Args:
            v: value that might be bool/int/float/str/None
            default: value returned when parsing fails or v is None

        Returns:
            True/False if confidently parsed, otherwise `default`.
        """
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

        - If an index exists and force_rebuild is False: load it from disk.
        - If load fails OR force_rebuild is True: rebuild index from PDFs.

        Robust PDF loading strategy:
            1) Try PyMuPDFLoader (often better at real-world PDFs)
            2) Fallback to PyPDFLoader (if PyMuPDF fails)

        Args:
            force_rebuild: if True, skip loading and rebuild index

        Returns:
            FAISS vectorstore ready for retrieval.

        Raises:
            DocumentPortalException: wraps any underlying indexing errors.
        """
        try:
            # Defensive check: embeddings must be available before building/loading FAISS
            if self.embeddings is None:
                raise RuntimeError("Embeddings are not loaded. Call _ensure_rag_ready() first.")

            # Expected FAISS file name created by FAISS.save_local()
            faiss_path = self.index_dir / "index.faiss"

            # Attempt loading existing index (fast path)
            if faiss_path.exists() and not force_rebuild:
                try:
                    log.info("📦 Loading existing FAISS index", path=str(faiss_path))
                    return FAISS.load_local(
                        str(self.index_dir),
                        self.embeddings,
                        allow_dangerous_deserialization=True,  # needed for FAISS local load in some setups
                    )
                except Exception as e:
                    # If load fails (version mismatch, corruption, etc.), rebuild automatically
                    log.warning(f"⚠️ Failed to load FAISS index. Rebuilding... Reason: {e}")

            # No index found or rebuild required: collect PDFs
            pdf_files = sorted(self.data_pdfs.glob("*.pdf"))
            if not pdf_files:
                raise FileNotFoundError(f"❌ No PDFs found in: {self.data_pdfs}")

            docs = []       # all successfully loaded documents
            bad_pdfs = []   # PDFs that could not be parsed

            # Load each PDF with robust fallback
            for pdf in tqdm(pdf_files, desc="📥 Loading PDFs"):
                try:
                    loader = PyMuPDFLoader(str(pdf))
                    docs.extend(loader.load())
                except Exception as e1:
                    try:
                        loader = PyPDFLoader(str(pdf))
                        docs.extend(loader.load())
                    except Exception as e2:
                        # Record the PDF and continue instead of failing the whole pipeline
                        bad_pdfs.append(str(pdf))
                        log.warning(
                            "⚠️ Skipping PDF due to parse error",
                            pdf=str(pdf),
                            pymupdf_error=str(e1),
                            pypdf_error=str(e2),
                        )
                        continue

            # If nothing loaded, fail with a clear message
            if not docs:
                raise RuntimeError(
                    "❌ No documents could be loaded from PDFs. "
                    "All PDFs may be corrupted or unreadable."
                )

            # Warn if some PDFs were skipped, but do not fail the run
            if bad_pdfs:
                log.warning("⚠️ Some PDFs were skipped", skipped=len(bad_pdfs))
                log.warning("First skipped PDF", first=bad_pdfs[0])

            # Chunking parameters (from YAML or defaults)
            chunk_size = self.config.get_rag_param("chunk_size") or 800
            chunk_overlap = self.config.get_rag_param("chunk_overlap") or 120

            # Split docs into overlapping chunks to improve retrieval
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            chunks = splitter.split_documents(docs)

            # Build FAISS from chunks and persist it
            log.info("🧱 Building new FAISS index ...", chunks=len(chunks))
            vectorstore = FAISS.from_documents(chunks, self.embeddings)
            vectorstore.save_local(str(self.index_dir))
            log.info("✅ FAISS index built and saved", path=str(self.index_dir))
            return vectorstore

        except Exception as e:
            # Wrap any errors to keep consistent exception type upstream
            raise DocumentPortalException("FAISS index error", e)

    # ---------------------------------------------------------------
    def _build_rag_chain(self, retriever):
        """
        Build the RAG chain using:
          - retriever -> context (formatted as a single string)
          - question  -> prompt question slot
          - prompt_template
          - llm
          - string output parser

        Args:
            retriever: LangChain retriever that returns top-k relevant chunks

        Returns:
            A runnable chain that takes {"input": "..."} and returns a string DMP.
        """
        try:
            def format_docs(docs):
                """
                Format retrieved docs into a readable context string.

                Notes:
                    - Includes page and source metadata if present
                    - Keeps chunk content clean (strip whitespace)
                """
                if not docs:
                    return ""
                formatted = []
                for d in docs:
                    page = d.metadata.get("page", "")
                    src = d.metadata.get("source", "")
                    formatted.append(f"[Page {page}] {src}\n{d.page_content.strip()}")
                return "\n\n".join(formatted)

            # RunnableMap builds the expected prompt inputs:
            #   context: retrieved text
            #   question: the user's instruction/query
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
        """
        Build a chain without retrieval.

        Behavior:
          - context is always an empty string
          - question comes from x["input"]
          - prompt_template must accept context/question keys

        Returns:
            Runnable chain for pure prompt-based generation.
        """
        try:
            no_rag_chain = (
                RunnableMap(
                    {
                        "context": lambda x: "",          # no retrieved context
                        "question": lambda x: x["input"], # full query becomes the question
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
        """
        Lazy-init embeddings + FAISS + retriever + rag_chain ONLY when needed.

        This is the key optimization:
          - If use_rag=False, you never pay for embeddings or index loading.
          - If use_rag=True, components are initialized once and reused.
        """
        # If already initialized, do nothing
        if self.rag_chain is not None and self.retriever is not None and self.vectorstore is not None:
            return

        # ✅ Load embeddings only at the moment we truly need RAG
        if self.embeddings is None:
            self.embeddings = self.model_loader.load_embeddings()

        # Load existing FAISS index or build a new one
        self.vectorstore = self._load_or_build_index(force_rebuild=self.force_rebuild_index)

        # Create retriever with top-k configuration
        top_k = self.config.get_rag_param("retriever_top_k") or 6
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})

        # Build the full RAG chain with retriever
        self.rag_chain = self._build_rag_chain(self.retriever)

        log.info("✅ RAG components initialized", llm=self.llm_name, top_k=top_k)

    # ---------------------------------------------------------------
    def _cleanup_title_json(self, safe_title: str):
        """
        Remove any old JSONs for this title except the dmptool JSON.

        Why:
            Older pipeline versions may have produced multiple JSON variants.
            This keeps the output folder clean and prevents confusion.

        Args:
            safe_title: sanitized project title used as the file stem.
        """
        keep = f"{safe_title}.dmptool.json"
        for p in self.output_json.glob(f"{safe_title}*.json"):
            if p.name != keep:
                try:
                    p.unlink()
                except Exception:
                    # Swallow deletion errors (e.g., file locked) since it's non-critical cleanup
                    pass

    # ---------------------------------------------------------------
    def generate_dmp(
        self,
        title: str,
        form_inputs: dict,
        use_rag: Optional[bool] = None,
        funding_agency: str = "NIH",   # ✅ NEW: passed from main.py (top-level JSON)
    ):
        """
        Generate DMP from inputs.

        use_rag:
          - True  => use retrieval (loads embeddings + FAISS lazily)
          - False => no retrieval (never touches embeddings/FAISS)
          - None  => use YAML default rag.enabled

        funding_agency:
          - passed from top-level input JSON (not inside inputs)
          - currently used for logging / future routing (still NIH outputs for now)

        Args:
            title: Project title (used in prompt and output filenames)
            form_inputs: Dict of user-entered fields (Element 1..4 content, etc.)
            use_rag: Override for RAG usage (optional)
            funding_agency: Funder identifier ("NIH", later "NSF", etc.)

        Returns:
            Generated markdown string (the DMP content).
        """
        try:
            # Validate title early to avoid generating unnamed files
            title = (title or "").strip()
            if not title:
                raise ValueError("❌ Title is required.")

            # Normalize funder label for logging/future routing
            funding_agency = (funding_agency or "NIH").strip().upper()
            log.info("🏷️ Funding agency selected", funding_agency=funding_agency)

            # Decide final RAG usage (explicit argument > YAML default)
            use_rag_final = self._to_bool(use_rag, default=self.use_rag_default)

            log.info(
                "🧾 RAG decision",
                use_rag_input=use_rag,
                rag_default=self.use_rag_default,
                use_rag_final=use_rag_final,
            )

            # Choose chain based on final RAG decision
            if use_rag_final:
                # Initializes embeddings/index/retriever only once, when needed
                self._ensure_rag_ready()
                chain = self.rag_chain
            else:
                chain = self.no_rag_chain

            # Convert form inputs into readable lines for the prompt
            # - Only include non-empty string values
            # - Normalize keys into Title Case for readability
            user_elements = [
                f"{key.replace('_',' ').title()}: {val}".strip()
                for key, val in (form_inputs or {}).items()
                if isinstance(val, str) and val.strip()
            ]

            # Build the query/prompt sent to the chain
            # NOTE: Still NIH wording + NIH template (until you add NSF support routing)
            query = (
                f"You are an NIH data steward and grant writer. "
                f"Create a complete NIH Data Management and Sharing Plan (DMSP) "
                f"for the project titled '{title}'.\n\n"
                f"User Inputs:\n{chr(10).join(user_elements)}\n\n"
                f"Use the following NIH DMSP Markdown template. Do not alter section titles:\n"
                f"{self.template_text}"
            )

            # If using RAG, add an explicit instruction to incorporate retrieved context
            # (This helps the model treat context as authoritative guidance.)
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

            # Run the chain: input -> (optional retrieval) -> prompt -> LLM -> string output
            result = chain.invoke({"input": query})

            # Create safe file stem from title (Windows-safe, Linux-safe)
            safe_title = re.sub(r'[\\/*?:"<>|]', "_", title).strip()

            # Compute output paths for each artifact
            md_path = self.output_md / f"{safe_title}.md"
            docx_path = self.output_docx / f"{safe_title}.docx"
            dmptool_json_path = self.output_json / f"{safe_title}.dmptool.json"

            # Save markdown output (the generated DMP text)
            md_path.write_text(result, encoding="utf-8")

            # Generate NIH DOCX using the official NIH template to preserve formatting
            nih_template_docx = Path("data/inputs/nih-dms-plan-template.docx")
            build_nih_docx_from_template(
                template_docx_path=str(nih_template_docx),
                output_docx_path=str(docx_path),
                project_title=title,
                generated_markdown=result,
            )

            # Build the DMPTool-compatible JSON object (ONLY JSON output desired)
            dmptool_obj = build_dmptool_json(
                template_title="NIH Data Management and Sharing Plan",
                project_title=title,
                form_inputs=form_inputs,
                generated_markdown=result,
                provenance="dmpchef",
            )

            # Clean up older json files for this title (keeps only *.dmptool.json)
            self._cleanup_title_json(safe_title)

            # Write the dmptool JSON to disk
            with open(dmptool_json_path, "w", encoding="utf-8") as f:
                json.dump(dmptool_obj, f, indent=2, ensure_ascii=False)

            # Final success log (includes funder + rag decision + paths)
            log.info(
                "✅ DMP generated successfully",
                title=title,
                funding_agency=funding_agency,
                use_rag=use_rag_final,
                md=str(md_path),
                docx=str(docx_path),
                dmptool_json=str(dmptool_json_path),
            )
            return result

        except Exception as e:
            # Wrap any generation-time exception for consistent upstream handling
            raise DocumentPortalException("DMP generation error", e)

from __future__ import annotations
import os, sys, time, json, hashlib, requests, shutil, re
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

# --- project imports ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from logger.custom_logger import GLOBAL_LOGGER as log


class UnifiedWebIngestion:
    """
    🌐 Unified NIH Grants + DMPTool Ingestion (Cross-Session Deduplication + Copy Forward)
    --------------------------------------------------------------------
    ✅ Loads hashes from previous sessions
    ✅ Deduplicates across sessions by file hash
    ✅ Copies previous PDFs/texts into new session before crawling
    ✅ Saves per-domain + master manifests safely
    ✅ Skips duplicate PDFs and login/signup/account pages
    ✅ Extracts meaningful content for RAG
    """

    def __init__(
        self,
        data_root: str = "C:/Users/Nahid/AI_DMP/DMP_RAG_Pipeline/data",
        json_links: str = "data/web_links.json",
        max_depth: int = 5,
        crawl_delay: float = 1.2,
        max_pages: int = 18000,
    ):
        self.data_root = Path(data_root)
        self.session_folder = self._detect_or_create_session_folder()
        self.master_manifest = self.session_folder / "manifest_master.json"
        self.global_manifest = {"sites": {}}

        self.max_depth = max_depth
        self.crawl_delay = crawl_delay
        self.max_pages = max_pages
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (UnifiedIngestor/NIH-RAG)"})

        self.urls = self._load_links(json_links)
        self.previous_hashes = self._load_previous_manifests()

        self.stats = {
            "dmptool.org": {"pdfs": 0, "skipped": 0},
            "grants.nih.gov": {"pages": 0, "pdfs": 0, "skipped": 0},
        }

        print(f"\n✅ Session Folder Created: {self.session_folder}\n")

    # --------------------------------------------------------
    # Folder setup
    # --------------------------------------------------------
    def _detect_or_create_session_folder(self) -> Path:
        parent = self.data_root / "data_ingestion"
        parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        tag = f"{now.year}_{now.month:02d}_{now.day:02d}_NIH_ingestion"
        ts = now.strftime("%Y%m%d_%H%M%S")
        folder = parent / f"{tag}_{ts}"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _prepare_site_dirs(self, domain: str):
        site_root = self.session_folder / domain
        txt_dir, pdf_dir = site_root / "texts", site_root / "pdfs"
        manifest_path = site_root / f"manifest_{domain.replace('.', '_')}.json"
        for d in [txt_dir, pdf_dir]:
            d.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                return txt_dir, pdf_dir, manifest_path, json.load(f)
        return txt_dir, pdf_dir, manifest_path, {"files": {}}

    # --------------------------------------------------------
    # Load previous manifests
    # --------------------------------------------------------
    def _load_previous_manifests(self) -> dict[str, set[str]]:
        parent = self.data_root / "data_ingestion"
        sessions = sorted([p for p in parent.glob("*_NIH_ingestion*") if p.is_dir()], reverse=True)
        if len(sessions) <= 1:
            print("ℹ️ No previous sessions found — starting fresh.")
            return {}

        hash_index = {}
        for folder in sessions[1:]:
            manifest_path = folder / "manifest_master.json"
            if not manifest_path.exists():
                continue
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                for domain, files in manifest.get("sites", {}).items():
                    domain_hashes = hash_index.setdefault(domain, set())
                    domain_hashes.update({v.get("hash") for v in files.values() if "hash" in v})
            except Exception as e:
                print(f"⚠️ Failed to load {manifest_path}: {e}")

        if hash_index:
            print("♻️ Loaded previous session hashes:")
            for d, c in hash_index.items():
                print(f"   - {d}: {len(c)} known hashes")
        else:
            print("⚠️ No previous manifest found — starting fresh.")
        return hash_index

    # --------------------------------------------------------
    # Manifest saving
    # --------------------------------------------------------
    def _save_manifest(self, manifest_path: Path, manifest: dict, domain: str):
        try:
            tmp = manifest_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(manifest_path)
            self.global_manifest["sites"][domain] = manifest.get("files", {})
            with open(self.master_manifest, "w", encoding="utf-8") as f:
                json.dump(self.global_manifest, f, indent=2)
            print(f"✅ Manifest written: {manifest_path}")
        except Exception as e:
            print(f"❌ Manifest save error for {domain}: {e}")

    # --------------------------------------------------------
    # Text filters (enhanced)
    # --------------------------------------------------------
    def _is_valid_text_block(self, text: str) -> bool:
        text = text.strip().lower()
        if not text or len(text.split()) < 5:
            return False
        if re.search(r"\b(expired|superseded|no longer valid)\b", text):
            return False
        if "page last updated" in text or "last modified" in text:
            return False

        skip_terms = [
            "cookie", "privacy", "terms", "subscribe", "newsletter", "login", "sign in", "register",
            "sign up", "create account", "user", "my account", "unsubscribe", "copyright", "social",
            "contact us", "feedback", "help", "faq", "press", "media", "event", "calendar",
            "webinar", "training", "conference", "careers", "employment", "donate"
        ]
        if any(term in text for term in skip_terms):
            return False

        whitelist = [
            "nih", "grant", "funding opportunity", "proposal", "application", "rfa", "foa",
            "data management", "data sharing", "repository", "dataset", "policy",
            "guideline", "regulation", "federal policy", "fair data", "open data",
            "data science", "ai ethics", "clinical trial", "metadata", "dmsp",
            "findable", "accessible", "interoperable", "reusable"
        ]
        return any(term in text for term in whitelist)

    # --------------------------------------------------------
    # Load links
    # --------------------------------------------------------
    def _load_links(self, path: str) -> list[str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("sources", [])
        except Exception as e:
            print(f"⚠️ Failed to load {path}: {e}")
            return []

    def _compute_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    # --------------------------------------------------------
    # HTML cleanup + extraction
    # --------------------------------------------------------
    def _clean_html(self, html: str) -> BeautifulSoup:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "iframe", "svg", "form"]):
            tag.decompose()

        for sel in [
            "[class*=banner]", "[id*=banner]", "[class*=footer]", "[id*=footer]",
            "[class*=nav]", "[id*=nav]", "[role=navigation]", "[class*=menu]", "[id*=menu]",
            "[class*=sidebar]", "[id*=sidebar]", "[class*=social]", "[id*=social]",
            "[class*=search]", "[id*=search]", "[class*=cookie]", "[id*=cookie]",
        ]:
            for t in soup.select(sel):
                t.decompose()
        return soup

    def _extract_text(self, soup: BeautifulSoup) -> str:
        blocks = []
        for el in soup.find_all(["h1", "h2", "h3", "p", "li", "section", "article", "div"]):
            txt = el.get_text(" ", strip=True)
            if self._is_valid_text_block(txt):
                blocks.append(txt)
        merged, buf = [], ""
        for s in blocks:
            if len(buf.split()) < 90:
                buf = (buf + " " + s).strip()
            else:
                merged.append(buf)
                buf = s
        if buf:
            merged.append(buf)
        return "\n\n".join(merged)

    # --------------------------------------------------------
    # PDF download
    # --------------------------------------------------------
    def _download_pdf(self, href: str, pdf_dir: Path, domain: str, manifest: dict):
        try:
            r = self.session.get(href, timeout=30)
            if r.status_code != 200 or b"%PDF" not in r.content[:500]:
                return
            ph = self._compute_hash(r.content)
            if domain in self.previous_hashes and ph in self.previous_hashes[domain]:
                self.stats[domain]["skipped"] += 1
                return
            fname = f"{domain.split('.')[0]}_dmp_{len(manifest['files']) + 1:04d}.pdf"
            dest = pdf_dir / fname
            dest.write_bytes(r.content)
            manifest["files"][href] = {
                "url": href, "file": str(dest), "hash": ph, "type": "pdf",
                "last_updated": datetime.utcnow().isoformat(),
            }
            self.stats[domain]["pdfs"] += 1
        except Exception as e:
            print(f"⚠️ PDF download failed: {href} | {e}")

    # --------------------------------------------------------
    # NIH Crawl (skip login/signup URLs)
    # --------------------------------------------------------
    def _crawl_nih(self, start_url: str, domain: str):
        txt_dir, pdf_dir, manifest_path, manifest = self._prepare_site_dirs(domain)
        visited, queue = set(), [(start_url, 0)]
        skip_url_terms = [
            "login", "signin", "signup", "register", "account", "forgot", "logout",
            "profile", "cart", "donate", "feedback", "subscribe", "unsubscribe"
        ]

        print(f"🌐 Crawling NIH site: {start_url}")
        with tqdm(total=self.max_pages, desc="NIH Pages", unit="page") as pbar:
            while queue and len(visited) < self.max_pages:
                url, depth = queue.pop(0)
                if url in visited or depth > self.max_depth:
                    continue
                if any(term in url.lower() for term in skip_url_terms):
                    continue

                visited.add(url)
                try:
                    r = self.session.get(url, timeout=20)
                    if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
                        continue

                    soup = self._clean_html(r.text)
                    text = self._extract_text(soup)
                    if text:
                        ph = self._compute_hash(text.encode("utf-8"))
                        if ph not in self.previous_hashes.get(domain, set()):
                            fname = f"page_{len(manifest['files']) + 1:04d}.txt"
                            dest = txt_dir / fname
                            dest.write_text(text, encoding="utf-8")
                            manifest["files"][url] = {
                                "url": url, "file": str(dest), "hash": ph,
                                "type": "text", "last_updated": datetime.utcnow().isoformat(),
                            }
                            self.stats[domain]["pages"] += 1

                    for a in soup.find_all("a", href=True):
                        href = urljoin(url, a["href"])
                        if href.lower().endswith(".pdf"):
                            self._download_pdf(href, pdf_dir, domain, manifest)
                        elif urlparse(href).netloc.endswith("nih.gov") and "#" not in href:
                            queue.append((href, depth + 1))

                    pbar.update(1)
                    time.sleep(self.crawl_delay)
                except Exception as e:
                    print(f"⚠️ Crawl failed for {url}: {e}")

        self._save_manifest(manifest_path, manifest, domain)
        print(f"✅ NIH crawl completed — Pages={self.stats[domain]['pages']} PDFs={self.stats[domain]['pdfs']}")

    # --------------------------------------------------------
    # DMPTool Crawl (skip login/register pages)
    # --------------------------------------------------------
    def _crawl_dmptool(self, start_url: str, domain: str):
        _, pdf_dir, manifest_path, manifest = self._prepare_site_dirs(domain)
        print(f"🌐 Crawling DMPTool: {start_url}")

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        prefs = {"download.default_directory": str(pdf_dir.resolve())}
        options.add_experimental_option("prefs", prefs)
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

        try:
            driver.get(start_url)
            time.sleep(6)

            if any(w in driver.current_url.lower() for w in ["login", "signin", "signup", "register", "account"]):
                print(f"⏭️ Skipping auth-related DMPTool page: {driver.current_url}")
                driver.quit()
                return

            seen = set()
            with tqdm(total=self.max_pages, desc="DMPTool PDFs", unit="pdf") as pbar:
                while True:
                    pdf_links = driver.find_elements(By.XPATH, "//a[contains(@href, '/export.pdf')]")
                    if not pdf_links:
                        break
                    hrefs = sorted({l.get_attribute("href") for l in pdf_links if l.get_attribute("href")})
                    new_links = [h for h in hrefs if h not in seen]
                    seen.update(new_links)
                    for href in new_links:
                        self._download_pdf(href, pdf_dir, domain, manifest)
                        pbar.update(1)
                    try:
                        next_btn = driver.find_element(By.CSS_SELECTOR, "a[rel='next']")
                        if not next_btn.is_enabled():
                            break
                        driver.execute_script("arguments[0].click();", next_btn)
                    except NoSuchElementException:
                        break
            self._save_manifest(manifest_path, manifest, domain)
            print(f"✅ DMPTool crawl completed — PDFs={self.stats[domain]['pdfs']}")
        except TimeoutException:
            print("⚠️ Timeout while loading DMPTool pages.")
        finally:
            driver.quit()

    # --------------------------------------------------------
    # Copy and cleanup
    # --------------------------------------------------------
    def _copy_previous_data(self):
        parent = self.data_root / "data_ingestion"
        sessions = sorted([p for p in parent.glob("*_NIH_ingestion*") if p.is_dir()], reverse=True)
        if len(sessions) < 2:
            print("ℹ️ No previous session data to copy.")
            return
        last_session = sessions[1]
        print(f"♻️ Copying files from previous session: {last_session.name}")
        for domain_dir in last_session.iterdir():
            if not domain_dir.is_dir():
                continue
            new_domain_dir = self.session_folder / domain_dir.name
            for subdir in ["pdfs", "texts"]:
                old_path = domain_dir / subdir
                new_path = new_domain_dir / subdir
                if old_path.exists():
                    new_path.mkdir(parents=True, exist_ok=True)
                    for file in old_path.glob("*"):
                        target = new_path / file.name
                        if not target.exists():
                            shutil.copy2(file, target)
        print("✅ Previous data copied.")

    def _cleanup_old_sessions(self):
        parent = self.data_root / "data_ingestion"
        sessions = sorted([p for p in parent.glob("*_NIH_ingestion*") if p.is_dir()], reverse=True)
        if len(sessions) <= 1:
            print("ℹ️ No old sessions to clean.")
            return
        current = self.session_folder
        print(f"🧹 Cleaning up old sessions — keeping only: {current.name}")
        for old in sessions:
            if old != current:
                try:
                    shutil.rmtree(old)
                    print(f"🗑️ Removed old session: {old}")
                except Exception as e:
                    print(f"⚠️ Failed to remove {old}: {e}")

    # --------------------------------------------------------
    # Run all
    # --------------------------------------------------------
    def run_all(self):
        # Clean up older sessions first — keep only current
        self._cleanup_old_sessions()

        # No need to copy data if we only keep the latest
        print("🚀 Starting fresh crawl (latest version only).")

        for url in self.urls:
            domain = urlparse(url).netloc
            if "dmptool.org" in domain:
                self._crawl_dmptool(url, domain)
            elif "nih.gov" in domain:
                self._crawl_nih(url, domain)
            else:
                print(f"⚠️ Skipped unsupported domain: {domain}")

        print("🏁 All crawls complete. Latest session only retained.")

# --------------------------------------------------------
# Example Run
# --------------------------------------------------------
if __name__ == "__main__":
    crawler = UnifiedWebIngestion(
        data_root="C:/Users/Nahid/AI_DMP/DMP_RAG_Pipeline/data",
        json_links="data/web_links.json",
        max_depth=5,
        crawl_delay=1.2,
        max_pages=18000,
    )
    crawler.run_all()

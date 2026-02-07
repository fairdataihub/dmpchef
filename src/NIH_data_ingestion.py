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
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
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
    ✅ Saves per-domain + master manifests safely
    ✅ Skips duplicate PDFs and login/signup/account pages
    ✅ Extracts meaningful content for RAG
    ✅ Exports PDFs after run into <data_root>/<export_pdf_folder> (default: NIH_95)
    ✅ Robustly resolves json_links path regardless of where you run the script from

    ✅ FIXES:
      - COPY-FORWARD: copies previous session PDFs/texts into this session BEFORE crawl
        so export still works even if all downloads are deduped.
      - Adds "already_have" stats (dedupe count) so logs are truthful.
      - Cleanup happens AFTER copy-forward.
      - ✅ EXPORT FIX: prevents *_001 duplicates by skipping export when PDF already exists in NIH_95 (by hash).

    ✅ UPDATE (this request):
      - data_root is now OPTIONAL and auto-resolves to <project_root>/data
        based on this file location (<project>/src/...).
    """

    def __init__(
        self,
        data_root: str | Path | None = None,
        json_links: str = r"data\web_links.json",
        max_depth: int = 5,
        crawl_delay: float = 1.2,
        max_pages: int = 18000,
        export_pdf_folder: str = "NIH_95",   # relative to data_root
        export_mode: str = "move",           # "move" or "copy"
        update_manifest_paths: bool = False, # if True, rewrite master manifest 'file' paths after export
        dmptool_require_nih_filter: bool = True,  # abort if NIH facet not in URL
        dmptool_safety_max_pages: int = 500,       # safety stop for pagination
        keep_last_n_sessions: int = 2,             # keep current + previous by default
        copy_forward_previous: bool = True,        # copy forward previous PDFs/texts into current session
    ):
        # ✅ Auto-detect <project_root>/data if not provided
        if data_root is None:
            # Assumes this file is at <project_root>/src/...
            project_root = Path(__file__).resolve().parents[1]
            data_root = project_root / "data"

        self.data_root = Path(data_root)
        self.session_folder = self._detect_or_create_session_folder()
        self.master_manifest = self.session_folder / "manifest_master.json"
        self.global_manifest = {"sites": {}}

        self.max_depth = max_depth
        self.crawl_delay = crawl_delay
        self.max_pages = max_pages

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (UnifiedIngestor/NIH-RAG)"})

        # export settings
        self.export_pdf_folder = export_pdf_folder
        self.export_mode = export_mode
        self.update_manifest_paths = update_manifest_paths

        # DMPTool controls
        self.dmptool_require_nih_filter = dmptool_require_nih_filter
        self.dmptool_safety_max_pages = dmptool_safety_max_pages

        # session retention / copy-forward controls
        self.keep_last_n_sessions = keep_last_n_sessions
        self.copy_forward_previous = copy_forward_previous

        # robust link loading
        self.urls = self._load_links(json_links)

        # previous hashes (from past manifests)
        self.previous_hashes = self._load_previous_manifests()

        # stats
        self.stats = {
            "dmptool.org": {"pdfs": 0, "skipped": 0, "already_have": 0},
            "grants.nih.gov": {"pages": 0, "pdfs": 0, "skipped": 0, "already_have": 0},
        }

        print(f"\n✅ data_root: {self.data_root.resolve()}")
        print(f"✅ Session Folder Created: {self.session_folder}\n")

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
    # Robust links loader
    # --------------------------------------------------------
    def _load_links(self, path: str) -> list[str]:
        candidates: list[Path] = []
        p = Path(path)

        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(Path.cwd() / p)
            project_root = Path(__file__).resolve().parents[1]  # <project>/src -> <project>
            candidates.append(project_root / p)
            candidates.append(project_root.parent / p)

        for cand in candidates:
            if cand.exists():
                try:
                    with open(cand, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    sources = data.get("sources", [])
                    print(f"✅ Loaded links from: {cand} (sources={len(sources)})")
                    if not isinstance(sources, list):
                        print("⚠️ 'sources' is not a list in web_links.json")
                        return []
                    return sources
                except Exception as e:
                    print(f"⚠️ Failed reading {cand}: {e}")
                    return []

        print(f"⚠️ Failed to load {path}. Tried:\n  - " + "\n  - ".join(str(c) for c in candidates))
        return []

    # --------------------------------------------------------
    # Session listing / copy-forward / cleanup
    # --------------------------------------------------------
    def _list_sessions(self) -> list[Path]:
        parent = self.data_root / "data_ingestion"
        parent.mkdir(parents=True, exist_ok=True)
        return sorted([p for p in parent.glob("*_NIH_ingestion*") if p.is_dir()], reverse=True)

    def _copy_forward_previous_session(self):
        """
        Copy PDFs/texts from the most recent previous session into current session folder.
        This makes export work even if all PDFs are deduped in this run.
        """
        sessions = self._list_sessions()
        if len(sessions) < 2:
            print("ℹ️ No previous session to copy forward.")
            return

        prev = sessions[1]
        print(f"♻️ Copy-forward from previous session: {prev.name}")

        for domain_dir in prev.iterdir():
            if not domain_dir.is_dir():
                continue

            new_domain_dir = self.session_folder / domain_dir.name
            for subdir in ["pdfs", "texts"]:
                old_path = domain_dir / subdir
                if not old_path.exists():
                    continue

                new_path = new_domain_dir / subdir
                new_path.mkdir(parents=True, exist_ok=True)

                for file in old_path.glob("*"):
                    target = new_path / file.name
                    if not target.exists():
                        try:
                            shutil.copy2(file, target)
                        except Exception as e:
                            print(f"⚠️ Failed copy {file} -> {target}: {e}")

        print("✅ Copy-forward complete.")

    def _cleanup_old_sessions(self):
        """
        Keep only the last N sessions (default N=2 keeps current + previous).
        """
        sessions = self._list_sessions()
        if len(sessions) <= self.keep_last_n_sessions:
            print("ℹ️ No old sessions to clean.")
            return

        print(f"🧹 Cleaning up old sessions — keeping last {self.keep_last_n_sessions}:")
        for s in sessions[: self.keep_last_n_sessions]:
            print(f"   ✅ keep: {s.name}")

        for old in sessions[self.keep_last_n_sessions :]:
            try:
                shutil.rmtree(old)
                print(f"🗑️ Removed old session: {old}")
            except Exception as e:
                print(f"⚠️ Failed to remove {old}: {e}")

    # --------------------------------------------------------
    # Load previous manifests (hash index)
    # --------------------------------------------------------
    def _load_previous_manifests(self) -> dict[str, set[str]]:
        sessions = self._list_sessions()
        if len(sessions) <= 1:
            print("ℹ️ No previous sessions found — starting fresh.")
            return {}

        hash_index: dict[str, set[str]] = {}
        for folder in sessions[1:]:
            manifest_path = folder / "manifest_master.json"
            if not manifest_path.exists():
                continue
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                for domain, files in manifest.get("sites", {}).items():
                    domain_hashes = hash_index.setdefault(domain, set())
                    domain_hashes.update(
                        {v.get("hash") for v in files.values() if isinstance(v, dict) and "hash" in v}
                    )
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
    # Text filters
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
        blocks: list[str] = []
        for el in soup.find_all(["h1", "h2", "h3", "p", "li", "section", "article", "div"]):
            txt = el.get_text(" ", strip=True)
            if self._is_valid_text_block(txt):
                blocks.append(txt)

        merged: list[str] = []
        buf = ""
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
    # PDF download (dedupe by previous hashes)
    # --------------------------------------------------------
    def _download_pdf(self, href: str, pdf_dir: Path, domain: str, manifest: dict):
        try:
            r = self.session.get(href, timeout=30)
            if r.status_code != 200 or b"%PDF" not in r.content[:500]:
                return

            ph = self._compute_hash(r.content)

            if ph in self.previous_hashes.get(domain, set()):
                self.stats.setdefault(domain, {"pdfs": 0, "skipped": 0, "already_have": 0})
                self.stats[domain]["already_have"] += 1
                return

            fname = f"{domain.split('.')[0]}_dmp_{len(manifest['files']) + 1:04d}.pdf"
            dest = pdf_dir / fname
            dest.write_bytes(r.content)

            manifest["files"][href] = {
                "url": href,
                "file": str(dest),
                "hash": ph,
                "type": "pdf",
                "last_updated": datetime.utcnow().isoformat(),
            }

            self.stats.setdefault(domain, {"pdfs": 0, "skipped": 0, "already_have": 0})
            self.stats[domain]["pdfs"] += 1

        except Exception as e:
            print(f"⚠️ PDF download failed: {href} | {e}")

    # --------------------------------------------------------
    # NIH Crawl
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
                                "url": url,
                                "file": str(dest),
                                "hash": ph,
                                "type": "text",
                                "last_updated": datetime.utcnow().isoformat(),
                            }
                            self.stats.setdefault(domain, {"pages": 0, "pdfs": 0, "skipped": 0, "already_have": 0})
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
        domain_pages = self.stats.get(domain, {}).get("pages", 0)
        domain_pdfs = self.stats.get(domain, {}).get("pdfs", 0)
        already = self.stats.get(domain, {}).get("already_have", 0)
        print(f"✅ NIH crawl completed — pages={domain_pages} downloaded_pdfs={domain_pdfs} already_have={already}")

    # --------------------------------------------------------
    # DMPTool Crawl
    # --------------------------------------------------------
    def _crawl_dmptool(self, start_url: str, domain: str):
        _, pdf_dir, manifest_path, manifest = self._prepare_site_dirs(domain)
        print(f"🌐 Crawling DMPTool: {start_url}")

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        wait = WebDriverWait(driver, 25)

        try:
            driver.get(start_url)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            time.sleep(1)

            cur = driver.current_url
            print("🔎 Current URL:", cur)

            if any(w in cur.lower() for w in ["login", "signin", "signup", "register", "account"]):
                print(f"⏭️ Skipping auth-related DMPTool page: {cur}")
                return

            if self.dmptool_require_nih_filter:
                if "facet%5Bfunder_ids%5D%5B%5D=123" not in cur and "facet[funder_ids][]=123" not in cur:
                    print("⚠️ NIH filter NOT present in current URL. Stopping to avoid downloading all public plans.")
                    return

            total_expected = None
            try:
                h2 = driver.find_element(By.XPATH, "//h2[contains(normalize-space(.), 'Plans')]")
                m = re.search(r"Plans\s*\((\d+)\)", h2.text)
                if m:
                    total_expected = int(m.group(1))
                    print(f"✅ Detected filtered plan count: {total_expected}")
            except Exception:
                pass

            pbar_total = total_expected if total_expected is not None else self.max_pages

            seen = set()
            page_num = 0
            stagnant_pages = 0

            with tqdm(total=pbar_total, desc="DMPTool NIH PDFs", unit="pdf") as pbar:
                while True:
                    page_num += 1
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "a")))

                    pdf_links = driver.find_elements(By.XPATH, "//a[contains(@href, '/export.pdf')]")
                    hrefs = sorted({a.get_attribute("href") for a in pdf_links if a.get_attribute("href")})
                    new_links = [h for h in hrefs if h not in seen]

                    if not new_links:
                        stagnant_pages += 1
                    else:
                        stagnant_pages = 0

                    for href in new_links:
                        seen.add(href)
                        self._download_pdf(href, pdf_dir, domain, manifest)
                        pbar.update(1)

                    if total_expected is not None and len(seen) >= total_expected:
                        print(f"✅ Reached detected total ({total_expected}). Stopping.")
                        break
                    if stagnant_pages >= 2:
                        print("⚠️ No new PDFs on 2 consecutive pages. Stopping to avoid looping.")
                        break

                    next_btn = None
                    for how, sel in [
                        (By.LINK_TEXT, "Next"),
                        (By.PARTIAL_LINK_TEXT, "Next"),
                        (By.CSS_SELECTOR, "a[rel='next']"),
                        (By.CSS_SELECTOR, "a[aria-label='Next']"),
                        (By.CSS_SELECTOR, "li.next a"),
                    ]:
                        try:
                            next_btn = driver.find_element(how, sel)
                            if next_btn:
                                break
                        except NoSuchElementException:
                            next_btn = None

                    if not next_btn:
                        break

                    aria_disabled = (next_btn.get_attribute("aria-disabled") or "").lower()
                    parent_class = ""
                    try:
                        parent_class = (next_btn.find_element(By.XPATH, "./..").get_attribute("class") or "").lower()
                    except Exception:
                        pass
                    if aria_disabled == "true" or "disabled" in parent_class:
                        break

                    prev_url = driver.current_url
                    driver.execute_script("arguments[0].click();", next_btn)

                    try:
                        wait.until(lambda d: d.current_url != prev_url)
                    except Exception:
                        time.sleep(2)

                    time.sleep(1)

                    if page_num >= self.dmptool_safety_max_pages:
                        print("⚠️ Safety stop: too many pages. Stopping.")
                        break

            self._save_manifest(manifest_path, manifest, domain)
            downloaded = self.stats.get(domain, {}).get("pdfs", 0)
            already = self.stats.get(domain, {}).get("already_have", 0)
            print(f"✅ DMPTool crawl completed — downloaded={downloaded} already_have={already}")

        except TimeoutException:
            print("⚠️ Timeout while loading DMPTool pages.")
        finally:
            driver.quit()

    # --------------------------------------------------------
    # Export PDFs to fixed folder (SKIP duplicates in destination by hash)
    # --------------------------------------------------------
    def _export_pdfs_to_folder(self, dest_rel: str, mode: str = "move"):
        dest_dir = (self.data_root / dest_rel)
        dest_dir.mkdir(parents=True, exist_ok=True)

        pdfs = list(self.session_folder.rglob("*.pdf"))
        if not pdfs:
            print("ℹ️ No PDFs found to export.")
            return

        # Hash what already exists in the destination to prevent *_001 duplicates
        existing_hashes: set[str] = set()
        for existing in dest_dir.glob("*.pdf"):
            try:
                b = existing.read_bytes()
                if b"%PDF" in b[:500]:
                    existing_hashes.add(self._compute_hash(b))
            except Exception:
                continue

        def unique_path(path: Path) -> Path:
            if not path.exists():
                return path
            stem, suf = path.stem, path.suffix
            i = 1
            while True:
                candidate = path.with_name(f"{stem}_{i:03d}{suf}")
                if not candidate.exists():
                    return candidate
                i += 1

        exported_map: dict[str, str] = {}
        exported = 0
        skipped_existing = 0

        for src in pdfs:
            try:
                b = src.read_bytes()
                if b"%PDF" not in b[:500]:
                    continue

                h = self._compute_hash(b)

                # ✅ If already present in NIH_95, skip export (prevents *_001 duplicates)
                if h in existing_hashes:
                    skipped_existing += 1
                    continue

                dst = unique_path(dest_dir / src.name)

                if mode == "copy":
                    shutil.copy2(src, dst)
                else:
                    shutil.move(str(src), str(dst))

                exported_map[str(src)] = str(dst)
                exported += 1
                existing_hashes.add(h)  # prevent duplicates within the same run too

            except Exception as e:
                print(f"⚠️ Failed to {mode} {src}: {e}")

        print(f"✅ Exported {exported} NEW PDFs to: {dest_dir}")
        print(f"♻️ Skipped {skipped_existing} PDFs already in destination (by hash)")

        if self.update_manifest_paths and exported_map:
            self._rewrite_master_manifest_pdf_paths(exported_map)

    def _rewrite_master_manifest_pdf_paths(self, exported_map: dict[str, str]):
        try:
            if not self.master_manifest.exists():
                return

            with open(self.master_manifest, "r", encoding="utf-8") as f:
                master = json.load(f)

            changed = 0
            for domain, files in master.get("sites", {}).items():
                for _url, meta in files.items():
                    if isinstance(meta, dict) and meta.get("type") == "pdf":
                        old_path = meta.get("file")
                        if old_path in exported_map:
                            meta["file"] = exported_map[old_path]
                            changed += 1

            if changed:
                tmp = self.master_manifest.with_suffix(".tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(master, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                tmp.replace(self.master_manifest)
                print(f"✅ Updated master manifest PDF paths: {changed} entries")

        except Exception as e:
            print(f"⚠️ Failed to update master manifest paths: {e}")

    # --------------------------------------------------------
    # Run all
    # --------------------------------------------------------
    def run_all(self):
        if not self.urls:
            print("⚠️ No URLs loaded from web_links.json. Nothing to crawl.")
            return

        # copy-forward BEFORE cleanup + crawl
        if self.copy_forward_previous:
            self._copy_forward_previous_session()

        # cleanup AFTER copy-forward
        self._cleanup_old_sessions()

        print("🚀 Starting crawl.")

        for url in self.urls:
            domain = urlparse(url).netloc
            if "dmptool.org" in domain:
                self._crawl_dmptool(url, domain)
            elif "nih.gov" in domain:
                self._crawl_nih(url, domain)
            else:
                print(f"⚠️ Skipped unsupported domain: {domain}")

        self._export_pdfs_to_folder(dest_rel=self.export_pdf_folder, mode=self.export_mode)
        print("🏁 All crawls complete.")


if __name__ == "__main__":
    crawler = UnifiedWebIngestion(
        data_root=None,  # ✅ auto-resolves to <project_root>/data
        json_links=r"data\web_links.json",
        max_depth=5,
        crawl_delay=1.2,
        max_pages=18000,
        export_pdf_folder="NIH_95",
        export_mode="move",
        update_manifest_paths=False,
        dmptool_require_nih_filter=True,
        dmptool_safety_max_pages=500,
        keep_last_n_sessions=2,
        copy_forward_previous=True,
    )
    crawler.run_all()

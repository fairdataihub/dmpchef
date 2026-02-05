# utils/nih_docx_writer.py

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from docx import Document
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph


# DOCX anchors (instruction paragraphs in the NIH template)
PROMPT_ANCHORS: List[Tuple[str, List[str]]] = [
    ("e1_q1", ["Summarize the types and estimated amount of scientific data expected to be generated in the project"]),
    ("e1_q2", ["Describe which scientific data from the project will be preserved and shared and provide the rationale for this decision"]),
    ("e1_q3", ["Briefly list the metadata, other relevant data, and any associated documentation"]),
    ("e2_q1", ["State whether specialized tools, software, and/or code are needed to access or manipulate shared scientific data"]),
    ("e3_q1", ["State what common data standards will be applied to the scientific data and associated metadata to enable interoperability of datasets and resources"]),
    ("e4_q1", ["Provide the name of the repository(ies) where scientific data and metadata arising from the project will be archived"]),
    ("e4_q2", ["Describe how the scientific data will be findable and identifiable"]),
    ("e4_q3", ["Describe when the scientific data will be made available to other users"]),
    ("e5_q1", ["NIH expects that in drafting Plans, researchers maximize the appropriate sharing of scientific data."]),
    ("e5_q2", ["State whether access to the scientific data will be controlled"]),
    ("e5_q3", ["If generating scientific data derived from humans, describe how the privacy, rights, and confidentiality of human research participants will be protected"]),
    ("e6_q1", ["Describe how compliance with this Plan will be monitored and managed, frequency of oversight, and by whom at your institution"]),
]


# ---------- DOCX helpers ----------

def _insert_paragraph_after(paragraph: Paragraph, text: str, style_name: Optional[str] = None) -> Paragraph:
    new_p = OxmlElement("w:p")
    paragraph._p.addnext(new_p)
    new_para = Paragraph(new_p, paragraph._parent)

    if style_name:
        try:
            new_para.style = style_name
        except Exception:
            pass

    if text:
        new_para.add_run(text)

    return new_para


def _delete_paragraph(paragraph: Paragraph) -> None:
    p = paragraph._element
    p.getparent().remove(p)
    paragraph._p = paragraph._element = None  # type: ignore


def _normalize_ws_after(doc: Document, idx: int) -> None:
    """Delete all consecutive blank paragraphs starting at idx."""
    j = idx
    while j < len(doc.paragraphs):
        p = doc.paragraphs[j]
        if (p.text or "").strip() == "":
            _delete_paragraph(p)
            continue
        break


def _find_anchor_paragraph_index(doc: Document, anchor_aliases: List[str]) -> Optional[int]:
    for i, p in enumerate(doc.paragraphs):
        pt = (p.text or "").strip()
        for a in anchor_aliases:
            if a.lower() in pt.lower():
                return i
    return None


def _write_answer_after_anchor(doc: Document, anchor_aliases: List[str], answer: str) -> None:
    idx = _find_anchor_paragraph_index(doc, anchor_aliases)
    if idx is None:
        return

    anchor_p = doc.paragraphs[idx]
    style_to_use = anchor_p.style.name if anchor_p.style else None

    _normalize_ws_after(doc, idx + 1)
    _insert_paragraph_after(anchor_p, answer.strip(), style_name=style_to_use)


def _remove_placeholder_only_paragraphs(doc: Document) -> None:
    placeholders = {k.lower() for k, _ in PROMPT_ANCHORS}
    for p in doc.paragraphs:
        if (p.text or "").strip().lower() in placeholders:
            p.text = ""


# ---------- Markdown parsing for YOUR output format ----------

def _strip_markdown_keep_structure(md: str) -> str:
    """Keep headings, remove links/emphasis, normalize whitespace."""
    if not md:
        return ""
    text = md.replace("\r\n", "\n").replace("\r", "\n")

    # [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # remove bold/italic markers but keep content
    text = text.replace("**", "").replace("*", "")

    # collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _split_into_heading_blocks(text: str) -> List[Tuple[str, str]]:
    """
    Split by headings:
      - 'Element X: ...' lines (may appear as plain after stripping **)
      - '### ...' lines
    Returns list of (heading, content_until_next_heading)
    """
    lines = text.split("\n")
    blocks: List[Tuple[str, List[str]]] = []

    def is_heading(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        if s.lower().startswith("element "):
            return True
        if s.startswith("### "):
            return True
        return False

    current_heading: Optional[str] = None
    current_content: List[str] = []

    for line in lines:
        if is_heading(line):
            if current_heading is not None:
                blocks.append((current_heading, current_content))
            current_heading = line.strip().lstrip("#").strip()
            current_content = []
        else:
            current_content.append(line)

    if current_heading is not None:
        blocks.append((current_heading, current_content))

    out: List[Tuple[str, str]] = []
    for h, content_lines in blocks:
        content = "\n".join(content_lines).strip()
        out.append((h, content))
    return out


def _extract_blocks_from_generated_markdown(generated_markdown: str) -> Dict[str, str]:
    """
    Map YOUR headings to e*_q* keys.

    Expected headings in your generated markdown:
      Element 1: Data Type
        ### Types and amount...
        ### Scientific data that will be preserved...
        ### Metadata, other relevant data...
      Element 2: Related Tools...
      Element 3: Standards
      Element 4:
        ### Repository...
        ### How scientific data will be findable...
        ### When and how long...
      Element 5:
        ### Factors affecting...
        ### Whether access...
        ### Protections...
      Element 6: Oversight...
    """
    plain = _strip_markdown_keep_structure(generated_markdown)
    heading_blocks = _split_into_heading_blocks(plain)

    # Normalize heading -> content
    hmap: Dict[str, str] = {}
    for h, c in heading_blocks:
        hmap[h.lower()] = c.strip()

    def get_h(contains: str) -> str:
        contains = contains.lower()
        for h in hmap.keys():
            if contains in h:
                return hmap[h]
        return ""

    blocks: Dict[str, str] = {}

    # Element 1 sub-headings
    blocks["e1_q1"] = get_h("types and amount of scientific data expected")
    blocks["e1_q2"] = get_h("scientific data that will be preserved and shared")
    blocks["e1_q3"] = get_h("metadata, other relevant data")

    # Element 2/3/6 are whole-element blocks (content after "Element X: ...")
    blocks["e2_q1"] = get_h("element 2: related tools")
    blocks["e3_q1"] = get_h("element 3: standards")
    blocks["e6_q1"] = get_h("element 6: oversight")

    # Element 4 sub-headings
    blocks["e4_q1"] = get_h("repository where scientific data and metadata will be archived")
    blocks["e4_q2"] = get_h("how scientific data will be findable and identifiable")
    blocks["e4_q3"] = get_h("when and how long the scientific data will be made available")

    # Element 5 sub-headings
    blocks["e5_q1"] = get_h("factors affecting subsequent access")
    blocks["e5_q2"] = get_h("whether access to scientific data will be controlled")
    blocks["e5_q3"] = get_h("protections for privacy, rights, and confidentiality")

    # Clean leading ":" etc
    for k, v in list(blocks.items()):
        blocks[k] = (v or "").strip().lstrip(" \t\n\r:.-").strip()

    return blocks


# ---------- Public API ----------

def build_nih_docx_from_template(
    template_docx_path: str,
    output_docx_path: str,
    project_title: str,
    generated_markdown: str,
) -> None:
    doc = Document(str(template_docx_path))

    # Insert Project Title under the main heading
    for p in doc.paragraphs:
        if (p.text or "").strip().upper() == "DATA MANAGEMENT AND SHARING PLAN":
            title_line = f"Project Title: {project_title.strip()}"
            _insert_paragraph_after(p, title_line, style_name=p.style.name if p.style else None)
            break

    # Extract answer blocks from YOUR markdown format
    blocks = _extract_blocks_from_generated_markdown(generated_markdown)

    # Remove old placeholder-only paragraphs if they exist
    _remove_placeholder_only_paragraphs(doc)

    # Insert each extracted answer after the NIH instruction paragraph
    for key, aliases in PROMPT_ANCHORS:
        answer = (blocks.get(key, "") or "").strip()
        if not answer:
            continue
        _write_answer_after_anchor(doc, aliases, answer)

    Path(str(output_docx_path)).parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_docx_path))

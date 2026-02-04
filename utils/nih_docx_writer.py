# utils/nih_docx_writer.py

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from docx import Document
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph


# These are the NIH prompt lines in the Word template (anchors).
# We will find these in the template, then write answers right after them.
PROMPT_ANCHORS: List[Tuple[str, List[str]]] = [
    ("e1_q1", ["Types and amount of scientific data expected to be generated in the project"]),
    ("e1_q2", ["Scientific data that will be preserved and shared, and the rationale for doing so"]),
    ("e1_q3", ["Metadata, other relevant data, and associated documentation"]),
    ("e2_q1", ["Related tools, software and/or code", "Element 2: Related Tools, Software and/or Code"]),
    ("e3_q1", ["Element 3: Standards", "Standards"]),
    ("e4_q1", ["Repository where scientific data and metadata will be archived"]),
    ("e4_q2", ["How scientific data will be findable and identifiable"]),
    ("e4_q3", ["When and how long the scientific data will be made available"]),
    ("e5_q1", ["Factors affecting subsequent access, distribution, or reuse of scientific data"]),
    ("e5_q2", ["Whether access to scientific data will be controlled"]),
    ("e5_q3", ["Protections for privacy, rights, and confidentiality of human research participants"]),
    ("e6_q1", ["Element 6: Oversight of Data Management and Sharing", "Oversight of Data Management and Sharing"]),
]


def _strip_markdown(md: str) -> str:
    """Lightweight markdown -> plain text."""
    if not md:
        return ""
    text = md

    # remove code fences
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    # remove headings markers
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)

    # bold/italic markers
    text = text.replace("**", "").replace("*", "")

    # collapse extra blank lines a bit
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_by_anchor_blocks(plain_text: str) -> Dict[str, str]:
    """
    Try to extract answer blocks from the generated text by scanning for anchor prompts.
    If an anchor isn't found, we leave it blank (you still keep the Word structure).
    """
    t = plain_text
    lower = t.lower()

    # Find positions of each anchor (first matching alias)
    positions: List[Tuple[str, int]] = []
    for key, aliases in PROMPT_ANCHORS:
        pos = -1
        for a in aliases:
            p = lower.find(a.lower())
            if p != -1:
                pos = p
                break
        positions.append((key, pos))

    # Build blocks in order using next anchor start as boundary
    blocks: Dict[str, str] = {}
    for i, (key, start) in enumerate(positions):
        if start == -1:
            blocks[key] = ""
            continue

        # Find end-of-line
        endline = t.find("\n", start)
        if endline == -1:
            endline = start

        # If the anchor line contains ":", start after it
        anchor_line = t[start:endline]
        colon_idx = anchor_line.find(":")
        content_start = (start + colon_idx + 1) if colon_idx != -1 else endline

        # Next anchor start
        next_start = len(t)
        for j in range(i + 1, len(positions)):
            _, ns = positions[j]
            if ns != -1:
                next_start = ns
                break

        chunk = t[content_start:next_start].strip()
        chunk = chunk.lstrip(" \n\r\t:-").strip()
        blocks[key] = chunk

    return blocks


def _insert_paragraph_after(paragraph: Paragraph, text: str, style_name: Optional[str] = None) -> Paragraph:
    """
    Safely insert a new paragraph after an existing paragraph (python-docx safe method).
    """
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


def _write_answer_after_anchor(doc: Document, anchor_aliases: List[str], answer: str) -> None:
    """
    Find the anchor paragraph in the template and write the answer right after it.
    If the next paragraph is blank, fill it. Otherwise insert a new paragraph after anchor.
    """
    if answer is None:
        answer = ""
    answer = answer.strip()

    # Find anchor paragraph
    anchor_idx = None
    for i, p in enumerate(doc.paragraphs):
        pt = (p.text or "").strip()
        for a in anchor_aliases:
            if a.lower() in pt.lower():
                anchor_idx = i
                break
        if anchor_idx is not None:
            break

    if anchor_idx is None:
        return  # anchor not found in this template

    anchor_p = doc.paragraphs[anchor_idx]
    style_to_use = anchor_p.style.name if anchor_p.style else None

    # Prefer to fill the next paragraph if it's empty
    if anchor_idx + 1 < len(doc.paragraphs):
        next_p = doc.paragraphs[anchor_idx + 1]
        if (next_p.text or "").strip() == "":
            if next_p.style:
                style_to_use = next_p.style.name
            next_p.text = ""
            next_p.add_run(answer)
            return

    # Otherwise insert a new paragraph after anchor
    _insert_paragraph_after(anchor_p, answer, style_name=style_to_use)


def build_nih_docx_from_template(
    template_docx_path: str,
    output_docx_path: str,
    project_title: str,
    generated_markdown: str,
) -> None:
    """
    Create a DOCX that matches the NIH blank template format by filling content into the template.
    """
    doc = Document(str(template_docx_path))

    # Insert Project Title under main heading (keeps NIH style)
    for p in doc.paragraphs:
        if (p.text or "").strip().upper() == "DATA MANAGEMENT AND SHARING PLAN":
            title_line = f"Project Title: {project_title.strip()}"
            _insert_paragraph_after(p, title_line, style_name=p.style.name if p.style else None)
            break

    plain = _strip_markdown(generated_markdown)
    blocks = _extract_by_anchor_blocks(plain)

    # Fill each question block
    for key, aliases in PROMPT_ANCHORS:
        _write_answer_after_anchor(doc, aliases, blocks.get(key, ""))

    Path(str(output_docx_path)).parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_docx_path))
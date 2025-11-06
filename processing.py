from __future__ import annotations

import io
import json
import re
import statistics
import tempfile
from pathlib import Path
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional dependency
    fitz = None

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:  # pragma: no cover - optional dependency
    pdfminer_extract_text = None

try:
    from indices.ai_corpus_index import parser as rich_parser
except Exception:  # pragma: no cover - optional dependency
    rich_parser = None

DEFAULT_WINDOW_SIZE = 3
DEFAULT_WINDOW_STRIDE = 2
MIN_SENTENCE_CHARS = 15

KNOWN_HEADINGS = [
    "abstract", "introduction", "background", "literature review", "literature survey", "related work", "related works",
    "problem statement", "approach", "model", "methodology", "methods", "materials and methods", "materials methods",
    "experimental setup", "experiments", "results", "results and discussion", "evaluation", "analysis", "discussion",
    "discussion and conclusion", "conclusion", "conclusions", "future work", "limitations", "acknowledgments", "acknowledgements",
    "references", "bibliography", "appendix", "supplementary materials", "supplementary material", "supplementary information",
]
STOP_WORDS_HEADINGS = {"and", "the", "of", "on", "in", "for", "to", "a", "an", "chapter", "section", "part"}

MAX_HEADING_WORDS = 8
MAX_HEADING_CHARS = 64
HEADING_SIM_THRESHOLD = 0.97

BLOCK_PARA = "PARA"
BLOCK_HEADING = "HEADING"
BLOCK_TABLE = "TABLE"
BLOCK_CAPTION = "CAPTION"
BLOCK_LIST = "LIST"

NARRATIVE_BLOCK_TYPES = {BLOCK_PARA, BLOCK_CAPTION}

LIST_BULLETS = {"-", "*", "+", "\u2022", "\u25E6", "\u25AA"}

INLINE_MATH_RE = re.compile(
    r"(\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]|\\begin\{equation\}.*?\\end\{equation\})",
    re.DOTALL,
)



SECTION_ALIASES: Dict[str, Sequence[str]] = {
    "ABSTRACT": ("ABSTRACT", "SUMMARY"),
    "INTRODUCTION": ("INTRODUCTION", "BACKGROUND", "1 INTRODUCTION", "1. INTRODUCTION"),
    "METHODS": (
        "METHODS",
        "MATERIALS AND METHODS",
        "MATERIALS & METHODS",
        "METHODOLOGY",
        "EXPERIMENTAL SETUP",
    ),
    "RESULTS": ("RESULTS", "FINDINGS", "EXPERIMENTS"),
    "DISCUSSION": ("DISCUSSION", "DISCUSSIONS", "ANALYSIS"),
    "CONCLUSION": ("CONCLUSION", "CONCLUSIONS", "SUMMARY AND CONCLUSION"),
    "REFERENCES": ("REFERENCES", "BIBLIOGRAPHY"),
    "ACKNOWLEDGMENTS": ("ACKNOWLEDGMENTS", "ACKNOWLEDGEMENTS"),
    "APPENDIX": ("APPENDIX", "APPENDICES", "SUPPLEMENTARY MATERIAL"),
}

HeadingDetector = Callable[[str], str]
SentenceSplitter = Callable[[str], List[str]]


@dataclass
class Block:
    block_id: int
    block_type: str
    text: str
    page: Optional[int] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class Section:
    section_id: int
    title: str
    normalized: str
    flags: Dict[str, bool]
    block_ids: List[int] = field(default_factory=list)


@dataclass
class Sentence:
    sentence_id: int
    section_id: int
    section_name: str
    section_offset: int
    block_id: int
    text: str
    clean_text: str
    block_type: str
    page: Optional[int] = None


@dataclass
class Window:
    window_id: int
    doc_id: str
    section_id: int
    section_name: str
    span_start: int
    span_end: int
    sentence_indices: List[int]
    text_raw: str
    text_clean: str


@dataclass
class DocumentData:
    doc_id: str
    source: str
    blocks: List[Block]
    sections: List[Section]
    sentences: List[Sentence]
    windows: List[Window]
    page_to_block_ids: Dict[int, List[int]] = field(default_factory=dict)


def normalize_section_name(title: str) -> str:
    raw = re.sub(r"[^A-Za-z0-9 ]+", " ", title.upper()).strip()
    raw = re.sub(r"\s+", " ", raw)
    if not raw:
        return "SECTION"
    for canonical, variants in SECTION_ALIASES.items():
        for variant in variants:
            if raw == variant:
                return canonical
            if raw.startswith(variant + " "):
                return canonical
    return raw


def strip_math(text: str) -> str:
    return INLINE_MATH_RE.sub(" ", text)


def _normalize_heading_candidate(text: str) -> str:
    cleaned = re.sub(r'[\\-_]+', ' ', text)
    cleaned = re.sub(r'[^A-Za-z0-9 ]+', ' ', cleaned)
    cleaned = cleaned.lower()
    tokens = []
    for tok in cleaned.split():
        if not tok or tok in STOP_WORDS_HEADINGS:
            continue
        if tok.isdigit():
            continue
        if re.fullmatch(r'[ivxlcdm]+', tok):
            continue
        tokens.append(tok)
    return ' '.join(tokens)

NORMALIZED_HEADINGS = { _normalize_heading_candidate(h): h for h in KNOWN_HEADINGS }

def _extract_lines_with_styles(page) -> List[Dict[str, Any]]:
    lines = []
    try:
        pdata = page.get_text("dict")
    except Exception:
        return lines
    for blk in pdata.get("blocks", []):
        if blk.get("type", 0) != 0:
            continue
        for ln in blk.get("lines", []):
            spans = ln.get("spans", [])
            text = "".join(sp.get("text", "") for sp in spans).strip()
            if not text:
                continue
            size = max((sp.get("size", 0) for sp in spans), default=0)
            flags = max((sp.get("flags", 0) for sp in spans), default=0)
            bbox = ln.get("bbox", [0, 0, 0, 0])
            lines.append({"text": text, "size": size, "flags": flags, "bbox": bbox})
    return lines


def _guess_heading_from_style(line: Dict[str, Any], base_size: float) -> bool:
    text = line.get("text", "").strip()
    if not text:
        return False
    if len(text) > MAX_HEADING_CHARS:
        return False
    tokens = text.split()
    if len(tokens) > MAX_HEADING_WORDS:
        return False
    size = float(line.get("size", 0.0) or 0.0)
    flags = int(line.get("flags", 0) or 0)
    is_bold = bool(flags & 2)
    font_bonus = base_size > 0 and size >= (base_size * 1.18)
    bold_bonus = base_size > 0 and is_bold and size >= (base_size * 1.06)
    numbered = bool(re.match(r"^(?:[0-9]+(?:\.[0-9]+)*|[IVXLCM]+[.])\s+", text))
    punct_ok = text.endswith(":") or not text.endswith(".")
    known = _match_known_heading(text)
    return punct_ok and (font_bonus or bold_bonus or numbered or known)


def _is_short_heading(line: str) -> bool:
    if not line:
        return False
    cand = line.strip()
    if ":" in cand:
        cand = cand.split(":",1)[0]
    cand = " ".join(cand.split())
    if len(cand) > MAX_HEADING_CHARS:
        return False
    words = [w for w in cand.split() if w]
    return len(words) <= MAX_HEADING_WORDS

def _match_known_heading(candidate: str) -> bool:
    if not candidate:
        return False
    raw = candidate.lower().strip()
    normalized = _normalize_heading_candidate(candidate)
    if raw in KNOWN_HEADINGS or normalized in NORMALIZED_HEADINGS:
        return True
    for heading in KNOWN_HEADINGS:
        ratio_raw = SequenceMatcher(None, raw, heading).ratio()
        ratio_norm = SequenceMatcher(None, normalized, heading).ratio()
        if max(ratio_raw, ratio_norm) >= HEADING_SIM_THRESHOLD:
            return True
    for heading in NORMALIZED_HEADINGS.keys():
        if SequenceMatcher(None, normalized, heading).ratio() >= HEADING_SIM_THRESHOLD:
            return True
    return False


class DocumentProcessor:
    def __init__(
        self,
        sentence_splitter: str = "regex",
        window_size: int = DEFAULT_WINDOW_SIZE,
        window_stride: int = DEFAULT_WINDOW_STRIDE,
        min_sentence_chars: int = MIN_SENTENCE_CHARS,
    ) -> None:
        self.splitter_kind = sentence_splitter.lower()
        self.window_size = window_size
        self.window_stride = window_stride
        self.min_sentence_chars = min_sentence_chars
        self._sentence_splitter = self._build_sentence_splitter(self.splitter_kind)

    def from_pdf(self, file_bytes: bytes, doc_id: str, max_pages: Optional[int] = None) -> DocumentData:
        if rich_parser is not None:
            try:
                return self._from_parser_pipeline(file_bytes, doc_id, max_pages)
            except Exception:
                pass  # fallback to legacy extractor
        pages = self._extract_pages_from_pdf(file_bytes, max_pages=max_pages)
        return self._from_pages(pages, doc_id=doc_id, source="pdf")

    def from_text(self, text: str, doc_id: str) -> DocumentData:
        pages = [(1, text)]
        return self._from_pages(pages, doc_id=doc_id, source="text")

    def _from_parser_pipeline(
        self,
        file_bytes: bytes,
        doc_id: str,
        max_pages: Optional[int],
    ) -> DocumentData:
        pdf_bytes = file_bytes
        if max_pages is not None and fitz is not None and max_pages > 0:
            src_doc = fitz.open(stream=file_bytes, filetype="pdf")
            try:
                total_pages = len(src_doc)
                if total_pages > max_pages:
                    dest_doc = fitz.open()
                    dest_doc.insert_pdf(src_doc, from_page=0, to_page=max_pages - 1)
                    pdf_bytes = dest_doc.tobytes()
                    dest_doc.close()
                else:
                    pdf_bytes = src_doc.tobytes()
            finally:
                src_doc.close()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            tmp_path = Path(tmp.name)
        try:
            parsed = rich_parser.process_document(str(tmp_path))
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
        blocks, sections, page_to_block_ids = self._convert_parser_sections(parsed.get("sections", []))
        if not blocks:
            raise ValueError("parser produced no blocks")
        sentences, windows = self._build_sentences_and_windows(doc_id, blocks, sections)
        return DocumentData(
            doc_id=doc_id,
            source="pdf",
            blocks=blocks,
            sections=sections,
            sentences=sentences,
            windows=windows,
            page_to_block_ids=page_to_block_ids,
        )

    def _convert_parser_sections(
        self, parser_sections: List[Dict[str, Any]]
    ) -> Tuple[List[Block], List[Section], Dict[int, List[int]]]:
        blocks: List[Block] = []
        sections: List[Section] = []
        page_to_block_ids: Dict[int, List[int]] = {}
        table_counter = 1

        def add_block(block_type: str, text: str, page: Optional[int], metadata: Dict[str, str]) -> int:
            block_idx = len(blocks)
            page_int: Optional[int] = None
            if page is not None:
                try:
                    page_int = int(page) + 1
                except (TypeError, ValueError):
                    page_int = None
            block = Block(
                block_id=block_idx,
                block_type=block_type,
                text=text,
                page=page_int,
                metadata=metadata,
            )
            blocks.append(block)
            if page_int is not None:
                page_to_block_ids.setdefault(page_int, []).append(block_idx)
            return block_idx

        for sec in parser_sections:
            raw_label = (sec.get("label") or "").strip()
            display_title = raw_label.replace("_", " " ).strip()
            if not display_title:
                display_title = "Section"
            title = display_title.title() if display_title.isupper() else display_title
            normalized = normalize_section_name(title)
            block_ids: List[int] = []
            for unit in sec.get("units", []):
                utype = str(unit.get("type", "PARA")).upper()
                text = (unit.get("text") or "").strip()
                metadata: Dict[str, str] = {"unit_type": utype, "source": "parser"}
                for key in ("x0", "x1", "top", "bottom", "font_size"):
                    val = unit.get(key)
                    if val is not None:
                        metadata[key] = str(val)
                if unit.get("bold"):
                    metadata["bold"] = "1"
                if unit.get("ital"):
                    metadata["ital"] = "1"
                page = unit.get("page")
                block_type = BLOCK_PARA
                block_text = text
                if utype == "TABLE":
                    block_type = BLOCK_TABLE
                    placeholder = unit.get("placeholder")
                    if not placeholder:
                        placeholder = f"[[TABLE {table_counter} on page {page if page is not None else '?'}]]"
                    metadata["placeholder"] = str(placeholder)
                    try:
                        metadata["table_json"] = json.dumps(unit, default=str)
                    except Exception:
                        metadata["table_json"] = str(unit)
                    block_text = str(placeholder)
                    table_counter += 1
                elif utype in {"FIGURE_CAPTION", "TABLE_CAPTION", "CAPTION", "CAPTION_CAND"}:
                    block_type = BLOCK_CAPTION
                elif utype in {"LIST", "BULLET"}:
                    block_type = BLOCK_LIST
                    if not block_text:
                        items = unit.get("items")
                        if isinstance(items, list):
                            block_text = "\n".join(str(it) for it in items if str(it).strip())
                elif utype in {"HEADING", "HEADING_CAND"}:
                    block_type = BLOCK_HEADING
                if not block_text and block_type != BLOCK_TABLE:
                    continue
                block_ids.append(add_block(block_type, block_text, page, metadata))
            if not block_ids:
                continue
            section = Section(
                section_id=len(sections),
                title=title,
                normalized=normalized,
                flags=self._section_flags(normalized, is_first=len(sections) == 0),
                block_ids=block_ids,
            )
            sections.append(section)
        if not sections and blocks:
            sections.append(
                Section(
                    section_id=0,
                    title="Document",
                    normalized="SECTION",
                    flags=self._section_flags("SECTION", True),
                    block_ids=[b.block_id for b in blocks],
                )
            )
        for idx, sec in enumerate(sections):
            sec.section_id = idx
        return blocks, sections, page_to_block_ids

    def _build_sentence_splitter(self, name: str) -> SentenceSplitter:
        if name == "punkt":
            try:
                import nltk

                try:
                    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
                except LookupError:
                    nltk.download("punkt", quiet=True)
                    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

                def punkt_split(text: str) -> List[str]:
                    return [s.strip() for s in tokenizer.tokenize(text) if s.strip()]

                return punkt_split
            except Exception:
                pass
        regex = re.compile(r"(?<=[.!?])\\s+")

        def regex_split(text: str) -> List[str]:
            return [s.strip() for s in re.split(regex, text) if s.strip()]

        return regex_split

    def _extract_pages_from_pdf(
        self,
        file_bytes: bytes,
        max_pages: Optional[int] = None,
    ) -> List[Tuple[int, str]]:
        if fitz is not None:
            return self._extract_with_pymupdf(file_bytes, max_pages)
        if pdfminer_extract_text is not None:
            return self._extract_with_pdfminer(file_bytes, max_pages)
        raise RuntimeError("No PDF extractor available. Install PyMuPDF or pdfminer.six.")

    def _extract_with_pymupdf(self, file_bytes: bytes, max_pages: Optional[int]) -> List[Tuple[int, str]]:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        try:
            total_pages = len(doc)
            stop = min(total_pages, max_pages or total_pages)
            pages: List[Tuple[int, str]] = []
            for i in range(stop):
                page = doc[i]
                lines = _extract_lines_with_styles(page) if fitz is not None else []
                if not lines:
                    txt = page.get_text("text")
                    pages.append((i + 1, self._normalize_page_text(txt)))
                    continue
                sizes = [ln["size"] for ln in lines if ln.get("size", 0) > 0 and len(ln.get("text", "").split()) >= 2]
                base = statistics.median(sizes) if sizes else 10.0
                out_chunks: List[Tuple[str, str]] = []
                buf: List[str] = []

                def flush_para():
                    nonlocal buf
                    if buf:
                        out_chunks.append(("PARA", " ".join(buf).strip()))
                        buf = []

                for ln in lines:
                    txt = ln.get("text", "").strip()
                    if not txt:
                        continue
                    if _guess_heading_from_style(ln, base):
                        flush_para()
                        out_chunks.append(("HEADING", txt))
                    else:
                        buf.append(txt)
                flush_para()
                page_text = "\n\n".join(text for _kind, text in out_chunks if text)
                pages.append((i + 1, page_text))
            return pages
        finally:
            doc.close()
    
    
    def _extract_with_pdfminer(self, file_bytes: bytes, max_pages: Optional[int]) -> List[Tuple[int, str]]:
        stream = io.BytesIO(file_bytes)
        text = pdfminer_extract_text(stream, maxpages=max_pages)
        chunks = [chunk.strip() for chunk in re.split(r"\f+", text) if chunk.strip()]
        pages: List[Tuple[int, str]] = []
        for i, chunk in enumerate(chunks, start=1):
            pages.append((i, self._normalize_page_text(chunk)))
        return pages

    def _normalize_page_text(self, text: str) -> str:
        text = text.replace("\r", "")
        text = re.sub(r"(?<=\w)-\s*\n(?=\w)", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[\t\u00a0]+", " ", text)
        return text

    def _from_pages(
        self,
        pages: List[Tuple[int, str]],
        doc_id: str,
        source: str,
    ) -> DocumentData:
        blocks: List[Block] = []
        page_to_block_ids: Dict[int, List[int]] = {}
        table_counter = 1
        block_id = 0

        def emit_block(page_no: int, block_type: str, line_items: Sequence[str]) -> None:
            nonlocal block_id, table_counter
            metadata: Dict[str, str] = {}
            if block_type == BLOCK_TABLE:
                placeholder = f"[[TABLE {table_counter} on page {page_no}]]"
                metadata["placeholder"] = placeholder
                metadata["raw"] = "\n".join(line_items)
                text = placeholder
                table_counter += 1
            elif block_type == BLOCK_LIST:
                text = "\n".join(line_items)
            else:
                text = " ".join(line_items)
            block = Block(
                block_id=block_id,
                block_type=block_type,
                text=text,
                page=page_no,
                metadata=metadata,
            )
            blocks.append(block)
            page_to_block_ids.setdefault(page_no, []).append(block_id)
            block_id += 1

        for page_no, page_text in pages:
            for chunk in re.split(r"\n\s*\n", page_text):
                lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
                if not lines:
                    continue
                pending: List[List[str]] = [lines]
                while pending:
                    current_lines = pending.pop(0)
                    block_type = self._classify_block(current_lines)
                    if block_type == BLOCK_HEADING and len(current_lines) > 1:
                        tail_lines = [ln for ln in current_lines[1:] if ln.strip()]
                        if tail_lines:
                            tail_joined = " ".join(tail_lines)
                            first_tail = tail_lines[0]
                            looks_like_heading_tail = self._is_known_heading(first_tail) or self._looks_like_heading(tail_joined)
                            narrative_tail = (
                                not looks_like_heading_tail
                                and (
                                    len(tail_joined.split()) > MAX_HEADING_WORDS
                                    or len(tail_joined) >= self.min_sentence_chars
                                    or any(ch in tail_joined for ch in ".?!")
                                )
                            )
                            if narrative_tail:
                                emit_block(page_no, BLOCK_HEADING, [current_lines[0]])
                                pending.insert(0, tail_lines)
                                continue
                    emit_block(page_no, block_type, current_lines)
        sections = self._build_sections(blocks)
        sentences, windows = self._build_sentences_and_windows(doc_id, blocks, sections)
        return DocumentData(
            doc_id=doc_id,
            source=source,
            blocks=blocks,
            sections=sections,
            sentences=sentences,
            windows=windows,
            page_to_block_ids=page_to_block_ids,
        )


    def _is_known_heading(self, line: str) -> bool:
        candidate = line.strip()
        if not candidate:
            return False
        if _match_known_heading(candidate):
            return True
        if ':' in candidate:
            prefix = candidate.split(':', 1)[0]
            if _match_known_heading(prefix):
                return True
        parts = candidate.split(maxsplit=1)
        if len(parts) == 2:
            leading, rest = parts
            if leading.rstrip('.').isdigit() or re.fullmatch(r'[IVXLCMivxlcm]+\.?', leading):
                if _match_known_heading(rest):
                    return True
        return False

    def _classify_block(self, lines: Sequence[str]) -> str:
        first = lines[0]
        joined = " ".join(lines)
        if re.match(r"^(figure|table)\s+\d+", first, re.IGNORECASE):
            return BLOCK_CAPTION
        if self._is_known_heading(first) or self._is_known_heading(joined):
            return BLOCK_HEADING
        if any(self._line_is_list(ln) for ln in lines):
            return BLOCK_LIST
        if self._looks_like_table(lines):
            return BLOCK_TABLE
        if self._looks_like_heading(joined):
            return BLOCK_HEADING
        return BLOCK_PARA

    def _line_is_list(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        token = stripped.split()[0]
        if token in LIST_BULLETS:
            return True
        if re.match(r'^\d+(?:\.\d+)*[\.)-]?$', token):
            return True
        return False

    def _looks_like_table(self, lines: Sequence[str]) -> bool:
        table_lines = 0
        for ln in lines:
            if "\t" in ln:
                table_lines += 1
            elif "|" in ln and ln.count("|") >= 2:
                table_lines += 1
            elif re.search(r"\s{2,}\S+\s{2,}", ln):
                table_lines += 1
        return table_lines >= max(2, len(lines) // 2)

    def _looks_like_heading(self, text: str) -> bool:
        words = text.split()
        if not words:
            return False
        # Prefer short candidates
        if _is_short_heading(text) and _match_known_heading(text):
            return True
        if ':' in text:
            prefix = text.split(':',1)[0]
            if _is_short_heading(prefix) and _match_known_heading(prefix):
                return True
        if _is_short_heading(text):
            letters = [ch for ch in text if ch.isalpha()]
            if letters and sum(ch.isupper() for ch in letters) / len(letters) >= 0.6:
                return True
            if __import__('re').match(r'^(\d+(?:\.\d+)*|[IVXLCM]+\.)', words[0]):
                return True
            if text.endswith(':') and len(words) <= 12:
                return True
        return False

    def _build_sections(self, blocks: List[Block]) -> List[Section]:
        sections: List[Section] = []
        current_blocks: List[int] = []
        current_title = "Front Matter"
        current_norm = "FRONTMATTER"
        section_id = 0
        for block in blocks:
            if block.block_type == BLOCK_HEADING:
                if current_blocks:
                    sections.append(
                        Section(
                            section_id=section_id,
                            title=current_title,
                            normalized=current_norm,
                            flags=self._section_flags(current_norm, section_id == 0),
                            block_ids=current_blocks.copy(),
                        )
                    )
                    section_id += 1
                current_title = block.text.strip()
                current_norm = normalize_section_name(block.text)
                current_blocks = []
                continue
            current_blocks.append(block.block_id)
        if current_blocks:
            sections.append(
                Section(
                    section_id=len(sections),
                    title=current_title,
                    normalized=normalize_section_name(current_title) if sections else "FRONTMATTER",
                    flags=self._section_flags(
                        normalize_section_name(current_title) if sections else "FRONTMATTER",
                        len(sections) == 0,
                    ),
                    block_ids=current_blocks.copy(),
                )
            )
        if not sections:
            sections.append(
                Section(
                    section_id=0,
                    title="Document",
                    normalized="SECTION",
                    flags=self._section_flags("SECTION", True),
                    block_ids=[b.block_id for b in blocks],
                )
            )
        for idx, sec in enumerate(sections):
            sec.section_id = idx
        return sections

    def _section_flags(self, normalized: str, is_first: bool) -> Dict[str, bool]:
        up = normalized.upper()
        return {
            "references": up in {"REFERENCES", "ACKNOWLEDGMENTS", "APPENDIX"},
            "frontmatter": is_first,
        }

    def _build_sentences_and_windows(
        self,
        doc_id: str,
        blocks: List[Block],
        sections: List[Section],
    ) -> Tuple[List[Sentence], List[Window]]:
        sentences: List[Sentence] = []
        windows: List[Window] = []
        sentence_id = 0
        window_id = 0
        for section in sections:
            section_sentence_indices: List[int] = []
            section_sentence_texts: List[str] = []
            section_sentence_clean: List[str] = []
            for block_id in section.block_ids:
                block = blocks[block_id]
                if block.block_type == BLOCK_TABLE:
                    continue
                block_sentences = self._sentence_splitter(block.text)
                block_sentences = self._glue_short_sentences(block_sentences)
                include_in_windows = block.block_type in NARRATIVE_BLOCK_TYPES
                for sent_text in block_sentences:
                    clean_text = strip_math(sent_text)
                    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                    offset = len(section_sentence_indices) if include_in_windows else -1
                    sentence = Sentence(
                        sentence_id=sentence_id,
                        section_id=section.section_id,
                        section_name=section.title,
                        section_offset=offset,
                        block_id=block.block_id,
                        text=sent_text,
                        clean_text=clean_text,
                        block_type=block.block_type,
                        page=block.page,
                    )
                    sentences.append(sentence)
                    if include_in_windows:
                        section_sentence_indices.append(sentence_id)
                        section_sentence_texts.append(sent_text)
                        section_sentence_clean.append(clean_text)
                    sentence_id += 1
            if not section_sentence_indices:
                continue
            local_idx = 0
            while local_idx < len(section_sentence_indices):
                end_idx = min(len(section_sentence_indices), local_idx + self.window_size)
                doc_sentence_indices = section_sentence_indices[local_idx:end_idx]
                raw_text = ' '.join(section_sentence_texts[local_idx:end_idx]).strip()
                clean_window = ' '.join(section_sentence_clean[local_idx:end_idx]).strip()
                windows.append(
                    Window(
                        window_id=window_id,
                        doc_id=doc_id,
                        section_id=section.section_id,
                        section_name=section.title,
                        span_start=local_idx,
                        span_end=end_idx - 1,
                        sentence_indices=doc_sentence_indices,
                        text_raw=raw_text,
                        text_clean=clean_window,
                    )
                )
                window_id += 1
                if end_idx == len(section_sentence_indices):
                    break
                local_idx += self.window_stride
        return sentences, windows
    def _glue_short_sentences(self, sentences: List[str]) -> List[str]:
        if not sentences:
            return []
        glued: List[str] = []
        buffer = ""
        for sent in sentences:
            txt = sent.strip()
            if len(txt) < self.min_sentence_chars:
                buffer = (buffer + " " + txt).strip()
                continue
            if buffer:
                glued.append((buffer + " " + txt).strip())
                buffer = ""
            else:
                glued.append(txt)
        if buffer:
            if glued:
                glued[-1] = (glued[-1] + " " + buffer).strip()
            else:
                glued.append(buffer)
        return glued


__all__ = [
    "Block",
    "Section",
    "Sentence",
    "Window",
    "DocumentData",
    "DocumentProcessor",
    "normalize_section_name",
    "strip_math",
]



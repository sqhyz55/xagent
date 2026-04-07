from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..core.config import DEFAULT_PROTECTED_PATTERNS, DEFAULT_TIKTOKEN_ENCODING
from ..utils.token_utils import (
    get_token_counter,
    split_text_by_tokens,
)

logger = logging.getLogger(__name__)

DEFAULT_SEPARATORS: List[str] = ["\n\n", "\n", "。", "！", "？", ". ", ", ", " "]


def _join_paragraphs(paragraphs: List[Dict[str, Any]]) -> str:
    texts = [p.get("text", "") for p in paragraphs if p.get("text")]
    return "\n\n".join(texts)


def _join_paragraphs_with_metadata(
    paragraphs: List[Dict[str, Any]],
) -> tuple[str, list[tuple[int, int, Dict[str, Any]]]]:
    """Join paragraphs while tracking character positions to source metadata.

    Returns:
        Tuple of (joined_text, intervals) where intervals is a list of
        (start_pos, end_pos, source_paragraph) tuples.
    """
    texts: list[str] = []
    intervals: list[tuple[int, int, Dict[str, Any]]] = []
    current_pos = 0

    for p in paragraphs:
        if not p.get("text"):
            continue
        text = p.get("text", "")
        if texts:
            # Add separator length (2 for "\n\n")
            current_pos += 2
        texts.append(text)
        start_pos = current_pos
        current_pos += len(text)
        end_pos = current_pos
        intervals.append((start_pos, end_pos, p))

    joined_text = "\n\n".join(texts)
    return joined_text, intervals


def _create_chunk_record(
    text: str, source_paragraph: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a chunk record with position information (no ID or timestamp).

    Args:
        text: Chunk text content
        source_paragraph: Source paragraph metadata for position info

    Returns:
        Chunk record with text, position fields, and full metadata dictionary
    """
    # Extract metadata from source paragraph if available
    metadata = (source_paragraph or {}).get("metadata", {})

    return {
        "text": text,
        "page_number": metadata.get("page_number"),
        "section": metadata.get("section"),
        "anchor": metadata.get("anchor"),
        "json_path": metadata.get("json_path"),
        "metadata": metadata,  # Preserve full metadata dictionary
    }


def _find_contributing_paragraphs_for_range(
    intervals: List[tuple[int, int, Dict[str, Any]]],
    range_start: int,
    range_end: int,
) -> List[Dict[str, Any]]:
    """Find all paragraphs that overlap with the given character range.

    Args:
        intervals: List of (start_pos, end_pos, source_paragraph) tuples
        range_start: Start of the character range
        range_end: End of the character range

    Returns:
        List of source paragraphs that overlap with the range
    """
    contributing: List[Dict[str, Any]] = []
    for interval_start, interval_end, para in intervals:
        # Check for overlap: two intervals [a, b) and [c, d) overlap if b > c and d > a
        if interval_end > range_start and interval_start < range_end:
            contributing.append(para)
    return contributing


def _collect_pages_from_paragraphs(
    paragraphs: Iterable[Dict[str, Any]],
) -> List[int]:
    """Collect sorted unique positive page numbers from paragraph metadata.

    Handles various edge cases:
    - None or non-dict paragraph entries
    - Missing or non-dict metadata
    - Non-integer or negative page numbers

    Args:
        paragraphs: Iterable of paragraph dictionaries that may contain metadata

    Returns:
        Sorted list of unique positive page numbers
    """
    pages: set[int] = set()
    for para in paragraphs:
        # Validate paragraph is a dict
        if not para or not isinstance(para, dict):
            continue

        # Validate metadata exists and is a dict
        meta = para.get("metadata")
        if not meta or not isinstance(meta, dict):
            continue

        # Validate and extract page_number
        page_num = meta.get("page_number")
        if isinstance(page_num, int) and page_num > 0:
            pages.add(page_num)

    return sorted(pages)


def _validate_positions_record(record: Dict[str, Any]) -> bool:
    """Validate the integrity of positions metadata in a chunk record.

    Checks:
    - Positions list is sorted in ascending order
    - No duplicate page numbers in positions
    - page_number (if set) is present in positions list

    Args:
        record: Chunk record to validate

    Returns:
        True if positions metadata is valid, False otherwise
    """
    positions = record.get("metadata", {}).get("positions")
    if not positions:
        return True  # No positions to validate

    # Check if sorted
    if positions != sorted(positions):
        logger.warning(f"Positions not sorted: {positions}")
        return False

    # Check for duplicates
    if len(positions) != len(set(positions)):
        logger.warning(f"Duplicate positions found: {positions}")
        return False

    # Check page_number consistency
    page_num = record.get("page_number")
    if page_num and page_num not in positions:
        logger.warning(f"page_number {page_num} not in positions {positions}")
        return False

    return True


def _apply_positions_to_record(
    record: Dict[str, Any],
    positions: Any,
) -> None:
    """Normalize positions list and apply to final chunk record metadata.

    - Ensures positions are unique positive integers.
    - Writes them to metadata['positions'] if non-empty.
    - Derives page_number from the smallest position when missing.
    """
    if not isinstance(positions, list):
        return

    pages: set[int] = set()
    for p in positions:
        if isinstance(p, int) and p > 0:
            pages.add(p)
    if not pages:
        return

    sorted_pages = sorted(pages)
    meta = record.get("metadata") or {}
    meta["positions"] = sorted_pages
    record["metadata"] = meta

    if not record.get("page_number"):
        record["page_number"] = sorted_pages[0]


def _split_by_separators_core(text: str, separators: Optional[List[str]]) -> List[str]:
    """Core splitter that preserves delimiters by attaching them to the previous unit.

    This function contains the shared logic for regex construction, splitting with
    capturing groups, handling None parts, and attaching delimiters to the
    previous chunk. It returns a list of pure text chunks and is reused by
    higher-level wrappers with/without metadata.

    Args:
        text: Text to split
        separators: List of separator strings to use for splitting (defaults applied inside)

    Returns:
        List of text chunks with delimiters attached to the previous chunk
    """
    if not separators:
        separators = DEFAULT_SEPARATORS

    escaped_separators = [re.escape(sep) for sep in separators]
    pattern = "|".join(f"({escaped_sep})" for escaped_sep in escaped_separators)

    parts = re.split(pattern, text)

    chunks: List[str] = []
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            text_part = parts[i] if parts[i] is not None else ""
            delimiter_part = parts[i + 1] if parts[i + 1] is not None else ""
            chunk = text_part + delimiter_part
            if chunk.strip():
                chunks.append(chunk)
        else:
            text_part = parts[i] if parts[i] is not None else ""
            if text_part.strip():
                chunks.append(text_part)

    return chunks


def _split_by_separators(text: str, separators: Optional[List[str]]) -> List[str]:
    """Wrapper: returns pure text chunks using the core splitter."""
    return _split_by_separators_core(text, separators)


def _split_by_separators_with_metadata(
    text: str,
    separators: Optional[List[str]],
    source_paragraph: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Wrapper: returns structured chunks with metadata using the core splitter."""
    units = _split_by_separators_core(text, separators)
    return [
        {"text": unit, "source_paragraph": source_paragraph}
        for unit in units
        if unit.strip()
    ]


def _find_protected_ranges(
    text: str,
    patterns: Optional[List[str]] = None,
) -> List[tuple[int, int]]:
    """Find protected regions (start, end) in text. Returns merged non-overlapping ranges."""
    if patterns is None:
        patterns = list(DEFAULT_PROTECTED_PATTERNS)
    if not patterns:
        return []
    ranges: List[tuple[int, int]] = []
    for pat in patterns:
        try:
            for m in re.finditer(pat, text, re.MULTILINE | re.DOTALL):
                ranges.append((m.start(), m.end()))
        except re.error:
            continue
    ranges.sort(key=lambda r: r[0])
    merged: List[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _split_by_separators_with_metadata_and_protection(
    text: str,
    separators: Optional[List[str]],
    source_paragraph: Optional[Dict[str, Any]],
    protected_patterns: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """Split text by separators but never split inside protected regions (P1)."""
    ranges = _find_protected_ranges(text, protected_patterns)
    if not ranges:
        return _split_by_separators_with_metadata(text, separators, source_paragraph)
    result: List[Dict[str, Any]] = []
    pos = 0
    for start, end in ranges:
        if pos < start:
            normal = text[pos:start]
            units = _split_by_separators_with_metadata(
                normal, separators, source_paragraph
            )
            result.extend(units)
        protected_text = text[start:end]
        if protected_text.strip():
            result.append(
                {"text": protected_text, "source_paragraph": source_paragraph}
            )
        pos = end
    if pos < len(text):
        units = _split_by_separators_with_metadata(
            text[pos:], separators, source_paragraph
        )
        result.extend(units)
    return result


def _window_with_overlap(
    tokens: List[str], chunk_size: int, chunk_overlap: int
) -> List[str]:
    if chunk_size <= 0:
        return []
    if chunk_overlap < 0:
        chunk_overlap = 0
    chunks: List[str] = []
    start = 0
    n = len(tokens)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = "".join(tokens[start:end])
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(start + chunk_size - chunk_overlap, start + 1)
    return chunks


def _window_with_overlap_and_metadata(
    chunk_records: List[Dict[str, Any]], chunk_size: int, chunk_overlap: int
) -> List[Dict[str, Any]]:
    """Apply sliding window with overlap to chunk records, preserving metadata.

    Args:
        chunk_records: List of chunk records with text and source_paragraph
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        List of windowed chunk records with preserved metadata
    """
    if chunk_size <= 0:
        return []
    if chunk_overlap < 0:
        chunk_overlap = 0

    # OPTIMIZATION: Use interval mapping instead of per-character metadata
    # Store (start_pos, end_pos, source_paragraph) tuples to avoid O(n) memory
    tokens: list[str] = []
    intervals: List[tuple[int, int, Optional[Dict[str, Any]]]] = []

    for chunk_record in chunk_records:
        text = chunk_record["text"]
        source_paragraph = chunk_record.get("source_paragraph")

        start_pos = len(tokens)
        tokens.extend(list(text))
        end_pos = len(tokens)

        # Store interval instead of per-character mapping
        intervals.append((start_pos, end_pos, source_paragraph))

    # Apply sliding window
    windows: List[Dict[str, Any]] = []
    start = 0
    n = len(tokens)

    while start < n:
        end = min(n, start + chunk_size)
        window_text = "".join(tokens[start:end])

        if window_text:
            # Find contributing paragraphs for this window using interval overlap check
            first_paragraph: Optional[Dict[str, Any]] = None
            contributing_paragraphs: List[Dict[str, Any]] = []

            # OPTIMIZATION: Use interval overlap check instead of iterating each character
            # Two intervals [a, b) and [c, d) overlap if b > c and d > a
            for interval_start, interval_end, para in intervals:
                if (
                    interval_end > start and interval_start < end
                ):  # Correct overlap condition
                    if para is not None:
                        contributing_paragraphs.append(para)
                        if first_paragraph is None:
                            first_paragraph = para

            window_record: Dict[str, Any] = {
                "text": window_text,
                "source_paragraph": first_paragraph,
            }
            positions = _collect_pages_from_paragraphs(contributing_paragraphs)
            if positions:
                window_record["positions"] = positions

            windows.append(window_record)

        if end == n:
            break
        start = max(start + chunk_size - chunk_overlap, start + 1)

    return windows


def _merge_units_by_token_limit(
    unit_records: List[Dict[str, Any]],
    chunk_token_size: int,
    chunk_token_overlap: int,
    num_tokens_fn: Callable[[str], int],
    tiktoken_encoding: str = DEFAULT_TIKTOKEN_ENCODING,
) -> List[Dict[str, Any]]:
    """Merge semantic units by token limit with greedy merge and overlap.

    Units are merged until adding the next would exceed chunk_token_size;
    then overlap is applied by carrying trailing units (that fit in
    chunk_token_overlap) to the next chunk. Single units that exceed
    chunk_token_size are split by token (fallback) and then merged.

    Args:
        unit_records: List of {"text": str, "source_paragraph": optional dict}.
        chunk_token_size: Max tokens per chunk.
        chunk_token_overlap: Overlap in tokens between consecutive chunks.
        num_tokens_fn: Function text -> token count.
        tiktoken_encoding: Encoding name for fallback split of long units.

    Returns:
        List of {"text": str, "source_paragraph": optional dict}.
    """
    if chunk_token_size <= 0 or not unit_records:
        return []

    if chunk_token_overlap < 0:
        chunk_token_overlap = 0
    if chunk_token_overlap >= chunk_token_size:
        chunk_token_overlap = max(0, chunk_token_size - 1)

    # Expand any unit that exceeds limit into token-sized sub-units
    expanded: List[Dict[str, Any]] = []
    for rec in unit_records:
        text = rec.get("text", "")
        para = rec.get("source_paragraph")
        n = num_tokens_fn(text)
        if n <= chunk_token_size:
            expanded.append({"text": text, "source_paragraph": para})
        else:
            for segment in split_text_by_tokens(
                text,
                max_tokens=chunk_token_size,
                overlap_tokens=chunk_token_overlap,
                encoding_name=tiktoken_encoding,
            ):
                if segment.strip():
                    expanded.append({"text": segment, "source_paragraph": para})

    if not expanded:
        return []

    # Greedy merge with overlap
    windows: List[Dict[str, Any]] = []
    current_units: List[Dict[str, Any]] = []
    current_tokens = 0
    i = 0
    while i < len(expanded):
        rec = expanded[i]
        t = num_tokens_fn(rec["text"])
        if current_tokens + t <= chunk_token_size:
            current_units.append(rec)
            current_tokens += t
            i += 1
            continue
        # Emit current chunk
        if current_units:
            chunk_text = "".join(u["text"] for u in current_units)
            first_para = current_units[0].get("source_paragraph")

            window_record: Dict[str, Any] = {
                "text": chunk_text,
                "source_paragraph": first_para,
            }
            # OPTIMIZATION: Direct iteration instead of generator to avoid overhead
            contributing_paragraphs: List[Dict[str, Any]] = []
            for u in current_units:
                para = u.get("source_paragraph")
                if para:
                    contributing_paragraphs.append(para)
            positions = _collect_pages_from_paragraphs(contributing_paragraphs)
            if positions:
                window_record["positions"] = positions

            windows.append(window_record)
        # Overlap: take trailing units that fit in chunk_token_overlap
        overlap_units: List[Dict[str, Any]] = []
        overlap_tokens = 0
        for u in reversed(current_units):
            ut = num_tokens_fn(u["text"])
            if overlap_tokens + ut <= chunk_token_overlap:
                overlap_units.insert(0, u)
                overlap_tokens += ut
            else:
                break
        current_units = overlap_units
        current_tokens = overlap_tokens
        # Do not advance i so we re-process rec in next iteration
    if current_units:
        chunk_text = "".join(u["text"] for u in current_units)
        first_para = current_units[0].get("source_paragraph")

        window_record = {
            "text": chunk_text,
            "source_paragraph": first_para,
        }
        # OPTIMIZATION: Direct iteration instead of generator to avoid overhead
        final_contributing_paragraphs: List[Dict[str, Any]] = []
        for u in current_units:
            para = u.get("source_paragraph")
            if para:
                final_contributing_paragraphs.append(para)
        positions = _collect_pages_from_paragraphs(final_contributing_paragraphs)
        if positions:
            window_record["positions"] = positions

        windows.append(window_record)

    return windows


def apply_recursive_strategy(
    paragraphs: List[Dict[str, Any]], params: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Apply recursive chunking strategy with metadata preservation.

    This strategy splits text by separators first, then applies sliding window
    with overlap. Metadata from source paragraphs is preserved and propagated
    to the final chunks.

    Thread-safety: This function is thread-safe as it does not modify shared
    state. However, the input paragraphs list should not be modified during
    execution to ensure consistent results.

    Deterministic: Given the same input, this function produces identical
    output, ensuring reproducible chunking for document versioning.
    """
    if not paragraphs:
        return []

    separators: Optional[List[str]] = params.get("separators")
    chunk_size_param = params.get("chunk_size")
    chunk_overlap: int = int(params.get("chunk_overlap", 200))

    # P1: optional protected content (code blocks, formulas, tables, etc.)
    enable_protected = params.get("enable_protected_content", True)
    protected_patterns: Optional[List[str]] = params.get("protected_patterns")

    all_chunk_records = []

    for paragraph in paragraphs:
        text = paragraph.get("text", "")
        if not text.strip():
            continue

        if enable_protected:
            paragraph_chunks = _split_by_separators_with_metadata_and_protection(
                text, separators, paragraph, protected_patterns
            )
        else:
            paragraph_chunks = _split_by_separators_with_metadata(
                text, separators, paragraph
            )
        all_chunk_records.extend(paragraph_chunks)

    if not all_chunk_records:
        return []

    # Apply size limit: token-based (P0) or character-based (legacy)
    use_token_count = bool(params.get("use_token_count"))
    if chunk_size_param is None:
        # User didn't set chunk_size, trust semantic splitting completely
        windows = all_chunk_records
    elif use_token_count:
        # P0: semantic unit merge by token limit (tiktoken)
        chunk_token_size = int(chunk_size_param)
        tiktoken_encoding = str(
            params.get("tiktoken_encoding", DEFAULT_TIKTOKEN_ENCODING)
        )
        num_tokens_fn = get_token_counter(tiktoken_encoding)
        windows = _merge_units_by_token_limit(
            all_chunk_records,
            chunk_token_size=chunk_token_size,
            chunk_token_overlap=chunk_overlap,
            num_tokens_fn=num_tokens_fn,
            tiktoken_encoding=tiktoken_encoding,
        )
    else:
        chunk_size = int(chunk_size_param)
        windows = _window_with_overlap_and_metadata(
            all_chunk_records, chunk_size, chunk_overlap
        )

    units_total_chars = sum(len(r.get("text", "")) for r in all_chunk_records)
    out_lens = [len(w.get("text", "")) for w in windows]
    logger.info(
        "[RAG][chunk] apply_recursive_strategy: semantic_units=%s units_total_chars=%s "
        "chunk_size=%r use_token_count=%s overlap=%s -> windows=%s "
        "window_char_min=%s max=%s sum=%s",
        len(all_chunk_records),
        units_total_chars,
        chunk_size_param,
        use_token_count,
        chunk_overlap,
        len(windows),
        min(out_lens, default=0),
        max(out_lens, default=0),
        sum(out_lens),
    )

    # Create final chunk records with preserved metadata and multi-page positions
    final_chunks: List[Dict[str, Any]] = []
    for w in windows:
        text = w.get("text", "").strip()
        if not text:
            continue

        source_paragraph = w.get("source_paragraph")
        record = _create_chunk_record(text, source_paragraph)

        positions = w.get("positions")
        _apply_positions_to_record(record, positions)

        # Validate positions metadata integrity (optional, for debugging)
        if not _validate_positions_record(record):
            logger.debug(f"Positions validation failed for chunk: {record}")

        final_chunks.append(record)

    return final_chunks


def _split_by_headers_with_positions(
    text: str,
    headers_to_split_on: Optional[List[tuple[str, str]]],
) -> List[tuple[str, str, int, int]]:
    """Split text into sections by header patterns, tracking positions in original text.

    Returns:
        List of (section_text, section_header, start_pos, end_pos) tuples.
    """
    lines = text.splitlines()
    if not headers_to_split_on:
        # Default: atx-style # to ######
        header_pattern = re.compile(r"^\s{0,3}#{1,6}\s+")
        sections_with_headers: List[tuple[str, str, int, int]] = []
        current: List[str] = []
        section_header = ""
        current_start_pos = 0
        current_line_number = 0

        for line in lines:
            line_with_newline = line + "\n"
            if header_pattern.match(line):
                if current:
                    section_text = "\n".join(current)
                    sections_with_headers.append(
                        (
                            section_text,
                            section_header,
                            current_start_pos,
                            current_line_number,
                        )
                    )
                    current = []
                section_header = line.strip()
                # Update start position for next section
                current_start_pos = current_line_number + len(line_with_newline)
            current.append(line)
            current_line_number += len(line_with_newline)

        if current:
            section_text = "\n".join(current)
            sections_with_headers.append(
                (section_text, section_header, current_start_pos, current_line_number)
            )
        return sections_with_headers

    # User-provided headers: e.g. [("# ", "H1"), ("## ", "H2")]. Try longest prefix first.
    sorted_headers = sorted(headers_to_split_on, key=lambda x: len(x[0]), reverse=True)
    sections_with_headers = []
    section_lines: List[str] = []
    section_header = ""
    current_start_pos = 0
    current_line_number = 0

    for line in lines:
        line_with_newline = line + "\n"
        matched = False
        for prefix, _ in sorted_headers:
            if line.strip().startswith(prefix) or line.startswith(prefix):
                if section_lines:
                    section_text = "\n".join(section_lines)
                    sections_with_headers.append(
                        (
                            section_text,
                            section_header,
                            current_start_pos,
                            current_line_number,
                        )
                    )
                    section_lines = []
                section_header = line.strip()
                section_lines.append(line)
                # Update start position for next section
                current_start_pos = current_line_number + len(line_with_newline)
                matched = True
                break
        if not matched:
            section_lines.append(line)
        current_line_number += len(line_with_newline)

    if section_lines:
        section_text = "\n".join(section_lines)
        sections_with_headers.append(
            (section_text, section_header, current_start_pos, current_line_number)
        )
    return sections_with_headers


def _split_by_headers(
    text: str,
    headers_to_split_on: Optional[List[tuple[str, str]]],
) -> List[tuple[str, str]]:
    """Split text into (section_text, section_header) by header patterns. P1."""
    lines = text.splitlines()
    if not headers_to_split_on:
        # Default: atx-style # to ######
        header_pattern = re.compile(r"^\s{0,3}#{1,6}\s+")
        sections_with_headers = []
        current: List[str] = []
        section_header = ""
        for line in lines:
            if header_pattern.match(line):
                if current:
                    sections_with_headers.append(("\n".join(current), section_header))
                    current = []
                section_header = line.strip()
            current.append(line)
        if current:
            sections_with_headers.append(("\n".join(current), section_header))
        return sections_with_headers

    # User-provided headers: e.g. [("# ", "H1"), ("## ", "H2")]. Try longest prefix first.
    sorted_headers = sorted(headers_to_split_on, key=lambda x: len(x[0]), reverse=True)
    sections_with_headers = []
    section_lines: List[str] = []
    section_header = ""
    for line in lines:
        matched = False
        for prefix, _ in sorted_headers:
            if line.strip().startswith(prefix) or line.startswith(prefix):
                if section_lines:
                    sections_with_headers.append(
                        ("\n".join(section_lines), section_header)
                    )
                    section_lines = []
                section_header = line.strip()
                section_lines.append(line)
                matched = True
                break
        if not matched:
            section_lines.append(line)
    if section_lines:
        sections_with_headers.append(("\n".join(section_lines), section_header))
    return sections_with_headers


def apply_markdown_strategy(
    paragraphs: List[Dict[str, Any]], params: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Apply markdown-aware chunking strategy with header detection.

    This strategy detects markdown headers (# ## ###) and splits documents
    into sections, preserving section metadata in chunks. Falls back to
    recursive strategy if no headers are found.

    Thread-safety: This function is thread-safe as it does not modify shared
    state. Input paragraphs should not be modified during execution.

    Deterministic: Produces identical output for the same input, ensuring
    reproducible chunking for markdown documents.
    """
    text = _join_paragraphs(paragraphs)
    if not text:
        return []
    chunk_size_param = params.get("chunk_size")
    chunk_overlap: int = int(params.get("chunk_overlap", 200))
    separators: Optional[List[str]] = params.get("separators")
    headers_to_split_on: Optional[List[tuple[str, str]]] = params.get(
        "headers_to_split_on"
    )

    # P1: Split by Markdown headers (configurable or default # to ######)
    sections_with_headers = _split_by_headers(text, headers_to_split_on)

    # If no headers found (single section with no header), fallback to recursive
    if len(sections_with_headers) == 1 and not sections_with_headers[0][1]:
        return apply_recursive_strategy(paragraphs, params)

    # Build text with metadata tracking for positions support
    try:
        text_with_intervals, intervals = _join_paragraphs_with_metadata(paragraphs)
    except Exception as e:
        logger.warning(
            f"Failed to join paragraphs with metadata, falling back to recursive: {e}"
        )
        return apply_recursive_strategy(paragraphs, params)

    # Split by headers while tracking positions in original text
    try:
        sections_with_positions = _split_by_headers_with_positions(
            text_with_intervals, headers_to_split_on
        )
    except Exception as e:
        logger.warning(
            f"Failed to split by headers with positions, falling back to recursive: {e}"
        )
        return apply_recursive_strategy(paragraphs, params)

    # For each section, further split and create chunks with section metadata (P1)
    chunks = []
    section_meta = {"section": ""}

    for sec, section_header, section_start, section_end in sections_with_positions:
        section_meta["section"] = section_header or ""
        source_para = {"metadata": dict(section_meta)}

        # Find contributing paragraphs for this section using actual positions
        # Validate section range to prevent empty range queries
        if section_end > section_start:
            contributing_paragraphs = _find_contributing_paragraphs_for_range(
                intervals, section_start, section_end
            )
        else:
            contributing_paragraphs = []

        if separators and separators != DEFAULT_SEPARATORS:
            section_chunks = _split_by_separators(sec, separators)
            if section_chunks:
                if chunk_size_param is None:
                    windows = section_chunks
                else:
                    chunk_size = int(chunk_size_param)
                    total_chars = sum(len(chunk) for chunk in section_chunks)
                    if total_chars <= chunk_size:
                        windows = section_chunks
                    else:
                        # Create metadata records for each chunk
                        chunk_records = [
                            {"text": chunk, "source_paragraph": None}
                            for chunk in section_chunks
                        ]
                        windowed_chunks = _window_with_overlap_and_metadata(
                            chunk_records,
                            chunk_size,
                            chunk_overlap,
                        )
                        windows = [w["text"] for w in windowed_chunks]
            else:
                windows = [sec]
        else:
            if chunk_size_param is None:
                windows = [sec]
            else:
                chunk_size = int(chunk_size_param)
                tokens = list(sec)
                windows = _window_with_overlap(tokens, chunk_size, chunk_overlap)

        if not windows:
            continue

        for w in windows:
            if w.strip():
                record = _create_chunk_record(w.strip(), source_paragraph=source_para)
                # Apply positions from contributing paragraphs
                positions = _collect_pages_from_paragraphs(contributing_paragraphs)
                _apply_positions_to_record(record, positions)
                chunks.append(record)

    return chunks


# P2: table/image context attachment
_TABLE_LINE_PATTERN = re.compile(r"\n\s*\|[^|\n]+\|", re.MULTILINE)
_IMAGE_PATTERN = re.compile(r"!\[.*?\]\(.*?\)", re.DOTALL)


def _is_table_chunk(text: str) -> bool:
    """True if text looks like a markdown table (has |...| row)."""
    # Primary: pattern match for markdown table rows
    if _TABLE_LINE_PATTERN.search(text):
        return True
    # Fallback: require at least 2 lines with pipe chars and a separator-like line
    lines = text.splitlines()
    pipe_lines = [ln for ln in lines if "|" in ln]
    if len(pipe_lines) >= 2 and text.count("|") >= 4:
        # Check for separator row pattern (e.g., |---|---|)
        for ln in lines:
            if re.search(r"\|[-:]+[-|:]*\|", ln):
                return True
    return False


def _is_image_chunk(text: str) -> bool:
    """True if text contains markdown image syntax."""
    return bool(_IMAGE_PATTERN.search(text))


def attach_media_context(
    chunks: List[Dict[str, Any]],
    table_context_size: int = 0,
    image_context_size: int = 0,
) -> None:
    """P2: Prepend/append surrounding context to table and image chunks (in-place).

    For each chunk that looks like a table or image, prepend the last N chars
    of the previous chunk and append the first N chars of the next chunk,
    so retrieval gets better context. N = table_context_size or image_context_size.

    Args:
        chunks: List of chunk dicts with "text" key; modified in place.
        table_context_size: Max chars from prev/next chunk to attach to table chunks; 0 = off.
        image_context_size: Max chars from prev/next chunk to attach to image chunks; 0 = off.
    """
    if not chunks or (table_context_size <= 0 and image_context_size <= 0):
        return
    n = len(chunks)
    for i in range(n):
        text = chunks[i].get("text", "")
        if not text:
            continue
        ctx_size = 0
        if table_context_size > 0 and _is_table_chunk(text):
            ctx_size = table_context_size
        elif image_context_size > 0 and _is_image_chunk(text):
            ctx_size = image_context_size
        if ctx_size <= 0:
            continue
        prefix = ""
        if i > 0:
            prev_text = chunks[i - 1].get("text", "")
            if prev_text:
                prefix = (
                    prev_text[-ctx_size:] if len(prev_text) > ctx_size else prev_text
                )
        suffix = ""
        if i + 1 < n:
            next_text = chunks[i + 1].get("text", "")
            if next_text:
                suffix = (
                    next_text[:ctx_size] if len(next_text) > ctx_size else next_text
                )
        if prefix:
            chunks[i]["text"] = prefix.strip() + "\n\n" + text
        if suffix:
            chunks[i]["text"] = chunks[i]["text"] + "\n\n" + suffix.strip()


def apply_fixed_size_strategy(
    paragraphs: List[Dict[str, Any]], params: Dict[str, Any]
) -> List[Dict[str, Any]]:
    text = _join_paragraphs(paragraphs)
    if not text:
        return []
    chunk_size_param = params.get("chunk_size")
    chunk_overlap: int = int(params.get("chunk_overlap", 0))

    if chunk_size_param is None:
        # User didn't set chunk_size, return whole text as one chunk
        return [_create_chunk_record(text.strip())]
    else:
        chunk_size = int(chunk_size_param)
        tokens = list(text)
        windows = _window_with_overlap(tokens, chunk_size, chunk_overlap)
        return [_create_chunk_record(w.strip()) for w in windows if w.strip()]

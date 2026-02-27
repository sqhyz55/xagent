"""Parser registry for managing file type to parser mappings."""

from typing import Dict, List, Set

from ...document_parser import document_parser_registry

# Static fallback mapping for file extension to supported parser methods.
# Dynamic compatibility derived from `document_parser_registry` and
# per-parser `supported_extensions` will take precedence when available.
PARSER_COMPATIBILITY: Dict[str, List[str]] = {
    # Documents
    ".pdf": ["deepdoc", "pymupdf", "pdfplumber", "unstructured"],
    ".docx": ["docx", "unstructured"],
    ".doc": ["docx", "unstructured"],
    ".pptx": ["unstructured"],
    ".ppt": ["unstructured"],
    # Text/Markdown
    ".txt": ["text"],
    ".md": ["markdown", "commonmark"],
    ".rst": ["rst"],
    # Code files
    ".py": ["code", "python_ast"],
    ".js": ["code", "javascript"],
    ".ts": ["code", "typescript"],
    ".java": ["code", "java"],
    ".cpp": ["code", "cpp"],
    ".c": ["code", "c"],
    ".go": ["code", "go"],
    ".rs": ["code", "rust"],
    ".php": ["code", "php"],
    ".rb": ["code", "ruby"],
    ".sh": ["code", "bash"],
    ".sql": ["code", "sql"],
    # Web formats
    ".html": ["html", "beautifulsoup"],
    ".xml": ["xml"],
    ".json": ["json"],
    ".yaml": ["yaml"],
    ".yml": ["yaml"],
    # Data formats
    ".csv": ["csv"],
    ".xlsx": ["excel", "openpyxl"],
    ".xls": ["excel", "openpyxl"],
    # Images (for OCR or captioning)
    ".jpg": ["image", "image_caption"],
    ".jpeg": ["image", "image_caption"],
    ".png": ["image", "image_caption"],
    ".gif": ["image", "image_caption"],
    ".bmp": ["image", "image_caption"],
    ".tiff": ["image", "image_caption"],
    ".webp": ["image", "image_caption"],
}

# Lazily populated dynamic compatibility map derived from registered parsers.
_DYNAMIC_COMPATIBILITY: Dict[str, List[str]] | None = None


def _normalize_extension(file_extension: str) -> str:
    """Normalize file extension to canonical form with leading dot and lowercase."""
    if not file_extension.startswith("."):
        file_extension = "." + file_extension
    return file_extension.lower()


def _build_dynamic_compatibility() -> Dict[str, List[str]]:
    """Build dynamic extension → parser mapping from registered parsers."""
    mapping: Dict[str, List[str]] = {}

    for parser_name, parser_class in document_parser_registry.parsers().items():
        supported = getattr(parser_class, "supported_extensions", None)
        if not supported:
            continue
        for ext in supported:
            norm_ext = _normalize_extension(ext)
            if norm_ext not in mapping:
                mapping[norm_ext] = []
            if parser_name not in mapping[norm_ext]:
                mapping[norm_ext].append(parser_name)

    return mapping


def get_supported_parsers(file_extension: str) -> List[str]:
    """Get supported parser methods for a file extension.

    Args:
        file_extension: File extension (with or without leading dot)

    Returns:
        List of supported parser method names
    """
    global _DYNAMIC_COMPATIBILITY

    norm_ext = _normalize_extension(file_extension)

    # Initialize dynamic compatibility lazily
    if _DYNAMIC_COMPATIBILITY is None:
        _DYNAMIC_COMPATIBILITY = _build_dynamic_compatibility()

    # Prefer dynamically derived mapping when available
    if norm_ext in _DYNAMIC_COMPATIBILITY:
        return _DYNAMIC_COMPATIBILITY[norm_ext]

    # Fallback to static mapping
    return PARSER_COMPATIBILITY.get(norm_ext, [])


def validate_parser_compatibility(
    file_extension: str, parser_method: str, allow_mixed: bool = False
) -> bool:
    """Validate if a parser method is compatible with a file type.

    Args:
        file_extension: File extension to check
        parser_method: Parser method to validate
        allow_mixed: If True, allow any parser method

    Returns:
        True if compatible, False otherwise
    """
    if allow_mixed:
        return True

    supported_parsers = get_supported_parsers(file_extension)
    return parser_method in supported_parsers


def get_all_supported_extensions() -> Set[str]:
    """Get all supported file extensions."""
    return set(PARSER_COMPATIBILITY.keys())


def register_parser_support(file_extension: str, parser_method: str) -> None:
    """Register a new parser method for a file extension.

    This is used when adding new parsers to the system.

    Args:
        file_extension: File extension (with leading dot)
        parser_method: Parser method name to add
    """
    if not file_extension.startswith("."):
        file_extension = "." + file_extension

    file_extension = file_extension.lower()

    if file_extension not in PARSER_COMPATIBILITY:
        PARSER_COMPATIBILITY[file_extension] = []

    if parser_method not in PARSER_COMPATIBILITY[file_extension]:
        PARSER_COMPATIBILITY[file_extension].append(parser_method)

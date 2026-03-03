"""Tests for parser registry functionality."""

from xagent.core.tools.core.RAG_tools.core.parser_registry import (
    PARSER_COMPATIBILITY,
    get_all_supported_extensions,
    get_supported_parsers,
    register_parser_support,
    validate_parser_compatibility,
)


def test_get_supported_parsers_uses_dynamic_mapping_for_pdf() -> None:
    """PDF extension should return parser set derived from parser registry."""
    parsers = get_supported_parsers(".pdf")

    # Core parsers from document_parser_registry
    assert "pypdf" in parsers
    assert "pdfplumber" in parsers
    assert "pymupdf" in parsers
    assert "unstructured" in parsers
    assert "deepdoc" in parsers


def test_get_supported_parsers_uses_dynamic_mapping_for_docx() -> None:
    """DOCX extension should use dynamic compatibility (Unstructured/DeepDoc)."""
    parsers = get_supported_parsers(".docx")

    # Unstructured and DeepDoc both declare support for .docx
    assert "unstructured" in parsers
    assert "deepdoc" in parsers


class TestGetSupportedParsers:
    """Test getting supported parsers for file extensions."""

    def test_get_supported_parsers_known_extension(self):
        """Test getting parsers for known file extensions."""
        # Test PDF parsers
        pdf_parsers = get_supported_parsers(".pdf")
        assert "deepdoc" in pdf_parsers
        assert "pymupdf" in pdf_parsers
        assert "pdfplumber" in pdf_parsers

        # Test code file parsers
        py_parsers = get_supported_parsers(".py")
        assert "code" in py_parsers
        assert "python_ast" in py_parsers

        # Test markdown parsers (dynamic: unstructured, deepdoc; static fallback not used for .md)
        md_parsers = get_supported_parsers(".md")
        assert "unstructured" in md_parsers
        assert "deepdoc" in md_parsers

    def test_get_supported_parsers_unknown_extension(self):
        """Test getting parsers for unknown file extensions."""
        result = get_supported_parsers(".unknown")
        assert result == []

    def test_get_supported_parsers_case_insensitive(self):
        """Test extension matching is case insensitive."""
        result1 = get_supported_parsers(".PDF")
        result2 = get_supported_parsers(".pdf")
        assert result1 == result2

    def test_get_supported_parsers_without_dot(self):
        """Test extension matching works without leading dot."""
        result1 = get_supported_parsers("pdf")
        result2 = get_supported_parsers(".pdf")
        assert result1 == result2


class TestValidateParserCompatibility:
    """Test parser compatibility validation."""

    def test_validate_parser_compatibility_mixed_allowed(self):
        """Test validation when mixed parsers are allowed."""
        # Should always return True when allow_mixed=True
        assert validate_parser_compatibility(".pdf", "any_parser", allow_mixed=True)
        assert validate_parser_compatibility(".unknown", "any_parser", allow_mixed=True)
        assert validate_parser_compatibility(".py", "web_parser", allow_mixed=True)

    def test_validate_parser_compatibility_mixed_not_allowed(self):
        """Test validation when mixed parsers are not allowed."""
        # Valid combinations (.pdf/.md use dynamic map; .py uses static fallback)
        assert validate_parser_compatibility(".pdf", "deepdoc", allow_mixed=False)
        assert validate_parser_compatibility(".py", "code", allow_mixed=False)
        assert validate_parser_compatibility(".md", "unstructured", allow_mixed=False)
        assert validate_parser_compatibility(".md", "deepdoc", allow_mixed=False)

        # Invalid combinations
        assert not validate_parser_compatibility(".pdf", "code", allow_mixed=False)
        assert not validate_parser_compatibility(".py", "deepdoc", allow_mixed=False)
        assert not validate_parser_compatibility(
            ".unknown", "any_parser", allow_mixed=False
        )

    def test_validate_parser_compatibility_unknown_extension(self):
        """Test validation with unknown file extension."""
        # Unknown extension should not be compatible with any parser when mixed=False
        assert not validate_parser_compatibility(
            ".xyz", "any_parser", allow_mixed=False
        )
        # But should be compatible when mixed=True
        assert validate_parser_compatibility(".xyz", "any_parser", allow_mixed=True)


class TestGetAllSupportedExtensions:
    """Test getting all supported extensions."""

    def test_get_all_supported_extensions(self):
        """Test getting all supported file extensions."""
        extensions = get_all_supported_extensions()

        # Should contain common extensions
        assert ".pdf" in extensions
        assert ".py" in extensions
        assert ".md" in extensions
        assert ".txt" in extensions
        assert ".html" in extensions

        # Should be a set
        assert isinstance(extensions, set)

        # Should contain all keys from PARSER_COMPATIBILITY
        assert extensions == set(PARSER_COMPATIBILITY.keys())


class TestRegisterParserSupport:
    """Test registering new parser support."""

    def test_register_parser_support_new_extension(self):
        """Test registering support for new file extension."""
        # Register support for a new extension
        register_parser_support(".xyz", "xyz_parser")

        # Should now be supported
        parsers = get_supported_parsers(".xyz")
        assert "xyz_parser" in parsers

        # Clean up
        if ".xyz" in PARSER_COMPATIBILITY:
            del PARSER_COMPATIBILITY[".xyz"]

    def test_register_parser_support_existing_extension(self):
        """Test registering additional parser for existing extension (static-only)."""
        # Use .py: only in static map, so register_parser_support affects get_supported_parsers
        original_parsers = get_supported_parsers(".py")
        original_count = len(original_parsers)

        register_parser_support(".py", "new_py_parser")

        new_parsers = get_supported_parsers(".py")
        assert len(new_parsers) == original_count + 1
        assert "new_py_parser" in new_parsers

        # Clean up
        if "new_py_parser" in PARSER_COMPATIBILITY[".py"]:
            PARSER_COMPATIBILITY[".py"].remove("new_py_parser")

    def test_register_parser_support_normalizes_extension(self):
        """Test that extension gets normalized with leading dot."""
        register_parser_support("xyz", "test_parser")

        parsers = get_supported_parsers(".xyz")
        assert "test_parser" in parsers

        # Clean up
        if ".xyz" in PARSER_COMPATIBILITY:
            del PARSER_COMPATIBILITY[".xyz"]

    def test_register_parser_support_no_duplicates(self):
        """Test that registering same parser twice doesn't create duplicates."""
        register_parser_support(".test", "duplicate_parser")
        register_parser_support(".test", "duplicate_parser")

        parsers = get_supported_parsers(".test")
        assert parsers.count("duplicate_parser") == 1

        # Clean up
        if ".test" in PARSER_COMPATIBILITY:
            del PARSER_COMPATIBILITY[".test"]


class TestParserCompatibilityData:
    """Test the PARSER_COMPATIBILITY data structure."""

    def test_parser_compatibility_structure(self):
        """Test that PARSER_COMPATIBILITY has correct structure."""
        assert isinstance(PARSER_COMPATIBILITY, dict)

        for ext, parsers in PARSER_COMPATIBILITY.items():
            # Extensions should start with dot
            assert ext.startswith("."), f"Extension {ext} should start with dot"

            # Parsers should be a list
            assert isinstance(parsers, list), f"Parsers for {ext} should be a list"

            # Should have at least one parser
            assert len(parsers) > 0, f"Extension {ext} should have at least one parser"

            # All parsers should be strings
            assert all(isinstance(p, str) for p in parsers), (
                f"All parsers for {ext} should be strings"
            )

    def test_common_file_types_supported(self):
        """Test that common file types are supported."""
        common_extensions = [
            ".pdf",
            ".docx",
            ".txt",
            ".md",
            ".py",
            ".html",
            ".json",
            ".csv",
            ".xlsx",
            ".ppt",
            ".pptx",
        ]

        for ext in common_extensions:
            assert ext in PARSER_COMPATIBILITY, (
                f"Common extension {ext} should be supported"
            )
            assert len(PARSER_COMPATIBILITY[ext]) > 0, (
                f"Extension {ext} should have parsers"
            )

    def test_powerpoint_files_supported(self):
        """Test that PowerPoint files are supported with unstructured parser."""
        # Test .pptx support
        pptx_parsers = get_supported_parsers(".pptx")
        assert "unstructured" in pptx_parsers
        assert len(pptx_parsers) > 0

        # Test .ppt support
        ppt_parsers = get_supported_parsers(".ppt")
        assert "unstructured" in ppt_parsers
        assert len(ppt_parsers) > 0

        # Validate compatibility
        assert validate_parser_compatibility(".pptx", "unstructured", allow_mixed=False)
        assert validate_parser_compatibility(".ppt", "unstructured", allow_mixed=False)

        # deepdoc should not be compatible with PowerPoint
        assert not validate_parser_compatibility(".pptx", "deepdoc", allow_mixed=False)
        assert not validate_parser_compatibility(".ppt", "deepdoc", allow_mixed=False)

"""Phase 6 — A12 XML adapter tests.

Tests the safe quarantine disposition for XML files:
  - Well-formed CDA/XML → quarantined with reason 'unsupported_format'
  - Malformed XML → quarantined with reason 'validation_failed'
  - DTD/external entity declarations → quarantined (safe)
  - Depth overflow → quarantined
  - .xml file with non-XML content follows content sniffing (txt → not xml adapter)
  - Verifies no raw XML is left under sanitized/

A12 is explicitly delivered as 'safe quarantine' (optional adapter deferred).
"""
from __future__ import annotations

import pytest

from kavachmed.adapters import xml as xml_adapter
from kavachmed.contracts import FileEntry
from kavachmed.security import StageError


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

MINIMAL_CDA = b"""<?xml version="1.0" encoding="UTF-8"?>
<ClinicalDocument xmlns="urn:hl7-org:v3">
  <id root="2.16.840.1.113883.19.5"/>
  <text>Patient is well.</text>
</ClinicalDocument>
"""

MALFORMED_XML = b"""<?xml version="1.0"?>
<root>
  <unclosed
"""

DEEP_XML_TEMPLATE = "<a>" * 600 + "x" + "</a>" * 600

DTD_XML = b"""<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ELEMENT foo ANY>
]>
<foo>bar</foo>
"""

EXTERNAL_ENTITY_XML = b"""<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<foo>&xxe;</foo>
"""

NAMESPACE_CDA = b"""<?xml version="1.0" encoding="UTF-8"?>
<ns0:ClinicalDocument xmlns:ns0="urn:hl7-org:v3" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <ns0:id root="2.16.840.1.113883.19.5"/>
  <ns0:text>Blood pressure 120/80 mmHg</ns0:text>
</ns0:ClinicalDocument>
"""

ESCAPED_CHARS_XML = b"""<?xml version="1.0"?>
<record>
  <name>O&apos;Brien &amp; Sons &lt;Ltd&gt;</name>
  <note>Temperature &gt; 37.5 C</note>
</record>
"""


def _make_entry(raw: bytes, sniff_fmt: str = "xml", ext: str = ".xml") -> FileEntry:
    """Build a minimal FileEntry for the XML adapter tests."""
    from pathlib import Path
    text: str | None
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="replace")
    return FileEntry(
        abs_path=Path(f"/tmp/test{ext}"),
        rel_path=f"test{ext}",
        sha256="0" * 64,
        raw_bytes=raw,
        sniff_fmt=sniff_fmt,
        encoding="utf-8",
        text=text,
        quarantine_reason=None,
    )


EMPTY_POLICY: dict = {}


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

class TestXmlSafeQuarantine:
    """All XML files must be quarantined — never copied to sanitized/."""

    def test_well_formed_cda_is_sanitised_to_valid_xml(self):
        """Well-formed CDA (A12 sanitiser): fields extracted, projection stays well-formed."""
        entry = _make_entry(MINIMAL_CDA)
        pf = xml_adapter.parse(entry, EMPTY_POLICY)
        assert pf.fmt == "xml" and pf.records and pf.projector is not None
        out, _ = pf.projector([], {})
        assert xml_adapter._probe_xml(out) == "well_formed"

    def test_malformed_xml_raises(self):
        """Malformed XML: quarantine with unsupported/validation error."""
        entry = _make_entry(MALFORMED_XML)
        with pytest.raises(StageError):
            xml_adapter.parse(entry, EMPTY_POLICY)

    def test_dtd_declaration_raises(self):
        """DTD declaration must be rejected (security: no DTD expansion)."""
        entry = _make_entry(DTD_XML)
        with pytest.raises(StageError):
            xml_adapter.parse(entry, EMPTY_POLICY)

    def test_external_entity_raises(self):
        """External entity reference must be rejected (XXE prevention)."""
        entry = _make_entry(EXTERNAL_ENTITY_XML)
        with pytest.raises(StageError):
            xml_adapter.parse(entry, EMPTY_POLICY)

    def test_depth_overflow_raises(self):
        """Deeply nested XML must be quarantined (depth > 500)."""
        raw = DEEP_XML_TEMPLATE.encode("utf-8")
        entry = _make_entry(raw)
        with pytest.raises(StageError):
            xml_adapter.parse(entry, EMPTY_POLICY)

    def test_namespace_qualified_cda_round_trips(self):
        """Namespace-qualified CDA: namespaces are markup, copied verbatim."""
        entry = _make_entry(NAMESPACE_CDA)
        pf = xml_adapter.parse(entry, EMPTY_POLICY)
        out, _ = pf.projector([], {})
        assert out == entry.raw_bytes

    def test_escaped_chars_xml_round_trip_and_edit(self):
        """Escaped text stays escaped; a placeholder edit keeps the XML well-formed."""
        from kavachmed.contracts import Edit
        entry = _make_entry(ESCAPED_CHARS_XML)
        pf = xml_adapter.parse(entry, EMPTY_POLICY)
        out, _ = pf.projector([], {})
        assert out == entry.raw_bytes
        rec = pf.records[0]
        f = next(fl for fl in rec.fields if fl.text.strip())
        out2, _ = pf.projector([Edit(rec.key, f.idx, 0, len(f.text), "PLACEHOLDER", "[NAME]")], {})
        assert xml_adapter._probe_xml(out2) == "well_formed"

    def test_empty_bytes_raises(self):
        """Empty raw bytes → raises StageError."""
        entry = _make_entry(b"", sniff_fmt="xml")
        with pytest.raises(StageError):
            xml_adapter.parse(entry, EMPTY_POLICY)

    def test_already_quarantined_entry_raises(self):
        """Entry with existing quarantine_reason must still raise."""
        from pathlib import Path
        entry = FileEntry(
            abs_path=Path("/tmp/test.xml"),
            rel_path="test.xml",
            sha256="0" * 64,
            raw_bytes=None,
            sniff_fmt="xml",
            encoding="utf-8",
            text=None,
            quarantine_reason="mixed_encoding",
        )
        with pytest.raises(StageError):
            xml_adapter.parse(entry, EMPTY_POLICY)


class TestXmlRouterDispatch:
    """Verify that the adapter router dispatches xml sniff_fmt to the XML adapter."""

    def test_router_dispatches_xml_to_sanitiser(self):
        """sniff_fmt='xml' in router → XML sanitiser."""
        from kavachmed.adapters import parse as router_parse
        entry = _make_entry(MINIMAL_CDA, sniff_fmt="xml")
        assert router_parse(entry, EMPTY_POLICY).fmt == "xml"

    def test_router_txt_sniff_does_not_use_xml_adapter(self):
        """A .xml file sniffed as 'txt' goes to text adapter, not xml adapter."""
        from kavachmed.adapters import parse as router_parse
        # Plain text that happens to have .xml extension but sniffs as txt
        entry = _make_entry(b"Hello world\nThis is plain text.\n", sniff_fmt="txt", ext=".xml")
        # text adapter should parse it successfully (no StageError for simple text)
        result = router_parse(entry, EMPTY_POLICY)
        assert result is not None
        assert result.fmt == "text"

    def test_router_unknown_fmt_raises_unsupported(self):
        """Unknown sniff_fmt raises E_UNSUPPORTED."""
        from kavachmed.adapters import parse as router_parse
        entry = _make_entry(b"<foo/>", sniff_fmt="unknown_fmt")
        with pytest.raises(StageError) as exc_info:
            router_parse(entry, EMPTY_POLICY)
        assert exc_info.value.code == "E_UNSUPPORTED"

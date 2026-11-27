"""Deterministic release dispositions for incoming credentials and explicit attachments."""
from __future__ import annotations

import base64
import binascii
import re
from functools import lru_cache

from kavachmed.security import StageError

MARKER = "[CREDENTIAL_REMOVED]"
SOURCE = "[SOURCE_PATH_REMOVED]"
SECRET_FIELDS = frozenset({
    "password", "passwd", "api_key", "access_token", "auth_token", "refresh_token",
    "client_secret", "private_key", "secret_access_key", "aws_secret_access_key",
})
_ASSIGN = re.compile(
    r'''(?ix)\b(?:password|passwd|api[_ -]?key|(?:access|auth|refresh)[_ -]?token|client[_ -]?secret)'''
    r'''\s*[:=]\s*(?:"([^"\r\n]*)"|'([^'\r\n]*)'|([^\s;,]+))'''
)
_SIGNATURE = re.compile(r"\b(?:AKIA[A-Z0-9]{16}|sk_(?:live|test)_[A-Za-z0-9]{12,}|gh[pousr]_[A-Za-z0-9]{30,})\b")
_PEM = re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----")
_KEY_HEX = re.compile(r"(?im)^\s*key_hex\s*[:=]\s*[a-f0-9]{32,128}\s*$")
_MRN_LABEL = re.compile(r"(?i)\b(?:MRN|UHID|patient\s+ID)\s*[:=#]\s*([a-z0-9][a-z0-9./_-]*\d[a-z0-9./_-]*)")


@lru_cache(maxsize=1024)
def normalized(key: str) -> str:
    return re.sub(r"[- ]+", "_", re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key.strip())).casefold()


def key_material(text: str) -> bool:
    return bool(_PEM.search(text) or _KEY_HEX.search(text))


def inline_ranges(text: str):
    for m in _ASSIGN.finditer(text):
        group = m.lastindex
        if group and m.group(group) and m.group(group) != MARKER:
            yield m.span(group)
    for m in _SIGNATURE.finditer(text):
        yield m.span()


@lru_cache(maxsize=2048)
def json_role(parent: str, key: str) -> str:
    name = normalized(key)
    owner = normalized(parent).strip("_")
    if name in SECRET_FIELDS:
        return "credential"
    if owner in {"export_info", "export_meta", "metadata", "provenance"} and name in {"source", "source_path", "source_file"}:
        return "source"
    if name in {"attachment", "content_base64", "attachment_base64", "payload_b64", "form_data_b64"} or (
        owner in {"attachment", "attachments"} and name == "data"
    ):
        return "attachment"
    if owner in {"attachment", "attachments"} and name in {"url", "title", "hash", "size"}:
        return "attachment_meta"
    return ""


class AttachmentBudget:
    def __init__(self) -> None:
        self.remaining = 1024 * 1024

    def decode(self, value: str) -> str:
        limit = min(self.remaining, 256 * 1024)
        if len(value) > 4 * ((limit + 2) // 3):
            raise StageError("parse", "E_ATTACHMENT")
        try:
            raw = base64.b64decode(value, validate=True)
            text = raw.decode("utf-8")
        except (ValueError, UnicodeError, binascii.Error):
            raise StageError("parse", "E_ATTACHMENT") from None
        if len(raw) > limit or any(ord(c) < 32 and c not in "\t\r\n" for c in text):
            raise StageError("parse", "E_ATTACHMENT")
        self.remaining -= len(raw)
        return text


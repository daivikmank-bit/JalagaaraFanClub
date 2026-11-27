import base64
import json
from unittest.mock import patch

from kavachmed import pipeline
from kavachmed.security import load_keys


def test_verified_mixed_fields_and_standalone_key(tmp_path):
    inp, out = tmp_path / "input", tmp_path / "out"
    inp.mkdir()
    data = {"metadata": {"source": "private/export.tsv"}, "password": "private value",
            "attachment": base64.b64encode(b"MRN: TEST-X34-123").decode(), "bp": "120/80"}
    (inp / "report.json").write_text(json.dumps(data))
    (inp / "config.txt").write_text("-----BEGIN PRIVATE KEY-----")
    keys = load_keys({"RAKSHA_SECRET": "synthetic-case"}, pbkdf2_iterations=1)
    pipeline.run(inp, out, keys, {})
    files = json.loads((out / "report.json").read_text())["files"]
    assert any(f.get("reason") == "key_material" for f in files)
    result, = (out / "sanitized").rglob("*.json")
    obj = json.loads(result.read_bytes())
    assert obj["metadata"]["source"] == "[SOURCE_PATH_REMOVED]"
    assert obj["password"] == "[CREDENTIAL_REMOVED]"
    assert b"TEST-X34-123" not in base64.b64decode(obj["attachment"])
    assert obj["bp"] == "120/80"

    with patch("kavachmed.release.verify", return_value=False):
        pipeline.run(inp, out, keys, {})
    assert not result.exists()
    assert not (out / "records.jsonl").read_text().strip()


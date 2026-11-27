"""Canary test suite (D6): planted tokens in all hiding places, unseen-name canaries, and disk-dirty fault."""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kavachmed import gate, residual_a, residual_b
from kavachmed.contracts import Edit, Field, LocEntry, LocMap, Record
from kavachmed.policy import load_policy


class TestCanarySuite(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="km-canary-"))
        self.inp = self.tmp / "input"
        self.out = self.tmp / "output"
        self.inp.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_unseen_name_and_channel_canaries_zero_survivors(self) -> None:
        # 1. Cell & name-bearing header canary in CSV
        (self.inp / "ward.csv").write_text(
            "time,ramesh\n08:00,UnseenLatinCanary\n",
            encoding="utf-8",
        )
        # 2. JSON name-bearing key, source_path, base64 attachment, Devanagari + lowercase gazetteer
        b64_payload = base64.b64encode("Secret patient अनदेखाकैनरी 98 76 54 32 10".encode("utf-8")).decode("ascii")
        (self.inp / "opd.json").write_text(
            json.dumps(
                {
                    "source_path": "D:\\HIS\\exports\\CanaryDirToken\\file.json",
                    "anita": "canarylover",
                    "attachment": b64_payload,
                    "visits": [{"notes": "Seen with अनदेखाकैनरी and anita in OPD"}],
                }
            ),
            encoding="utf-8",
        )
        # 3. Filename canary
        (self.inp / "Report_CanaryFileToken_9988776655.txt").write_text(
            "Patient UnseenFilePerson with ९८७६५४३२१० and ambulance UP32AB9999\n",
            encoding="utf-8",
        )
        # 4. FHIR text.div canary & meta.source
        fhir_b = {
            "resourceType": "Bundle",
            "id": "b-can",
            "meta": {"profile": ["https://nrces.in/ndhm/fhir/r4/StructureDefinition/DocumentBundle"], "source": "http://bad/CanaryMetaSource"},
            "type": "document",
            "entry": [
                {
                    "fullUrl": "urn:uuid:p1",
                    "resource": {
                        "resourceType": "Patient",
                        "id": "p1",
                        "name": [{"text": "FhirCanaryName"}],
                        "text": {"status": "generated", "div": "<div>FhirCanaryName 9988776655</div>"},
                    },
                }
            ],
        }
        (self.inp / "fhir_canary.json").write_text(json.dumps(fhir_b), encoding="utf-8")
        # 5. Ambulance remarks canary (vehicle plate + unseen attendant name + phone)
        (self.inp / "ambulance_log.csv").write_text(
            "trip,from,to,remarks\nT1,District,Hospital,Picked up AmbulanceCanaryPerson phone 9123456789 plate UP32CD4321; anita accompanied\n",
            encoding="utf-8",
        )
        # 6. Malformed file carrying a canary (must quarantine, never leak into output or logs)
        (self.inp / "broken.txt").write_bytes(b"\xff\xfeMalformedCanaryToken\x80\x81")

        iso_tmp = self.tmp / "iso_tmp"
        iso_home = self.tmp / "iso_home"
        iso_tmp.mkdir()
        iso_home.mkdir()
        env = dict(os.environ, RAKSHA_SECRET="canary-test-secret-01", TMPDIR=str(iso_tmp), HOME=str(iso_home))

        repo_root = Path(__file__).resolve().parents[2]
        proc = subprocess.run(
            [sys.executable, str(repo_root / "main.py"), "--input", str(self.inp), "--output", str(self.out)],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )

        canary_tokens = [
            "UnseenLatinCanary",
            "CanaryDirToken",
            "अनदेखाकैनरी",
            "98 76 54 32 10",
            "९८७६५४३२१०",
            "UnseenFilePerson",
            "9988776655",
            "UP32AB9999",
            "FhirCanaryName",
            "CanaryMetaSource",
            "AmbulanceCanaryPerson",
            "9123456789",
            "UP32CD4321",
            "anita",
            "ramesh",
            "canarylover",
            "MalformedCanaryToken",
            "canary-test-secret-01",
        ]
        haystacks = [proc.stdout, proc.stderr]
        for root_dir in (self.out, iso_tmp, iso_home):
            for p in root_dir.rglob("*"):
                if p.is_file():
                    haystacks.append(p.name)
                    haystacks.append(p.read_text(encoding="utf-8", errors="ignore"))
        joined = "\n".join(haystacks)
        for tok in canary_tokens:
            self.assertNotIn(tok, joined, f"Canary leaked: {tok}")

    def test_disk_dirty_writer_fault_caught_and_fail_closed(self) -> None:
        _ = load_policy()
        dest = self.out / "sanitized" / "dirty.txt"
        clean_mem = b"PSN-12345678 stable with fever."
        rk = ("dirty.txt", "doc")
        rec = Record(key=rk, fields=[Field(0, "doc", clean_mem.decode("utf-8"))])

        # Persistent dirty disk bug must be caught by read-back verification and fail closed (dest removed)
        def _persistent_dirty_writer(path: Path, _mem_bytes: bytes) -> None:
            path.write_bytes(b"UnseenDiskBugName 9876543210 leaked on disk.")

        ok_persist, final_p, _, counts_p = gate.run_with_repair(
            dest_path=dest,
            initial_bytes=clean_mem,
            rel_out="sanitized/dirty.txt",
            initial_loc_map=LocMap([LocEntry(0, len(clean_mem), rk, 0, 0, len(clean_mem), None)]),
            leak_keys={},
            scan_a=residual_a.scan,
            scan_b=residual_b.scan,
            reproject_fn=lambda _rep: (clean_mem, LocMap([LocEntry(0, len(clean_mem), rk, 0, 0, len(clean_mem), None)])),
            records_by_key={rk: rec},
            max_passes=3,
            writer_fn=_persistent_dirty_writer,
        )
        self.assertFalse(ok_persist)
        self.assertIsNone(final_p)
        self.assertFalse(dest.exists())
        self.assertGreater(counts_p["B1"] + counts_p["B2"] + counts_p["B4"], 0)

        # Transient disk fault on attempt 0 is detected by Layer B, retried, and repaired cleanly on attempt 1
        attempts = {"n": 0}

        def _transient_fault_writer(path: Path, mem_bytes: bytes) -> None:
            attempts["n"] += 1
            if attempts["n"] == 1:
                path.write_bytes(b"UnseenDiskBugName 9876543210 leaked on disk.")
            else:
                path.write_bytes(mem_bytes)

        ok_trans, final_t, _, counts_t = gate.run_with_repair(
            dest_path=dest,
            initial_bytes=clean_mem,
            rel_out="sanitized/dirty.txt",
            initial_loc_map=LocMap([LocEntry(0, len(clean_mem), rk, 0, 0, len(clean_mem), None)]),
            leak_keys={},
            scan_a=residual_a.scan,
            scan_b=residual_b.scan,
            reproject_fn=lambda _rep: (clean_mem, LocMap([LocEntry(0, len(clean_mem), rk, 0, 0, len(clean_mem), None)])),
            records_by_key={rk: rec},
            max_passes=3,
            writer_fn=_transient_fault_writer,
        )
        self.assertTrue(ok_trans)
        self.assertEqual(final_t, clean_mem)
        self.assertGreater(counts_t["B1"] + counts_t["B2"] + counts_t["B4"], 0)


if __name__ == "__main__":
    unittest.main()

"""Refresh the sha256 checksums in croissant.json after any listed file changes.

Every cr:FileObject whose contentUrl points into this repository's main branch
gets the checksum of the local copy of that file. Run from anywhere:

    python scripts/update_croissant_checksums.py
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "/Composable_Causality/main/"


def main() -> None:
    path = ROOT / "croissant.json"
    meta = json.loads(path.read_text())
    for obj in meta.get("distribution", []):
        url = obj.get("contentUrl", "")
        if obj.get("@type") != "cr:FileObject" or MARKER not in url:
            continue
        local = ROOT / url.split(MARKER, 1)[1]
        digest = hashlib.sha256(local.read_bytes()).hexdigest()
        if obj.get("sha256") != digest:
            print(f"updated {local.relative_to(ROOT)}")
            obj["sha256"] = digest
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

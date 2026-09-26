#!/usr/bin/env python3
"""Audit catalog checkboxes and manifest evidence without requiring a build."""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CATALOGS = (ROOT / "docs" / "catalogs" / "AUGMENTATION_CATALOG.md", ROOT / "docs" / "catalogs" / "NOISE_CATALOG.md")
MARKER_STATUS = {" ": "todo", "~": "partial", "x": "implemented"}
BULLET_RE = re.compile(r"^\s*- \[([ x~])\]\s+(.+?)\s*$")
HEADING_RE = re.compile(r"^\s*#{2,4}\s+(.+?)\s*$")
IGNORED_PATHS = {"", "host-only", "none", "n/a"}


def catalog_entries(path: Path):
    category = ""
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        heading = HEADING_RE.match(raw)
        if heading:
            category = heading.group(1)
            continue
        match = BULLET_RE.match(raw)
        if not match:
            continue
        marker, item = match.groups()
        item = item.replace("`", "").rstrip(".")
        item = re.sub(r"\s+\(.*\)\s*$", "", item)
        yield path.name, category, item, MARKER_STATUS[marker], line_number


def main() -> int:
    manifest_path = ROOT / "IMPLEMENTATION_MANIFEST.tsv"
    with manifest_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))

    errors: list[str] = []
    umbrella = ROOT / "include" / "augmatch" / "augmatch.hpp"
    if not umbrella.exists():
        errors.append(f"missing umbrella header: {umbrella.relative_to(ROOT)}")
    else:
        umbrella_text = umbrella.read_text()
        header_root = ROOT.joinpath("include", "augmatch")
        for public_header in sorted(header_root.rglob("*.hpp")):
            relative = public_header.relative_to(header_root).as_posix()
            if relative == "augmatch.hpp":
                continue
            if "/" not in relative:
                errors.append(f"public header outside a topic folder: {relative}")
            include = f'#include "augmatch/{relative}"'
            if include not in umbrella_text:
                errors.append(f"umbrella header omits public header: {relative}")
    manifest_keys = Counter((r["catalog"], r["category"], r["item"]) for r in rows)
    catalog_rows = []
    for catalog in CATALOGS:
        catalog_rows.extend(catalog_entries(catalog))
    catalog_keys = Counter((c, category, item) for c, category, item, _, _ in catalog_rows)

    if catalog_keys != manifest_keys:
        for key in sorted(set(catalog_keys) | set(manifest_keys)):
            if catalog_keys[key] != manifest_keys[key]:
                errors.append(
                    f"catalog/manifest multiplicity mismatch {key!r}: "
                    f"catalog={catalog_keys[key]} manifest={manifest_keys[key]}"
                )

    marker_by_key = {(c, category, item): status for c, category, item, status, _ in catalog_rows}
    for row in rows:
        key = (row["catalog"], row["category"], row["item"])
        if row["status"] == "todo":
            errors.append(f"todo manifest row: {key!r}")
        expected = marker_by_key.get(key)
        if expected is not None and expected != row["status"]:
            errors.append(
                f"status mismatch {key!r}: catalog={expected} manifest={row['status']}"
            )
        for column in ("cpp", "cuda", "test"):
            for relative in filter(None, (part.strip() for part in row[column].split(";"))):
                if relative in IGNORED_PATHS:
                    continue
                if not (ROOT / relative).exists():
                    errors.append(f"missing {column} artifact for {key!r}: {relative}")

    print(f"catalog rows: {len(catalog_rows)}")
    print(f"manifest rows: {len(rows)}")
    print("status counts:", dict(Counter(row["status"] for row in rows)))
    print(f"blank CPU/CUDA fields (documented task rows may be intentional): "
          f"{sum(not row['cpp'] or not row['cuda'] for row in rows)}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("manifest audit: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
scraper/merge_bib.py

Script para fusionar entradas bibliográficas de IEEE y Web of Science (WOS).
- Salidas:
    * .bib fusionado
    * Reporte CSV con el mapeo (tipo de match, score, archivos fuente)
Uso:
python3 utils/merge_bib.py \
        --ieee-dir data/raw/IEEE \
        --wos-dirs data/raw/WoS \
        --out-dir data/processed
"""

from __future__ import annotations
import argparse
import csv
import logging
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import bibtexparser
from bibtexparser.bibdatabase import BibDatabase
from bibtexparser.bwriter import BibTexWriter
from rapidfuzz import process, fuzz

# ---- Configuración de logging ----
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("merge_bib")


# ---- Utilidades ----
def project_root() -> Path:
    """Retorna la raíz del proyecto asumiendo que este archivo está en <project>/utils/."""
    return Path(__file__).resolve().parent.parent


def read_bib_files(folder: Path) -> List[Dict]:
    """Lee todos los archivos .bib en una carpeta y retorna una lista de entradas."""
    entries: List[Dict] = []
    if not folder.exists():
        logger.warning("Carpeta no existe: %s", folder)
        return entries
    for p in sorted(folder.glob("*.bib")):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            db = bibtexparser.loads(text)
            for e in db.entries:
                e.setdefault("_source_file", str(p.name))
                entries.append(e)
        except Exception as ex:
            logger.error("Error leyendo %s: %s", p, ex)
    logger.info("Leidos %d entries desde %s", len(entries), folder)
    return entries


def normalize_isbn(isbn_raw: Optional[str]) -> Optional[List[str]]:
    if not isbn_raw:
        return None
    s = re.sub(r'[{}\s"]', "", isbn_raw)
    parts = re.split(r'[;,/|]', s)
    parts = [p.strip().lower() for p in parts if p.strip()]
    return parts if parts else None


def normalize_title(title: Optional[str]) -> str:
    if not title:
        return ""
    t = title.lower()
    t = unicodedata.normalize("NFD", t)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def get_isbns_from_entry(e: Dict) -> Optional[List[str]]:
    for key in ("isbn", "ISBN", "issn", "ISSN"):
        if key in e and str(e[key]).strip():
            return normalize_isbn(str(e[key]))
    return None


def merge_entries(e_list: List[Dict]) -> Dict:
    merged: Dict = {}
    merged["ENTRYTYPE"] = e_list[0].get("ENTRYTYPE", e_list[0].get("type", "inproceedings"))
    fields = set().union(*[set(e.keys()) for e in e_list])
    fields = [f for f in fields if not f.startswith("_")]

    for f in fields:
        vals = [str(e.get(f, "")).strip() for e in e_list if str(e.get(f, "")).strip()]
        if not vals:
            continue
        if f.lower() == "keywords":
            kwset = set()
            for v in vals:
                for tok in re.split(r"[;,]", v):
                    tok = tok.strip()
                    if tok:
                        kwset.add(tok)
            merged[f] = "; ".join(sorted(kwset))
        elif f.lower() == "abstract":
            merged[f] = max(vals, key=len)
        else:
            merged[f] = max(vals, key=len)
    return merged


# ---- Lógica principal de merge ----
def build_isbn_map(entries: List[Dict]) -> Dict[str, List[Tuple[int, Dict]]]:
    m: Dict[str, List[Tuple[int, Dict]]] = {}
    for idx, e in enumerate(entries):
        isbns = get_isbns_from_entry(e)
        if isbns:
            for isb in isbns:
                m.setdefault(isb, []).append((idx, e))
    return m


def merge_collections(
    ieee_entries: List[Dict],
    wos_entries: List[Dict],
    title_threshold: int = 88,
) -> Tuple[List[Dict], List[Dict]]:

    used_wos = set()
    used_ieee = set()
    merged_results: List[Dict] = []
    mapping_rows: List[Dict] = []

    wos_isbn_map = build_isbn_map(wos_entries)

    # 1) Match por ISBN
    for i_idx, e_ieee in enumerate(ieee_entries):
        isbns = get_isbns_from_entry(e_ieee)
        if not isbns:
            continue
        for isb in isbns:
            if isb in wos_isbn_map:
                for w_idx, e_wos in wos_isbn_map[isb]:
                    if w_idx in used_wos:
                        continue
                    merged_results.append(merge_entries([e_ieee, e_wos]))
                    used_ieee.add(i_idx)
                    used_wos.add(w_idx)
                    mapping_rows.append({
                        "merged_key": "",
                        "ieee_key": e_ieee.get("ID", ""),
                        "wos_key": e_wos.get("ID", ""),
                        "match_type": "ISBN",
                        "score": 100,
                        "isbn": isb,
                        "ieee_file": e_ieee.get("_source_file", ""),
                        "wos_file": e_wos.get("_source_file", ""),
                    })
                break

    remaining_ieee = [(i, e) for i, e in enumerate(ieee_entries) if i not in used_ieee]
    remaining_wos = [(i, e) for i, e in enumerate(wos_entries) if i not in used_wos]

    wos_pool_idx = [i for i, _ in remaining_wos]
    wos_pool_titles = [normalize_title(e.get("title", "")) for _, e in remaining_wos]

    # 2) Match por título difuso
    for i_idx, e_ieee in remaining_ieee:
        title_ieee = normalize_title(e_ieee.get("title", ""))
        if not title_ieee or not wos_pool_titles:
            continue
        best = process.extractOne(title_ieee, wos_pool_titles, scorer=fuzz.token_set_ratio)
        if best:
            _, score, pos = best
            if score >= title_threshold:
                w_idx = wos_pool_idx[pos]
                e_wos = wos_entries[w_idx]
                merged_results.append(merge_entries([e_ieee, e_wos]))
                used_ieee.add(i_idx)
                used_wos.add(w_idx)
                mapping_rows.append({
                    "merged_key": "",
                    "ieee_key": e_ieee.get("ID", ""),
                    "wos_key": e_wos.get("ID", ""),
                    "match_type": "TITLE",
                    "score": int(score),
                    "isbn": ";".join(get_isbns_from_entry(e_ieee) or []) or ";".join(get_isbns_from_entry(e_wos) or []),
                    "ieee_file": e_ieee.get("_source_file", ""),
                    "wos_file": e_wos.get("_source_file", ""),
                })
                del wos_pool_titles[pos]
                del wos_pool_idx[pos]

    # 3) No emparejados
    for i, e in enumerate(ieee_entries):
        if i not in used_ieee:
            merged_results.append(merge_entries([e]))
            mapping_rows.append({
                "merged_key": "",
                "ieee_key": e.get("ID", ""),
                "wos_key": "",
                "match_type": "UNMATCHED_IEEE",
                "score": 0,
                "isbn": ";".join(get_isbns_from_entry(e) or []),
                "ieee_file": e.get("_source_file", ""),
                "wos_file": "",
            })

    for i, e in enumerate(wos_entries):
        if i not in used_wos:
            merged_results.append(merge_entries([e]))
            mapping_rows.append({
                "merged_key": "",
                "ieee_key": "",
                "wos_key": e.get("ID", ""),
                "match_type": "UNMATCHED_WOS",
                "score": 0,
                "isbn": ";".join(get_isbns_from_entry(e) or []),
                "ieee_file": "",
                "wos_file": e.get("_source_file", ""),
            })

    return merged_results, mapping_rows


# ---- Escritura ----
def write_bib_and_csv(out_dir: Path, merged_results: List[Dict], mapping_rows: List[Dict], out_bib: str, out_csv: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, e in enumerate(merged_results, start=1):
        e["ID"] = f"merged{idx}"

    bibdb = BibDatabase()
    bibdb.entries = merged_results
    writer = BibTexWriter()
    writer.order_entries_by = None

    (out_dir / out_bib).write_text(writer.write(bibdb), encoding="utf-8")

    with (out_dir / out_csv).open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["merged_key", "ieee_key", "wos_key", "match_type", "score", "isbn", "ieee_file", "wos_file"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in mapping_rows:
            w.writerow(r)


# ---- CLI ----
def parse_args() -> argparse.Namespace:
    pr = project_root()
    parser = argparse.ArgumentParser(description="Merge IEEE and Web of Science .bib files")
    parser.add_argument("--ieee-dir", default=str(pr / "data/raw/IEEE"))
    parser.add_argument("--wos-dirs", default=str(pr / "data/raw/WoS"))
    parser.add_argument("--out-dir", default=str(pr / "data/processed"))
    parser.add_argument("--out-bib", default="merged.bib")
    parser.add_argument("--out-csv", default="merge_map.csv")
    parser.add_argument("--title-threshold", type=int, default=88)
    return parser.parse_args()


def merge_main():
    args = parse_args()
    pr = project_root()

    ieee_dir = Path(args.ieee_dir)
    if not ieee_dir.is_absolute():
        ieee_dir = (pr / ieee_dir).resolve()

    wos_dirs = []
    for d in args.wos_dirs.split(","):
        p = Path(d.strip())
        if not p.is_absolute():
            p = (pr / p).resolve()
        wos_dirs.append(p)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (pr / out_dir).resolve()

    ieee_entries = read_bib_files(ieee_dir)
    wos_entries = []
    for d in wos_dirs:
        wos_entries.extend(read_bib_files(d))

    merged_results, mapping_rows = merge_collections(
        ieee_entries, wos_entries, title_threshold=args.title_threshold
    )

    write_bib_and_csv(out_dir, merged_results, mapping_rows, args.out_bib, args.out_csv)


if __name__ == "__main__":
    merge_main()
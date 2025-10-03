
from __future__ import annotations

import pandas as pd
import typer
import re
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

app = typer.Typer(add_completion=False)

CLIENT_NAME = "Intellyse bePro"
CARRIER_NAME = "NTR Freight"
DEFAULT_START = "2025-01-01"
DEFAULT_END = "2025-12-31"

EXPECTED_SHEETS = {
    "express_export": "CH TD Exp WW",
    "economy_export": "CH DD Exp Economy",
    "domestic_tariffs": "CH TD 3rdCty Domestic",
}

BAND_VALIDATOR = re.compile(r"^\d+_up_to_\d+\[\w+\](?:\[\w+\]|\[flat\])$", re.IGNORECASE)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ntr_converter")

@dataclass(frozen=True)
class Region:
    country: Optional[str] = None
    zipcode: Optional[str] = None
    city: Optional[str] = None
    airport: Optional[str] = None
    seaport: Optional[str] = None
    identifier_string: Optional[str] = None

    def as_row(self, id_: int) -> Dict[str, Any]:
        return {
            "id": id_,
            "client": CLIENT_NAME,
            "carrier": CARRIER_NAME,
            "country": self.country or "",
            "zipcode": self.zipcode or "",
            "city": self.city or "",
            "airport": self.airport or "",
            "seaport": self.seaport or "",
            "identifierstring": self.identifier_string or "",
        }

def load_sheet(xl: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    candidates = [s for s in xl.sheet_names if s.strip().lower() == sheet_name.strip().lower()]
    if not candidates:
        candidates = [s for s in xl.sheet_names if sheet_name.strip().lower() in s.strip().lower()]
    if not candidates:
        raise KeyError(f"Sheet '{sheet_name}' not found in workbook")
    chosen = candidates[0]
    logger.info("Reading sheet: %s", chosen)
    return xl.parse(chosen, header=None)

def detect_header_row(df: pd.DataFrame, unit_keywords: List[str] = ("kg", "ldm", "cbm")) -> int:
    best_idx = 0
    best_score = -1
    for i, row in df.iterrows():
        tokens = [str(c).strip() for c in row if pd.notna(c)]
        score = sum(3 for t in tokens if any(k in t.lower() for k in unit_keywords))
        score += sum(2 for t in tokens if re.search(r"\d+", t))
        if score > best_score:
            best_score, best_idx = score, i
    return best_idx

def tidy_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    header_row = detect_header_row(raw)
    df = raw.iloc[header_row:].reset_index(drop=True)
    df.columns = df.iloc[0].fillna("")
    df = df.drop(index=0).reset_index(drop=True)
    df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed", na=False)]
    df.columns = [str(c).strip() for c in df.columns]
    return df

def numeric_or_nan(v: Any) -> Optional[float]:
    if pd.isna(v):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s in {"", "-", "n/a", "na"}:
        return None
    s_clean = re.sub(r"[^0-9.,-]", "", s)
    if s_clean.count(",") > 0 and s_clean.count(".") == 0:
        s_clean = s_clean.replace(",", ".")
    if s_clean.count(".") > 1:
        parts = s_clean.split(".")
        s_clean = "".join(parts[:-1]) + "." + parts[-1]
    try:
        return float(s_clean)
    except ValueError:
        return None

def extract_number_pairs_from_label(label: str) -> Optional[Tuple[Optional[int], Optional[int]]]:
    if not isinstance(label, str):
        return None
    s = label.strip()
    m = re.search(r"(\d+)\s*(?:-|to)\s*(\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    m2 = re.search(r"(\d{1,7})", s)
    if m2:
        return None, int(m2.group(1))
    return None

def infer_bands_from_price_columns(price_cols: List[str], unit_guess: str = "kg") -> List[str]:
    pairs, prev_high = [], 0
    for raw in price_cols:
        parsed = extract_number_pairs_from_label(str(raw))
        if parsed is None:
            low, high = prev_high, 9999999
        else:
            low, high = parsed if parsed[0] is not None else (prev_high, parsed[1] or 9999999)
        if high < low:
            high = low
        pairs.append((low, high))
        prev_high = high
    return [f"{low}_up_to_{high}[{unit_guess}][{unit_guess}]" for low, high in pairs]

def build_regions_from_tokens(tokens: List[str], existing: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
    unique = sorted({(t or "").strip() for t in tokens if t and str(t).strip() != ""})
    if existing is not None and not existing.empty:
        existing_tokens = set(existing["identifierstring"].tolist())
    else:
        existing_tokens = set()
    new_tokens = [t for t in unique if t not in existing_tokens]
    start_id = existing["id"].max() if existing is not None and not existing.empty else 0
    rows, id_map = [], {}
    for i, tok in enumerate(new_tokens, start=start_id + 1):
        r = Region(identifier_string=tok)
        rows.append(r.as_row(i))
        id_map[tok] = i
    if existing is not None and not existing.empty:
        regions_df = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    else:
        regions_df = pd.DataFrame(rows)
    for t in existing_tokens:
        idx = existing.loc[existing["identifierstring"] == t, "id"].iloc[0]
        id_map[t] = idx
    return regions_df, id_map

def build_tariffs_from_table(df: pd.DataFrame, region_map: Dict[str, int], service_type: str,
                             startdate: str, enddate: str, currency: str, origin_default: str = "Switzerland") -> pd.DataFrame:
    headers = list(df.columns)
    price_cols = [h for h in headers if re.search(r"\d", str(h))]
    id_cols = [h for h in headers if h not in price_cols]
    if not price_cols or not id_cols:
        return pd.DataFrame()
    identifier_col = id_cols[0]
    bands = infer_bands_from_price_columns(price_cols)
    rows = []
    for _, r in df.iterrows():
        dest_token = str(r[identifier_col]).strip()
        if not dest_token:
            continue
        origin = origin_default if "exp" in service_type.lower() else "World"
        origin_id, dest_id = region_map.get(origin), region_map.get(dest_token)
        if not origin_id or not dest_id:
            continue
        route = json.dumps([origin_id, dest_id])
        base = {
            "startdate": startdate,
            "enddate": enddate,
            "client": CLIENT_NAME,
            "carrier": CARRIER_NAME,
            "service_type": service_type,
            "currency": currency,
            "ldm_conversion": "",
            "cbm_conversion": "",
            "min_price": "",
            "max_price": "",
            "route": route,
        }
        for pc, band in zip(price_cols, bands):
            val = numeric_or_nan(r[pc])
            if val is not None:
                base[band] = val
        rows.append(base)
    if not rows:
        return pd.DataFrame()
    df_tariffs = pd.DataFrame(rows)
    df_tariffs.insert(0, "id", range(1, len(df_tariffs) + 1))
    return df_tariffs

def empty_surcharges() -> pd.DataFrame:
    return pd.DataFrame(columns=["id", "client", "carrier", "description", "amount", "currency"])

def validate_output(regions_df: pd.DataFrame, tariffs_df: pd.DataFrame) -> bool:
    ok = True
    if "id" not in regions_df.columns:
        return False
    if not tariffs_df.empty and "route" in tariffs_df.columns:
        region_ids = set(regions_df["id"].tolist())
        for _, r in tariffs_df.iterrows():
            try:
                route = json.loads(r["route"])
            except Exception:
                ok = False
                continue
            for rid in route:
                if rid not in region_ids:
                    ok = False
    return ok

@app.command()
def convert(in_file: Path = typer.Option(..., "--in"), out_file: Path = typer.Option(..., "--out"),
            startdate: str = typer.Option(DEFAULT_START), enddate: str = typer.Option(DEFAULT_END), currency: str = typer.Option("EUR")):
    if not in_file.exists():
        raise typer.Exit(code=1, message=f"Input file not found: {in_file}")
    xl = pd.ExcelFile(in_file)
    all_regions = pd.DataFrame()
    all_tariffs = []
    for sheet_key, service_name in [
        ("express_export", "Express Worldwide"),
        ("economy_export", "Economy Select"),
        ("domestic_tariffs", "Domestic")]:
        sheet_name = EXPECTED_SHEETS.get(sheet_key)
        if not sheet_name:
            continue
        try:
            raw = load_sheet(xl, sheet_name)
        except KeyError:
            continue
        tidy = tidy_dataframe(raw)
        headers = list(tidy.columns)
        price_headers = [h for h in headers if re.search(r"\d", str(h))]
        id_cols = [h for h in headers if h not in price_headers]
        if not id_cols:
            continue
        dest_col = id_cols[0]
        tokens = [str(v).strip() for v in tidy[dest_col].tolist() if pd.notna(v) and str(v).strip() != ""]
        if sheet_key == "express_export":
            tokens.append("Switzerland")
        all_regions, region_map = build_regions_from_tokens(tokens, existing=all_regions)
        tariffs_df = build_tariffs_from_table(tidy, region_map, service_name, startdate, enddate, currency)
        if not tariffs_df.empty:
            all_tariffs.append(tariffs_df)
    if not all_tariffs:
        logger.error("No tariffs extracted from workbook")
        return
    tariffs_df = pd.concat(all_tariffs, ignore_index=True)
    surcharges_df = empty_surcharges()
    if not validate_output(all_regions, tariffs_df):
        logger.warning("Validation failed; output may contain inconsistencies")
    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        all_regions.to_excel(writer, index=False, sheet_name="Regions")
        tariffs_df.to_excel(writer, index=False, sheet_name="Tariffs")
        surcharges_df.to_excel(writer, index=False, sheet_name="Surcharges")
    logger.info("Conversion completed: %s", out_file)

if __name__ == "__main__":
    app()

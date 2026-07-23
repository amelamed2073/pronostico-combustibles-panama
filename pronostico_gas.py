from __future__ import annotations

from html import unescape
from io import BytesIO
import os
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Optional
import re
import sys
import unicodedata
import warnings

PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_CACHE_DIR = PROJECT_DIR / ".cache"
MPL_CACHE_DIR = LOCAL_CACHE_DIR / "matplotlib"
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import pearsonr
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import coint, grangercausalitytests
from streamlit.runtime.scriptrunner import get_script_run_ctx


BASE_DIR = PROJECT_DIR
DEFAULT_INPUT_FILE_NAMES = [
    "Precio historico de la gasolina (2015-2026) (2).xlsx",
]
DEFAULT_SECTORAL_IPC_NAME = "ipc_sectorial_panama_2017_2025.xlsx"
DEFAULT_OUTPUT_NAME = "pronostico_combustibles_panama_web.xlsx"
IPC_SOURCE_URL = "https://www.inec.gob.pa/publicaciones/Default3.aspx?ID_CATEGORIA=4&ID_PUBLICACION=1396&ID_SUBCATEGORIA=82"
WORLD_BANK_API_BASE = "https://api.worldbank.org/v2"
WORLD_BANK_DATA_URL = "https://data.worldbank.org"
EIA_BRENT_SOURCE_URL = "https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?f=m&n=PET&s=RBRTE"
OFFICIAL_FUEL_POSTS_API = "https://www.energia.gob.pa/wp-json/wp/v2/posts"
PRICE_UNIT = "B/. / litro"
MIN_REASONABLE_PRICE = 0.05
MAX_REASONABLE_PRICE = 10.0
MIN_TARGET_MEDIAN_PRICE = 0.20
MAX_TARGET_MEDIAN_PRICE = 1.80
FUEL_NAMES = {
    "gasolina_95": "Gasolina 95",
    "gasolina_91": "Gasolina 91",
    "diesel": "Diésel",
}
FUEL_CANDIDATES = {
    "gasolina_95": ["gasolina_95", "gasolina95", "gasolina_95_octanos", "95_octanos", "95"],
    "gasolina_91": ["gasolina_91", "gasolina91", "gasolina_91_octanos", "91_octanos", "91"],
    "diesel": ["diesel", "diesel_bajo_azufre", "ultra_bajo_azufre", "diessel"],
}
IPC_SECTORS = {
    "ipc_general": "TOTAL",
    "ipc_alimentos": "ALIMENTOS Y BEBIDAS NO ALCOHÓLICAS",
    "ipc_vivienda": "VIVIENDA, AGUA, ELECTRICIDAD Y GAS",
    "ipc_transporte": "TRANSPORTE",
    "ipc_restaurantes": "RESTAURANTES Y HOTELES",
    "ipc_bienes_servicios": "BIENES Y SERVICIOS DIVERSOS",
}
IPC_SECTOR_NAMES = {
    "ipc_general": "IPC general",
    "ipc_alimentos": "Alimentos y bebidas no alcohólicas",
    "ipc_vivienda": "Vivienda, agua, electricidad y gas",
    "ipc_transporte": "Transporte",
    "ipc_restaurantes": "Restaurantes y hoteles",
    "ipc_bienes_servicios": "Bienes y servicios diversos",
}
REGIONAL_COUNTRIES = {
    "PAN": "Panamá",
    "CRI": "Costa Rica",
    "DOM": "República Dominicana",
}
REGIONAL_CURRENCY_LABELS = {
    "PAN": "Balboa/USD",
    "CRI": "Colón/USD",
    "DOM": "Peso dominicano/USD",
}
REGIONAL_INDICATORS = {
    "LP.LPI.OVRL.XQ": "Índice de desempeño logístico (1–5)",
    "TM.VAL.FUEL.ZS.UN": "Importaciones de combustibles (% de mercancías)",
    "PA.NUS.FCRF": "Tipo de cambio oficial (moneda local/USD)",
    "FP.CPI.TOTL.ZG": "Inflación general anual (%)",
}
REGIONAL_FALLBACK_ROWS = [
    {"codigo_pais": "PAN", "indicador": "LP.LPI.OVRL.XQ", "anio": 2022, "valor": 3.1},
    {"codigo_pais": "CRI", "indicador": "LP.LPI.OVRL.XQ", "anio": 2022, "valor": 2.9},
    {"codigo_pais": "DOM", "indicador": "LP.LPI.OVRL.XQ", "anio": 2022, "valor": 2.6},
    {"codigo_pais": "PAN", "indicador": "TM.VAL.FUEL.ZS.UN", "anio": 2024, "valor": 19.9706022981},
    {"codigo_pais": "CRI", "indicador": "TM.VAL.FUEL.ZS.UN", "anio": 2024, "valor": 10.2627314710},
    {"codigo_pais": "DOM", "indicador": "TM.VAL.FUEL.ZS.UN", "anio": 2024, "valor": 16.6433167112},
    {"codigo_pais": "PAN", "indicador": "FP.CPI.TOTL.ZG", "anio": 2024, "valor": 0.6932255510},
    {"codigo_pais": "CRI", "indicador": "FP.CPI.TOTL.ZG", "anio": 2024, "valor": -0.4128530010},
    {"codigo_pais": "DOM", "indicador": "FP.CPI.TOTL.ZG", "anio": 2024, "valor": 3.3022333895},
    {"codigo_pais": "PAN", "indicador": "PA.NUS.FCRF", "anio": 2023, "valor": 1.0},
    {"codigo_pais": "CRI", "indicador": "PA.NUS.FCRF", "anio": 2023, "valor": 544.0507755056},
    {"codigo_pais": "DOM", "indicador": "PA.NUS.FCRF", "anio": 2023, "valor": 56.1576},
    {"codigo_pais": "PAN", "indicador": "PA.NUS.FCRF", "anio": 2024, "valor": 1.0},
    {"codigo_pais": "CRI", "indicador": "PA.NUS.FCRF", "anio": 2024, "valor": 515.1097719214},
    {"codigo_pais": "DOM", "indicador": "PA.NUS.FCRF", "anio": 2024, "valor": 59.5651333333},
]
MONTH_NUMBERS = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}


def normalize_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", text.lower())).strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_name(col) for col in out.columns]
    return out


def find_column(columns: Iterable[str], candidates: list[str]) -> str:
    cols = list(columns)
    normalized_candidates = [normalize_name(candidate) for candidate in candidates]
    for candidate in normalized_candidates:
        if candidate in cols:
            return candidate

    for candidate in normalized_candidates:
        pattern = re.compile(rf"(?:^|_){re.escape(candidate)}(?:_|$)")
        for col in cols:
            if pattern.search(col):
                return col
    raise KeyError(f"No se encontró ninguna columna compatible con: {candidates}")


def try_find_column(columns: Iterable[str], candidates: list[str]) -> Optional[str]:
    try:
        return find_column(columns, candidates)
    except KeyError:
        return None


def parse_number(value: Any) -> float:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)

    raw = re.sub(r"[^0-9,\.\-+]", "", str(value).strip())
    if not raw or raw in {"-", "+"}:
        return np.nan

    comma_count = raw.count(",")
    dot_count = raw.count(".")
    if comma_count and dot_count:
        decimal_sep = "," if raw.rfind(",") > raw.rfind(".") else "."
        thousands_sep = "." if decimal_sep == "," else ","
        raw = raw.replace(thousands_sep, "").replace(decimal_sep, ".")
    elif comma_count:
        parts = raw.split(",")
        raw = "".join(parts) if comma_count > 1 and len(parts[-1]) == 3 else ".".join(parts)
    elif dot_count > 1:
        parts = raw.split(".")
        raw = "".join(parts) if len(parts[-1]) == 3 else "".join(parts[:-1]) + "." + parts[-1]

    try:
        return float(raw)
    except ValueError:
        return np.nan


def to_numeric_series(series: pd.Series) -> pd.Series:
    return series.map(parse_number).astype(float)


def parse_dates_safely(series: pd.Series, dayfirst: bool = False) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    numeric_version = pd.to_numeric(series, errors="coerce")
    if numeric_version.notna().mean() >= 0.8:
        plausible_excel = numeric_version.between(1, 100_000)
        converted = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        converted.loc[plausible_excel] = pd.to_datetime(
            numeric_version.loc[plausible_excel], errors="coerce", unit="D", origin="1899-12-30"
        )
        return converted

    text_series = series.astype(str).str.strip()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(text_series, errors="coerce", dayfirst=dayfirst)


def extract_price_candidates(value: Any) -> list[float]:
    values: list[float] = []
    for match in re.findall(r"(?<!\d)[+-]?\d+(?:[\.,]\d+)+(?!\d)", str(value)):
        parsed = parse_number(match)
        if np.isfinite(parsed) and MIN_REASONABLE_PRICE <= parsed <= MAX_REASONABLE_PRICE:
            values.append(float(parsed))
    return values


def detect_fuel(text: str) -> Optional[str]:
    normalized = normalize_name(text)
    if "gasolina" in normalized and re.search(r"(?:^|_)95(?:_|$)", normalized):
        return "gasolina_95"
    if "gasolina" in normalized and re.search(r"(?:^|_)91(?:_|$)", normalized):
        return "gasolina_91"
    if "diesel" in normalized or "diessel" in normalized:
        return "diesel"
    return None


def detect_live_prices_from_table(df: pd.DataFrame) -> dict[str, float]:
    detected: dict[str, float] = {}
    if df.empty:
        return detected

    table = df.copy()
    table.columns = [normalize_name(" ".join(map(str, col)) if isinstance(col, tuple) else col) for col in table.columns]
    price_columns = [col for col in table.columns if any(token in col for token in ("precio", "price", "balboa", "litro"))]

    for _, row in table.iterrows():
        row_text = " ".join(str(value) for value in row.tolist() if pd.notna(value))
        fuel = detect_fuel(row_text)
        if fuel is None:
            continue

        candidates: list[float] = []
        for col in price_columns:
            candidates.extend(extract_price_candidates(row[col]))
        if not candidates:
            for value in row.tolist():
                candidates.extend(extract_price_candidates(value))
        if candidates:
            detected.setdefault(fuel, candidates[0])
    return detected


def normalize_live_price_token(token: str) -> float:
    token = str(token).strip()
    direct = parse_number(token)
    if np.isfinite(direct) and MIN_REASONABLE_PRICE <= direct <= MAX_REASONABLE_PRICE:
        return float(direct)

    digits = re.sub(r"[^0-9]", "", token)
    if len(digits) == 4:
        inferred = int(digits) / 1000
        if MIN_REASONABLE_PRICE <= inferred <= MAX_REASONABLE_PRICE:
            return float(inferred)
    return np.nan


def fetch_latest_official_fuel_post() -> dict[str, str]:
    response = requests.get(
        OFFICIAL_FUEL_POSTS_API,
        params={"search": "combustibles", "per_page": 10},
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0 (compatible; FuelForecastPrototype/4.0)"},
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("La API oficial de comunicados no devolvió una lista de publicaciones.")

    for post in payload:
        title = unescape(str(post.get("title", {}).get("rendered", "")))
        slug = str(post.get("slug", ""))
        if "actualizacion-de-los-precios-de-los-combustibles" not in slug:
            continue
        html = unescape(str(post.get("content", {}).get("rendered", "")))
        image_match = re.search(r'href="([^"]*Precio-por-localidad[^"]+)"', html, flags=re.IGNORECASE)
        if not image_match:
            image_match = re.search(r'src="([^"]*Precio-por-localidad[^"]+)"', html, flags=re.IGNORECASE)
        if not image_match:
            continue
        return {
            "title": title,
            "post_url": str(post.get("link", "")),
            "image_url": image_match.group(1).replace("http://", "https://"),
        }
    raise ValueError("No se encontró una publicación oficial reciente con la tabla de precios por localidad.")


def extract_panama_prices_from_official_image(image_bytes: bytes) -> dict[str, float]:
    if shutil.which("tesseract") is None:
        raise ValueError("Tesseract no está disponible para leer la tabla oficial por imagen.")

    with Image.open(BytesIO(image_bytes)) as raw_image:
        image = raw_image.convert("RGB")
    width, height = image.size
    crop = image.crop((0, int(height * 0.1875), width, int(height * 0.2657)))
    crop = ImageOps.grayscale(crop).resize((crop.width * 5, crop.height * 5))
    crop = crop.filter(ImageFilter.SHARPEN)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        crop.save(temp_path)
        result = subprocess.run(
            ["tesseract", str(temp_path), "stdout", "--psm", "6"],
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    panama_line = next(
        (
            line.strip()
            for line in result.stdout.splitlines()
            if "panama" in normalize_name(line) and "colon" in normalize_name(line)
        ),
        "",
    )
    if not panama_line:
        raise ValueError("No se pudo leer la fila de Panamá / Colón en la tabla oficial.")

    tokens = re.findall(r"\d+(?:\.\d+)?", panama_line)
    if len(tokens) < 3:
        raise ValueError("La fila de Panamá / Colón no contiene tres precios legibles.")
    values = [normalize_live_price_token(token) for token in tokens[-3:]]
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Los precios detectados en la tabla oficial no son válidos.")
    return {
        "gasolina_95": float(values[0]),
        "gasolina_91": float(values[1]),
        "diesel": float(values[2]),
    }


def fetch_live_panama_prices_from_official_post() -> tuple[dict[str, float], str, str]:
    post = fetch_latest_official_fuel_post()
    response = requests.get(
        post["image_url"],
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0 (compatible; FuelForecastPrototype/4.0)"},
    )
    response.raise_for_status()
    prices = extract_panama_prices_from_official_image(response.content)
    return prices, post["post_url"], post["title"]


def fetch_live_panama_prices_from_tables() -> tuple[dict[str, float], Optional[str]]:
    candidate_urls = [
        "https://www.energia.gob.pa/precios-de-combustibles/",
        "https://energia.gob.pa/precios-de-combustibles/",
    ]
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FuelForecastPrototype/1.0)"}

    for url in candidate_urls:
        try:
            response = requests.get(url, timeout=12, headers=headers)
            response.raise_for_status()
            tables = pd.read_html(BytesIO(response.content))
        except (requests.RequestException, ValueError, ImportError):
            continue

        merged: dict[str, float] = {}
        for table in tables:
            merged.update(detect_live_prices_from_table(table))
        if merged:
            return merged, url
    return {}, None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_panama_prices() -> tuple[dict[str, float], Optional[str], str]:
    try:
        prices, source_url, source_label = fetch_live_panama_prices_from_official_post()
        return prices, source_url, source_label
    except (requests.RequestException, ValueError, OSError, subprocess.SubprocessError):
        pass

    prices, source_url = fetch_live_panama_prices_from_tables()
    if prices:
        return prices, source_url, "Fuente oficial tabular"
    return {}, None, ""


def load_sheet_with_header_detection(file_obj: Any, sheet_name: str) -> tuple[pd.DataFrame, int]:
    for header_row in range(8):
        temp = normalize_columns(pd.read_excel(file_obj, sheet_name=sheet_name, header=header_row))
        if temp.empty:
            continue
        date_col = try_find_column(temp.columns, ["fecha", "mes", "periodo", "vigencia"])
        fuel_cols = [try_find_column(temp.columns, candidates) for candidates in FUEL_CANDIDATES.values()]
        if date_col and any(fuel_cols):
            return temp, header_row
    return normalize_columns(pd.read_excel(file_obj, sheet_name=sheet_name)), 0


def monthly_time_weighted_average(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    result: list[pd.Series] = []
    for col in value_cols:
        observations = df[["fecha", col]].dropna().sort_values("fecha")
        if observations.empty:
            continue

        conflicts = observations.groupby("fecha")[col].agg(["min", "max"])
        conflicts = conflicts[(conflicts["max"] - conflicts["min"]).abs() > 0.02]
        if not conflicts.empty:
            dates = ", ".join(date.strftime("%Y-%m-%d") for date in conflicts.index[:5])
            raise ValueError(f"Hay precios incompatibles para {FUEL_NAMES[col]} en la misma fecha: {dates}.")

        effective = observations.groupby("fecha")[col].median().sort_index()
        daily_index = pd.date_range(effective.index.min(), effective.index.max(), freq="D")
        daily = effective.reindex(daily_index).ffill()
        monthly = daily.resample("MS").mean().rename(col)
        result.append(monthly)

    if not result:
        raise ValueError("No hay series numéricas utilizables.")
    return pd.concat(result, axis=1).rename_axis("periodo").reset_index()


def load_fuel_data(input_sources: list[Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    logs: list[dict[str, Any]] = []

    for source_order, source in enumerate(input_sources):
        source_name = getattr(source, "name", str(source))
        xls = pd.ExcelFile(source)
        for sheet in xls.sheet_names:
            try:
                raw, detected_header = load_sheet_with_header_detection(source, sheet)
            except (ValueError, TypeError) as exc:
                logs.append({"archivo": source_name, "hoja": sheet, "estado": "omitida", "motivo": str(exc), "filas_cargadas": 0})
                continue

            date_col = try_find_column(raw.columns, ["fecha", "mes", "periodo", "vigencia"])
            fuel_map: dict[str, str] = {}
            if date_col:
                for standard_name, candidates in FUEL_CANDIDATES.items():
                    detected = try_find_column(raw.columns, candidates)
                    if detected and detected != date_col:
                        fuel_map[detected] = standard_name

            if not date_col or not fuel_map:
                logs.append({"archivo": source_name, "hoja": sheet, "estado": "omitida", "motivo": "Faltan fecha o combustibles", "filas_cargadas": 0})
                continue

            temp = raw[[date_col, *fuel_map]].copy().rename(columns={date_col: "fecha", **fuel_map})
            temp["fecha"] = parse_dates_safely(temp["fecha"], dayfirst=True)
            for col in fuel_map.values():
                temp[col] = to_numeric_series(temp[col])
                invalid = temp[col].notna() & ~temp[col].between(MIN_REASONABLE_PRICE, MAX_REASONABLE_PRICE)
                temp.loc[invalid, col] = np.nan

            numeric_cols = list(dict.fromkeys(fuel_map.values()))
            incompatible_cols: list[str] = []
            for col in numeric_cols:
                median_price = temp[col].median(skipna=True)
                if pd.notna(median_price) and not MIN_TARGET_MEDIAN_PRICE <= median_price <= MAX_TARGET_MEDIAN_PRICE:
                    incompatible_cols.append(col)
                    temp[col] = np.nan

            temp = temp.dropna(subset=["fecha"]).dropna(subset=numeric_cols, how="all")
            if temp.empty:
                reason = (
                    f"Escala incompatible con {PRICE_UNIT}: {', '.join(FUEL_NAMES[col] for col in incompatible_cols)}"
                    if incompatible_cols
                    else "Sin registros útiles"
                )
                logs.append({"archivo": source_name, "hoja": sheet, "estado": "omitida", "motivo": reason, "filas_cargadas": 0})
                continue

            temp["_source_order"] = source_order
            frames.append(temp)
            logs.append({
                "archivo": source_name,
                "hoja": sheet,
                "estado": "incluida",
                "motivo": (
                    "OK; series omitidas por escala incompatible: "
                    + ", ".join(FUEL_NAMES[col] for col in incompatible_cols)
                    if incompatible_cols
                    else "OK"
                ),
                "header_detectado": detected_header,
                "filas_cargadas": len(temp),
            })

    if not frames:
        raise ValueError("No se pudieron cargar datos válidos de los archivos de combustibles.")

    df = pd.concat(frames, ignore_index=True).sort_values(["fecha", "_source_order"])
    value_cols = [col for col in FUEL_NAMES if col in df.columns]
    monthly = monthly_time_weighted_average(df, value_cols)
    return monthly, pd.DataFrame(logs)


@st.cache_data(show_spinner=False)
def load_sectoral_ipc(source: Any) -> pd.DataFrame:
    """Carga los índices mensuales por división desde el Cuadro 2 oficial del INEC."""
    xls = pd.ExcelFile(source)
    records: list[dict[str, Any]] = []

    for sheet_name in xls.sheet_names:
        year_text = str(sheet_name).strip()
        if not year_text.isdigit():
            continue
        year = int(year_text)
        raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        if raw.shape[0] < 6 or raw.shape[1] < 15:
            continue

        normalized_labels = raw.iloc[:, 0].map(normalize_name)
        row_by_sector: dict[str, int] = {}
        for sector, official_label in IPC_SECTORS.items():
            matches = normalized_labels[normalized_labels == normalize_name(official_label)].index
            if len(matches):
                row_by_sector[sector] = int(matches[0])

        if "ipc_general" not in row_by_sector or len(row_by_sector) < 3:
            continue

        for column_index in range(14, raw.shape[1]):
            month_name = normalize_name(raw.iat[4, column_index])
            month = MONTH_NUMBERS.get(month_name)
            if month is None:
                continue
            row: dict[str, Any] = {"periodo": pd.Timestamp(year=year, month=month, day=1)}
            for sector, row_index in row_by_sector.items():
                row[sector] = parse_number(raw.iat[row_index, column_index])
            records.append(row)

    if not records:
        raise ValueError("El archivo de IPC no coincide con el formato sectorial esperado del INEC.")

    result = pd.DataFrame(records).sort_values("periodo").drop_duplicates("periodo", keep="last")
    sector_cols = [col for col in IPC_SECTORS if col in result.columns]
    result = result.dropna(subset=sector_cols, how="all").reset_index(drop=True)
    if len(result) < 36:
        raise ValueError("Se requieren al menos 36 meses de IPC sectorial para evaluar rezagos.")
    return result


def _ols_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    train_x = np.column_stack([np.ones(len(train)), *[train[col].to_numpy() for col in features]])
    test_x = np.column_stack([np.ones(len(test)), *[test[col].to_numpy() for col in features]])
    coefficients, _, _, _ = np.linalg.lstsq(train_x, train["objetivo"].to_numpy(), rcond=None)
    return test_x @ coefficients


def run_granger_test(
    pair: pd.DataFrame,
    target_col: str,
    driver_col: str,
    max_lag: int = 6,
) -> dict[str, Any]:
    cleaned = pair[[target_col, driver_col]].dropna().copy()
    cleaned = cleaned[np.isfinite(cleaned[target_col]) & np.isfinite(cleaned[driver_col])]
    if len(cleaned) < max(24, (max_lag + 1) * 4):
        return {
            "mejor_rezago": np.nan,
            "p_valor_min": np.nan,
            "estadistico_f": np.nan,
            "rezagos_significativos": 0,
            "observaciones": len(cleaned),
            "significativo_5_pct": False,
        }

    if np.isclose(cleaned[target_col].std(ddof=0), 0) or np.isclose(cleaned[driver_col].std(ddof=0), 0):
        return {
            "mejor_rezago": np.nan,
            "p_valor_min": np.nan,
            "estadistico_f": np.nan,
            "rezagos_significativos": 0,
            "observaciones": len(cleaned),
            "significativo_5_pct": False,
        }

    granger_input = cleaned[[target_col, driver_col]].astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = grangercausalitytests(granger_input, maxlag=max_lag, verbose=False)

    lag_rows: list[dict[str, Any]] = []
    for lag, lag_result in result.items():
        f_stat, p_value, _, _ = lag_result[0]["ssr_ftest"]
        lag_rows.append(
            {
                "lag": lag,
                "f_stat": float(f_stat),
                "p_value": float(p_value),
            }
        )

    best = min(lag_rows, key=lambda row: row["p_value"])
    significant_count = sum(row["p_value"] < 0.05 for row in lag_rows)
    return {
        "mejor_rezago": int(best["lag"]),
        "p_valor_min": float(best["p_value"]),
        "estadistico_f": float(best["f_stat"]),
        "rezagos_significativos": int(significant_count),
        "observaciones": len(cleaned),
        "significativo_5_pct": bool(best["p_value"] < 0.05),
    }


def run_cointegration_test(
    pair: pd.DataFrame,
    left_col: str,
    right_col: str,
    max_lag: int = 6,
) -> dict[str, Any]:
    cleaned = pair[[left_col, right_col]].dropna().copy()
    cleaned = cleaned[np.isfinite(cleaned[left_col]) & np.isfinite(cleaned[right_col])]
    cleaned = cleaned[(cleaned[left_col] > 0) & (cleaned[right_col] > 0)]
    if len(cleaned) < max(36, (max_lag + 1) * 6):
        return {
            "estadistico": np.nan,
            "p_valor": np.nan,
            "critico_5_pct": np.nan,
            "cointegrada_5_pct": False,
            "observaciones": len(cleaned),
        }

    left_log = np.log(cleaned[left_col].astype(float))
    right_log = np.log(cleaned[right_col].astype(float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p_value, critical_values = coint(
            left_log,
            right_log,
            trend="c",
            maxlag=max_lag,
            autolag="aic",
        )

    critical_5 = float(critical_values[1]) if len(critical_values) >= 2 else np.nan
    return {
        "estadistico": float(stat),
        "p_valor": float(p_value),
        "critico_5_pct": critical_5,
        "cointegrada_5_pct": bool(np.isfinite(stat) and np.isfinite(critical_5) and stat < critical_5),
        "observaciones": len(cleaned),
    }


def analyze_sectoral_ipc(
    ipc_df: pd.DataFrame,
    monthly_fuel_df: pd.DataFrame,
    fuel_col: str,
    max_lag: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evalúa transmisión descriptiva y capacidad predictiva sin afirmar causalidad."""
    if fuel_col not in monthly_fuel_df.columns:
        raise ValueError(f"No existe la serie {fuel_col} en los datos de combustibles.")

    merged = (
        ipc_df.merge(monthly_fuel_df[["periodo", fuel_col]], on="periodo", how="inner")
        .sort_values("periodo")
        .reset_index(drop=True)
    )
    merged["combustible_var_pct"] = merged[fuel_col].pct_change(fill_method=None) * 100

    sector_cols = [col for col in IPC_SECTORS if col in merged.columns]
    for sector in sector_cols:
        merged[f"{sector}_var_pct"] = merged[sector].pct_change(fill_method=None) * 100

    ccf_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for sector in sector_cols:
        target_col = f"{sector}_var_pct"
        base = merged[["periodo", "combustible_var_pct", target_col]].rename(columns={target_col: "objetivo"})
        level_pair = merged[["periodo", fuel_col, sector]].rename(
            columns={fuel_col: "combustible_nivel", sector: "ipc_nivel"}
        )

        sector_ccf: list[dict[str, Any]] = []
        for lag in range(max_lag + 1):
            pair = pd.DataFrame(
                {
                    "combustible": base["combustible_var_pct"].shift(lag),
                    "objetivo": base["objetivo"],
                }
            ).dropna()
            if len(pair) < 12 or pair["combustible"].std() == 0 or pair["objetivo"].std() == 0:
                correlation, p_value = np.nan, np.nan
            else:
                correlation, p_value = pearsonr(pair["combustible"], pair["objetivo"])
            ccf_row = {
                "sector": sector,
                "sector_nombre": IPC_SECTOR_NAMES[sector],
                "combustible": fuel_col,
                "rezago_meses": lag,
                "correlacion": float(correlation) if np.isfinite(correlation) else np.nan,
                "p_valor": float(p_value) if np.isfinite(p_value) else np.nan,
                "observaciones": len(pair),
                "banda_95_aprox": 1.96 / np.sqrt(len(pair)) if len(pair) else np.nan,
            }
            sector_ccf.append(ccf_row)
            ccf_rows.append(ccf_row)

        valid_ccf = [row for row in sector_ccf if np.isfinite(row["correlacion"])]
        best_ccf = max(valid_ccf, key=lambda row: abs(row["correlacion"]))

        granger_forward = run_granger_test(base, "objetivo", "combustible_var_pct", max_lag=min(6, max_lag))
        granger_reverse = run_granger_test(base, "combustible_var_pct", "objetivo", max_lag=min(6, max_lag))
        cointegration = run_cointegration_test(level_pair, "ipc_nivel", "combustible_nivel", max_lag=min(6, max_lag))

        model_data = base.copy()
        model_data["objetivo_lag_1"] = model_data["objetivo"].shift(1)
        for lag in range(max_lag + 1):
            model_data[f"combustible_lag_{lag}"] = model_data["combustible_var_pct"].shift(lag)
        required = ["objetivo", "objetivo_lag_1", *[f"combustible_lag_{lag}" for lag in range(max_lag + 1)]]
        model_data = model_data.dropna(subset=required).reset_index(drop=True)
        if len(model_data) < 36:
            raise ValueError(f"{IPC_SECTOR_NAMES[sector]} no tiene suficientes observaciones comparables.")

        final_holdout = min(12, max(6, len(model_data) // 5))
        training_pool = model_data.iloc[:-final_holdout]
        final_test = model_data.iloc[-final_holdout:]
        selection_holdout = min(12, max(6, len(training_pool) // 5))
        selection_train = training_pool.iloc[:-selection_holdout]
        selection_test = training_pool.iloc[-selection_holdout:]

        lag_validation_scores: list[tuple[int, float]] = []
        for lag in range(max_lag + 1):
            feature = f"combustible_lag_{lag}"
            predictions = _ols_predict(selection_train, selection_test, ["objetivo_lag_1", feature])
            score = float(np.mean(np.abs(selection_test["objetivo"].to_numpy() - predictions)))
            lag_validation_scores.append((lag, score))
        selected_lag, selection_mae = min(lag_validation_scores, key=lambda item: item[1])

        baseline_predictions = _ols_predict(training_pool, final_test, ["objetivo_lag_1"])
        extended_predictions = _ols_predict(
            training_pool,
            final_test,
            ["objetivo_lag_1", f"combustible_lag_{selected_lag}"],
        )
        baseline_mae = float(np.mean(np.abs(final_test["objetivo"].to_numpy() - baseline_predictions)))
        extended_mae = float(np.mean(np.abs(final_test["objetivo"].to_numpy() - extended_predictions)))
        improvement = (baseline_mae - extended_mae) / baseline_mae * 100 if baseline_mae else np.nan

        summary_rows.append(
            {
                "sector": sector,
                "sector_nombre": IPC_SECTOR_NAMES[sector],
                "combustible": fuel_col,
                "rezago_ccf_meses": best_ccf["rezago_meses"],
                "correlacion_ccf": round(best_ccf["correlacion"], 4),
                "p_valor_ccf": round(best_ccf["p_valor"], 4),
                "rezago_predictivo_meses": selected_lag,
                "mae_seleccion": round(selection_mae, 4),
                "mae_base": round(baseline_mae, 4),
                "mae_con_combustible": round(extended_mae, 4),
                "mejora_mae_pct": round(float(improvement), 2) if np.isfinite(improvement) else np.nan,
                "meses_prueba_final": final_holdout,
                "periodo_inicio": merged["periodo"].min(),
                "periodo_fin": merged["periodo"].max(),
                "granger_combustible_ipc_rezago": granger_forward["mejor_rezago"],
                "granger_combustible_ipc_p_valor": (
                    round(granger_forward["p_valor_min"], 4)
                    if np.isfinite(granger_forward["p_valor_min"])
                    else np.nan
                ),
                "granger_combustible_ipc_f_stat": (
                    round(granger_forward["estadistico_f"], 4)
                    if np.isfinite(granger_forward["estadistico_f"])
                    else np.nan
                ),
                "granger_combustible_ipc_significativo": granger_forward["significativo_5_pct"],
                "granger_ipc_combustible_rezago": granger_reverse["mejor_rezago"],
                "granger_ipc_combustible_p_valor": (
                    round(granger_reverse["p_valor_min"], 4)
                    if np.isfinite(granger_reverse["p_valor_min"])
                    else np.nan
                ),
                "granger_ipc_combustible_significativo": granger_reverse["significativo_5_pct"],
                "cointegracion_estadistico": (
                    round(cointegration["estadistico"], 4)
                    if np.isfinite(cointegration["estadistico"])
                    else np.nan
                ),
                "cointegracion_p_valor": (
                    round(cointegration["p_valor"], 4)
                    if np.isfinite(cointegration["p_valor"])
                    else np.nan
                ),
                "cointegracion_critico_5_pct": (
                    round(cointegration["critico_5_pct"], 4)
                    if np.isfinite(cointegration["critico_5_pct"])
                    else np.nan
                ),
                "cointegracion_5_pct": cointegration["cointegrada_5_pct"],
                "observaciones_cointegracion": cointegration["observaciones"],
            }
        )

    return merged, pd.DataFrame(summary_rows), pd.DataFrame(ccf_rows)


def create_ipc_lag_figure(ccf_df: pd.DataFrame, sector: str):
    data = ccf_df[ccf_df["sector"] == sector].sort_values("rezago_meses")
    if data.empty:
        raise ValueError("No hay resultados de rezagos para el sector seleccionado.")
    best_index = data["correlacion"].abs().idxmax()
    colors = ["#2f78bd" if idx == best_index else "#a8b6c8" for idx in data.index]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(data["rezago_meses"], data["correlacion"], color=colors)
    ax.plot(data["rezago_meses"], data["banda_95_aprox"], color="#ef6a4c", linewidth=1.4)
    ax.plot(data["rezago_meses"], -data["banda_95_aprox"], color="#ef6a4c", linewidth=1.4)
    ax.axhline(0, color="#667085", linewidth=0.9)
    ax.set(
        title=f"Cómo se transmite {FUEL_NAMES[data['combustible'].iloc[0]]} hacia {IPC_SECTOR_NAMES[sector]}",
        xlabel="Meses en que tarda el efecto del combustible",
        ylabel="Relación temporal de las variaciones",
    )
    ax.set_xticks(data["rezago_meses"])
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def describe_ipc_relationship(correlation: float, p_value: float) -> str:
    if not np.isfinite(correlation):
        return "Sin señal"
    strength = abs(float(correlation))
    if np.isfinite(p_value) and p_value < 0.05 and strength >= 0.4:
        return "Alta"
    if np.isfinite(p_value) and p_value < 0.10 and strength >= 0.2:
        return "Media"
    if strength >= 0.1:
        return "Baja"
    return "No concluyente"


def describe_ipc_confidence(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "Sin dato"
    if p_value < 0.01:
        return "Alta"
    if p_value < 0.05:
        return "Media"
    if p_value < 0.10:
        return "Baja"
    return "Muy baja"


def describe_ipc_predictive_effect(improvement: float) -> str:
    if not np.isfinite(improvement):
        return "Sin dato"
    if improvement > 5:
        return "Sí mejora"
    if improvement > 0:
        return "Mejora leve"
    if improvement > -5:
        return "No mejora"
    return "Empeora"


def describe_ipc_direction(correlation: float) -> str:
    if not np.isfinite(correlation):
        return "Sin dirección clara"
    return "Positiva" if correlation >= 0 else "Negativa"


def render_ipc_sector_cards(summary_df: pd.DataFrame, selected_sector: str) -> None:
    ordered_df = summary_df.copy()
    ordered_df["_selected_first"] = ordered_df["sector"] != selected_sector
    ordered_df = ordered_df.sort_values(["_selected_first", "sector_nombre"]).drop(columns="_selected_first")
    card_columns = st.columns(2)
    for index, (_, row) in enumerate(ordered_df.iterrows()):
        relationship = describe_ipc_relationship(row["correlacion_ccf"], row["p_valor_ccf"])
        confidence = describe_ipc_confidence(row["p_valor_ccf"])
        predictive = describe_ipc_predictive_effect(row["mejora_mae_pct"])
        direction = describe_ipc_direction(row["correlacion_ccf"])
        lag_text = f"{int(row['rezago_predictivo_meses'])} meses"
        long_term = "Sí" if bool(row["cointegracion_5_pct"]) else "No clara"
        predictive_detail = (
            f"{predictive} ({float(row['mejora_mae_pct']):+.1f}% en MAE)"
            if np.isfinite(row["mejora_mae_pct"])
            else predictive
        )
        with card_columns[index % len(card_columns)]:
            with st.container(border=True):
                if row["sector"] == selected_sector:
                    st.caption("Sector seleccionado")
                st.markdown(f"**{row['sector_nombre']}**")
                st.write(f"Relación: **{relationship}**")
                st.write(f"Dirección: **{direction}**")
                st.write(f"Tiempo de efecto: **{lag_text}**")
                st.write(f"Pronóstico: **{predictive_detail}**")
                st.caption(f"Confianza: {confidence} · Largo plazo: {long_term}")


@st.cache_data(ttl=86_400, show_spinner=False)
def fetch_world_bank_indicator(indicator: str, start_year: int = 2018) -> pd.DataFrame:
    """Consulta una serie anual comparable para los tres países del bloque regional."""
    if indicator not in REGIONAL_INDICATORS:
        raise ValueError(f"Indicador regional no permitido: {indicator}")

    country_codes = ";".join(REGIONAL_COUNTRIES)
    url = f"{WORLD_BANK_API_BASE}/country/{country_codes}/indicator/{indicator}"
    response = requests.get(
        url,
        params={
            "format": "json",
            "per_page": 500,
            "date": f"{start_year}:{pd.Timestamp.today().year}",
        },
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0 (compatible; FuelForecastPrototype/2.0)"},
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        raise ValueError("La API del Banco Mundial no devolvió observaciones válidas.")

    records: list[dict[str, Any]] = []
    for observation in payload[1]:
        value = observation.get("value")
        country_code = observation.get("countryiso3code")
        year = observation.get("date")
        if country_code not in REGIONAL_COUNTRIES or value is None or not str(year).isdigit():
            continue
        records.append(
            {
                "codigo_pais": country_code,
                "pais": REGIONAL_COUNTRIES[country_code],
                "indicador": indicator,
                "indicador_nombre": REGIONAL_INDICATORS[indicator],
                "anio": int(year),
                "valor": float(value),
            }
        )
    if not records:
        raise ValueError(f"No hay observaciones regionales para {indicator}.")
    return pd.DataFrame(records).sort_values(["codigo_pais", "anio"]).reset_index(drop=True)


def regional_fallback_data() -> pd.DataFrame:
    fallback = pd.DataFrame(REGIONAL_FALLBACK_ROWS)
    fallback["pais"] = fallback["codigo_pais"].map(REGIONAL_COUNTRIES)
    fallback["indicador_nombre"] = fallback["indicador"].map(REGIONAL_INDICATORS)
    return fallback[["codigo_pais", "pais", "indicador", "indicador_nombre", "anio", "valor"]]


@st.cache_data(ttl=86_400, show_spinner=False)
def load_regional_benchmark() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Construye un benchmark regional; usa una instantánea oficial si la API no responde."""
    frames: list[pd.DataFrame] = []
    source_status = "API del Banco Mundial"
    try:
        for indicator in REGIONAL_INDICATORS:
            frames.append(fetch_world_bank_indicator(indicator))
        raw = pd.concat(frames, ignore_index=True)
    except (requests.RequestException, ValueError, TypeError):
        raw = regional_fallback_data()
        source_status = "instantánea oficial de respaldo (consulta 2026-07-15)"

    rows: list[dict[str, Any]] = []
    for country_code, country_name in REGIONAL_COUNTRIES.items():
        country_data = raw[raw["codigo_pais"] == country_code]
        row: dict[str, Any] = {"codigo_pais": country_code, "pais": country_name}
        for indicator, prefix in {
            "LP.LPI.OVRL.XQ": "lpi",
            "TM.VAL.FUEL.ZS.UN": "importacion_combustible_pct",
            "FP.CPI.TOTL.ZG": "inflacion_pct",
        }.items():
            latest = country_data[country_data["indicador"] == indicator].sort_values("anio").tail(1)
            if latest.empty:
                row[prefix] = np.nan
                row[f"{prefix}_anio"] = np.nan
            else:
                row[prefix] = float(latest["valor"].iloc[0])
                row[f"{prefix}_anio"] = int(latest["anio"].iloc[0])

        fx = country_data[country_data["indicador"] == "PA.NUS.FCRF"].sort_values("anio").tail(2)
        latest_fx = fx.tail(1)
        if latest_fx.empty:
            row["tipo_cambio_oficial"] = np.nan
            row["tipo_cambio_anio"] = np.nan
        else:
            row["tipo_cambio_oficial"] = float(latest_fx["valor"].iloc[0])
            row["tipo_cambio_anio"] = int(latest_fx["anio"].iloc[0])
        if len(fx) == 2 and float(fx["valor"].iloc[0]) != 0:
            row["variacion_cambiaria_pct"] = (
                float(fx["valor"].iloc[1]) / float(fx["valor"].iloc[0]) - 1
            ) * 100
            row["variacion_cambiaria_anio"] = int(fx["anio"].iloc[1])
        else:
            row["variacion_cambiaria_pct"] = np.nan
            row["variacion_cambiaria_anio"] = np.nan
        rows.append(row)

    summary = pd.DataFrame(rows)
    normalized = summary[["codigo_pais", "pais"]].copy()

    def minmax(series: pd.Series, invert: bool = False, absolute: bool = False) -> pd.Series:
        values = series.abs() if absolute else series.copy()
        valid = values.dropna()
        if valid.empty or np.isclose(valid.max(), valid.min()):
            score = pd.Series(0.0, index=series.index)
        else:
            score = (values - valid.min()) / (valid.max() - valid.min()) * 100
        return 100 - score if invert else score

    normalized["Dependencia importadora"] = minmax(summary["importacion_combustible_pct"])
    normalized["Fricción logística"] = minmax(summary["lpi"], invert=True)
    normalized["Presión inflacionaria"] = minmax(summary["inflacion_pct"])
    normalized["Exposición cambiaria"] = minmax(summary["variacion_cambiaria_pct"], absolute=True)
    normalized["Presión fletes/logística"] = (
        normalized["Dependencia importadora"] + normalized["Fricción logística"]
    ) / 2
    summary["presion_fletes_logistica_score"] = normalized["Presión fletes/logística"].to_numpy()
    return raw, summary, normalized, source_status


def create_regional_exposure_figure(normalized_df: pd.DataFrame):
    metrics = [
        "Dependencia importadora",
        "Fricción logística",
        "Presión inflacionaria",
        "Exposición cambiaria",
    ]
    countries = normalized_df["pais"].tolist()
    y = np.arange(len(metrics), dtype=float)
    width = 0.23
    colors = ["#2f78bd", "#ef8a47", "#53a567"]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for index, country in enumerate(countries):
        row = normalized_df[normalized_df["pais"] == country].iloc[0]
        offset = (index - (len(countries) - 1) / 2) * width
        ax.barh(y + offset, [row[metric] for metric in metrics], height=width * 0.9, label=country, color=colors[index])
    ax.set_yticks(y)
    ax.set_yticklabels(metrics)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Presión relativa dentro del bloque (0–100)")
    ax.set_title("Dónde se concentra la presión relativa en el bloque")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    return fig


def describe_regional_exposure(score: float) -> str:
    if not np.isfinite(score):
        return "Sin dato"
    if score >= 67:
        return "Alta"
    if score >= 34:
        return "Media"
    return "Baja"


def pick_regional_driver(row: pd.Series, metric_columns: list[str]) -> str:
    values = {column: float(row[column]) for column in metric_columns if np.isfinite(row[column])}
    if not values:
        return "Sin dato"
    return max(values, key=values.get)


def build_regional_country_view(summary_df: pd.DataFrame, normalized_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "Dependencia importadora",
        "Fricción logística",
        "Presión inflacionaria",
        "Exposición cambiaria",
    ]
    country_view = summary_df.merge(normalized_df, on=["codigo_pais", "pais"], how="left")
    country_view["exposicion_promedio_score"] = country_view[metric_columns].mean(axis=1)
    country_view["exposicion_relativa"] = country_view["exposicion_promedio_score"].apply(describe_regional_exposure)
    country_view["principal_presion"] = country_view.apply(
        lambda row: pick_regional_driver(row, metric_columns),
        axis=1,
    )
    country_view["puesto_bloque"] = (
        country_view["exposicion_promedio_score"].rank(method="min", ascending=False).astype("Int64")
    )
    return country_view


def render_regional_country_cards(country_view: pd.DataFrame) -> None:
    ordered_view = country_view.copy()
    ordered_view["_panama_first"] = ordered_view["pais"] != "Panamá"
    ordered_view = ordered_view.sort_values(
        ["_panama_first", "exposicion_promedio_score"],
        ascending=[True, False],
    ).drop(columns="_panama_first")
    total_countries = len(ordered_view)
    card_columns = st.columns(3)
    for index, (_, row) in enumerate(ordered_view.iterrows()):
        inflation_text = f"{float(row['inflacion_pct']):+.2f}%" if np.isfinite(row["inflacion_pct"]) else "s/d"
        fx_text = (
            f"{float(row['variacion_cambiaria_pct']):+.2f}%"
            if np.isfinite(row["variacion_cambiaria_pct"])
            else "s/d"
        )
        with card_columns[index % len(card_columns)]:
            with st.container(border=True):
                if row["pais"] == "Panamá":
                    st.caption("País de referencia")
                st.markdown(f"**{row['pais']}**")
                st.write(f"Exposición relativa: **{row['exposicion_relativa']}**")
                st.write(f"Principal presión: **{row['principal_presion']}**")
                st.write(f"Posición en el bloque: **{int(row['puesto_bloque'])} de {total_countries}**")
                st.write(f"Dependencia importadora: **{float(row['importacion_combustible_pct']):.2f}%**")
                st.write(f"Logística: **LPI {float(row['lpi']):.1f}**")
                st.caption(f"Inflación: {inflation_text} · Tipo de cambio: {fx_text}")


def create_exchange_rate_figure(summary_df: pd.DataFrame):
    display = summary_df.dropna(subset=["variacion_cambiaria_pct"]).copy()
    colors = ["#2f78bd", "#ef8a47", "#53a567"]
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    ax.bar(
        display["pais"],
        display["variacion_cambiaria_pct"],
        color=colors[: len(display)],
        width=0.58,
    )
    ax.axhline(0, color="#667085", linewidth=1.0)
    ax.set_ylabel("Variación anual frente al USD (%)")
    ax.set_title("Qué tanto cambió cada moneda frente al dólar")
    ax.grid(axis="y", alpha=0.25)
    for index, row in display.reset_index(drop=True).iterrows():
        value = float(row["variacion_cambiaria_pct"])
        offset = 0.35 if value >= 0 else -0.55
        va = "bottom" if value >= 0 else "top"
        ax.text(index, value + offset, f"{value:+.2f}%", ha="center", va=va, fontsize=9)
    fig.tight_layout()
    return fig


def describe_fx_pressure(delta: float) -> str:
    if not np.isfinite(delta):
        return "Sin dato"
    magnitude = abs(float(delta))
    if magnitude >= 5:
        return "Alta"
    if magnitude >= 1:
        return "Media"
    return "Baja"


def describe_fx_direction(delta: float) -> str:
    if not np.isfinite(delta):
        return "Sin dato"
    if delta >= 0.25:
        return "Se debilitó"
    if delta <= -0.25:
        return "Se fortaleció"
    return "Se mantuvo estable"


def build_fx_country_view(summary_df: pd.DataFrame) -> pd.DataFrame:
    fx_view = summary_df[
        [
            "codigo_pais",
            "pais",
            "tipo_cambio_oficial",
            "tipo_cambio_anio",
            "variacion_cambiaria_pct",
            "variacion_cambiaria_anio",
        ]
    ].copy()
    fx_view["presion_cambiaria"] = fx_view["variacion_cambiaria_pct"].apply(describe_fx_pressure)
    fx_view["movimiento_cambiario"] = fx_view["variacion_cambiaria_pct"].apply(describe_fx_direction)
    fx_view["magnitud_movimiento"] = fx_view["variacion_cambiaria_pct"].abs()
    fx_view["puesto_presion"] = (
        fx_view["magnitud_movimiento"].rank(method="min", ascending=False).astype("Int64")
    )
    return fx_view


def render_fx_country_cards(fx_view: pd.DataFrame) -> None:
    ordered_view = fx_view.copy()
    ordered_view["_panama_first"] = ordered_view["pais"] != "Panamá"
    ordered_view = ordered_view.sort_values(
        ["_panama_first", "magnitud_movimiento"],
        ascending=[True, False],
    ).drop(columns="_panama_first")
    total_countries = len(ordered_view)
    card_columns = st.columns(3)
    for index, (_, row) in enumerate(ordered_view.iterrows()):
        value_text = f"{float(row['tipo_cambio_oficial']):.2f}" if np.isfinite(row["tipo_cambio_oficial"]) else "s/d"
        delta_text = (
            f"{float(row['variacion_cambiaria_pct']):+.2f}%"
            if np.isfinite(row["variacion_cambiaria_pct"])
            else "s/d"
        )
        with card_columns[index % len(card_columns)]:
            with st.container(border=True):
                if row["pais"] == "Panamá":
                    st.caption("País de referencia")
                st.markdown(f"**{row['pais']}**")
                st.write(f"Presión cambiaria: **{row['presion_cambiaria']}**")
                st.write(f"Movimiento reciente: **{row['movimiento_cambiario']}**")
                st.write(f"Posición en el bloque: **{int(row['puesto_presion'])} de {total_countries}**")
                st.write(f"Tipo oficial: **{value_text}**")
                st.caption(
                    f"{REGIONAL_CURRENCY_LABELS[row['codigo_pais']]} · Variación anual: {delta_text}"
                )


def create_logistics_freight_figure(summary_df: pd.DataFrame):
    display = summary_df.copy()
    colors = ["#2f78bd", "#ef8a47", "#53a567"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    axes[0].bar(
        display["pais"],
        display["lpi"],
        color=colors[: len(display)],
        width=0.58,
    )
    axes[0].set_title("Desempeño logístico")
    axes[0].set_ylabel("LPI (1-5)")
    axes[0].set_ylim(0, max(3.8, float(display["lpi"].max()) + 0.5))
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        display["pais"],
        display["importacion_combustible_pct"],
        color=colors[: len(display)],
        width=0.58,
    )
    axes[1].set_title("Dependencia importadora de combustibles")
    axes[1].set_ylabel("% de mercancías")
    axes[1].grid(axis="y", alpha=0.25)

    for axis in axes:
        axis.tick_params(axis="x", rotation=0)
    fig.suptitle("Qué tan presionado está el abastecimiento", y=1.02)
    fig.tight_layout()
    return fig


def describe_logistics_pressure(score: float) -> str:
    if not np.isfinite(score):
        return "Sin dato"
    if score >= 67:
        return "Alta"
    if score >= 34:
        return "Media"
    return "Baja"


def build_logistics_country_view(summary_df: pd.DataFrame, normalized_df: pd.DataFrame) -> pd.DataFrame:
    logistics_view = summary_df[
        [
            "codigo_pais",
            "pais",
            "lpi",
            "lpi_anio",
            "importacion_combustible_pct",
            "importacion_combustible_pct_anio",
            "presion_fletes_logistica_score",
        ]
    ].copy()
    normalized_subset = normalized_df[
        [
            "codigo_pais",
            "pais",
            "Dependencia importadora",
            "Fricción logística",
        ]
    ].copy()
    logistics_view = logistics_view.merge(
        normalized_subset,
        on=["codigo_pais", "pais"],
        how="left",
    )
    logistics_view["presion_logistica"] = logistics_view["presion_fletes_logistica_score"].apply(
        describe_logistics_pressure
    )
    logistics_view["principal_factor"] = np.where(
        logistics_view["Dependencia importadora"] >= logistics_view["Fricción logística"],
        "Dependencia importadora",
        "Fricción logística",
    )
    logistics_view["puesto_presion"] = (
        logistics_view["presion_fletes_logistica_score"].rank(method="min", ascending=False).astype("Int64")
    )
    return logistics_view


def render_logistics_country_cards(logistics_view: pd.DataFrame) -> None:
    ordered_view = logistics_view.copy()
    ordered_view["_panama_first"] = ordered_view["pais"] != "Panamá"
    ordered_view = ordered_view.sort_values(
        ["_panama_first", "presion_fletes_logistica_score"],
        ascending=[True, False],
    ).drop(columns="_panama_first")
    total_countries = len(ordered_view)
    card_columns = st.columns(3)
    for index, (_, row) in enumerate(ordered_view.iterrows()):
        import_text = (
            f"{float(row['importacion_combustible_pct']):.2f}%"
            if np.isfinite(row["importacion_combustible_pct"])
            else "s/d"
        )
        lpi_text = f"{float(row['lpi']):.1f}" if np.isfinite(row["lpi"]) else "s/d"
        with card_columns[index % len(card_columns)]:
            with st.container(border=True):
                if row["pais"] == "Panamá":
                    st.caption("País de referencia")
                st.markdown(f"**{row['pais']}**")
                st.write(f"Presión logística: **{row['presion_logistica']}**")
                st.write(f"Principal factor: **{row['principal_factor']}**")
                st.write(f"Posición en el bloque: **{int(row['puesto_presion'])} de {total_countries}**")
                st.write(f"Dependencia importadora: **{import_text}**")
                st.write(f"Desempeño logístico: **LPI {lpi_text}**")
                st.caption(
                    f"Presión compuesta: {float(row['presion_fletes_logistica_score']):.1f}/100"
                    if np.isfinite(row["presion_fletes_logistica_score"])
                    else "Presión compuesta: s/d"
                )


def load_regional_elasticity_data(source: Any) -> pd.DataFrame:
    source_name = str(getattr(source, "name", "")).lower()
    raw = pd.read_csv(source) if source_name.endswith(".csv") else pd.read_excel(source)
    raw = normalize_columns(raw)
    date_col = find_column(raw.columns, ["periodo", "fecha", "mes"])
    country_col = find_column(raw.columns, ["pais", "country"])
    fuel_col = find_column(raw.columns, ["precio_combustible", "combustible", "fuel_price"])
    basket_col = find_column(raw.columns, ["indice_cba", "cba", "canasta_basica", "basket_index"])

    data = raw[[date_col, country_col, fuel_col, basket_col]].rename(
        columns={
            date_col: "periodo",
            country_col: "pais",
            fuel_col: "precio_combustible",
            basket_col: "indice_cba",
        }
    )
    data["periodo"] = parse_dates_safely(data["periodo"], dayfirst=True).dt.to_period("M").dt.to_timestamp()
    data["pais"] = data["pais"].astype(str).str.strip()
    data["precio_combustible"] = to_numeric_series(data["precio_combustible"])
    data["indice_cba"] = to_numeric_series(data["indice_cba"])
    data = data.dropna().query("precio_combustible > 0 and indice_cba > 0")
    data = data.groupby(["pais", "periodo"], as_index=False)[["precio_combustible", "indice_cba"]].mean()
    valid_countries = data.groupby("pais").size()
    valid_countries = valid_countries[valid_countries >= 24].index
    data = data[data["pais"].isin(valid_countries)].sort_values(["pais", "periodo"]).reset_index(drop=True)
    if data.empty:
        raise ValueError("Se requieren al menos 24 meses positivos por país para estimar elasticidades.")
    return data


def analyze_regional_elasticity(data: pd.DataFrame) -> pd.DataFrame:
    results: list[dict[str, Any]] = []
    for country, group in data.groupby("pais"):
        group = group.sort_values("periodo").copy()
        group["combustible_log_var"] = np.log(group["precio_combustible"]).diff()
        group["cba_log_var"] = np.log(group["indice_cba"]).diff()
        for lag in [0, 1, 3, 6]:
            pair = pd.DataFrame(
                {
                    "x": group["combustible_log_var"].shift(lag),
                    "y": group["cba_log_var"],
                }
            ).dropna()
            if len(pair) < 18 or np.isclose(pair["x"].var(), 0) or np.isclose(pair["y"].var(), 0):
                continue
            elasticity = float(np.cov(pair["x"], pair["y"], ddof=1)[0, 1] / pair["x"].var(ddof=1))
            correlation, p_value = pearsonr(pair["x"], pair["y"])
            results.append(
                {
                    "pais": country,
                    "rezago_meses": lag,
                    "elasticidad_combustible_cba": elasticity,
                    "correlacion": float(correlation),
                    "p_valor": float(p_value),
                    "observaciones": len(pair),
                }
            )
    result = pd.DataFrame(results)
    if result.empty:
        raise ValueError("Las series no tienen variación suficiente para estimar elasticidades.")
    result["rezago_destacado"] = False
    for country, indexes in result.groupby("pais").groups.items():
        best_index = result.loc[list(indexes), "correlacion"].abs().idxmax()
        result.loc[best_index, "rezago_destacado"] = True
    return result


def create_regional_elasticity_figure(elasticity_df: pd.DataFrame):
    countries = elasticity_df["pais"].drop_duplicates().tolist()
    lags = [0, 1, 3, 6]
    x = np.arange(len(lags), dtype=float)
    width = min(0.24, 0.72 / max(1, len(countries)))
    colors = ["#2f78bd", "#ef8a47", "#53a567", "#9b6cc2"]
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for index, country in enumerate(countries):
        country_data = elasticity_df[elasticity_df["pais"] == country].set_index("rezago_meses")
        values = [country_data["elasticidad_combustible_cba"].get(lag, np.nan) for lag in lags]
        offset = (index - (len(countries) - 1) / 2) * width
        ax.bar(x + offset, values, width=width * 0.9, label=country, color=colors[index % len(colors)])
    ax.axhline(0, color="#667085", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([str(lag) for lag in lags])
    ax.set_xlabel("Rezago del combustible (meses)")
    ax.set_ylabel("Elasticidad de variaciones mensuales")
    ax.set_title("Elasticidad combustible–CBA por país y rezago")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def regularize_series(series: pd.Series) -> pd.Series:
    regular = series.sort_index().asfreq("MS")
    if regular.notna().sum() < 12:
        raise ValueError("Se requieren al menos 12 meses con datos.")
    regular = regular.interpolate(method="time", limit=2, limit_area="inside")
    if regular.isna().any():
        missing = regular.index[regular.isna()].strftime("%Y-%m").tolist()
        raise ValueError(f"Hay meses faltantes sin imputación segura: {', '.join(missing[:6])}.")
    return regular.astype(float)


@st.cache_data(ttl=86_400, show_spinner=False)
def fetch_brent_monthly() -> pd.DataFrame:
    """Descarga la tabla mensual oficial de Brent publicada por la EIA."""
    response = requests.get(
        EIA_BRENT_SOURCE_URL,
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0 (compatible; FuelForecastPrototype/3.0)"},
    )
    response.raise_for_status()
    tables = pd.read_html(BytesIO(response.content))
    raw = next(
        (
            table
            for table in tables
            if "Year" in table.columns and {"Jan", "Feb", "Mar", "Dec"}.issubset(table.columns)
        ),
        None,
    )
    if raw is None:
        raise ValueError("La tabla mensual de Brent no tiene la estructura esperada.")

    month_numbers = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }
    monthly = raw.melt(id_vars="Year", value_vars=list(month_numbers), var_name="mes", value_name="brent_usd_barril")
    monthly["Year"] = pd.to_numeric(monthly["Year"], errors="coerce")
    monthly["brent_usd_barril"] = pd.to_numeric(monthly["brent_usd_barril"], errors="coerce")
    monthly["periodo"] = pd.to_datetime(
        {
            "year": monthly["Year"],
            "month": monthly["mes"].map(month_numbers),
            "day": 1,
        },
        errors="coerce",
    )
    monthly = (
        monthly[["periodo", "brent_usd_barril"]]
        .dropna()
        .query("brent_usd_barril > 0")
        .sort_values("periodo")
        .reset_index(drop=True)
    )
    if len(monthly) < 36:
        raise ValueError("La serie mensual de Brent tiene menos de 36 observaciones.")
    return monthly


def fit_ets(train: pd.Series, seasonal: bool):
    return ExponentialSmoothing(
        train,
        trend="add",
        damped_trend=True,
        seasonal="add" if seasonal else None,
        seasonal_periods=12 if seasonal else None,
        initialization_method="estimated",
    ).fit(optimized=True, use_brute=True)


def naive_forecast(train: pd.Series, horizon: int, seasonal: bool) -> pd.Series:
    if seasonal:
        values = np.resize(train.iloc[-12:].to_numpy(), horizon)
    else:
        values = np.repeat(train.iloc[-1], horizon)
    index = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    return pd.Series(values, index=index, dtype=float)


def forecast_baseline(series: pd.Series, horizon: int) -> tuple[pd.Series, pd.Series, pd.Series, dict[str, Any]]:
    regular = regularize_series(series)
    holdout = min(12, max(3, len(regular) // 5))
    train, test = regular.iloc[:-holdout], regular.iloc[-holdout:]
    if len(train) < 12:
        raise ValueError("No hay suficientes meses para separar entrenamiento y validación.")

    candidates: list[tuple[str, float]] = []
    naive_pred = naive_forecast(train, holdout, seasonal=False)
    candidates.append(("Naive: último valor", float(np.mean(np.abs(test.to_numpy() - naive_pred.to_numpy())))))

    seasonal_naive = len(train) >= 24
    if seasonal_naive:
        pred = naive_forecast(train, holdout, seasonal=True)
        candidates.append(("Naive estacional", float(np.mean(np.abs(test.to_numpy() - pred.to_numpy())))))

    for seasonal, name in [(False, "Holt amortiguado"), (True, "Holt-Winters aditivo")]:
        if seasonal and len(train) < 24:
            continue
        try:
            pred = fit_ets(train, seasonal).forecast(holdout)
            candidates.append((name, float(np.mean(np.abs(test.to_numpy() - np.asarray(pred))))))
        except (ValueError, np.linalg.LinAlgError):
            continue

    model_name, validation_mae = min(candidates, key=lambda item: item[1])
    if model_name == "Naive: último valor":
        forecast = naive_forecast(regular, horizon, seasonal=False)
        residuals = regular.diff().dropna()
    elif model_name == "Naive estacional":
        forecast = naive_forecast(regular, horizon, seasonal=True)
        residuals = (regular - regular.shift(12)).dropna()
    else:
        fitted = fit_ets(regular, seasonal=model_name == "Holt-Winters aditivo")
        forecast = pd.Series(np.asarray(fitted.forecast(horizon)), index=pd.date_range(regular.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"))
        residuals = pd.Series(fitted.resid).dropna()

    residual_sigma = float(residuals.std(ddof=1)) if len(residuals) > 1 else 0.0
    scale = np.sqrt(np.arange(1, horizon + 1))
    lower = pd.Series(np.maximum(MIN_REASONABLE_PRICE, forecast.to_numpy() - 1.96 * residual_sigma * scale), index=forecast.index)
    upper = pd.Series(forecast.to_numpy() + 1.96 * residual_sigma * scale, index=forecast.index)
    metrics = {
        "modelo": model_name,
        "mae_validacion": validation_mae,
        "ultimo_valor": float(regular.iloc[-1]),
        "promedio": float(regular.mean()),
        "tendencia_reciente": float(regular.iloc[-1] - regular.iloc[-4]),
        "meses_validacion": holdout,
    }
    return forecast, lower, upper, metrics


def project_external_series(series: pd.Series, end_period: pd.Timestamp) -> pd.Series:
    """Extiende una variable externa hasta el mes requerido usando la misma selección temporal base."""
    regular = regularize_series(series)
    end_period = pd.Timestamp(end_period).to_period("M").to_timestamp()
    if regular.index[-1] >= end_period:
        return regular.loc[:end_period]

    missing_horizon = len(pd.date_range(regular.index[-1] + pd.offsets.MonthBegin(1), end_period, freq="MS"))
    projected, _, _, _ = forecast_baseline(regular, missing_horizon)
    return pd.concat([regular, projected]).sort_index()


def forecast_dynamic_with_brent(
    series: pd.Series,
    brent_series: pd.Series,
    horizon: int,
    holdout: int,
) -> tuple[pd.Series, pd.Series, pd.Series, dict[str, Any]]:
    """Modelo dinámico en diferencias con un rezago autorregresivo y variación del Brent."""
    target = regularize_series(series)
    brent = regularize_series(brent_series)
    aligned_brent = brent.reindex(target.index).interpolate(method="time").ffill().bfill()
    if aligned_brent.isna().any() or len(target) < max(36, holdout + 24):
        raise ValueError("No hay suficiente período común entre Brent y el combustible.")

    frame = pd.DataFrame({"nivel": target, "brent": aligned_brent})
    frame["variacion"] = frame["nivel"].diff()
    frame["variacion_rezagada"] = frame["variacion"].shift(1)
    frame["variacion_log_brent"] = np.log(frame["brent"]).diff()
    train_end = target.index[-holdout - 1]
    test_index = target.index[-holdout:]
    candidates: list[dict[str, Any]] = []

    for lag in [0, 1, 2, 3]:
        driver_col = f"brent_rezago_{lag}"
        frame[driver_col] = frame["variacion_log_brent"].shift(lag)
        train_rows = frame.loc[:train_end, ["variacion", "variacion_rezagada", driver_col]].dropna()
        if len(train_rows) < 24:
            continue
        design = np.column_stack(
            [
                np.ones(len(train_rows)),
                train_rows["variacion_rezagada"].to_numpy(),
                train_rows[driver_col].to_numpy(),
            ]
        )
        coefficients = np.linalg.lstsq(design, train_rows["variacion"].to_numpy(), rcond=None)[0]
        previous_level = float(target.loc[train_end])
        previous_change = float(frame.loc[train_end, "variacion"])
        validation_predictions: list[float] = []
        valid_candidate = True
        for period in test_index:
            driver_value = frame.loc[period, driver_col]
            if pd.isna(driver_value):
                valid_candidate = False
                break
            predicted_change = float(
                coefficients[0] + coefficients[1] * previous_change + coefficients[2] * driver_value
            )
            previous_level = max(MIN_REASONABLE_PRICE, previous_level + predicted_change)
            validation_predictions.append(previous_level)
            previous_change = predicted_change
        if not valid_candidate:
            continue
        mae = float(np.mean(np.abs(target.loc[test_index].to_numpy() - np.asarray(validation_predictions))))
        candidates.append({"rezago": lag, "mae": mae})

    if not candidates:
        raise ValueError("No se pudo validar ningún rezago de Brent.")
    best = min(candidates, key=lambda item: item["mae"])
    selected_lag = int(best["rezago"])
    selected_col = f"brent_rezago_{selected_lag}"
    final_rows = frame[["variacion", "variacion_rezagada", selected_col]].dropna()
    final_design = np.column_stack(
        [
            np.ones(len(final_rows)),
            final_rows["variacion_rezagada"].to_numpy(),
            final_rows[selected_col].to_numpy(),
        ]
    )
    coefficients = np.linalg.lstsq(final_design, final_rows["variacion"].to_numpy(), rcond=None)[0]
    fitted_changes = final_design @ coefficients
    residuals = final_rows["variacion"].to_numpy() - fitted_changes

    future_index = pd.date_range(target.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    projected_brent = project_external_series(brent, future_index[-1])
    driver = np.log(projected_brent).diff().shift(selected_lag)
    previous_level = float(target.iloc[-1])
    previous_change = float(frame["variacion"].dropna().iloc[-1])
    forecast_values: list[float] = []
    for period in future_index:
        driver_value = driver.get(period, np.nan)
        if pd.isna(driver_value):
            raise ValueError("La proyección de Brent no cubre todo el horizonte solicitado.")
        predicted_change = float(
            coefficients[0] + coefficients[1] * previous_change + coefficients[2] * driver_value
        )
        previous_level = max(MIN_REASONABLE_PRICE, previous_level + predicted_change)
        forecast_values.append(previous_level)
        previous_change = predicted_change

    forecast = pd.Series(forecast_values, index=future_index, dtype=float)
    residual_sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
    scale = np.sqrt(np.arange(1, horizon + 1))
    lower = pd.Series(
        np.maximum(MIN_REASONABLE_PRICE, forecast.to_numpy() - 1.96 * residual_sigma * scale),
        index=future_index,
    )
    upper = pd.Series(forecast.to_numpy() + 1.96 * residual_sigma * scale, index=future_index)
    metrics = {
        "mae_externo": float(best["mae"]),
        "rezago_brent": selected_lag,
        "coeficiente_brent": float(coefficients[2]),
        "impacto_brent_10_pct": float(coefficients[2] * np.log(1.10)),
    }
    return forecast, lower, upper, metrics


def forecast_series(
    series: pd.Series,
    horizon: int,
    brent_series: Optional[pd.Series] = None,
) -> tuple[pd.Series, pd.Series, pd.Series, dict[str, Any]]:
    baseline_forecast, baseline_lower, baseline_upper, metrics = forecast_baseline(series, horizon)
    metrics.update(
        {
            "modelo_historico": metrics["modelo"],
            "mae_historico": metrics["mae_validacion"],
            "mae_externo": np.nan,
            "mejora_mae_brent_pct": np.nan,
            "rezago_brent": np.nan,
            "coeficiente_brent": np.nan,
            "impacto_brent_10_pct": np.nan,
            "modelo_ganador": "Histórico",
        }
    )
    if brent_series is None or brent_series.empty:
        return baseline_forecast, baseline_lower, baseline_upper, metrics

    try:
        external_forecast, external_lower, external_upper, external_metrics = forecast_dynamic_with_brent(
            series,
            brent_series,
            horizon,
            int(metrics["meses_validacion"]),
        )
    except (ValueError, np.linalg.LinAlgError):
        return baseline_forecast, baseline_lower, baseline_upper, metrics

    external_mae = float(external_metrics["mae_externo"])
    baseline_mae = float(metrics["mae_historico"])
    improvement = (baseline_mae - external_mae) / baseline_mae * 100 if baseline_mae > 0 else 0.0
    metrics.update(external_metrics)
    metrics["mejora_mae_brent_pct"] = improvement
    if external_mae < baseline_mae:
        metrics["modelo"] = f"Dinámico con Brent (rezago {int(external_metrics['rezago_brent'])})"
        metrics["mae_validacion"] = external_mae
        metrics["modelo_ganador"] = "Brent"
        return external_forecast, external_lower, external_upper, metrics
    return baseline_forecast, baseline_lower, baseline_upper, metrics


def build_forecast_output(
    monthly_df: pd.DataFrame,
    horizon: int,
    brent_series: Optional[pd.Series] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    last_period = monthly_df["periodo"].max()
    future_periods = pd.date_range(last_period + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    output = monthly_df.copy()
    output["tipo_registro"] = "historico"
    forecast_df = pd.DataFrame({"periodo": future_periods, "tipo_registro": "pronostico", "mes_horizonte": range(1, horizon + 1)})
    summary_rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for col in FUEL_NAMES:
        if col not in monthly_df.columns:
            continue
        series = pd.Series(monthly_df[col].values, index=pd.DatetimeIndex(monthly_df["periodo"]), name=col)
        try:
            pred, lower, upper, metrics = forecast_series(series, horizon, brent_series)
        except ValueError as exc:
            errors.append(f"{FUEL_NAMES[col]}: {exc}")
            continue

        forecast_df[col] = pred.to_numpy()
        forecast_df[f"{col}_lim_inf"] = lower.to_numpy()
        forecast_df[f"{col}_lim_sup"] = upper.to_numpy()
        row: dict[str, Any] = {
            "serie": col,
            "modelo": metrics["modelo"],
            "ultimo_valor_historico": round(metrics["ultimo_valor"], 4),
            "promedio_historico": round(metrics["promedio"], 4),
            "tendencia_reciente_ultimos_4_meses": round(metrics["tendencia_reciente"], 4),
            "mae_validacion": round(metrics["mae_validacion"], 4),
            "modelo_historico": metrics["modelo_historico"],
            "mae_historico": round(metrics["mae_historico"], 4),
            "mae_con_brent": round(metrics["mae_externo"], 4) if pd.notna(metrics["mae_externo"]) else np.nan,
            "mejora_mae_brent_pct": (
                round(metrics["mejora_mae_brent_pct"], 2)
                if pd.notna(metrics["mejora_mae_brent_pct"])
                else np.nan
            ),
            "rezago_brent_meses": (
                int(metrics["rezago_brent"]) if pd.notna(metrics["rezago_brent"]) else np.nan
            ),
            "impacto_brent_10_pct": (
                round(metrics["impacto_brent_10_pct"], 4)
                if pd.notna(metrics["impacto_brent_10_pct"])
                else np.nan
            ),
            "modelo_ganador": metrics["modelo_ganador"],
            "meses_validacion": metrics["meses_validacion"],
            "horizonte_meses": horizon,
            "direccion_esperada": "alza" if pred.iloc[-1] > metrics["ultimo_valor"] else "baja" if pred.iloc[-1] < metrics["ultimo_valor"] else "estable",
        }
        for i in range(horizon):
            row[f"pronostico_mes_{i + 1}"] = round(float(pred.iloc[i]), 4)
            row[f"lim_inf_mes_{i + 1}"] = round(float(lower.iloc[i]), 4)
            row[f"lim_sup_mes_{i + 1}"] = round(float(upper.iloc[i]), 4)
        summary_rows.append(row)

    if not summary_rows:
        raise ValueError("No se pudo pronosticar ninguna serie. " + " ".join(errors))
    if errors:
        st.warning("Series omitidas: " + " ".join(errors))

    combined = pd.concat([output, forecast_df], ignore_index=True, sort=False).sort_values("periodo").reset_index(drop=True)
    return combined, pd.DataFrame(summary_rows)


def create_historical_figure(combined_df: pd.DataFrame):
    historical = combined_df[combined_df["tipo_registro"] == "historico"].sort_values("periodo")
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for col, label in FUEL_NAMES.items():
        if col in historical:
            ax.plot(historical["periodo"], historical[col], linewidth=2, label=label)
    ax.set(title="Histórico de precios de combustibles en Panamá", xlabel="Periodo", ylabel=f"Precio ({PRICE_UNIT})")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def create_forecast_figure(combined_df: pd.DataFrame, horizon: int):
    historical = combined_df[combined_df["tipo_registro"] == "historico"].sort_values("periodo")
    forecast = combined_df[combined_df["tipo_registro"] == "pronostico"].sort_values("periodo")
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for col, label in FUEL_NAMES.items():
        if col not in forecast or forecast[col].isna().all():
            continue
        last = historical[["periodo", col]].dropna().tail(1)
        x = pd.concat([last["periodo"], forecast["periodo"]], ignore_index=True)
        y = pd.concat([last[col], forecast[col]], ignore_index=True)
        line = ax.plot(x, y, linestyle="--", linewidth=2, label=label)[0]
        lower_col, upper_col = f"{col}_lim_inf", f"{col}_lim_sup"
        if lower_col in forecast and upper_col in forecast:
            ax.fill_between(forecast["periodo"], forecast[lower_col], forecast[upper_col], color=line.get_color(), alpha=0.12)
    ax.set(title=f"Pronóstico con intervalo aproximado del 95 % ({horizon} meses)", xlabel="Periodo", ylabel=f"Precio ({PRICE_UNIT})")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def build_brent_projection(brent_df: pd.DataFrame, end_period: pd.Timestamp) -> pd.DataFrame:
    observed = pd.Series(
        brent_df["brent_usd_barril"].to_numpy(),
        index=pd.DatetimeIndex(brent_df["periodo"]),
        dtype=float,
    )
    extended = project_external_series(observed, end_period)
    result = extended.rename("brent_usd_barril").rename_axis("periodo").reset_index()
    result["tipo_registro"] = np.where(
        result["periodo"] <= observed.index.max(),
        "observado",
        "proyeccion_auxiliar",
    )
    return result


def create_brent_phase3_figure(brent_projection_df: pd.DataFrame, history_start: pd.Timestamp):
    display = brent_projection_df[brent_projection_df["periodo"] >= history_start].copy()
    observed = display[display["tipo_registro"] == "observado"]
    projected = display[display["tipo_registro"] == "proyeccion_auxiliar"]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.plot(observed["periodo"], observed["brent_usd_barril"], linewidth=2, label="Brent observado")
    if not projected.empty:
        bridge = pd.concat([observed.tail(1), projected], ignore_index=True)
        ax.plot(
            bridge["periodo"],
            bridge["brent_usd_barril"],
            linestyle="--",
            linewidth=2,
            label="Proyección auxiliar de Brent",
        )
        ax.axvspan(projected["periodo"].min(), projected["periodo"].max(), alpha=0.08)
    ax.set(
        title="Brent utilizado como variable externa",
        xlabel="Periodo",
        ylabel="USD por barril",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def load_default_files() -> list[Any]:
    return [BASE_DIR / name for name in DEFAULT_INPUT_FILE_NAMES if (BASE_DIR / name).exists()]


def dataframe_download_bytes(df_map: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in df_map.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return output.getvalue()


def build_local_reference_prices(monthly_df: pd.DataFrame) -> tuple[dict[str, float], dict[str, pd.Timestamp]]:
    history = monthly_df.sort_values("periodo")
    prices: dict[str, float] = {}
    periods: dict[str, pd.Timestamp] = {}
    for fuel_key in FUEL_NAMES:
        if fuel_key not in history.columns:
            continue
        valid = history[["periodo", fuel_key]].dropna()
        if valid.empty:
            continue
        latest = valid.iloc[-1]
        prices[fuel_key] = float(latest[fuel_key])
        periods[fuel_key] = pd.Timestamp(latest["periodo"])
    return prices, periods


def render_overview_metrics(monthly_df: pd.DataFrame, summary_df: pd.DataFrame, selected_horizon: int) -> None:
    last_period = pd.Timestamp(monthly_df["periodo"].max())
    winners = int((summary_df["modelo_ganador"] == "Brent").sum()) if "modelo_ganador" in summary_df else 0
    cols = st.columns(4)
    cols[0].metric("Último mes histórico", f"{last_period:%Y-%m}")
    cols[1].metric("Horizonte activo", f"{selected_horizon} meses")
    cols[2].metric("Combustibles modelados", f"{len(summary_df)}")
    cols[3].metric("Modelos con Brent", f"{winners}")


def render_validation_table(summary_df: pd.DataFrame) -> None:
    validation_df = summary_df[
        ["serie", "modelo", "mae_validacion", "meses_validacion", "direccion_esperada"]
    ].copy()
    validation_df["serie"] = validation_df["serie"].map(FUEL_NAMES)
    validation_df["mae_validacion"] = validation_df["mae_validacion"].round(4)
    validation_df["direccion_esperada"] = validation_df["direccion_esperada"].str.capitalize()
    st.dataframe(
        validation_df,
        width="stretch",
        hide_index=True,
        column_config={
            "serie": st.column_config.TextColumn("Combustible", width="small"),
            "modelo": st.column_config.TextColumn("Modelo", width="medium"),
            "mae_validacion": st.column_config.NumberColumn("MAE (error absoluto medio)", format="%.4f", width="small"),
            "meses_validacion": st.column_config.NumberColumn("Meses val.", format="%d", width="small"),
            "direccion_esperada": st.column_config.TextColumn("Señal", width="small"),
        },
    )


def describe_brent_effect(improvement: float) -> str:
    if not np.isfinite(improvement):
        return "Sin dato"
    if improvement >= 20:
        return "Alta"
    if improvement >= 5:
        return "Media"
    if improvement > 0:
        return "Leve"
    if improvement > -5:
        return "No mejora"
    return "Empeora"


def build_brent_summary_view(summary_df: pd.DataFrame) -> pd.DataFrame:
    brent_view = summary_df[
        [
            "serie",
            "mae_historico",
            "mae_con_brent",
            "mejora_mae_brent_pct",
            "rezago_brent_meses",
            "modelo_ganador",
        ]
    ].copy()
    brent_view["combustible"] = brent_view["serie"].map(FUEL_NAMES)
    brent_view["apoyo_brent"] = brent_view["mejora_mae_brent_pct"].apply(describe_brent_effect)
    brent_view["mejora_abs"] = brent_view["mejora_mae_brent_pct"].abs()
    return brent_view


def render_brent_model_cards(brent_view: pd.DataFrame) -> None:
    ordered_view = brent_view.copy()
    ordered_view["_brent_first"] = ordered_view["modelo_ganador"] != "Brent"
    ordered_view = ordered_view.sort_values(
        ["_brent_first", "mejora_mae_brent_pct"],
        ascending=[True, False],
    ).drop(columns="_brent_first")
    card_columns = st.columns(3)
    for index, (_, row) in enumerate(ordered_view.iterrows()):
        lag_text = (
            f"{int(row['rezago_brent_meses'])} mes(es)"
            if np.isfinite(row["rezago_brent_meses"])
            else "s/d"
        )
        improvement_text = (
            f"{float(row['mejora_mae_brent_pct']):+.1f}%"
            if np.isfinite(row["mejora_mae_brent_pct"])
            else "s/d"
        )
        with card_columns[index % len(card_columns)]:
            with st.container(border=True):
                st.markdown(f"**{row['combustible']}**")
                st.write(f"Apoyo de Brent: **{row['apoyo_brent']}**")
                st.write(f"Modelo ganador: **{row['modelo_ganador']}**")
                st.write(f"Rezago útil: **{lag_text}**")
                st.write(f"MAE histórico: **{float(row['mae_historico']):.4f}**")
                st.write(f"MAE con Brent: **{float(row['mae_con_brent']):.4f}**")
                st.caption(f"Cambio en MAE: {improvement_text}")


def render_forecast_cards_final(summary_df: pd.DataFrame, selected_horizon: int) -> None:
    forecast_col = f"pronostico_mes_{selected_horizon}"
    lower_col = f"lim_inf_mes_{selected_horizon}"
    upper_col = f"lim_sup_mes_{selected_horizon}"
    columns = st.columns(len(summary_df))
    for column, (_, row) in zip(columns, summary_df.iterrows()):
        name = FUEL_NAMES.get(str(row["serie"]), str(row["serie"]))
        current_price = float(row["ultimo_valor_historico"])
        projected_price = float(row[forecast_col])
        delta_amount = projected_price - current_price
        delta_pct = (delta_amount / current_price * 100) if current_price else 0.0
        if np.isclose(delta_amount, 0.0, atol=0.0001):
            trend_label = "Se mantiene"
            delta_label = f"{delta_amount:+.4f} {PRICE_UNIT}"
        elif delta_amount > 0:
            trend_label = "Sube"
            delta_label = f"+{abs(delta_amount):.4f} {PRICE_UNIT}"
        else:
            trend_label = "Baja"
            delta_label = f"-{abs(delta_amount):.4f} {PRICE_UNIT}"
        with column:
            with st.container(border=True):
                st.caption(f"Mes {selected_horizon}")
                st.metric(name, trend_label, delta_label)
                st.write(f"Cambio estimado: {delta_pct:+.1f}%")
                st.caption(f"Último precio: {current_price:.4f} {PRICE_UNIT}")
                st.caption(f"Precio estimado en mes {selected_horizon}: {projected_price:.4f} {PRICE_UNIT}")
                st.caption("Rango aproximado del 95 %")
                st.write(f"{float(row[lower_col]):.4f}–{float(row[upper_col]):.4f} {PRICE_UNIT}")


def render_live_prices_box(monthly_df: pd.DataFrame) -> None:
    st.subheader("Precios de referencia en Panamá")
    with st.spinner("Consultando precios vigentes..."):
        prices, source_url, source_label = fetch_live_panama_prices()
    reference_mode = "live"
    source_text = ""
    if not prices:
        prices, local_periods = build_local_reference_prices(monthly_df)
        if not prices:
            st.info("No hay precios disponibles ni en línea ni en el historial local.")
            return
        reference_mode = "local"
        unique_periods = sorted({period.strftime("%Y-%m") for period in local_periods.values()})
        if len(unique_periods) == 1:
            source_text = f"Respaldo local · último mes disponible: {unique_periods[0]}"
        else:
            period_by_fuel = ", ".join(
                f"{FUEL_NAMES[key]} {local_periods[key]:%Y-%m}" for key in prices if key in local_periods
            )
            source_text = f"Respaldo local por serie: {period_by_fuel}"
        st.info("No se pudieron validar precios vigentes en línea; se muestran los últimos valores del historial local.")
    cols = st.columns(len(prices))
    for column, (key, value) in zip(cols, prices.items()):
        with column:
            st.metric(FUEL_NAMES[key], f"{value:.4f}")
            st.write(f"**Unidad:** {PRICE_UNIT}")
    if source_url and reference_mode == "live":
        source_note = f"{source_label} · {source_url}" if source_label else source_url
        st.caption(f"Fuente: {source_note} · Consulta almacenada durante una hora")
    elif source_text:
        st.caption(source_text)


def main() -> None:
    st.set_page_config(page_title="Pronóstico de combustibles Panamá", layout="wide")
    st.title("Pronóstico web de combustibles en Panamá")
    st.caption(f"Selección por validación temporal · precios expresados en {PRICE_UNIT}")

    with st.sidebar:
        st.header("Configuración")
        selected_horizon = st.slider("Meses a pronosticar", 3, 18, 12)
        use_uploaded = st.toggle("Usar archivos cargados", value=False)
        uploaded_files = st.file_uploader("Cargar archivos Excel", type=["xlsx"], accept_multiple_files=True)
        uploaded_ipc_sectoral = st.file_uploader(
            "IPC sectorial del INEC (opcional)",
            type=["xlsx"],
            accept_multiple_files=False,
            help="Si no cargas otro archivo, se utiliza el Cuadro 2 oficial incluido con el proyecto.",
        )

    input_sources = uploaded_files if use_uploaded and uploaded_files else load_default_files()
    if not input_sources:
        st.warning("Carga los Excel o colócalos en la misma carpeta del programa.")
        return

    brent_monthly_df = pd.DataFrame()
    brent_projection_df = pd.DataFrame()
    brent_series: Optional[pd.Series] = None
    phase3_error: Optional[str] = None
    try:
        with st.spinner("Consultando Brent para la Fase 3..."):
            brent_monthly_df = fetch_brent_monthly()
        brent_series = pd.Series(
            brent_monthly_df["brent_usd_barril"].to_numpy(),
            index=pd.DatetimeIndex(brent_monthly_df["periodo"]),
            name="brent_usd_barril",
            dtype=float,
        )
    except (requests.RequestException, ValueError, OSError, pd.errors.ParserError) as exc:
        phase3_error = str(exc)

    try:
        monthly_df, sheet_log = load_fuel_data(input_sources)
        combined_df, summary_df = build_forecast_output(monthly_df, selected_horizon, brent_series)
    except (ValueError, OSError, KeyError) as exc:
        st.error(f"Error al procesar los archivos: {exc}")
        return

    render_overview_metrics(monthly_df, summary_df, selected_horizon)

    top_col1, top_col2 = st.columns([1.3, 1])
    with top_col1:
        st.pyplot(create_historical_figure(combined_df), clear_figure=True)
    with top_col2:
        st.subheader("Validación del modelo")
        render_validation_table(summary_df)

    bottom_col1, bottom_col2 = st.columns([1.3, 1])
    with bottom_col1:
        st.pyplot(create_forecast_figure(combined_df, selected_horizon), clear_figure=True)
    with bottom_col2:
        render_live_prices_box(monthly_df)
        st.subheader("Pronóstico final e intervalo aproximado")
        render_forecast_cards_final(summary_df, selected_horizon)

    st.divider()
    st.header("Pronóstico con variable externa")
    st.caption(
        "Compara el mejor modelo histórico con una regresión dinámica que incorpora la variación mensual del Brent. "
        "La validación utiliza el mismo bloque temporal final y selecciona automáticamente el menor MAE (error absoluto medio)."
    )
    if brent_series is None or brent_monthly_df.empty:
        st.warning(
            "No se pudo consultar Brent en línea; el pronóstico continúa con los modelos históricos. "
            f"Detalle: {phase3_error or 'fuente no disponible'}"
        )
    else:
        try:
            forecast_end = monthly_df["periodo"].max() + pd.offsets.MonthBegin(selected_horizon)
            brent_projection_df = build_brent_projection(brent_monthly_df, forecast_end)
            brent_view = build_brent_summary_view(summary_df)
            brent_winners = int((brent_view["modelo_ganador"] == "Brent").sum())
            best_brent_row = brent_view.sort_values("mejora_mae_brent_pct", ascending=False).iloc[0]
            winning_lags = brent_view.loc[brent_view["modelo_ganador"] == "Brent", "rezago_brent_meses"].dropna()
            if winning_lags.empty:
                common_brent_lag = (
                    int(best_brent_row["rezago_brent_meses"])
                    if np.isfinite(best_brent_row["rezago_brent_meses"])
                    else None
                )
            else:
                common_brent_lag = int(winning_lags.mode().iloc[0])
            phase3_col1, phase3_col2 = st.columns([1.25, 1])
            with phase3_col1:
                history_start = max(
                    pd.Timestamp("2015-01-01"),
                    pd.Timestamp(monthly_df["periodo"].min()),
                )
                st.pyplot(
                    create_brent_phase3_figure(brent_projection_df, history_start),
                    clear_figure=True,
                )
            with phase3_col2:
                latest_brent = brent_monthly_df.sort_values("periodo").iloc[-1]
                st.metric(
                    "Último promedio mensual de Brent",
                    f"US${latest_brent['brent_usd_barril']:.2f}/barril",
                    help=f"Último mes completo disponible: {latest_brent['periodo']:%Y-%m}",
                )
                st.subheader("Lectura rápida")
                brent_quick_col1, brent_quick_col2 = st.columns(2)
                with brent_quick_col1:
                    st.metric("Brent ayuda en", f"{brent_winners} de {len(brent_view)}")
                    st.metric("Mayor mejora", best_brent_row["combustible"])
                with brent_quick_col2:
                    st.metric(
                        "Mejora máxima",
                        f"{float(best_brent_row['mejora_mae_brent_pct']):.1f}%",
                    )
                    st.metric(
                        "Rezago más útil",
                        f"{common_brent_lag} mes(es)" if common_brent_lag is not None else "s/d",
                    )
                st.info(
                    f"En esta comparación, **Brent** mejora el pronóstico en **{brent_winners} de {len(brent_view)}** combustibles. "
                    f"La mayor reducción del error aparece en **{best_brent_row['combustible']}** "
                    f"con **{float(best_brent_row['mejora_mae_brent_pct']):.1f}%**."
                )

            with st.expander("Resumen por combustible", expanded=False):
                render_brent_model_cards(brent_view)

            with st.expander("Ver detalle técnico de Brent"):
                phase3_display = brent_view[
                    [
                        "combustible",
                        "mae_historico",
                        "mae_con_brent",
                        "mejora_mae_brent_pct",
                        "rezago_brent_meses",
                        "modelo_ganador",
                    ]
                ].rename(
                    columns={
                        "combustible": "Combustible",
                        "mae_historico": "MAE histórico (error absoluto medio)",
                        "mae_con_brent": "MAE con Brent (error absoluto medio)",
                        "mejora_mae_brent_pct": "Mejora MAE (error absoluto medio, %)",
                        "rezago_brent_meses": "Rezago Brent",
                        "modelo_ganador": "Modelo ganador",
                    }
                )
                st.dataframe(phase3_display, width="stretch", hide_index=True)

            st.caption(
                f"Fuente y descarga mensual automatizada: [U.S. Energy Information Administration]({EIA_BRENT_SOURCE_URL}). "
                "Para meses futuros, Brent se extiende con un modelo auxiliar validado; no representa una cotización futura de mercado."
            )
        except (ValueError, OSError, KeyError, IndexError) as exc:
            phase3_error = str(exc)
            st.warning(f"No se pudo completar la comparación con Brent: {exc}")

    sectoral_ipc_df = pd.DataFrame()
    sectoral_merged_df = pd.DataFrame()
    sectoral_summary_df = pd.DataFrame()
    sectoral_ccf_df = pd.DataFrame()
    st.divider()
    st.header("IPC sectorial: transmisión y rezagos")
    st.caption(
        "Compara variaciones mensuales del combustible con divisiones del IPC. "
        "Los resultados muestran asociación temporal, precedencia tipo Granger y cointegración de largo plazo; "
        "no demuestran causalidad estructural por sí solos."
    )
    sector_source: Any = uploaded_ipc_sectoral
    if sector_source is None:
        default_sector_path = BASE_DIR / DEFAULT_SECTORAL_IPC_NAME
        sector_source = default_sector_path if default_sector_path.exists() else None

    if sector_source is None:
        st.info("Carga el Excel sectorial del INEC o coloca el archivo oficial en la carpeta del programa.")
    else:
        try:
            sectoral_ipc_df = load_sectoral_ipc(sector_source)
            available_fuels = [col for col in FUEL_NAMES if col in monthly_df.columns]
            selected_ipc_fuel = st.selectbox(
                "Combustible para analizar",
                available_fuels,
                format_func=lambda value: FUEL_NAMES[value],
                key="ipc_sector_fuel",
            )
            sectoral_merged_df, sectoral_summary_df, sectoral_ccf_df = analyze_sectoral_ipc(
                sectoral_ipc_df,
                monthly_df,
                selected_ipc_fuel,
            )
            available_sectors = sectoral_summary_df["sector"].tolist()
            selected_sector = st.selectbox(
                "Componente del IPC",
                available_sectors,
                format_func=lambda value: IPC_SECTOR_NAMES[value],
                key="ipc_sector_name",
            )
            selected_result = sectoral_summary_df[sectoral_summary_df["sector"] == selected_sector].iloc[0]
            relationship = describe_ipc_relationship(
                selected_result["correlacion_ccf"],
                selected_result["p_valor_ccf"],
            )
            predictive = describe_ipc_predictive_effect(selected_result["mejora_mae_pct"])
            long_term = "Sí" if bool(selected_result["cointegracion_5_pct"]) else "No clara"
            direction = describe_ipc_direction(selected_result["correlacion_ccf"])
            ccf_lag = int(selected_result["rezago_ccf_meses"])
            predictive_lag = int(selected_result["rezago_predictivo_meses"])
            improvement = float(selected_result["mejora_mae_pct"])

            ipc_col1, ipc_col2 = st.columns([1.25, 1])
            with ipc_col1:
                st.pyplot(create_ipc_lag_figure(sectoral_ccf_df, selected_sector), clear_figure=True)
            with ipc_col2:
                st.subheader("Lectura rápida")
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("¿Hay relación?", relationship)
                    st.metric("¿Mejora el pronóstico?", predictive)
                with metric_col2:
                    st.metric("¿Cuánto tarda en sentirse?", f"{predictive_lag} meses")
                    st.metric("¿Hay señal de largo plazo?", long_term)
                effect_text = (
                    f"mejora el MAE (error absoluto medio) en {improvement:.1f}%"
                    if improvement > 0
                    else f"no mejora el MAE (error absoluto medio) y cambia {improvement:.1f}%"
                )
                st.info(
                    f"Para **{selected_result['sector_nombre']}**, la señal observada es **{direction.lower()}**. "
                    f"La relación más visible aparece alrededor de **{ccf_lag} meses** y, al usar esta señal en el modelo, "
                    f"el mejor desfase práctico es **{predictive_lag} meses**. En términos predictivos, esta variable **{effect_text}**."
                )

            with st.expander("Resumen por sector", expanded=False):
                render_ipc_sector_cards(sectoral_summary_df, selected_sector)

            with st.expander("Ver detalle técnico del análisis"):
                display_summary = sectoral_summary_df[
                    [
                        "sector_nombre",
                        "rezago_ccf_meses",
                        "correlacion_ccf",
                        "p_valor_ccf",
                        "rezago_predictivo_meses",
                        "mae_base",
                        "mae_con_combustible",
                        "mejora_mae_pct",
                    ]
                ].rename(
                    columns={
                        "sector_nombre": "Componente IPC",
                        "rezago_ccf_meses": "Meses donde se ve más la relación",
                        "correlacion_ccf": "Relación temporal",
                        "p_valor_ccf": "Confianza estadística (p-valor)",
                        "rezago_predictivo_meses": "Meses usados en el pronóstico",
                        "mae_base": "MAE base (error absoluto medio)",
                        "mae_con_combustible": "MAE con combustible (error absoluto medio)",
                        "mejora_mae_pct": "Cambio porcentual del MAE",
                    }
                )
                st.dataframe(display_summary, width="stretch", hide_index=True)

                st.subheader("Granger y cointegración")
                causal_display = sectoral_summary_df[
                    [
                        "sector_nombre",
                        "granger_combustible_ipc_rezago",
                        "granger_combustible_ipc_p_valor",
                        "granger_ipc_combustible_rezago",
                        "granger_ipc_combustible_p_valor",
                        "cointegracion_p_valor",
                        "cointegracion_5_pct",
                    ]
                ].rename(
                    columns={
                        "sector_nombre": "Componente IPC",
                        "granger_combustible_ipc_rezago": "Rezago Granger comb.→IPC",
                        "granger_combustible_ipc_p_valor": "p-valor comb.→IPC",
                        "granger_ipc_combustible_rezago": "Rezago Granger IPC→comb.",
                        "granger_ipc_combustible_p_valor": "p-valor IPC→comb.",
                        "cointegracion_p_valor": "p-valor cointegración",
                        "cointegracion_5_pct": "Cointegrada al 5%",
                    }
                )
                st.dataframe(causal_display, width="stretch", hide_index=True)

                granger_forward_sig = bool(selected_result["granger_combustible_ipc_significativo"])
                granger_reverse_sig = bool(selected_result["granger_ipc_combustible_significativo"])
                coint_sig = bool(selected_result["cointegracion_5_pct"])
                if granger_forward_sig:
                    granger_forward_text = (
                        f"Sí hay evidencia de precedencia temporal tipo Granger desde el combustible hacia "
                        f"{selected_result['sector_nombre']} con mejor rezago de "
                        f"{int(selected_result['granger_combustible_ipc_rezago'])} meses."
                    )
                else:
                    granger_forward_text = (
                        f"No hay evidencia suficiente al 5 % de precedencia tipo Granger desde el combustible hacia "
                        f"{selected_result['sector_nombre']}."
                    )

                if granger_reverse_sig:
                    granger_reverse_text = (
                        f"En sentido inverso, el IPC sectorial también muestra señal temporal sobre el combustible "
                        f"con mejor rezago de {int(selected_result['granger_ipc_combustible_rezago'])} meses."
                    )
                else:
                    granger_reverse_text = (
                        "En sentido inverso no aparece evidencia fuerte al 5 % en la prueba de Granger."
                    )

                if coint_sig:
                    coint_text = (
                        "Además, las series en niveles lucen cointegradas al 5 %, lo que sugiere una relación "
                        "de equilibrio de largo plazo."
                    )
                else:
                    coint_text = (
                        "No aparece cointegración al 5 % en niveles, así que la relación observada parece más "
                        "de corto/mediano plazo que de equilibrio estable."
                    )
                st.info(f"{granger_forward_text} {granger_reverse_text} {coint_text}")
            st.caption(
                f"Período común: {selected_result['periodo_inicio']:%Y-%m} a "
                f"{selected_result['periodo_fin']:%Y-%m}. Fuente del IPC: [INEC Panamá]({IPC_SOURCE_URL})."
            )
        except (ValueError, OSError, KeyError, IndexError) as exc:
            st.error(f"No se pudo construir el análisis de IPC sectorial: {exc}")

    regional_raw_df = pd.DataFrame()
    regional_summary_df = pd.DataFrame()
    regional_normalized_df = pd.DataFrame()
    st.divider()
    st.header("Comparación regional estructural")
    try:
        with st.spinner("Consultando indicadores regionales oficiales..."):
            regional_raw_df, regional_summary_df, regional_normalized_df, regional_source_status = (
                load_regional_benchmark()
            )
        regional_country_view = build_regional_country_view(
            regional_summary_df,
            regional_normalized_df,
        )
        most_exposed_row = regional_country_view.sort_values(
            "exposicion_promedio_score",
            ascending=False,
        ).iloc[0]
        panama_row = regional_country_view[regional_country_view["pais"] == "Panamá"].iloc[0]
        highest_logistics_row = regional_country_view.sort_values(
            "Fricción logística",
            ascending=False,
        ).iloc[0]
        highest_fx_row = regional_country_view.sort_values(
            "Exposición cambiaria",
            ascending=False,
        ).iloc[0]

        st.caption(
            "Benchmark Panamá–Costa Rica–República Dominicana con indicadores comparables. "
            "La escala 0–100 solo compara estos tres países y ayuda a ubicar dónde se concentra más presión relativa."
        )
        regional_col1, regional_col2 = st.columns([1.25, 1])
        with regional_col1:
            st.pyplot(create_regional_exposure_figure(regional_normalized_df), clear_figure=True)
        with regional_col2:
            st.subheader("Lectura rápida")
            quick_col1, quick_col2 = st.columns(2)
            with quick_col1:
                st.metric("País más expuesto", most_exposed_row["pais"])
                st.metric(
                    "Panamá en el bloque",
                    f"{panama_row['exposicion_relativa']} ({int(panama_row['puesto_bloque'])} de {len(regional_country_view)})",
                )
            with quick_col2:
                st.metric("Mayor presión logística", highest_logistics_row["pais"])
                st.metric("Mayor presión cambiaria", highest_fx_row["pais"])
            st.info(
                f"En esta comparación, **{most_exposed_row['pais']}** es el país con mayor exposición relativa. "
                f"**Panamá** se ubica en la posición **{int(panama_row['puesto_bloque'])} de {len(regional_country_view)}** "
                f"y su principal presión proviene de **{str(panama_row['principal_presion']).lower()}**."
            )

        with st.expander("Resumen por país", expanded=False):
            render_regional_country_cards(regional_country_view)

        with st.expander("Ver detalle técnico de la comparación"):
            regional_display = regional_summary_df[
                [
                    "pais",
                    "lpi",
                    "lpi_anio",
                    "importacion_combustible_pct",
                    "importacion_combustible_pct_anio",
                    "inflacion_pct",
                    "inflacion_pct_anio",
                    "variacion_cambiaria_pct",
                    "variacion_cambiaria_anio",
                ]
            ].rename(
                columns={
                    "pais": "País",
                    "lpi": "LPI",
                    "lpi_anio": "Año LPI",
                    "importacion_combustible_pct": "Combustible / importaciones (%)",
                    "importacion_combustible_pct_anio": "Año importaciones",
                    "inflacion_pct": "Inflación general (%)",
                    "inflacion_pct_anio": "Año inflación",
                    "variacion_cambiaria_pct": "Variación cambiaria (%)",
                    "variacion_cambiaria_anio": "Año cambiario",
                }
            )
            numeric_columns = [
                "LPI",
                "Combustible / importaciones (%)",
                "Inflación general (%)",
                "Variación cambiaria (%)",
            ]
            regional_display[numeric_columns] = regional_display[numeric_columns].round(2)
            st.dataframe(regional_display, width="stretch", hide_index=True)
            st.info(
                "Un valor alto en la escala normalizada indica mayor exposición relativa dentro del bloque. "
                "La fricción logística invierte el LPI: menor desempeño logístico equivale a mayor fricción. "
                "La exposición cambiaria usa la magnitud de la variación anual de la moneda frente al dólar."
            )

        st.subheader("Tipo de cambio")
        st.caption(
            "Se muestra el tipo de cambio oficial más reciente frente al dólar y su variación anual. "
            "Una mayor variación sugiere más riesgo de traspaso cambiario hacia combustibles importados."
        )
        fx_country_view = build_fx_country_view(regional_summary_df)
        fx_most_exposed_row = fx_country_view.sort_values(
            "magnitud_movimiento",
            ascending=False,
        ).iloc[0]
        fx_stable_row = fx_country_view.sort_values(
            "magnitud_movimiento",
            ascending=True,
        ).iloc[0]
        panama_fx_row = fx_country_view[fx_country_view["pais"] == "Panamá"].iloc[0]

        fx_col1, fx_col2 = st.columns([1.15, 1])
        with fx_col1:
            st.pyplot(create_exchange_rate_figure(regional_summary_df), clear_figure=True)
        with fx_col2:
            st.subheader("Lectura rápida")
            fx_quick_col1, fx_quick_col2 = st.columns(2)
            with fx_quick_col1:
                st.metric("Mayor presión cambiaria", fx_most_exposed_row["pais"])
                st.metric(
                    "Panamá frente al USD",
                    panama_fx_row["movimiento_cambiario"],
                )
            with fx_quick_col2:
                st.metric("País más estable", fx_stable_row["pais"])
                st.metric(
                    "Mayor movimiento anual",
                    f"{float(fx_most_exposed_row['magnitud_movimiento']):.2f}%",
                )
            st.info(
                f"En este bloque, **{fx_most_exposed_row['pais']}** muestra la mayor presión cambiaria reciente. "
                f"**Panamá** aparece como referencia **{str(panama_fx_row['movimiento_cambiario']).lower()}**, "
                f"mientras **{fx_stable_row['pais']}** es el país con menor variación anual."
            )

        with st.expander("Resumen cambiario por país", expanded=False):
            render_fx_country_cards(fx_country_view)

        with st.expander("Ver detalle técnico del tipo de cambio"):
            fx_display = regional_summary_df[
                [
                    "pais",
                    "tipo_cambio_oficial",
                    "tipo_cambio_anio",
                    "variacion_cambiaria_pct",
                    "variacion_cambiaria_anio",
                ]
            ].rename(
                columns={
                    "pais": "País",
                    "tipo_cambio_oficial": "Tipo de cambio oficial",
                    "tipo_cambio_anio": "Año tipo de cambio",
                    "variacion_cambiaria_pct": "Variación anual (%)",
                    "variacion_cambiaria_anio": "Año variación",
                }
            )
            fx_display["Unidad"] = regional_summary_df["codigo_pais"].map(REGIONAL_CURRENCY_LABELS)
            fx_display[
                ["Tipo de cambio oficial", "Variación anual (%)"]
            ] = fx_display[["Tipo de cambio oficial", "Variación anual (%)"]].round(2)
            st.dataframe(fx_display, width="stretch", hide_index=True)
            st.info(
                "La lectura usa la variación anual de la moneda local por USD. "
                "Un aumento indica más unidades de moneda local por dólar y, por tanto, mayor presión cambiaria."
            )

        st.subheader("Fletes y logística")
        st.caption(
            "Se aproxima con dos señales estructurales oficiales: desempeño logístico (LPI) y dependencia "
            "importadora de combustibles. Mayor dependencia y menor LPI implican más presión potencial de fletes."
        )
        logistics_country_view = build_logistics_country_view(
            regional_summary_df,
            regional_normalized_df,
        )
        logistics_most_exposed_row = logistics_country_view.sort_values(
            "presion_fletes_logistica_score",
            ascending=False,
        ).iloc[0]
        logistics_lowest_lpi_row = logistics_country_view.sort_values("lpi", ascending=True).iloc[0]
        logistics_highest_import_row = logistics_country_view.sort_values(
            "importacion_combustible_pct",
            ascending=False,
        ).iloc[0]
        panama_logistics_row = logistics_country_view[logistics_country_view["pais"] == "Panamá"].iloc[0]

        logistics_col1, logistics_col2 = st.columns([1.15, 1])
        with logistics_col1:
            st.pyplot(create_logistics_freight_figure(regional_summary_df), clear_figure=True)
        with logistics_col2:
            st.subheader("Lectura rápida")
            logistics_quick_col1, logistics_quick_col2 = st.columns(2)
            with logistics_quick_col1:
                st.metric("Mayor presión logística", logistics_most_exposed_row["pais"])
                st.metric(
                    "Panamá en el bloque",
                    f"{panama_logistics_row['presion_logistica']} ({int(panama_logistics_row['puesto_presion'])} de {len(logistics_country_view)})",
                )
            with logistics_quick_col2:
                st.metric("Mayor dependencia importadora", logistics_highest_import_row["pais"])
                st.metric("Menor desempeño logístico", logistics_lowest_lpi_row["pais"])
            st.info(
                f"En esta comparación, **{logistics_most_exposed_row['pais']}** concentra la mayor presión logística. "
                f"**Panamá** ocupa la posición **{int(panama_logistics_row['puesto_presion'])} de {len(logistics_country_view)}** "
                f"y su principal factor es **{str(panama_logistics_row['principal_factor']).lower()}**."
            )

        with st.expander("Resumen logístico por país", expanded=False):
            render_logistics_country_cards(logistics_country_view)

        with st.expander("Ver detalle técnico de fletes y logística"):
            logistics_display = regional_summary_df[
                [
                    "pais",
                    "lpi",
                    "lpi_anio",
                    "importacion_combustible_pct",
                    "importacion_combustible_pct_anio",
                    "presion_fletes_logistica_score",
                ]
            ].rename(
                columns={
                    "pais": "País",
                    "lpi": "LPI",
                    "lpi_anio": "Año LPI",
                    "importacion_combustible_pct": "Combustible / importaciones (%)",
                    "importacion_combustible_pct_anio": "Año importaciones",
                    "presion_fletes_logistica_score": "Presión fletes/logística (0-100)",
                }
            )
            logistics_display[
                ["LPI", "Combustible / importaciones (%)", "Presión fletes/logística (0-100)"]
            ] = logistics_display[
                ["LPI", "Combustible / importaciones (%)", "Presión fletes/logística (0-100)"]
            ].round(2)
            st.dataframe(logistics_display, width="stretch", hide_index=True)
            st.info(
                "La presión de fletes y logística combina dos señales: menor LPI implica más fricción, "
                "y una mayor dependencia importadora aumenta la exposición del abastecimiento."
            )
        st.caption(
            f"Fuente: [World Development Indicators – Banco Mundial]({WORLD_BANK_DATA_URL}). "
            f"Modo de consulta: {regional_source_status}. Inflación general se usa como proxy macroeconómico y no sustituye "
            "una serie armonizada de inflación alimentaria."
        )
    except (ValueError, OSError, KeyError, IndexError, requests.RequestException) as exc:
        st.error(f"No se pudo construir la comparación regional: {exc}")

    tab1, tab2, tab3 = st.tabs(["Serie mensual", "Histórico y pronóstico", "Control de hojas"])
    with tab1:
        st.dataframe(monthly_df, width="stretch")
    with tab2:
        st.dataframe(combined_df, width="stretch")
    with tab3:
        st.dataframe(sheet_log, width="stretch")

    export_sheets = {
        "serie_mensual": monthly_df,
        "historico_y_pronostico": combined_df,
        "resumen_pronostico": summary_df,
        "control_hojas": sheet_log,
    }
    if not sectoral_ipc_df.empty:
        export_sheets["ipc_sectorial"] = sectoral_ipc_df
    if not sectoral_merged_df.empty:
        export_sheets["ipc_y_combustible"] = sectoral_merged_df
    if not sectoral_summary_df.empty:
        export_sheets["resumen_rezagos_ipc"] = sectoral_summary_df
    if not sectoral_ccf_df.empty:
        export_sheets["ccf_ipc"] = sectoral_ccf_df
    if not regional_raw_df.empty:
        export_sheets["regional_datos_bm"] = regional_raw_df
    if not regional_summary_df.empty:
        export_sheets["regional_resumen"] = regional_summary_df
    if not regional_normalized_df.empty:
        export_sheets["regional_normalizado"] = regional_normalized_df
    if not brent_monthly_df.empty:
        export_sheets["brent_mensual"] = brent_monthly_df
    if not brent_projection_df.empty:
        export_sheets["brent_y_proyeccion"] = brent_projection_df
    phase3_columns = [
        "serie",
        "modelo_historico",
        "mae_historico",
        "mae_con_brent",
        "mejora_mae_brent_pct",
        "rezago_brent_meses",
        "impacto_brent_10_pct",
        "modelo_ganador",
    ]
    if all(column in summary_df.columns for column in phase3_columns):
        export_sheets["comparacion_modelos_fase3"] = summary_df[phase3_columns]
    excel_bytes = dataframe_download_bytes(export_sheets)
    st.download_button("Descargar resultados en Excel", excel_bytes, DEFAULT_OUTPUT_NAME, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
    if get_script_run_ctx() is None:
        print(f"Esta es una app Streamlit. Ejecútala así:\n\n{sys.executable} -m streamlit run {Path(__file__).resolve()}")
        sys.exit(0)
    main()

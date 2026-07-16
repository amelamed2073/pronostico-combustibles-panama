from __future__ import annotations

from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Optional
import re
import sys
import unicodedata
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from scipy.stats import pearsonr
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from streamlit.runtime.scriptrunner import get_script_run_ctx


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_FILE_NAMES = [
    "Precio historico de la gasolina (2015-2026) (2).xlsx",
]
DEFAULT_SECTORAL_IPC_NAME = "ipc_sectorial_panama_2017_2025.xlsx"
DEFAULT_OUTPUT_NAME = "pronostico_combustibles_panama_web.xlsx"
IPC_SOURCE_URL = "https://www.inec.gob.pa/publicaciones/Default3.aspx?ID_CATEGORIA=4&ID_PUBLICACION=1396&ID_SUBCATEGORIA=82"
WORLD_BANK_API_BASE = "https://api.worldbank.org/v2"
WORLD_BANK_DATA_URL = "https://data.worldbank.org"
EIA_BRENT_SOURCE_URL = "https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?f=m&n=PET&s=RBRTE"
PRICE_UNIT = "B/./litro"
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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_panama_prices() -> tuple[dict[str, float], Optional[str]]:
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
        title=f"CCF: {FUEL_NAMES[data['combustible'].iloc[0]]} → {IPC_SECTOR_NAMES[sector]}",
        xlabel="Rezago del combustible (meses)",
        ylabel="Correlación de variaciones mensuales",
    )
    ax.set_xticks(data["rezago_meses"])
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


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
    ax.set_xlabel("Exposición relativa dentro del bloque (0–100)")
    ax.set_title("Comparación estructural regional normalizada")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    return fig


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


def render_forecast_cards_final(summary_df: pd.DataFrame, selected_horizon: int) -> None:
    forecast_col = f"pronostico_mes_{selected_horizon}"
    lower_col = f"lim_inf_mes_{selected_horizon}"
    upper_col = f"lim_sup_mes_{selected_horizon}"
    cards = [
        """
        <style>
        :root { color-scheme: light dark; }
        html, body {
            margin: 0;
            padding: 0;
            background: transparent;
            color: #1f2937;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        .forecast-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 12px;
        }
        .forecast-card {
            border: 1px solid rgba(31, 41, 55, 0.22);
            border-radius: 14px;
            padding: 15px 18px;
            background: rgba(255, 255, 255, 0.78);
        }
        .forecast-card-title {
            color: #374151;
            font-size: 0.95rem;
            font-weight: 600;
            margin-bottom: 10px;
        }
        .forecast-card-label {
            color: #6b7280;
            font-size: 0.78rem;
            margin-top: 8px;
        }
        .forecast-card-value {
            color: #111827;
            font-size: 1.8rem;
            font-weight: 700;
            margin: 2px 0 7px;
        }
        .forecast-card-detail {
            color: #4b5563;
            font-size: 0.88rem;
            font-weight: 500;
        }
        @media (prefers-color-scheme: dark) {
            html, body { color: #f3f4f6; }
            .forecast-card {
                background: rgba(255, 255, 255, 0.07);
                border-color: rgba(255, 255, 255, 0.22);
            }
            .forecast-card-title { color: #e5e7eb; }
            .forecast-card-label { color: #9ca3af; }
            .forecast-card-value { color: #ffffff; }
            .forecast-card-detail { color: #d1d5db; }
        }
        </style>
        <div class="forecast-grid">
        """
    ]
    for _, row in summary_df.iterrows():
        name = escape(FUEL_NAMES.get(str(row["serie"]), str(row["serie"])))
        cards.append(
            f"""
            <div class="forecast-card">
                <div class="forecast-card-title">{name}</div>
                <div class="forecast-card-label">Precio pronosticado · mes {selected_horizon}</div>
                <div class="forecast-card-value">{float(row[forecast_col]):.4f} {escape(PRICE_UNIT)}</div>
                <div class="forecast-card-label">Intervalo aproximado del 95 %</div>
                <div class="forecast-card-detail">
                    {float(row[lower_col]):.4f}–{float(row[upper_col]):.4f} {escape(PRICE_UNIT)}
                </div>
                <div class="forecast-card-label">Tendencia central</div>
                <div class="forecast-card-detail">{escape(str(row['direccion_esperada']).capitalize())}</div>
            </div>
            """
        )
    cards.append("</div>")
    component_height = max(190, 190 * len(summary_df))
    components.html("".join(cards), height=component_height, scrolling=False)


def render_live_prices_box() -> None:
    st.subheader("Precios vigentes en línea de Panamá")
    with st.spinner("Consultando precios vigentes..."):
        prices, source_url = fetch_live_panama_prices()
    if not prices:
        st.info("No se pudieron validar precios vigentes en la fuente configurada.")
        return
    cols = st.columns(len(prices))
    for column, (key, value) in zip(cols, prices.items()):
        column.metric(FUEL_NAMES[key], f"{value:.4f} {PRICE_UNIT}")
    if source_url:
        st.caption(f"Fuente: {source_url} · Consulta almacenada durante una hora")


def main() -> None:
    st.set_page_config(page_title="Pronóstico de combustibles Panamá", layout="wide")
    st.title("Pronóstico web de combustibles en Panamá")
    st.caption(f"Selección por validación temporal · precios expresados en {PRICE_UNIT}")

    with st.sidebar:
        st.header("Configuración")
        selected_horizon = st.slider("Meses a pronosticar", 3, 18, 6)
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

    top_col1, top_col2 = st.columns([1.3, 1])
    with top_col1:
        st.pyplot(create_historical_figure(combined_df), clear_figure=True)
    with top_col2:
        st.subheader("Validación del modelo")
        st.dataframe(summary_df[["serie", "modelo", "mae_validacion", "meses_validacion", "direccion_esperada"]], width="stretch", hide_index=True)

    bottom_col1, bottom_col2 = st.columns([1.3, 1])
    with bottom_col1:
        st.pyplot(create_forecast_figure(combined_df, selected_horizon), clear_figure=True)
    with bottom_col2:
        render_live_prices_box()
        st.subheader("Pronóstico final e intervalo aproximado")
        render_forecast_cards_final(summary_df, selected_horizon)

    st.divider()
    st.header("Pronóstico con variable externa")
    st.caption(
        "Compara el mejor modelo histórico con una regresión dinámica que incorpora la variación mensual del Brent. "
        "La validación utiliza el mismo bloque temporal final y selecciona automáticamente el menor MAE."
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
                phase3_display = summary_df[
                    [
                        "serie",
                        "mae_historico",
                        "mae_con_brent",
                        "mejora_mae_brent_pct",
                        "rezago_brent_meses",
                        "modelo_ganador",
                    ]
                ].copy()
                phase3_display["serie"] = phase3_display["serie"].map(FUEL_NAMES)
                phase3_display = phase3_display.rename(
                    columns={
                        "serie": "Combustible",
                        "mae_historico": "MAE histórico",
                        "mae_con_brent": "MAE con Brent",
                        "mejora_mae_brent_pct": "Mejora MAE (%)",
                        "rezago_brent_meses": "Rezago Brent",
                        "modelo_ganador": "Modelo ganador",
                    }
                )
                st.dataframe(phase3_display, width="stretch", hide_index=True)

            winners = int((summary_df["modelo_ganador"] == "Brent").sum())
            st.info(
                f"Resultado de selección: el modelo con Brent ganó para {winners} de "
                f"{len(summary_df)} combustibles. Cuando no reduce el MAE, la aplicación conserva el modelo histórico. "
                "El rezago indica cuántos meses tarda la variación del Brent en aportar señal predictiva."
            )
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
        "Los resultados muestran asociación temporal y capacidad predictiva; no demuestran causalidad."
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

            ipc_col1, ipc_col2 = st.columns([1.25, 1])
            with ipc_col1:
                st.pyplot(create_ipc_lag_figure(sectoral_ccf_df, selected_sector), clear_figure=True)
            with ipc_col2:
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
                        "rezago_ccf_meses": "Rezago CCF",
                        "correlacion_ccf": "Correlación",
                        "p_valor_ccf": "p-valor",
                        "rezago_predictivo_meses": "Rezago predictivo",
                        "mae_base": "MAE base",
                        "mae_con_combustible": "MAE con combustible",
                        "mejora_mae_pct": "Mejora MAE (%)",
                    }
                )
                st.dataframe(display_summary, width="stretch", hide_index=True)

            selected_result = sectoral_summary_df[sectoral_summary_df["sector"] == selected_sector].iloc[0]
            significance = "estadísticamente distinguible de cero al 5 %" if selected_result["p_valor_ccf"] < 0.05 else "no significativa al 5 %"
            direction = "positiva" if selected_result["correlacion_ccf"] >= 0 else "negativa"
            improvement = selected_result["mejora_mae_pct"]
            predictive_text = (
                f"El modelo con combustible redujo el MAE final en {improvement:.1f} %."
                if improvement > 0
                else f"El combustible no mejoró el MAE final ({improvement:.1f} %)."
            )
            st.info(
                f"Lectura: la asociación más intensa es {direction} con un rezago de "
                f"{int(selected_result['rezago_ccf_meses'])} meses y resulta {significance}. "
                f"El rezago predictivo elegido fue {int(selected_result['rezago_predictivo_meses'])} meses. "
                f"{predictive_text}"
            )
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
    st.caption(
        "Benchmark Panamá–Costa Rica–República Dominicana con indicadores comparables. "
        "Las escalas normalizadas describen exposición relativa dentro de este bloque; no son un ranking mundial."
    )
    try:
        with st.spinner("Consultando indicadores regionales oficiales..."):
            regional_raw_df, regional_summary_df, regional_normalized_df, regional_source_status = (
                load_regional_benchmark()
            )

        regional_col1, regional_col2 = st.columns([1.25, 1])
        with regional_col1:
            st.pyplot(create_regional_exposure_figure(regional_normalized_df), clear_figure=True)
        with regional_col2:
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
            "Lectura: un valor alto en la escala normalizada indica mayor exposición dentro de los tres países. "
            "La fricción logística invierte el LPI: menor desempeño logístico equivale a mayor fricción. "
            "La exposición cambiaria usa la magnitud de la variación anual de la moneda frente al dólar."
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

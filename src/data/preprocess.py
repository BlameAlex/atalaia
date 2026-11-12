"""Preprocesamiento: panel condado-año -> dataset de pronóstico (features en t, nivel de riesgo en t+1).

Uso:
    .venv/bin/python src/data/preprocess.py --raw data/raw/county_year_climate_flood_risk.csv --out data/processed/
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ANIO_FINAL = 2026
TRAIN_YEARS = (2001, 2020)
TEST_YEARS = (2021, 2025)
MIN_ANIOS_REALES = 3

ESTADOS = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR", "CALIFORNIA": "CA",
    "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE", "DISTRICT OF COLUMBIA": "DC",
    "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI", "IDAHO": "ID", "ILLINOIS": "IL",
    "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS", "KENTUCKY": "KY", "LOUISIANA": "LA",
    "MAINE": "ME", "MARYLAND": "MD", "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN",
    "MISSISSIPPI": "MS", "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK", "OREGON": "OR",
    "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT", "VERMONT": "VT", "VIRGINIA": "VA",
    "WASHINGTON": "WA", "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY",
    "PUERTO RICO": "PR", "VIRGIN ISLANDS": "VI", "GUAM": "GU", "AMERICAN SAMOA": "AS",
    "NORTHERN MARIANA ISLANDS": "MP", "FEDERATED STATES OF MICRONESIA": "FM",
    "MARSHALL ISLANDS": "MH", "PALAU": "PW",
}

CONTEOS = ["nfip_claim_count", "high_severity_claim_count", "flood_zone_count", "storm_event_count",
           "flood_related_event_count", "storm_injuries", "storm_deaths", "event_type_count",
           "fema_declaration_count", "fema_incident_type_count", "paid_loss_to_claim_coverage"]
MONTOS = ["nfip_paid_loss_usd", "nfip_coverage_usd", "average_claim_severity_usd",
          "storm_property_damage_usd", "storm_total_damage_usd"]
PERCENTILES = ["claim_loss_percentile", "storm_damage_percentile", "event_frequency_percentile",
               "human_impact_percentile", "declaration_percentile"]
SCORE = "composite_risk_score"
LAGS = [SCORE, "nfip_claim_count", "flood_related_event_count", "fema_declaration_count"]
MAS = [SCORE, "nfip_claim_count", "flood_related_event_count", "log_nfip_paid_loss_usd", "fema_declaration_count"]

ETIQUETAS_BASE = {
    "nfip_claim_count": "Reclamos NFIP", "high_severity_claim_count": "Reclamos NFIP de alta severidad",
    "flood_zone_count": "Zonas de inundación con reclamos", "storm_event_count": "Eventos de tormenta NOAA",
    "flood_related_event_count": "Eventos de inundación NOAA", "storm_injuries": "Heridos por tormentas",
    "storm_deaths": "Muertes por tormentas", "event_type_count": "Tipos de evento NOAA",
    "fema_declaration_count": "Declaraciones FEMA", "fema_incident_type_count": "Tipos de incidente FEMA",
    "paid_loss_to_claim_coverage": "Pérdida pagada / cobertura NFIP",
    "log_nfip_paid_loss_usd": "Pérdida pagada NFIP (log USD)", "log_nfip_coverage_usd": "Cobertura NFIP (log USD)",
    "log_average_claim_severity_usd": "Severidad media de reclamo (log USD)",
    "log_storm_property_damage_usd": "Daño a propiedad por tormentas (log USD)",
    "log_storm_total_damage_usd": "Daño total por tormentas (log USD)",
    SCORE: "Puntaje de riesgo compuesto", "claim_loss_percentile": "Percentil de pérdida NFIP",
    "storm_damage_percentile": "Percentil de daño por tormentas",
    "event_frequency_percentile": "Percentil de frecuencia de eventos",
    "human_impact_percentile": "Percentil de impacto humano", "declaration_percentile": "Percentil de declaraciones FEMA",
}


def etiqueta(f):
    for suf, txt in [("_lag1", " (año anterior)"), ("_lag2", " (hace 2 años)"), ("_ma3", " (media 3 años)"),
                     ("_ma5", " (media 5 años)"), ("_hist_mean", " (media histórica)")]:
        if f.endswith(suf):
            return ETIQUETAS_BASE[f[: -len(suf)]] + txt
    if f == "anios_historia":
        return "Años de historia del condado"
    return ETIQUETAS_BASE[f]


def limpiar(df):
    df["county_fips"] = df["county_fips"].str.zfill(5)
    st = df["state"].str.strip().str.upper()
    df["state"] = st.map(ESTADOS).fillna(st)
    no_mapeados = sorted(set(df.loc[df["state"].str.len() != 2, "state"]))
    df = df[~df["county_fips"].str.endswith("000")].copy()
    df["paid_loss_to_claim_coverage"] = df["paid_loss_to_claim_coverage"].fillna(0)
    # hay pagos negativos (ajustes contables NFIP); log1p no los admite -> recorte a 0
    neg = int((df[MONTOS + ["paid_loss_to_claim_coverage"]] < 0).any(axis=1).sum())
    print(f"[INFO] filas con montos negativos recortados a 0: {neg}")
    df[MONTOS + ["paid_loss_to_claim_coverage"]] = df[MONTOS + ["paid_loss_to_claim_coverage"]].clip(lower=0)
    # un estado por condado (el más frecuente) por si quedara mezcla
    df["state"] = df.groupby("county_fips")["state"].transform(lambda s: s.mode().iat[0])
    return df, no_mapeados


def reindexar(df, piso, pct_min):
    primer = df.groupby("county_fips")["year"].min()
    idx = pd.MultiIndex.from_tuples([(c, y) for c, y0 in primer.items() for y in range(y0, ANIO_FINAL + 1)],
                                    names=["county_fips", "year"])
    estado = df.groupby("county_fips")["state"].first()
    p = df.set_index(["county_fips", "year"]).reindex(idx)
    p["fila_imputada"] = p[SCORE].isna().astype(int)
    p[SCORE] = p[SCORE].fillna(piso)
    for c, v in pct_min.items():
        p[c] = p[c].fillna(v)
    p = p.fillna({c: 0 for c in CONTEOS + MONTOS + ["storm_crop_damage_usd"]}).reset_index()
    p["state"] = p["county_fips"].map(estado)
    return p


def construir_features(p, piso):
    """Features del año t usando solo filas <= t de cada condado (panel contiguo, ordenado)."""
    p = p.sort_values(["county_fips", "year"]).reset_index(drop=True)
    f = p[["county_fips", "state", "year"] + CONTEOS + [SCORE] + PERCENTILES].copy()
    for c in MONTOS:
        f[f"log_{c}"] = np.log1p(p[c])
    g = f.groupby("county_fips")
    for c in LAGS:
        relleno = piso if c == SCORE else 0
        for k in (1, 2):
            f[f"{c}_lag{k}"] = g[c].shift(k).fillna(relleno)
    for c in MAS:
        for w in (3, 5):
            f[f"{c}_ma{w}"] = g[c].transform(lambda s: s.rolling(w, min_periods=1).mean())
    f[f"{SCORE}_hist_mean"] = g[SCORE].transform(lambda s: s.expanding().mean())
    f["anios_historia"] = g.cumcount() + 1
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="models")
    a = ap.parse_args()
    out, mdir = Path(a.out), Path(a.models)
    out.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(a.raw, dtype={"county_fips": str})
    n_raw = len(raw)
    df, no_mapeados = limpiar(raw)
    print(f"[INFO] filas crudas={n_raw}, tras quitar xx000={len(df)}, estados={df['state'].nunique()}")
    print(f"[INFO] valores de state no mapeados a 2 letras: {no_mapeados or 'ninguno'}")

    anios_reales = df.groupby("county_fips")["year"].nunique()
    pocos = anios_reales[anios_reales < MIN_ANIOS_REALES].index
    df = df[~df["county_fips"].isin(pocos)]
    print(f"[INFO] condados eliminados por < {MIN_ANIOS_REALES} años reales: {len(pocos)}")

    piso = float(df[SCORE].min())
    pct_min = {c: float(df[c].min()) for c in PERCENTILES}
    panel = reindexar(df, piso, pct_min)
    n_imp = int(panel["fila_imputada"].sum())

    feats = construir_features(panel, piso)
    features = [c for c in feats.columns if c not in ("county_fips", "state", "year")]

    # target: score de t+1 del mismo condado
    feats["target_year"] = feats["year"] + 1
    feats["composite_risk_score_t1"] = feats.groupby("county_fips")[SCORE].shift(-1)
    ds = feats[feats["target_year"].between(TRAIN_YEARS[0], TEST_YEARS[1])
               & feats["composite_risk_score_t1"].notna()].copy()
    ds["split"] = np.where(ds["target_year"] <= TRAIN_YEARS[1], "train", "test")
    tr = ds["split"] == "train"
    p33, p67 = ds.loc[tr, "composite_risk_score_t1"].quantile([1 / 3, 2 / 3]).tolist()
    ds["nivel_riesgo_t1"] = np.select([ds["composite_risk_score_t1"] < p33,
                                       ds["composite_risk_score_t1"] < p67], [0, 1], 2)
    cols = ["county_fips", "state", "year", "target_year", "composite_risk_score_t1", "nivel_riesgo_t1", "split"]
    ds = ds[cols + features]
    ds.to_csv(out / "dataset_riesgo.csv", index=False)

    latest = feats.loc[feats.groupby("county_fips")["year"].idxmax(), ["county_fips", "state", "year"] + features]
    latest.to_csv(mdir / "feature_store_latest.csv", index=False)

    balance = {s: {str(k): int(v) for k, v in ds.loc[ds["split"] == s, "nivel_riesgo_t1"].value_counts().sort_index().items()}
               for s in ("train", "test")}
    meta = {
        "features": features,
        "etiquetas_features": {f: etiqueta(f) for f in features},
        "umbrales": {"p33": p33, "p67": p67},
        "clases": {"0": "Bajo", "1": "Medio", "2": "Alto"},
        "score_piso": piso,
        "train_target_years": list(TRAIN_YEARS),
        "test_target_years": list(TEST_YEARS),
        "p75_train": ds.loc[tr, features].quantile(0.75).to_dict(),
        "medianas_train": ds.loc[tr, features].median().to_dict(),
        "n_filas": {"train": int(tr.sum()), "test": int((~tr).sum())},
        "balance_clases": balance,
    }
    (mdir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    # --- verificación ---
    assert not ds[features + ["composite_risk_score_t1", "nivel_riesgo_t1"]].isna().any().any(), "NaN en dataset"
    assert ds.loc[tr, "target_year"].max() <= 2020 and ds.loc[~tr, "target_year"].min() >= 2021
    assert (ds["target_year"] != ANIO_FINAL).all()
    assert set(ds["nivel_riesgo_t1"].unique()) == {0, 1, 2}
    assert not latest[features].isna().any().any() and latest["county_fips"].is_unique
    # no-fuga: recalcular truncando a años <= t debe dar las mismas features
    for fips in ds["county_fips"].drop_duplicates().sample(5, random_state=42):
        sub = panel[panel["county_fips"] == fips]
        for t in sub["year"].sample(3, random_state=42):
            trunc = construir_features(sub[sub["year"] <= t], piso)
            a_ = trunc[trunc["year"] == t][features].reset_index(drop=True)
            b_ = feats[(feats["county_fips"] == fips) & (feats["year"] == t)][features].reset_index(drop=True)
            pd.testing.assert_frame_equal(a_, b_, check_dtype=False)

    print(f"[RESULTADOS] filas: train={meta['n_filas']['train']}, test={meta['n_filas']['test']}")
    print(f"[RESULTADOS] balance (0=Bajo,1=Medio,2=Alto): {balance}")
    print(f"[RESULTADOS] umbrales p33={p33:.4f} p67={p67:.4f}, piso={piso:.4f}")
    print(f"[RESULTADOS] condados={ds['county_fips'].nunique()}, features={len(features)}, "
          f"filas imputadas en panel={n_imp} de {len(panel)}, feature store={len(latest)} condados")
    print("[SUCCESS] preprocesamiento completo, asserts OK")


if __name__ == "__main__":
    main()

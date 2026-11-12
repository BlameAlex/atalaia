"""Inferencia del botón SOS: predecir(entrada) -> nivel de riesgo de inundación del condado.

Uso: .venv/bin/python src/models/predict.py   (ejecuta el demo con asserts)
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
MODELS = RAIZ / "models"

# --- Artefactos: se cargan una sola vez al importar el módulo ---
METADATA = json.loads((MODELS / "metadata.json").read_text(encoding="utf-8"))
FEATURES = METADATA["features"]
ETIQUETAS = METADATA.get("etiquetas_features", {})
P75 = METADATA["p75_train"]
CLASES = {int(k): v for k, v in METADATA["clases"].items()}
_imp = METADATA.get("importancias", {})  # {feature: importancia}
IMPORTANCIAS = sorted(_imp, key=_imp.get, reverse=True) or FEATURES

# joblib = pickle: solo se cargan artefactos propios generados por src/models/train.py.
MODELO = joblib.load(MODELS / "modelo_candidato.joblib")
if hasattr(MODELO, "n_jobs"):
    MODELO.n_jobs = 1  # una fila por SOS: el paralelismo de hilos solo agrega latencia
SCALER = joblib.load(MODELS / "scaler.joblib") if METADATA.get("requiere_scaler") else None

_fs = pd.read_csv(MODELS / "feature_store_latest.csv", dtype={"county_fips": str})
_fs["county_fips"] = _fs["county_fips"].str.zfill(5)
STORE = _fs.set_index("county_fips")
X_STORE = STORE[FEATURES]  # orden de columnas igual al de entrenamiento

UMBRAL_CONFIANZA = 70.0  # CA-08 / RNF-02
AVISO = "La IA no puede estimar con seguridad el riesgo de esta zona; requiere revisión manual."


def _factores(fila: pd.Series) -> list:
    def texto(f):
        return f"{ETIQUETAS.get(f, f)}: {round(float(fila[f]), 2)} (p75 train: {round(float(P75[f]), 2)})"
    sobre = [f for f in IMPORTANCIAS if f in P75 and fila[f] > P75[f]][:3]
    return [texto(f) for f in sobre] or [texto(IMPORTANCIAS[0])]


def predecir(entrada: dict) -> dict:
    t0 = time.perf_counter()
    if not isinstance(entrada, dict):
        return {"id_solicitud": None, "status": "error", "mensaje": "La entrada debe ser un objeto JSON (dict)."}
    id_sol = entrada.get("id_solicitud")
    fips = entrada.get("county_fips")
    if fips is None or str(fips).strip() == "":
        return {"id_solicitud": id_sol, "status": "error", "mensaje": "Falta el campo obligatorio county_fips."}
    fips = str(fips).strip()
    if not fips.isdigit():
        return {"id_solicitud": id_sol, "status": "error", "mensaje": f"county_fips no es numérico: '{fips}'."}
    fips = fips.zfill(5)
    if fips not in STORE.index:
        return {"id_solicitud": id_sol, "status": "error", "mensaje": f"Condado {fips} no encontrado en el feature store."}

    x = X_STORE.loc[[fips]]
    if SCALER is not None:
        x = SCALER.transform(x)
    proba = MODELO.predict_proba(x)[0]
    clases_modelo = [int(c) for c in MODELO.classes_]
    probs = {CLASES[c]: round(float(proba[clases_modelo.index(c)]), 4) for c in sorted(CLASES)}
    clase = int(clases_modelo[int(np.argmax(proba))])
    confianza = round(float(proba.max()) * 100, 2)
    fila = STORE.loc[fips]
    anio = int(fila["year"])

    salida = {
        "id_solicitud": id_sol, "county_fips": fips, "state": fila["state"],
        "anio_base": anio, "anio_pronostico": anio + 1,
        "nivel_riesgo": CLASES[clase], "clase_predicha": clase,
        "probabilidades": probs, "confianza": confianza,
        "requiere_revision": confianza < UMBRAL_CONFIANZA,
        "factores_clave": _factores(fila),
    }
    if salida["requiere_revision"]:
        salida["aviso"] = AVISO
    salida["tiempo_procesamiento_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    salida["status"] = "success"
    return salida


if __name__ == "__main__":
    lineas = []

    def log(s=""):
        print(s)
        lineas.append(str(s))

    def dump(o):
        return json.dumps(o, ensure_ascii=False, indent=2)

    orden = STORE["composite_risk_score_ma3"].sort_values()
    casos = [
        ("Condado de riesgo alto (mayor composite_risk_score_ma3)", {"id_solicitud": "SOS-0001", "county_fips": orden.index[-1], "lat": None, "lng": None}),
        ("Condado de riesgo bajo (menor composite_risk_score_ma3)", {"id_solicitud": "SOS-0002", "county_fips": int(orden.index[0])}),
        ("Condado inexistente", {"id_solicitud": "SOS-0003", "county_fips": "99999"}),
    ]
    log(f"[INFO] Modelo: {type(MODELO).__name__} | ganador: {METADATA.get('ganador')} | scaler: {SCALER is not None}")
    salidas = []
    for titulo, ent in casos:
        sal = predecir(ent)
        salidas.append(sal)
        log(f"\n=== {titulo} ===\n[ENTRADA]\n{dump(ent)}\n[SALIDA]\n{dump(sal)}")

    # Asserts de forma
    claves = {"id_solicitud", "county_fips", "state", "anio_base", "anio_pronostico", "nivel_riesgo", "clase_predicha",
              "probabilidades", "confianza", "requiere_revision", "factores_clave", "tiempo_procesamiento_ms", "status"}
    for sal in salidas[:2]:
        assert sal["status"] == "success", sal
        assert claves <= sal.keys(), claves - sal.keys()
        assert list(sal["probabilidades"]) == ["Bajo", "Medio", "Alto"]
        assert abs(sum(sal["probabilidades"].values()) - 1) < 1e-3
        assert sal["nivel_riesgo"] in {"Bajo", "Medio", "Alto"}
        assert 1 <= len(sal["factores_clave"]) <= 3
        assert sal["requiere_revision"] == (sal["confianza"] < 70) == ("aviso" in sal)
    assert salidas[2]["status"] == "error" and "mensaje" in salidas[2]
    assert predecir({"id_solicitud": "SOS-X"})["status"] == "error"
    assert predecir("no soy dict")["status"] == "error"
    assert predecir({"county_fips": "12a86"})["status"] == "error"
    log("\n[SUCCESS] Asserts de forma OK")

    # Latencia: 100 llamadas con condados aleatorios
    rng = np.random.default_rng(42)
    muestra = rng.choice(STORE.index.to_numpy(), size=100)
    tiempos = []
    for f in muestra:
        t0 = time.perf_counter()
        assert predecir({"id_solicitud": "LAT", "county_fips": f})["status"] == "success"
        tiempos.append((time.perf_counter() - t0) * 1000)
    media, p95 = float(np.mean(tiempos)), float(np.percentile(tiempos, 95))
    log(f"[RESULTADOS] Latencia 100 llamadas: media = {media:.2f} ms | p95 = {p95:.2f} ms")
    assert p95 < 100, f"p95 {p95:.2f} ms >= 100 ms"
    log("[SUCCESS] p95 < 100 ms")

    (RAIZ / "reports").mkdir(exist_ok=True)
    (RAIZ / "reports" / "demo_inferencia.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")

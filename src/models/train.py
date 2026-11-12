"""Entrenamiento y comparación de modelos: persistencia, LR, RF por defecto y RF ajustado.

Uso:
    .venv/bin/python src/models/train.py --data data/processed/dataset_riesgo.csv --output models/ | tee reports/train_log.txt
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

NOMBRES = ["Bajo", "Medio", "Alto"]
REPORTS = Path("reports")
FIGS = REPORTS / "figures"


def metricas(y, p):
    return {"accuracy": float(accuracy_score(y, p)),
            "f1_macro": float(f1_score(y, p, average="macro")),
            "recall_alto": float(recall_score(y, p, labels=[2], average=None)[0])}


def diagnostico(tr, te):
    if tr["f1_macro"] - te["f1_macro"] > 0.10:
        return "Overfitting"
    if tr["f1_macro"] < 0.6:
        return "Underfitting"
    return "Generalización adecuada"


def evaluar(nombre, modelo_txt, hiper, predict, Xtr, ytr, Xte, yte, t_train):
    p_tr = predict(Xtr)
    t0 = time.perf_counter()
    p_te = predict(Xte)
    t_inf = time.perf_counter() - t0
    tr, te = metricas(ytr, p_tr), metricas(yte, p_te)
    print(f"[INFO] Entrenamiento finalizado en {t_train:.2f} segundos.")
    print(f"[RESULTADOS EN TEST SET] {nombre}")
    print(classification_report(yte, p_te, labels=[0, 1, 2], target_names=NOMBRES, digits=4))
    print(f"[INFO] Train: acc={tr['accuracy']:.4f} f1_macro={tr['f1_macro']:.4f} recall_alto={tr['recall_alto']:.4f}")
    print(f"[INFO] Test:  acc={te['accuracy']:.4f} f1_macro={te['f1_macro']:.4f} recall_alto={te['recall_alto']:.4f}")
    print(f"[INFO] Inferencia test: {t_inf:.4f} s ({1000 * t_inf / len(yte):.5f} ms/muestra)")
    cm = confusion_matrix(yte, p_te, labels=[0, 1, 2])
    ConfusionMatrixDisplay(cm, display_labels=NOMBRES).plot(cmap="Blues", values_format="d")
    plt.title(f"Matriz de confusión (test) — {nombre}")
    plt.tight_layout()
    plt.savefig(FIGS / f"cm_{nombre.split()[0].lower()}.png", dpi=120)
    plt.close("all")
    d = diagnostico(tr, te)
    print(f"[INFO] Diagnóstico: {d}\n")
    return {"variante": nombre, "modelo": modelo_txt, "hiperparametros": hiper, "train": tr, "test": te,
            "tiempo_entrenamiento_s": round(t_train, 4), "tiempo_inferencia_test_s": round(t_inf, 4),
            "inferencia_ms_por_muestra": round(1000 * t_inf / len(yte), 6), "diagnostico_auto": d,
            "diagnostico": d, "matriz_confusion_test": cm.tolist(),
            "classification_report_test": classification_report(yte, p_te, labels=[0, 1, 2], target_names=NOMBRES,
                                                                 output_dict=True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    out = Path(a.output)
    FIGS.mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata.json"
    meta = json.loads(meta_path.read_text())
    feats = meta["features"]

    df = pd.read_csv(a.data, dtype={"county_fips": str})
    print(f"[INFO] Cargando dataset: {a.data} ({len(df)} registros, {len(feats)} features)...")
    train = df[df["split"] == "train"].sort_values(["target_year", "county_fips"], kind="stable").reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    print(f"[INFO] Conjunto Train: {len(train)} registros (target {train.target_year.min()}–{train.target_year.max()}) | "
          f"Conjunto Test: {len(test)} registros (target {test.target_year.min()}–{test.target_year.max()})\n")
    Xtr, ytr, Xte, yte = train[feats], train["nivel_riesgo_t1"].to_numpy(), test[feats], test["nivel_riesgo_t1"].to_numpy()
    res = {}

    # Exp 0: persistencia (nivel de t con los umbrales de train)
    p33, p67 = meta["umbrales"]["p33"], meta["umbrales"]["p67"]
    persist = lambda X: np.select([X["composite_risk_score"] < p33, X["composite_risk_score"] < p67], [0, 1], 2)
    print("[INFO] Entrenando Exp0 Persistencia (sin entrenamiento: nivel t+1 = nivel t)...")
    res["exp0"] = evaluar("Exp0 Persistencia", "Persistencia", {"umbrales": meta["umbrales"]}, persist,
                          Xtr, ytr, Xte, yte, 0.0)

    # Exp 1: LR multinomial con escalado ajustado solo en train
    print("[INFO] Entrenando Exp1 LogisticRegression + StandardScaler (C=1.0, max_iter=1000)...")
    t0 = time.perf_counter()
    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(scaler.transform(Xtr), ytr)
    t_lr = time.perf_counter() - t0
    res["exp1"] = evaluar("Exp1 LogisticRegression", "LogisticRegression",
                          {"C": 1.0, "max_iter": 1000, "random_state": 42, "scaler": "StandardScaler"},
                          lambda X: lr.predict(scaler.transform(X)), Xtr, ytr, Xte, yte, t_lr)
    joblib.dump(scaler, out / "scaler.joblib")
    joblib.dump(lr, out / "modelo_baseline_lr.joblib")
    print(f"[SUCCESS] Escalador guardado en: {out / 'scaler.joblib'}")
    print(f"[SUCCESS] Modelo serializado con éxito en: {out / 'modelo_baseline_lr.joblib'}\n")

    # Exp 2: RF por defecto
    print("[INFO] Entrenando Exp2 RandomForest por defecto (n_estimators=100, max_depth=None)...")
    t0 = time.perf_counter()
    rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    t_rf = time.perf_counter() - t0
    res["exp2"] = evaluar("Exp2 RandomForest default", "RandomForestClassifier",
                          {"n_estimators": 100, "max_depth": None, "random_state": 42}, rf.predict,
                          Xtr, ytr, Xte, yte, t_rf)

    # Exp 3: RF ajustado con GridSearchCV + TimeSeriesSplit sobre train ordenado por target_year
    grid = {"max_depth": [8, 12, 16], "min_samples_leaf": [5, 20, 50], "max_features": ["sqrt", 0.5],
            "class_weight": [None, "balanced"]}
    print(f"[INFO] Entrenando Exp3 RandomForest ajustado (GridSearchCV, TimeSeriesSplit(3), 36 combinaciones x 3 folds)...")
    t0 = time.perf_counter()
    # Paralelismo a nivel de grid (n_jobs=-1) y RF secuencial dentro para no sobresuscribir CPUs.
    gs = GridSearchCV(RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1), grid,
                      cv=TimeSeriesSplit(n_splits=3), scoring="f1_macro", n_jobs=-1, refit=True).fit(Xtr, ytr)
    t_gs = time.perf_counter() - t0
    rf_aj = gs.best_estimator_.set_params(n_jobs=-1)
    best_params = {k: v for k, v in gs.best_params_.items()}
    print(f"[INFO] Grid completo en {t_gs:.2f} s (refit {gs.refit_time_:.2f} s). best_params={best_params} "
          f"(f1_macro CV={gs.best_score_:.4f})")
    res["exp3"] = evaluar("Exp3 RandomForest ajustado", "RandomForestClassifier",
                          {"n_estimators": 200, "random_state": 42, **best_params}, rf_aj.predict,
                          Xtr, ytr, Xte, yte, t_gs)
    res["exp3"]["tiempo_refit_s"] = round(gs.refit_time_, 4)
    res["exp3"]["f1_macro_cv"] = float(gs.best_score_)

    # Ganador entre Exp 1–3
    modelos = {"exp1": lr, "exp2": rf, "exp3": rf_aj}
    ganador = max(modelos, key=lambda k: (res[k]["test"]["f1_macro"], res[k]["test"]["recall_alto"]))
    res[ganador]["diagnostico"] = "Modelo Ganador"
    cand = modelos[ganador]
    joblib.dump(cand, out / "modelo_candidato.joblib")
    print(f"[INFO] Ganador: {res[ganador]['variante']} (f1_macro test={res[ganador]['test']['f1_macro']:.4f})")
    print(f"[SUCCESS] Modelo serializado con éxito en: {out / 'modelo_candidato.joblib'}")
    cm = pd.DataFrame(res[ganador]["matriz_confusion_test"], index=[f"real_{n}" for n in NOMBRES],
                      columns=[f"pred_{n}" for n in NOMBRES])
    print(f"[RESULTADOS EN TEST SET] Matriz de confusión del ganador ({res[ganador]['variante']}):\n{cm.to_string()}\n")

    if ganador == "exp1":
        w = np.abs(lr.coef_).mean(axis=0)
        imp = w / w.sum()
    else:
        imp = cand.feature_importances_
    importancias = dict(sorted(zip(feats, map(float, imp)), key=lambda kv: -kv[1]))
    rf_fig = cand if ganador != "exp1" else rf_aj
    top = pd.Series(rf_fig.feature_importances_, index=feats).nlargest(15)[::-1]
    plt.figure(figsize=(9, 6))
    plt.barh([meta["etiquetas_features"].get(f, f) for f in top.index], top.values, color="#2b6cb0")
    plt.xlabel("Importancia (reducción media de impureza)")
    plt.title(f"Top 15 features — {res[ganador]['variante'] if ganador != 'exp1' else res['exp3']['variante']}")
    plt.tight_layout()
    plt.savefig(FIGS / "feature_importance.png", dpi=120)
    plt.close("all")

    metrics = {"experimentos": res, "best_params_exp3": best_params, "ganador": ganador,
               "ganador_variante": res[ganador]["variante"], "n_train": len(train), "n_test": len(test)}
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    pd.DataFrame([{"exp": k, "variante": r["variante"], "modelo": r["modelo"],
                   "hiperparametros": json.dumps(r["hiperparametros"], ensure_ascii=False),
                   "acc_train": r["train"]["accuracy"], "f1_macro_train": r["train"]["f1_macro"],
                   "recall_alto_train": r["train"]["recall_alto"], "acc_test": r["test"]["accuracy"],
                   "f1_macro_test": r["test"]["f1_macro"], "recall_alto_test": r["test"]["recall_alto"],
                   "tiempo_entrenamiento_s": r["tiempo_entrenamiento_s"],
                   "tiempo_inferencia_test_s": r["tiempo_inferencia_test_s"],
                   "inferencia_ms_por_muestra": r["inferencia_ms_por_muestra"], "diagnostico": r["diagnostico"]}
                  for k, r in res.items()]).to_csv(REPORTS / "experimentos.csv", index=False)

    meta.update({"ganador": res[ganador]["variante"], "modelo_candidato": "models/modelo_candidato.joblib",
                 "requiere_scaler": ganador == "exp1", "importancias": importancias, "best_params": best_params})
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[SUCCESS] Metadata actualizada en: {meta_path}")

    # Verificación
    for f in [out / "modelo_candidato.joblib", out / "modelo_baseline_lr.joblib", out / "scaler.joblib", meta_path,
              REPORTS / "metrics.json", REPORTS / "experimentos.csv", FIGS / "feature_importance.png",
              REPORTS / "train_log.txt", *[FIGS / f"cm_exp{i}.png" for i in range(4)]]:
        assert f.exists(), f"Falta artefacto: {f}"
    m = joblib.load(out / "modelo_candidato.joblib")
    X5 = Xte.head(5)
    p5 = m.predict(joblib.load(out / "scaler.joblib").transform(X5) if meta["requiere_scaler"] else X5)
    assert set(np.unique(p5)) <= {0, 1, 2}
    assert set(json.loads((REPORTS / "metrics.json").read_text())["experimentos"]) == {"exp0", "exp1", "exp2", "exp3"}
    print("[SUCCESS] Verificación OK: artefactos presentes, predicciones en {0,1,2}, 4 experimentos en metrics.json.")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Construye 04. entrenamiento.ipynb (Fase 4)."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

md("""# 04. Entrenamiento — Fase 4

Modelo de **incidencia** de cáncer de mama (predice nuevos casos en T+1 sobre población sin
diagnóstico en T). Desbalance extremo (~2.300:1, event rate 0,043%).

**Técnicas aplicadas:**
- **GroupKFold(5) por `key`** — la misma paciente puede ser negativa en un par y positiva en
  otro (2021→2022 vs 2022→2023); evita que esté en train y validación del mismo fold.
- **SMOTE + undersampling** del majority, **solo dentro del train de cada fold** (imblearn
  Pipeline → sin fuga hacia el fold de validación).
- **class_weight / scale_pos_weight** como alternativa a SMOTE.
- **Predicciones out-of-fold (OOF)** para métricas honestas.
- **Optimización de umbral** (F2, prioriza recall) y **recall@top-k** (operación de tamización).
- **Dos variantes de imputación:** lineal sobre `_impute` (mediana), árboles sobre `_native`
  (NaN nativo).
- **Validación temporal final:** refit en train completo, evaluación en el par 2023→2024.

**Métrica primaria:** AUC-PR (área bajo precision-recall) — informativa bajo desbalance extremo,
a diferencia de AUC-ROC. Secundaria: Recall (minimizar falsos negativos).
""")

code("""import json, warnings, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import lightgbm as lgb
import xgboost as xgb

B = "bases"
RNG = 42
""")

code("""# Cargar ambas variantes (mismas filas/orden/columnas; difieren solo en imputables)
imp_tr = pd.read_parquet(f"{B}/prediccion_mama_train_impute.parquet")
nat_tr = pd.read_parquet(f"{B}/prediccion_mama_train_native.parquet")
imp_va = pd.read_parquet(f"{B}/prediccion_mama_val_impute.parquet")
nat_va = pd.read_parquet(f"{B}/prediccion_mama_val_native.parquet")

FEAT = [c for c in imp_tr.columns if c not in ('key', 'label')]
y_tr = imp_tr['label'].values.astype(int)
g_tr = imp_tr['key'].values
y_va = imp_va['label'].values.astype(int)

Xi_tr = imp_tr[FEAT].astype('float32').values   # imputado (LR)
Xn_tr = nat_tr[FEAT].astype('float32').values   # NaN nativo (árboles)
Xi_va = imp_va[FEAT].astype('float32').values
Xn_va = nat_va[FEAT].astype('float32').values

print(f"train {Xi_tr.shape} pos {y_tr.sum()} ({y_tr.mean()*100:.4f}%) | "
      f"val {Xi_va.shape} pos {y_va.sum()} ({y_va.mean()*100:.4f}%)")
print(f"features: {len(FEAT)} | grupos únicos (key) en train: {pd.Series(g_tr).nunique():,}")
print(f"ratio neg/pos train: {(y_tr==0).sum()/(y_tr==1).sum():.0f}:1")
""")

md("""## 1. Métricas y framework OOF

`run_cv` entrena con GroupKFold por `key` y devuelve predicciones out-of-fold. El `resampler`
(SMOTE+undersampling) se aplica **dentro** del pipeline, solo al train de cada fold. Para los
árboles se usa `scale_pos_weight`; para LR-balanced, `class_weight`.
""")

code("""def metrics_at_best_f2(y, p, beta=2.0):
    prec, rec, thr = precision_recall_curve(y, p)
    f2 = (1 + beta**2) * prec * rec / (beta**2 * prec + rec + 1e-12)
    i = int(np.nanargmax(f2[:-1])) if len(thr) > 0 else 0
    return {'thr': float(thr[i]), 'recall': float(rec[i]),
            'precision': float(prec[i]), 'f2': float(f2[i])}

def recall_at_topk(y, p, frac):
    k = max(1, int(len(p) * frac))
    idx = np.argpartition(-p, k)[:k]
    return float(y[idx].sum() / y.sum())

def summarize(y, p, name):
    m = {'modelo': name,
         'AUC_PR': float(average_precision_score(y, p)),
         'AUC_ROC': float(roc_auc_score(y, p))}
    m.update(metrics_at_best_f2(y, p))
    for f in (0.005, 0.01, 0.05, 0.10):
        m[f'recall@top{f*100:.1f}%'] = recall_at_topk(y, p, f)
    return m

def run_cv(make_model, X, y, groups, resampler=None, n_splits=5, name=""):
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(y))
    t0 = time.time()
    for k, (tr, va) in enumerate(gkf.split(X, y, groups)):
        Xtr, ytr = X[tr], y[tr]
        if resampler is not None:
            Xtr, ytr = resampler().fit_resample(Xtr, ytr)
        mdl = make_model(ytr)
        mdl.fit(Xtr, ytr)
        oof[va] = mdl.predict_proba(X[va])[:, 1]
    print(f"  {name}: OOF listo en {time.time()-t0:.0f}s")
    return oof
""")

md("""## 2. Resamplers y definición de modelos

`under 0.05 → SMOTE 0.5`: submuestrea el majority a 20× el minority, luego SMOTE eleva el
minority a la mitad del majority (≈3:1 final). Mantiene el set tratable y de calidad (evita
sobre-interpolar 1.485 positivos a cientos de miles).
""")

code("""def make_resampler():
    return ImbPipeline([
        ('under', RandomUnderSampler(sampling_strategy=0.05, random_state=RNG)),
        ('smote', SMOTE(sampling_strategy=0.5, random_state=RNG, k_neighbors=5)),
    ])

def lr_balanced(_y):
    return SkPipeline([('sc', StandardScaler()),
                       ('clf', SGDClassifier(loss='log_loss', class_weight='balanced',
                                             alpha=1e-4, max_iter=20, random_state=RNG))])

def lr_smote(_y):
    return SkPipeline([('sc', StandardScaler()),
                       ('clf', LogisticRegression(max_iter=500, random_state=RNG))])

def lgb_spw(y):
    spw = (y == 0).sum() / max(1, (y == 1).sum())
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                              subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                              random_state=RNG, n_jobs=-1, verbose=-1)

def lgb_smote(_y):
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=RNG, n_jobs=-1, verbose=-1)

def xgb_spw(y):
    spw = (y == 0).sum() / max(1, (y == 1).sum())
    return xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                             tree_method='hist', eval_metric='aucpr', random_state=RNG, n_jobs=-1)
""")

md("""## 3. Validación cruzada (OOF) de los modelos

| Modelo | Variante | Desbalance |
|--------|----------|-----------|
| LR-balanced | impute | class_weight |
| LR-SMOTE | impute | SMOTE+under |
| LightGBM-spw | native | scale_pos_weight |
| XGBoost-spw | native | scale_pos_weight |
| LightGBM-SMOTE | impute | SMOTE+under |
""")

code("""results = {}
print("Corriendo GroupKFold OOF (5 folds)...")
results['LR-balanced']    = run_cv(lr_balanced, Xi_tr, y_tr, g_tr, name='LR-balanced')
results['LR-SMOTE']       = run_cv(lr_smote,   Xi_tr, y_tr, g_tr, resampler=make_resampler, name='LR-SMOTE')
results['LightGBM-spw']   = run_cv(lgb_spw,    Xn_tr, y_tr, g_tr, name='LightGBM-spw')
results['XGBoost-spw']    = run_cv(xgb_spw,    Xn_tr, y_tr, g_tr, name='XGBoost-spw')
results['LightGBM-SMOTE'] = run_cv(lgb_smote,  Xi_tr, y_tr, g_tr, resampler=make_resampler, name='LightGBM-SMOTE')
print("OK")
""")

code("""rows = [summarize(y_tr, p, name) for name, p in results.items()]
cv = pd.DataFrame(rows).set_index('modelo').sort_values('AUC_PR', ascending=False)
pd.set_option('display.width', 200, 'display.max_columns', 30)
print("=== Comparación OOF (ordenado por AUC-PR) ===")
print(cv.to_string(float_format='{:.4f}'.format))
""")

code("""# Curvas Precision-Recall OOF
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
for name, p in results.items():
    prec, rec, _ = precision_recall_curve(y_tr, p)
    ax[0].plot(rec, prec, label=f"{name} (AP={average_precision_score(y_tr,p):.3f})")
ax[0].set_xlabel('Recall'); ax[0].set_ylabel('Precision'); ax[0].set_title('Precision-Recall (OOF)')
ax[0].legend(fontsize=8); ax[0].set_xlim(0, 1)
ax[0].axhline(y_tr.mean(), ls='--', c='gray', lw=1, label='baseline')

# Recall acumulado vs fracción de población priorizada (curva de captura para tamización)
for name, p in results.items():
    order = np.argsort(-p)
    cap = np.cumsum(y_tr[order]) / y_tr.sum()
    frac = np.arange(1, len(p)+1) / len(p)
    ax[1].plot(frac, cap, label=name)
ax[1].plot([0, 1], [0, 1], 'k--', lw=1, label='aleatorio')
ax[1].set_xlabel('Fracción de población priorizada'); ax[1].set_ylabel('Recall acumulado')
ax[1].set_title('Curva de captura (tamización)'); ax[1].legend(fontsize=8)
ax[1].set_xlim(0, 0.2)  # foco en top 20%
plt.tight_layout(); plt.show()
""")

md("""## 4. Evaluación en validación temporal (par 2023→2024)

El test honesto del modelo prospectivo: entrenar con todo el train (2021→2022 + 2022→2023) y
evaluar en un año futuro no visto. Cada modelo se reajusta sobre el train completo y se evalúa
sobre la validación.
""")

code("""def fit_full_and_eval(make_model, Xtr, Xva, resampler=None):
    Xt, yt = Xtr, y_tr
    if resampler is not None:
        Xt, yt = resampler().fit_resample(Xtr, y_tr)
    mdl = make_model(yt)
    mdl.fit(Xt, yt)
    return mdl, mdl.predict_proba(Xva)[:, 1]

configs = {
    'LR-balanced':    (lr_balanced, Xi_tr, Xi_va, None),
    'LR-SMOTE':       (lr_smote,    Xi_tr, Xi_va, make_resampler),
    'LightGBM-spw':   (lgb_spw,     Xn_tr, Xn_va, None),
    'XGBoost-spw':    (xgb_spw,     Xn_tr, Xn_va, None),
    'LightGBM-SMOTE': (lgb_smote,   Xi_tr, Xi_va, make_resampler),
}
val_pred, fitted = {}, {}
for name, (mk, Xt, Xv, rs) in configs.items():
    t0 = time.time()
    fitted[name], val_pred[name] = fit_full_and_eval(mk, Xt, Xv, rs)
    print(f"  {name}: refit+pred val en {time.time()-t0:.0f}s")

val_rows = [summarize(y_va, p, name) for name, p in val_pred.items()]
val = pd.DataFrame(val_rows).set_index('modelo').sort_values('AUC_PR', ascending=False)
print("\\n=== Validación 2023→2024 (ordenado por AUC-PR) ===")
print(val.to_string(float_format='{:.4f}'.format))
""")

md("""## 5. Modelo ganador — importancia de variables y curva PR final

Se selecciona por AUC-PR en validación. Importancia por ganancia (LightGBM). SHAP queda para
Fase 5.
""")

code("""best_name = val['AUC_PR'].idxmax()
print("Modelo ganador (AUC-PR val):", best_name)
best = fitted[best_name]

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
# PR final del ganador
prec, rec, _ = precision_recall_curve(y_va, val_pred[best_name])
ax[0].plot(rec, prec, lw=2)
ax[0].axhline(y_va.mean(), ls='--', c='gray', lw=1)
ax[0].set_xlabel('Recall'); ax[0].set_ylabel('Precision')
ax[0].set_title(f'PR validación — {best_name} (AP={average_precision_score(y_va,val_pred[best_name]):.3f})')

# importancia (si es árbol)
if hasattr(best, 'feature_importances_'):
    imp = pd.Series(best.feature_importances_, index=FEAT).sort_values(ascending=False).head(20)
    imp[::-1].plot(kind='barh', ax=ax[1], color='steelblue')
    ax[1].set_title(f'Top 20 importancia — {best_name}')
else:
    coef = pd.Series(np.abs(best.named_steps['clf'].coef_[0]), index=FEAT).sort_values(ascending=False).head(20)
    coef[::-1].plot(kind='barh', ax=ax[1], color='steelblue')
    ax[1].set_title(f'Top 20 |coef| — {best_name}')
plt.tight_layout(); plt.show()
""")

md("""## 6. Guardar modelos, predicciones y métricas""")

code("""import joblib
joblib.dump(fitted[best_name], f"{B}/modelo_{best_name}.joblib")
np.savez_compressed(f"{B}/oof_predicciones.npz",
                    y_train=y_tr, **{k.replace('-', '_'): v for k, v in results.items()})

metrics_out = {
    'cv_oof': cv.reset_index().to_dict(orient='records'),
    'validacion_2023_2024': val.reset_index().to_dict(orient='records'),
    'modelo_ganador': best_name,
}
with open(f"{B}/metricas_fase4.json", "w", encoding="utf-8") as f:
    json.dump(metrics_out, f, indent=2, ensure_ascii=False)

print("Guardado:")
print(f"  {B}/modelo_{best_name}.joblib")
print(f"  {B}/oof_predicciones.npz")
print(f"  {B}/metricas_fase4.json")
print(f"\\nGanador: {best_name}")
print(f"  AUC-PR val: {val.loc[best_name,'AUC_PR']:.4f} | recall@top1%: {val.loc[best_name,'recall@top1.0%']:.4f}")
""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
with open("04. entrenamiento.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook escrito: 04. entrenamiento.ipynb con", len(cells), "celdas")

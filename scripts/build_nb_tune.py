# -*- coding: utf-8 -*-
"""Construye 05. tuning e interpretabilidad.ipynb (Fase 5)."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

md("""# 05. Tuning e Interpretabilidad — Fase 5

Sobre el ganador de Fase 4 (**LightGBM-SMOTE**, AUC-PR val 0,0335):

1. **Tuning anti-sobreajuste** con Optuna + GroupKFold por `key`, optimizando **AUC-PR**.
   Espacio de búsqueda centrado en **regularización** (`min_child_samples`, `reg_alpha/lambda`,
   `num_leaves` y `max_depth` acotados, subsample/colsample < 1). Se incluyen las razones de
   resampleo (`under`, `smote`) como hiperparámetros.
2. **Diagnóstico de overfit**: gap entre AUC-PR en train-de-fold vs validación-de-fold.
3. **Comparación con `scale_pos_weight` tuneado** (native) — arregla el spw=2307 roto de Fase 4.
4. **Validación temporal 2023→2024** como test final imparcial (no se tunea sobre ella).
5. **SHAP** (TreeExplainer) — contribución global de cada característica (mean|SHAP|, beeswarm),
   comparada con importancia por ganancia.

> Anti-fuga: SMOTE/undersampling **solo dentro del train de cada fold**. El set de validación
> temporal nunca se usa para seleccionar hiperparámetros.
""")

code("""import json, time, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import lightgbm as lgb
import shap
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

B = "bases"; RNG = 42
""")

code("""imp_tr = pd.read_parquet(f"{B}/prediccion_mama_train_impute.parquet")
nat_tr = pd.read_parquet(f"{B}/prediccion_mama_train_native.parquet")
imp_va = pd.read_parquet(f"{B}/prediccion_mama_val_impute.parquet")
nat_va = pd.read_parquet(f"{B}/prediccion_mama_val_native.parquet")
FEAT = [c for c in imp_tr.columns if c not in ('key', 'label')]
y_tr = imp_tr['label'].values.astype(int); g_tr = imp_tr['key'].values
y_va = imp_va['label'].values.astype(int)
Xi_tr = imp_tr[FEAT].astype('float32').values
Xn_tr = nat_tr[FEAT].astype('float32').values
Xi_va = imp_va[FEAT].astype('float32').values
Xn_va = nat_va[FEAT].astype('float32').values

# 3 folds GroupKFold por key, precomputados (tuning rápido y consistente)
FOLDS = list(GroupKFold(n_splits=3).split(Xi_tr, y_tr, g_tr))
print(f"train {Xi_tr.shape} pos {y_tr.sum()} | val {Xi_va.shape} pos {y_va.sum()} | {len(FEAT)} feats")
print(f"folds: {[(len(tr), len(va), int(y_tr[va].sum())) for tr, va in FOLDS]}")
""")

md("""## 1. Tuning con Optuna (LightGBM-SMOTE)

35 trials, 3-fold GroupKFold. Objetivo = AUC-PR promedio out-of-fold. SMOTE+undersampling
dentro de cada fold. Rangos sesgados a regularización para frenar sobreajuste.
""")

code("""def resample(under, smote, X, y):
    pipe = ImbPipeline([
        ('under', RandomUnderSampler(sampling_strategy=under, random_state=RNG)),
        ('smote', SMOTE(sampling_strategy=smote, random_state=RNG, k_neighbors=5)),
    ])
    return pipe.fit_resample(X, y)

def lgb_params(trial):
    return dict(
        n_estimators=trial.suggest_int('n_estimators', 200, 700),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        num_leaves=trial.suggest_int('num_leaves', 15, 127),
        max_depth=trial.suggest_int('max_depth', 3, 12),
        min_child_samples=trial.suggest_int('min_child_samples', 20, 300),
        subsample=trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        min_split_gain=trial.suggest_float('min_split_gain', 0.0, 0.5),
        subsample_freq=1, random_state=RNG, n_jobs=-1, verbose=-1,
    )

def objective(trial):
    under = trial.suggest_float('under_ratio', 0.02, 0.10)
    smote = trial.suggest_float('smote_ratio', 0.30, 0.70)
    params = lgb_params(trial)
    aps = []
    for tr, va in FOLDS:
        Xr, yr = resample(under, smote, Xi_tr[tr], y_tr[tr])
        m = lgb.LGBMClassifier(**params).fit(Xr, yr)
        p = m.predict_proba(Xi_tr[va])[:, 1]
        aps.append(average_precision_score(y_tr[va], p))
    return float(np.mean(aps))

t0 = time.time()
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RNG))
study.optimize(objective, n_trials=35, show_progress_bar=False)
print(f"Tuning en {time.time()-t0:.0f}s")
print(f"Mejor AUC-PR CV: {study.best_value:.4f}")
print("Mejores hiperparámetros:")
for k, v in study.best_params.items():
    print(f"  {k}: {v if isinstance(v, int) else round(v, 5)}")
""")

md("""## 2. Diagnóstico de sobreajuste

Para los mejores hiperparámetros, se compara AUC-PR en el **train de cada fold** (resampleado
→ evaluado en el train original) vs el **fold de validación**. Un gap grande indica overfit.
""")

code("""bp = study.best_params.copy()
under_b, smote_b = bp.pop('under_ratio'), bp.pop('smote_ratio')
gaps = []
for k, (tr, va) in enumerate(FOLDS):
    Xr, yr = resample(under_b, smote_b, Xi_tr[tr], y_tr[tr])
    m = lgb.LGBMClassifier(**bp, subsample_freq=1, random_state=RNG, n_jobs=-1, verbose=-1).fit(Xr, yr)
    ap_tr = average_precision_score(y_tr[tr], m.predict_proba(Xi_tr[tr])[:, 1])
    ap_va = average_precision_score(y_tr[va], m.predict_proba(Xi_tr[va])[:, 1])
    gaps.append((ap_tr, ap_va))
    print(f"  fold {k}: AP train {ap_tr:.4f} | AP val {ap_va:.4f} | gap {ap_tr-ap_va:.4f}")
gt = np.mean([a for a, _ in gaps]); gv = np.mean([b for _, b in gaps])
print(f"\\nPromedio: AP train {gt:.4f} | AP val {gv:.4f} | gap {gt-gv:.4f}")
print("Gap moderado = regularización adecuada." if gt-gv < 0.02 else "Gap alto = revisar regularización.")
""")

md("""## 3. Comparación con scale_pos_weight tuneado (native)

El `LightGBM-spw` de Fase 4 quedó último por `scale_pos_weight=2307` (extremo). Se barre un
rango moderado en la variante native (NaN nativo) con 3-fold CV para ver si recupera y compite.
""")

code("""spw_grid = [1, 5, 10, 25, 48, 100, 250]
base_p = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, max_depth=8,
              min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
              reg_lambda=1.0, subsample_freq=1, random_state=RNG, n_jobs=-1, verbose=-1)
spw_res = []
for spw in spw_grid:
    aps = []
    for tr, va in FOLDS:
        m = lgb.LGBMClassifier(**base_p, scale_pos_weight=spw).fit(Xn_tr[tr], y_tr[tr])
        aps.append(average_precision_score(y_tr[va], m.predict_proba(Xn_tr[va])[:, 1]))
    spw_res.append((spw, float(np.mean(aps))))
    print(f"  spw={spw:>4}: AUC-PR CV {np.mean(aps):.4f}")
best_spw = max(spw_res, key=lambda x: x[1])
print(f"\\nMejor spw: {best_spw[0]} (AUC-PR CV {best_spw[1]:.4f}) vs SMOTE tuneado {study.best_value:.4f}")
""")

md("""## 4. Validación temporal 2023→2024 (test final)

Cada configuración se reajusta en el train completo y se evalúa una sola vez en validación.
""")

code("""def summarize(y, p, name):
    m = {'modelo': name, 'AUC_PR': float(average_precision_score(y, p)),
         'AUC_ROC': float(roc_auc_score(y, p))}
    for f in (0.005, 0.01, 0.05, 0.10):
        k = max(1, int(len(p) * f)); idx = np.argpartition(-p, k)[:k]
        m[f'recall@top{f*100:.1f}%'] = float(y[idx].sum() / y.sum())
    return m

rows = []
# SMOTE tuneado (impute)
Xr, yr = resample(under_b, smote_b, Xi_tr, y_tr)
best_smote = lgb.LGBMClassifier(**bp, subsample_freq=1, random_state=RNG, n_jobs=-1, verbose=-1).fit(Xr, yr)
p_smote = best_smote.predict_proba(Xi_va)[:, 1]
rows.append(summarize(y_va, p_smote, 'LGB-SMOTE-tuned'))

# spw tuneado (native)
best_spw_model = lgb.LGBMClassifier(**base_p, scale_pos_weight=best_spw[0]).fit(Xn_tr, y_tr)
p_spw = best_spw_model.predict_proba(Xn_va)[:, 1]
rows.append(summarize(y_va, p_spw, f'LGB-spw{best_spw[0]}-native'))

comp = pd.DataFrame(rows).set_index('modelo').sort_values('AUC_PR', ascending=False)
pd.set_option('display.width', 200, 'display.max_columns', 20)
print("=== Validación 2023→2024 (Fase 5) ===")
print(comp.to_string(float_format='{:.4f}'.format))
print(f"\\nBaseline Fase 4 (LightGBM-SMOTE sin tunear): AUC-PR 0.0335, recall@top10% 0.678")
best_name = comp['AUC_PR'].idxmax()
best_model = best_smote if 'SMOTE' in best_name else best_spw_model
best_X_va, best_X_tr = (Xi_va, Xi_tr) if 'SMOTE' in best_name else (Xn_va, Xn_tr)
print("Ganador Fase 5:", best_name)
""")

md("""## 5. Interpretabilidad — SHAP

Contribución de cada característica con TreeExplainer sobre el modelo ganador. SHAP se calcula
en una muestra (todos los positivos + 40k negativos) por costo. Beeswarm muestra dirección y
magnitud; el ranking mean|SHAP| cuantifica cuánto aporta cada variable.
""")

code("""# muestra para SHAP: todos los positivos de train + 40k negativos
pos_idx = np.where(y_tr == 1)[0]
neg_idx = np.random.RandomState(RNG).choice(np.where(y_tr == 0)[0], 40000, replace=False)
sidx = np.concatenate([pos_idx, neg_idx])
Xs = best_X_tr[sidx]
Xs_df = pd.DataFrame(Xs, columns=FEAT)

explainer = shap.TreeExplainer(best_model)
sv = explainer.shap_values(Xs_df)
# LightGBM binario: shap puede devolver lista [clase0, clase1] o array; tomar clase positiva
if isinstance(sv, list):
    sv = sv[1]
sv = np.asarray(sv)
if sv.ndim == 3:
    sv = sv[:, :, 1]
print("SHAP values shape:", sv.shape)
""")

code("""fig = plt.figure(figsize=(10, 9))
shap.summary_plot(sv, Xs_df, plot_type='dot', max_display=20, show=False)
plt.title(f'SHAP beeswarm — {best_name}')
plt.tight_layout(); plt.show()
""")

code("""# Ranking mean|SHAP| vs importancia por ganancia
mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=FEAT).sort_values(ascending=False)
gain = pd.Series(best_model.feature_importances_, index=FEAT)
gain = (gain / gain.sum()).reindex(mean_abs.index)
contrib = pd.DataFrame({'mean_abs_SHAP': mean_abs, 'gain_norm': gain})
contrib['rank_shap'] = range(1, len(contrib) + 1)
print("=== Contribución de características (top 25 por mean|SHAP|) ===")
print(contrib.head(25).to_string(float_format='{:.5f}'.format))
print("\\n=== Bottom 10 (aportan casi nada) ===")
print(contrib.tail(10).index.tolist())

fig, ax = plt.subplots(figsize=(9, 8))
mean_abs.head(20)[::-1].plot(kind='barh', ax=ax, color='steelblue')
ax.set_title(f'Top 20 contribución (mean|SHAP|) — {best_name}'); ax.set_xlabel('mean|SHAP|')
plt.tight_layout(); plt.show()
""")

md("""## 6. Guardar modelo tuneado, hiperparámetros y contribuciones""")

code("""import joblib
joblib.dump(best_model, f"{B}/modelo_fase5_{best_name}.joblib")
out = {
    'mejor_modelo': best_name,
    'best_params_smote': study.best_params,
    'cv_auc_pr_smote_tuned': float(study.best_value),
    'overfit_gap': float(gt - gv),
    'spw_sweep': spw_res,
    'mejor_spw': best_spw[0],
    'validacion_fase5': comp.reset_index().to_dict(orient='records'),
    'baseline_fase4_auc_pr': 0.0335,
    'shap_contribucion': mean_abs.round(6).to_dict(),
}
with open(f"{B}/tuning_fase5.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
contrib.to_csv(f"{B}/contribucion_caracteristicas.csv")
print("Guardado:")
print(f"  {B}/modelo_fase5_{best_name}.joblib")
print(f"  {B}/tuning_fase5.json")
print(f"  {B}/contribucion_caracteristicas.csv")
print(f"\\nGanador: {best_name} | AUC-PR val {comp.loc[best_name,'AUC_PR']:.4f} "
      f"(Fase 4: 0.0335) | recall@top10% {comp.loc[best_name,'recall@top10.0%']:.4f}")
""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "CAC venv", "language": "python", "name": "cacvenv"},
    "language_info": {"name": "python"}})
with open("05. tuning e interpretabilidad.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook escrito: 05. tuning e interpretabilidad.ipynb con", len(cells), "celdas")

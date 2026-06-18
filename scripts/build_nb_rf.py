# -*- coding: utf-8 -*-
"""Construye 15. random forest.ipynb (cierre comparativa — modelo de ensamble bagging).

Random Forest se mencionó en la metodología (ensamble de árboles) pero no se había entrenado.
Mismo protocolo que los demás modelos: Optuna + GroupKFold(3) por key sobre subsample
prevalencia-aprox, refit en train completo, validación temporal 2023->2024.

RF de sklearn NO admite NaN -> se usa la variante IMPUTADA (mediana train + flags), 59 feats,
comparable al XGBoost ganador (AUC-PR 0.0380, recall@top10% 0.704) y al LightGBM spw=1 (0.0346).
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

md("""# 15. Random Forest — cierre de la comparativa

Random Forest (ensamble por *bagging*) figuraba en la metodología pero no se había entrenado.
Aquí se entrena con el **mismo protocolo** que el resto de modelos para situarlo en la
comparativa frente al boosting.

- Variante **imputada** (mediana de train + indicadores), pues `RandomForestClassifier` de
  scikit-learn no maneja NaN. 59 características (igual que el XGBoost ganador).
- Optuna + GroupKFold(3) por `key` sobre subsample que preserva prevalencia; refit en train
  completo; validación temporal 2023->2024.
- `class_weight` incluido como hiperparámetro (balanced / balanced_subsample / None), dado el
  hallazgo de que la reponderación fuerte degrada el ranking en boosting.

**Bar a comparar:** XGBoost tuneado 0.0380 / LightGBM spw=1 0.0346 (AUC-PR validación).
""")

code("""import json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score
import joblib

B = "bases"; RNG = 42

imp_tr = pd.read_parquet(f"{B}/prediccion_mama_train_impute.parquet")
imp_va = pd.read_parquet(f"{B}/prediccion_mama_val_impute.parquet")
FEAT = [c for c in imp_tr.columns if c not in ('key', 'label')]
y_tr = imp_tr['label'].values.astype(int); g_tr = imp_tr['key'].values
y_va = imp_va['label'].values.astype(int)
Xi_tr = imp_tr[FEAT].astype('float32').values
Xi_va = imp_va[FEAT].astype('float32').values

# subsample tuning (preserva prevalencia): todos los positivos + 800k negativos
N_NEG = 800000
_rs = np.random.RandomState(RNG)
_pos = np.where(y_tr == 1)[0]
_neg = _rs.choice(np.where(y_tr == 0)[0], min(N_NEG, int((y_tr == 0).sum())), replace=False)
SUB = np.sort(np.concatenate([_pos, _neg]))
Xs, ys, gs = Xi_tr[SUB], y_tr[SUB], g_tr[SUB]
FOLDS = list(GroupKFold(n_splits=3).split(Xs, ys, gs))
print(f"train {Xi_tr.shape} pos {y_tr.sum()} | val {Xi_va.shape} pos {y_va.sum()} | {len(FEAT)} feats")
print(f"subsample {Xs.shape} pos {ys.sum()} (prev {ys.mean()*100:.4f}% vs full {y_tr.mean()*100:.4f}%)")

def summarize(y, p, name):
    m = {'modelo': name, 'AUC_PR': float(average_precision_score(y, p)),
         'AUC_ROC': float(roc_auc_score(y, p))}
    for f in (0.005, 0.01, 0.05, 0.10):
        k = max(1, int(len(p) * f)); idx = np.argpartition(-p, k)[:k]
        m[f'recall@top{f*100:.1f}%'] = float(y[idx].sum() / y.sum())
    return m
""")

md("""## 1. Tuning con Optuna""")

code("""def rf_params(trial):
    return dict(
        n_estimators=trial.suggest_int('n_estimators', 200, 600),
        max_depth=trial.suggest_int('max_depth', 6, 30),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 200, log=True),
        max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5]),
        class_weight=trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
        max_samples=trial.suggest_float('max_samples', 0.5, 1.0),
        n_jobs=-1, random_state=RNG, bootstrap=True,
    )

def objective(trial):
    params = rf_params(trial)
    aps = []
    for tr, va in FOLDS:
        m = RandomForestClassifier(**params).fit(Xs[tr], ys[tr])
        p = m.predict_proba(Xs[va])[:, 1]
        aps.append(average_precision_score(ys[va], p))
    return float(np.mean(aps))

t0 = time.time()
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RNG))
study.optimize(objective, n_trials=30, timeout=1800, show_progress_bar=False)
print(f"RF tuning en {time.time()-t0:.0f}s | {len(study.trials)} trials | mejor AUC-PR CV: {study.best_value:.4f}")
for k, v in study.best_params.items():
    print(f"  {k}: {v if isinstance(v, (int, str, type(None))) else round(v, 5)}")
""")

md("""## 2. Validación temporal 2023->2024 (refit en train completo)""")

code("""bp = dict(study.best_params)
model = RandomForestClassifier(**bp, n_jobs=-1, random_state=RNG, bootstrap=True).fit(Xi_tr, y_tr)
p_va = model.predict_proba(Xi_va)[:, 1]
res = summarize(y_va, p_va, 'RandomForest-tuned')

comp = pd.DataFrame([
    res,
    {'modelo': 'XGBoost tuneado (GANADOR)', 'AUC_PR': 0.0380, 'AUC_ROC': 0.900,
     'recall@top0.5%': 0.266, 'recall@top1.0%': 0.325, 'recall@top5.0%': 0.563, 'recall@top10.0%': 0.704},
    {'modelo': 'LightGBM spw=1', 'AUC_PR': 0.0346, 'AUC_ROC': 0.888,
     'recall@top0.5%': 0.244, 'recall@top1.0%': 0.306, 'recall@top5.0%': 0.547, 'recall@top10.0%': 0.673},
    {'modelo': 'LR-SMOTE (tuneado)', 'AUC_PR': 0.0099, 'AUC_ROC': 0.878,
     'recall@top0.5%': 0.207, 'recall@top1.0%': 0.268, 'recall@top5.0%': 0.502, 'recall@top10.0%': 0.651},
]).set_index('modelo').sort_values('AUC_PR', ascending=False)
pd.set_option('display.width', 200, 'display.max_columns', 20)
print("=== Validación 2023->2024 — RF vs referencia ===")
print(comp.to_string(float_format='{:.4f}'.format))
print(f"\\nRandom Forest: AUC-PR {res['AUC_PR']:.4f} | recall@top10% {res['recall@top10.0%']:.4f}")
pos = '>' if res['AUC_PR'] > 0.0346 else '<'
print(f"RF {pos} LightGBM spw=1 (0.0346); vs XGBoost ganador 0.0380")
""")

md("""## 3. Guardar modelo y métricas""")

code("""joblib.dump(model, f"{B}/modelo_RandomForest-tuned.joblib")
imp = pd.Series(model.feature_importances_, index=FEAT).sort_values(ascending=False)
out = {
    'modelo': 'RandomForest-tuned',
    'variante': 'impute (59 feats)',
    'best_params': study.best_params,
    'cv_auc_pr': float(study.best_value),
    'validacion': res,
    'top15_importancia_impureza': imp.head(15).round(5).to_dict(),
}
with open(f"{B}/metricas_rf.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False, default=str)
print("Guardado: bases/modelo_RandomForest-tuned.joblib, bases/metricas_rf.json")
print("\\nTop 10 importancia (impureza):", imp.head(10).index.tolist())
""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "CAC venv", "language": "python", "name": "cacvenv"},
    "language_info": {"name": "python"}})
with open("15. random forest.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook escrito: 15. random forest.ipynb con", len(cells), "celdas")

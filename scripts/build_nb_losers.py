# -*- coding: utf-8 -*-
"""Construye 07. tuning modelos perdedores.ipynb (Fase 5b).

Objetivo: tunear con Optuna los modelos que perdieron en Fase 4/5
(XGBoost-spw, LR-SMOTE, LR-balanced) y ver si superan al ganador actual:
LightGBM nativo spw=1 -> AUC-PR val 0.0346, recall@top10% 0.673.
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

BAR_PR = 0.0346
BAR_TOP10 = 0.673

md(f"""# 07. Tuning de modelos perdedores — Fase 5b

Los modelos que NO ganaron en Fase 4/5 se tunean con Optuna para ver si alcanzan al
ganador actual. **Bar a superar:** LightGBM nativo spw=1 -> **AUC-PR validacion {BAR_PR}**,
recall@top10% {BAR_TOP10}.

Contendientes:
1. **XGBoost** (native, NaN nativo) — el realista (0.0309 sin tunear). Se tunea regularizacion
   y se deja a Optuna elegir `scale_pos_weight` (incluido =1, por el hallazgo de Fase 5 de que
   reponderar degrada el ranking).
2. **LR-SMOTE** (impute + escalado) — 0.0098 sin tunear. Tunea C, penalty, razones SMOTE.
3. **LR-balanced** (impute + escalado, class_weight='balanced') — 0.0020 sin tunear. Tunea C, penalty.

Metodo: Optuna + GroupKFold(3) por `key`, objetivo **AUC-PR** out-of-fold. Validacion temporal
2023->2024 como test final imparcial (nunca se tunea sobre ella).

> Anti-fuga: escalado y SMOTE/undersampling **solo dentro del train de cada fold**.
""")

code("""import json, time, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

B = "bases"; RNG = 42
BAR_PR, BAR_TOP10 = %r, %r
""" % (BAR_PR, BAR_TOP10))

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

# Subsample para TUNING (preserva prevalencia aprox): todos los positivos + 800k negativos.
# 4,28M filas x 3 folds x 40 trials de XGBoost no termina en 1h; subsample ~5x mas rapido.
# Solo se usa para PICK de hiperparametros (comparacion relativa). El refit final y la
# validacion temporal 2023->2024 usan el train COMPLETO -> eval honesta.
N_NEG_TUNE = 800000
_rs = np.random.RandomState(RNG)
_pos = np.where(y_tr == 1)[0]
_neg = _rs.choice(np.where(y_tr == 0)[0], min(N_NEG_TUNE, int((y_tr == 0).sum())), replace=False)
SUB = np.sort(np.concatenate([_pos, _neg]))
Xi_s, Xn_s, y_s, g_s = Xi_tr[SUB], Xn_tr[SUB], y_tr[SUB], g_tr[SUB]
FOLDS = list(GroupKFold(n_splits=3).split(Xi_s, y_s, g_s))
print(f"train {Xi_tr.shape} pos {y_tr.sum()} | val {Xi_va.shape} pos {y_va.sum()} | {len(FEAT)} feats")
print(f"subsample tuning {Xi_s.shape} pos {y_s.sum()} (prev {y_s.mean()*100:.4f}% vs full {y_tr.mean()*100:.4f}%)")
print(f"folds: {[(len(tr), len(va), int(y_s[va].sum())) for tr, va in FOLDS]}")

def summarize(y, p, name):
    m = {'modelo': name, 'AUC_PR': float(average_precision_score(y, p)),
         'AUC_ROC': float(roc_auc_score(y, p))}
    for f in (0.005, 0.01, 0.05, 0.10):
        k = max(1, int(len(p) * f)); idx = np.argpartition(-p, k)[:k]
        m[f'recall@top{f*100:.1f}%'] = float(y[idx].sum() / y.sum())
    return m
""")

md("""## 1. XGBoost (native) — Optuna

Hasta 40 trials con tope de 1500s, 3-fold GroupKFold sobre el subsample de tuning.
`scale_pos_weight` tunable en [1, 50] (log) para confirmar el hallazgo de Fase 5 (reponderar
fuerte degrada el ranking). `tree_method='hist'`, NaN nativo. Refit final en train completo.
""")

code("""def xgb_params(trial):
    return dict(
        n_estimators=trial.suggest_int('n_estimators', 150, 500),
        learning_rate=trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
        max_depth=trial.suggest_int('max_depth', 3, 8),
        min_child_weight=trial.suggest_float('min_child_weight', 1.0, 200.0, log=True),
        subsample=trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        gamma=trial.suggest_float('gamma', 0.0, 5.0),
        reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        scale_pos_weight=trial.suggest_float('scale_pos_weight', 1.0, 50.0, log=True),
        tree_method='hist', eval_metric='aucpr', n_jobs=-1, random_state=RNG,
    )

def xgb_objective(trial):
    params = xgb_params(trial)
    aps = []
    for tr, va in FOLDS:
        m = xgb.XGBClassifier(**params).fit(Xn_s[tr], y_s[tr])
        p = m.predict_proba(Xn_s[va])[:, 1]
        aps.append(average_precision_score(y_s[va], p))
    return float(np.mean(aps))

t0 = time.time()
study_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RNG))
study_xgb.optimize(xgb_objective, n_trials=40, timeout=1500, show_progress_bar=False)
print(f"XGBoost tuning en {time.time()-t0:.0f}s | {len(study_xgb.trials)} trials | mejor AUC-PR CV: {study_xgb.best_value:.4f}")
for k, v in study_xgb.best_params.items():
    print(f"  {k}: {v if isinstance(v, (int, str)) else round(v, 5)}")
""")

md("""## 2. LR-SMOTE (impute + escalado) — Optuna

Pipeline por fold: StandardScaler -> RandomUnderSampler -> SMOTE -> LogisticRegression.
Tunea C, penalty (l1/l2, solver saga), y razones de resampleo. 30 trials.
""")

code("""def lr_smote_pipe(C, penalty, under, smote):
    return ImbPipeline([
        ('scaler', StandardScaler()),
        ('under', RandomUnderSampler(sampling_strategy=under, random_state=RNG)),
        ('smote', SMOTE(sampling_strategy=smote, random_state=RNG, k_neighbors=5)),
        ('lr', LogisticRegression(C=C, penalty=penalty, solver='saga',
                                  max_iter=2000, n_jobs=-1, random_state=RNG)),
    ])

def lr_smote_objective(trial):
    C = trial.suggest_float('C', 1e-3, 10.0, log=True)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    under = trial.suggest_float('under_ratio', 0.02, 0.10)
    smote = trial.suggest_float('smote_ratio', 0.30, 0.70)
    aps = []
    for tr, va in FOLDS:
        m = lr_smote_pipe(C, penalty, under, smote).fit(Xi_s[tr], y_s[tr])
        p = m.predict_proba(Xi_s[va])[:, 1]
        aps.append(average_precision_score(y_s[va], p))
    return float(np.mean(aps))

t0 = time.time()
study_lrs = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RNG))
study_lrs.optimize(lr_smote_objective, n_trials=30, timeout=900, show_progress_bar=False)
print(f"LR-SMOTE tuning en {time.time()-t0:.0f}s | {len(study_lrs.trials)} trials | mejor AUC-PR CV: {study_lrs.best_value:.4f}")
for k, v in study_lrs.best_params.items():
    print(f"  {k}: {v if isinstance(v, (int, str)) else round(v, 5)}")
""")

md("""## 3. LR-balanced (impute + escalado) — Optuna

Pipeline: StandardScaler -> LogisticRegression(class_weight='balanced'). Tunea C y penalty.
20 trials.
""")

code("""from sklearn.pipeline import Pipeline as SkPipeline

def lr_bal_pipe(C, penalty):
    return SkPipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=C, penalty=penalty, solver='saga', class_weight='balanced',
                                  max_iter=2000, n_jobs=-1, random_state=RNG)),
    ])

def lr_bal_objective(trial):
    C = trial.suggest_float('C', 1e-3, 10.0, log=True)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    aps = []
    for tr, va in FOLDS:
        m = lr_bal_pipe(C, penalty).fit(Xi_s[tr], y_s[tr])
        p = m.predict_proba(Xi_s[va])[:, 1]
        aps.append(average_precision_score(y_s[va], p))
    return float(np.mean(aps))

t0 = time.time()
study_lrb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RNG))
study_lrb.optimize(lr_bal_objective, n_trials=20, timeout=600, show_progress_bar=False)
print(f"LR-balanced tuning en {time.time()-t0:.0f}s | {len(study_lrb.trials)} trials | mejor AUC-PR CV: {study_lrb.best_value:.4f}")
for k, v in study_lrb.best_params.items():
    print(f"  {k}: {v if isinstance(v, (int, str)) else round(v, 5)}")
""")

md(f"""## 4. Validacion temporal 2023->2024 (test final)

Cada modelo se reajusta en el train completo y se evalua una sola vez en validacion,
contra el bar de Fase 5 (AUC-PR {BAR_PR}, recall@top10% {BAR_TOP10}).
""")

code("""rows = []

# XGBoost tuneado (native) — refit con best_params (reconstruir dict, no se puede re-suggest)
xgb_bp = dict(study_xgb.best_params)
xgb_bp.update(tree_method='hist', eval_metric='aucpr', n_jobs=-1, random_state=RNG)
m_xgb = xgb.XGBClassifier(**xgb_bp).fit(Xn_tr, y_tr)
p_xgb = m_xgb.predict_proba(Xn_va)[:, 1]
rows.append(summarize(y_va, p_xgb, 'XGBoost-tuned-native'))

# LR-SMOTE tuneado (impute)
lrs_bp = dict(study_lrs.best_params)
m_lrs = lr_smote_pipe(lrs_bp['C'], lrs_bp['penalty'], lrs_bp['under_ratio'], lrs_bp['smote_ratio']).fit(Xi_tr, y_tr)
p_lrs = m_lrs.predict_proba(Xi_va)[:, 1]
rows.append(summarize(y_va, p_lrs, 'LR-SMOTE-tuned'))

# LR-balanced tuneado (impute)
lrb_bp = dict(study_lrb.best_params)
m_lrb = lr_bal_pipe(lrb_bp['C'], lrb_bp['penalty']).fit(Xi_tr, y_tr)
p_lrb = m_lrb.predict_proba(Xi_va)[:, 1]
rows.append(summarize(y_va, p_lrb, 'LR-balanced-tuned'))

comp = pd.DataFrame(rows).set_index('modelo').sort_values('AUC_PR', ascending=False)
pd.set_option('display.width', 200, 'display.max_columns', 20)
print("=== Validacion 2023->2024 (Fase 5b) ===")
print(comp.to_string(float_format='{:.4f}'.format))
print(f"\\nBAR (LightGBM nativo spw=1, Fase 5): AUC-PR {BAR_PR}, recall@top10% {BAR_TOP10}")
best_name = comp['AUC_PR'].idxmax()
best_pr = comp.loc[best_name, 'AUC_PR']
veredicto = 'SUPERA el bar' if best_pr > BAR_PR else 'NO supera el bar'
print(f"Mejor perdedor tuneado: {best_name} (AUC-PR {best_pr:.4f}) -> {veredicto}")
""")

md("""## 5. Guardar resultados""")

code("""import joblib
best_model = {'XGBoost-tuned-native': m_xgb, 'LR-SMOTE-tuned': m_lrs, 'LR-balanced-tuned': m_lrb}[best_name]
joblib.dump(best_model, f"{B}/modelo_fase5b_{best_name}.joblib")
out = {
    'bar_lightgbm_spw1_native': {'AUC_PR': BAR_PR, 'recall@top10%': BAR_TOP10},
    'xgboost': {'best_params': study_xgb.best_params, 'cv_auc_pr': float(study_xgb.best_value)},
    'lr_smote': {'best_params': study_lrs.best_params, 'cv_auc_pr': float(study_lrs.best_value)},
    'lr_balanced': {'best_params': study_lrb.best_params, 'cv_auc_pr': float(study_lrb.best_value)},
    'validacion_fase5b': comp.reset_index().to_dict(orient='records'),
    'mejor_perdedor': best_name,
    'supera_bar': bool(best_pr > BAR_PR),
}
with open(f"{B}/tuning_perdedores.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("Guardado:")
print(f"  {B}/modelo_fase5b_{best_name}.joblib")
print(f"  {B}/tuning_perdedores.json")
print(f"\\nConclusion: {best_name} AUC-PR {best_pr:.4f} vs bar {BAR_PR} -> "
      f"{'SUPERA' if best_pr > BAR_PR else 'NO supera, LightGBM sigue ganador'}")
""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "CAC venv", "language": "python", "name": "cacvenv"},
    "language_info": {"name": "python"}})
with open("07. tuning modelos perdedores.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook escrito: 07. tuning modelos perdedores.ipynb con", len(cells), "celdas")

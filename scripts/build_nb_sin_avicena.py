# -*- coding: utf-8 -*-
"""Construye 09. sin tiene_avicena.ipynb (Fase 6 - depuracion de features).

Decision usuario 2026-06-17: eliminar `tiene_avicena` (= n_consultas_T.notna(), senal de
vigilancia redundante). Refit del XGBoost ganador con sus MISMOS hiperparametros tuneados,
sin esa feature, y evaluacion en validacion temporal 2023->2024 vs el bar actual
(AUC-PR 0.0380, recall@top10% 0.704). Si las metricas aguantan, queda como nuevo modelo.
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

md("""# 09. Refit XGBoost sin `tiene_avicena` — Fase 6

`tiene_avicena` = `n_consultas_T.notna()` — indicador puro de tener registro Avicena, senal de
vigilancia redundante con `n_consultas_T`. Se elimina (decision usuario).

Refit del ganador **XGBoost-tuned-native** con sus mismos hiperparametros (de
`tuning_perdedores.json`), sin esa feature. Evaluacion en validacion temporal 2023->2024.

**Bar a igualar/superar:** AUC-PR 0.0380, recall@top10% 0.704 (XGBoost CON tiene_avicena).
""")

code("""import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import xgboost as xgb, joblib
from sklearn.metrics import average_precision_score, roc_auc_score

B = "bases"; RNG = 42
DROP = ['tiene_avicena']

nat_tr = pd.read_parquet(f"{B}/prediccion_mama_train_native.parquet")
nat_va = pd.read_parquet(f"{B}/prediccion_mama_val_native.parquet")
FEAT = [c for c in nat_tr.columns if c not in ('key', 'label') + tuple(DROP)]
y_tr = nat_tr['label'].values.astype(int)
y_va = nat_va['label'].values.astype(int)
Xn_tr = nat_tr[FEAT].astype('float32').values
Xn_va = nat_va[FEAT].astype('float32').values
print(f"feats: {len(FEAT)} (eliminadas: {DROP})")
print(f"train {Xn_tr.shape} pos {y_tr.sum()} | val {Xn_va.shape} pos {y_va.sum()}")

with open(f"{B}/tuning_perdedores.json", encoding="utf-8") as f:
    bp = json.load(f)['xgboost']['best_params']
print("best_params XGBoost:", bp)
""")

code("""def summarize(y, p, name):
    m = {'modelo': name, 'AUC_PR': float(average_precision_score(y, p)),
         'AUC_ROC': float(roc_auc_score(y, p))}
    for f in (0.005, 0.01, 0.05, 0.10):
        k = max(1, int(len(p) * f)); idx = np.argpartition(-p, k)[:k]
        m[f'recall@top{f*100:.1f}%'] = float(y[idx].sum() / y.sum())
    return m

params = dict(bp); params.update(tree_method='hist', eval_metric='aucpr', n_jobs=-1, random_state=RNG)
model = xgb.XGBClassifier(**params).fit(Xn_tr, y_tr)
p_va = model.predict_proba(Xn_va)[:, 1]
res = summarize(y_va, p_va, 'XGBoost-sin-avicena')

comp = pd.DataFrame([
    res,
    {'modelo': 'XGBoost-CON-avicena (bar)', 'AUC_PR': 0.0380, 'AUC_ROC': 0.900,
     'recall@top0.5%': 0.266, 'recall@top1.0%': 0.325, 'recall@top5.0%': 0.563, 'recall@top10.0%': 0.704},
]).set_index('modelo')
pd.set_option('display.width', 200, 'display.max_columns', 20)
print("=== Validacion 2023->2024 — sin vs con tiene_avicena ===")
print(comp.to_string(float_format='{:.4f}'.format))
d_pr = res['AUC_PR'] - 0.0380; d_top10 = res['recall@top10.0%'] - 0.704
print(f"\\nDelta AUC-PR: {d_pr:+.4f} | Delta recall@top10%: {d_top10:+.4f}")
print("Veredicto:", "aguanta (impacto despreciable)" if abs(d_pr) < 0.002 else "impacto NO despreciable, revisar")
""")

md("""## Guardar modelo y metricas""")

code("""joblib.dump(model, f"{B}/modelo_fase6_XGBoost-sin-avicena.joblib")
out = {
    'modelo': 'XGBoost-sin-avicena',
    'features_eliminadas': DROP,
    'n_features': len(FEAT),
    'best_params': bp,
    'validacion': res,
    'bar_con_avicena': {'AUC_PR': 0.0380, 'recall@top10%': 0.704},
    'delta_auc_pr': float(res['AUC_PR'] - 0.0380),
    'delta_recall_top10': float(res['recall@top10.0%'] - 0.704),
}
with open(f"{B}/metricas_sin_avicena.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("Guardado:")
print(f"  {B}/modelo_fase6_XGBoost-sin-avicena.joblib")
print(f"  {B}/metricas_sin_avicena.json")
print(f"\\nAUC-PR {res['AUC_PR']:.4f} | recall@top10% {res['recall@top10.0%']:.4f} | {len(FEAT)} feats")
""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "CAC venv", "language": "python", "name": "cacvenv"},
    "language_info": {"name": "python"}})
with open("09. sin tiene_avicena.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook escrito: 09. sin tiene_avicena.ipynb con", len(cells), "celdas")

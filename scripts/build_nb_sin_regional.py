# -*- coding: utf-8 -*-
"""Construye 10. sin regional.ipynb (Fase 6 - equidad geografica).

Decision usuario 2026-06-17: eliminar la regional (`REG_*` one-hot) por equidad geografica
-- preocupa que el modelo priorice leads de Bogota por VOLUMEN de poblacion atendida, no por
riesgo individual. Test acumulativo: tambien sin `tiene_avicena` (ya eliminada).

Refit XGBoost mismos hiperparametros tuneados, sin REG_* ni tiene_avicena. Validacion temporal
2023->2024 vs bar actual (XGBoost-sin-avicena: AUC-PR 0.0375, recall@top10% 0.703).
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

md("""# 10. Refit XGBoost sin regional (`REG_*`) — Fase 6 equidad geografica

`REG_1 REG BOGOTA` salia #3 en SHAP. Es legitimo (refleja donde se concentra la atencion),
pero por **equidad geografica** preocupa que el modelo priorice leads de Bogota por volumen de
poblacion, no por riesgo individual. Se elimina la regional (6 dummies `REG_*`).

Test **acumulativo**: tambien sin `tiene_avicena` (ya eliminada en notebook 09).

Refit del XGBoost ganador con sus mismos hiperparametros. Evaluacion validacion temporal
2023->2024.

**Bar a igualar:** XGBoost-sin-avicena → AUC-PR 0.0375, recall@top10% 0.703.
""")

code("""import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import xgboost as xgb, joblib
from sklearn.metrics import average_precision_score, roc_auc_score

B = "bases"; RNG = 42

nat_tr = pd.read_parquet(f"{B}/prediccion_mama_train_native.parquet")
nat_va = pd.read_parquet(f"{B}/prediccion_mama_val_native.parquet")
REG_COLS = [c for c in nat_tr.columns if c.startswith('REG_')]
DROP = ['tiene_avicena'] + REG_COLS
FEAT = [c for c in nat_tr.columns if c not in ('key', 'label') and c not in DROP]
y_tr = nat_tr['label'].values.astype(int)
y_va = nat_va['label'].values.astype(int)
Xn_tr = nat_tr[FEAT].astype('float32').values
Xn_va = nat_va[FEAT].astype('float32').values
print(f"REG_* eliminadas: {REG_COLS}")
print(f"feats: {len(FEAT)} (eliminadas: tiene_avicena + {len(REG_COLS)} REG_*)")
print(f"train {Xn_tr.shape} pos {y_tr.sum()} | val {Xn_va.shape} pos {y_va.sum()}")

with open(f"{B}/tuning_perdedores.json", encoding="utf-8") as f:
    bp = json.load(f)['xgboost']['best_params']
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
res = summarize(y_va, p_va, 'XGBoost-sin-avicena-sin-regional')

comp = pd.DataFrame([
    res,
    {'modelo': 'XGBoost-sin-avicena (bar)', 'AUC_PR': 0.0375, 'AUC_ROC': 0.899,
     'recall@top0.5%': 0.267, 'recall@top1.0%': 0.330, 'recall@top5.0%': 0.563, 'recall@top10.0%': 0.703},
    {'modelo': 'XGBoost-completo (Fase 5b)', 'AUC_PR': 0.0380, 'AUC_ROC': 0.900,
     'recall@top0.5%': 0.266, 'recall@top1.0%': 0.325, 'recall@top5.0%': 0.563, 'recall@top10.0%': 0.704},
]).set_index('modelo')
pd.set_option('display.width', 200, 'display.max_columns', 20)
print("=== Validacion 2023->2024 — sin regional vs bars ===")
print(comp.to_string(float_format='{:.4f}'.format))
d_pr = res['AUC_PR'] - 0.0375; d_top10 = res['recall@top10.0%'] - 0.703
print(f"\\nDelta vs sin-avicena -> AUC-PR: {d_pr:+.4f} | recall@top10%: {d_top10:+.4f}")
print("Veredicto:", "aguanta (impacto despreciable)" if abs(d_pr) < 0.002 else "impacto NO despreciable")
""")

md("""## Distribucion geografica de los leads top-decil (equidad)

Se compara la composicion regional del top 10% de riesgo con y sin la feature regional,
usando `DESCRIP_REGIONAL` de la base original de validacion (si esta disponible) o las dummies.
El objetivo de quitar `REG_*` es que Bogota no domine el top por volumen.
""")

code("""# proporcion de positivos reales captados por regional en el top-decil (sin REG_* en el modelo)
k = max(1, int(len(p_va) * 0.10))
top_idx = np.argpartition(-p_va, k)[:k]
reg_dummies = [c for c in nat_va.columns if c.startswith('REG_')]
if reg_dummies:
    reg_va = nat_va[reg_dummies].iloc[top_idx]
    share_top = reg_va.mean().sort_values(ascending=False)
    share_all = nat_va[reg_dummies].mean().sort_values(ascending=False)
    tab = pd.DataFrame({'%_en_top10': share_top * 100, '%_en_poblacion': share_all.reindex(share_top.index) * 100})
    tab['sobre_representacion'] = tab['%_en_top10'] / tab['%_en_poblacion']
    print("=== Composicion regional del top-decil (modelo SIN regional) ===")
    print(tab.to_string(float_format='{:.2f}'.format))
    print("\\nsobre_representacion ~1 = proporcional a poblacion; >1 = sobre-priorizada")
else:
    print("No hay columnas REG_* en la base de validacion para el diagnostico.")
""")

md("""## Guardar modelo y metricas""")

code("""joblib.dump(model, f"{B}/modelo_fase6_XGBoost-sin-regional.joblib")
out = {
    'modelo': 'XGBoost-sin-avicena-sin-regional',
    'features_eliminadas': DROP,
    'n_features': len(FEAT),
    'best_params': bp,
    'validacion': res,
    'bar_sin_avicena': {'AUC_PR': 0.0375, 'recall@top10%': 0.703},
    'delta_auc_pr': float(res['AUC_PR'] - 0.0375),
    'delta_recall_top10': float(res['recall@top10.0%'] - 0.703),
}
with open(f"{B}/metricas_sin_regional.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("Guardado:")
print(f"  {B}/modelo_fase6_XGBoost-sin-regional.joblib")
print(f"  {B}/metricas_sin_regional.json")
print(f"\\nAUC-PR {res['AUC_PR']:.4f} | recall@top10% {res['recall@top10.0%']:.4f} | {len(FEAT)} feats")
""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "CAC venv", "language": "python", "name": "cacvenv"},
    "language_info": {"name": "python"}})
with open("10. sin regional.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook escrito: 10. sin regional.ipynb con", len(cells), "celdas")

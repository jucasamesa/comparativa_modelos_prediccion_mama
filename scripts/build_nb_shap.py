# -*- coding: utf-8 -*-
"""Construye 08. shap xgboost.ipynb (Fase 6 - interpretabilidad del ganador).

SHAP (TreeExplainer) sobre el ganador del proyecto: XGBoost-tuned-native
(AUC-PR val 0.0380, recall@top10% 0.704). El SHAP de Fase 5 fue sobre LightGBM;
este recalcula la contribucion sobre el modelo que efectivamente se usaria.
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

md("""# 08. Interpretabilidad SHAP — XGBoost (ganador del proyecto)

SHAP sobre **XGBoost-tuned-native** (AUC-PR val 0.0380, recall@top10% 0.704), que superó al
LightGBM de Fase 5. El SHAP de Fase 5 se calculo sobre LightGBM; aqui se recalcula sobre el
modelo que efectivamente entraria a produccion.

- TreeExplainer sobre `bases/modelo_fase5b_XGBoost-tuned-native.joblib`.
- Muestra: todos los positivos de train + 40k negativos (costo).
- Datos `_native` (NaN nativo, igual que en entrenamiento).
- Salidas: beeswarm, ranking mean|SHAP| vs ganancia, `bases/contribucion_xgboost.csv`,
  `bases/shap_xgboost.json`.
""")

code("""import json, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import joblib, shap, xgboost as xgb

B = "bases"; RNG = 42
model = joblib.load(f"{B}/modelo_fase5b_XGBoost-tuned-native.joblib")
nat_tr = pd.read_parquet(f"{B}/prediccion_mama_train_native.parquet")
FEAT = [c for c in nat_tr.columns if c not in ('key', 'label')]
y_tr = nat_tr['label'].values.astype(int)
Xn_tr = nat_tr[FEAT].astype('float32').values
print("modelo:", type(model).__name__, "| feats:", len(FEAT), "| train pos:", int(y_tr.sum()))
""")

md("""## 1. Calculo de valores SHAP""")

code("""pos_idx = np.where(y_tr == 1)[0]
neg_idx = np.random.RandomState(RNG).choice(np.where(y_tr == 0)[0], 40000, replace=False)
sidx = np.concatenate([pos_idx, neg_idx])
Xs = Xn_tr[sidx]
Xs_df = pd.DataFrame(Xs, columns=FEAT)

explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(Xs_df)
if isinstance(sv, list):
    sv = sv[1]
sv = np.asarray(sv)
if sv.ndim == 3:
    sv = sv[:, :, 1]
print("SHAP values shape:", sv.shape, "| muestra:", Xs_df.shape,
      f"({len(pos_idx)} pos + {len(neg_idx)} neg)")
""")

md("""## 2. Beeswarm — direccion y magnitud por caracteristica""")

code("""fig = plt.figure(figsize=(10, 9))
shap.summary_plot(sv, Xs_df, plot_type='dot', max_display=20, show=False)
plt.title('SHAP beeswarm — XGBoost-tuned-native')
plt.tight_layout(); plt.show()
""")

md("""## 3. Ranking mean|SHAP| vs importancia por ganancia""")

code("""mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=FEAT).sort_values(ascending=False)
gain = pd.Series(model.feature_importances_, index=FEAT)
gain = (gain / gain.sum()).reindex(mean_abs.index)
contrib = pd.DataFrame({'mean_abs_SHAP': mean_abs, 'gain_norm': gain})
contrib['rank_shap'] = range(1, len(contrib) + 1)
pd.set_option('display.width', 200, 'display.max_columns', 20)
print("=== Contribucion (top 25 por mean|SHAP|) ===")
print(contrib.head(25).to_string(float_format='{:.5f}'.format))
print("\\n=== Bottom 10 (aportan casi nada) ===")
print(contrib.tail(10).index.tolist())

fig, ax = plt.subplots(figsize=(9, 8))
mean_abs.head(20)[::-1].plot(kind='barh', ax=ax, color='darkorange')
ax.set_title('Top 20 contribucion (mean|SHAP|) — XGBoost-tuned-native'); ax.set_xlabel('mean|SHAP|')
plt.tight_layout(); plt.show()
""")

md("""## 4. Comparacion con SHAP de LightGBM (Fase 5)

Se contrasta el ranking del ganador actual (XGBoost) con el de LightGBM de Fase 5
(`contribucion_caracteristicas.csv`) para ver si la historia de features es estable
entre modelos (robustez) o cambia (sensible al algoritmo).
""")

code("""try:
    lgb_contrib = pd.read_csv(f"{B}/contribucion_caracteristicas.csv", index_col=0)
    lgb_rank = lgb_contrib['rank_shap'] if 'rank_shap' in lgb_contrib else \
               lgb_contrib['mean_abs_SHAP'].rank(ascending=False)
    cmp = pd.DataFrame({
        'rank_xgb': contrib['rank_shap'],
        'rank_lgb': lgb_rank.reindex(contrib.index),
    })
    cmp['delta'] = cmp['rank_lgb'] - cmp['rank_xgb']
    print("=== Top 15 XGBoost vs su rank en LightGBM (Fase 5) ===")
    print(cmp.head(15).to_string(float_format='{:.0f}'.format))
except FileNotFoundError:
    print("No se encontro contribucion_caracteristicas.csv (Fase 5); se omite comparacion.")
""")

md("""## 5. Guardar contribuciones""")

code("""contrib.to_csv(f"{B}/contribucion_xgboost.csv")
out = {
    'modelo': 'XGBoost-tuned-native',
    'auc_pr_val': 0.0380,
    'recall_top10': 0.704,
    'shap_muestra': {'positivos': int(len(pos_idx)), 'negativos': int(len(neg_idx))},
    'mean_abs_shap': mean_abs.round(6).to_dict(),
    'top10_features': mean_abs.head(10).index.tolist(),
    'bottom10_features': mean_abs.tail(10).index.tolist(),
}
with open(f"{B}/shap_xgboost.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("Guardado:")
print(f"  {B}/contribucion_xgboost.csv")
print(f"  {B}/shap_xgboost.json")
print("\\nTop 10 contribucion:", mean_abs.head(10).index.tolist())
""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "CAC venv", "language": "python", "name": "cacvenv"},
    "language_info": {"name": "python"}})
with open("08. shap xgboost.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook escrito: 08. shap xgboost.ipynb con", len(cells), "celdas")

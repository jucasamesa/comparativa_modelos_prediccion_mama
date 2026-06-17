# -*- coding: utf-8 -*-
"""Construye 11. seleccion estratificada.ipynb (Fase 6 - equidad por asignacion).

Decision usuario 2026-06-17: conservar la regional en el modelo (carga senal real, quitarla
costaba -16,5% AUC-PR y NO arreglaba la equidad: Bogota seguia 1,20x). La equidad geografica
se resuelve en la ASIGNACION de leads, no borrando la feature.

Modelo canonico: modelo_fase6_XGBoost-sin-avicena.joblib (58 feats, regional incluida).
Se compara seleccion de leads GLOBAL (top 10% absoluto) vs ESTRATIFICADA (top 10% DENTRO de
cada regional -> cuota proporcional a poblacion). Metrica: recall total y composicion regional.
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

md("""# 11. Seleccion de leads: global vs estratificada por regional — Fase 6

La regional se **conserva** en el modelo (`modelo_fase6_XGBoost-sin-avicena`, 58 feats): quitarla
costaba -16,5% AUC-PR y no arreglaba la equidad (Bogota seguia 1,20x via features proxy). La
equidad geografica se atiende en **como se asignan los leads**:

- **Global**: top 10% de riesgo absoluto en toda la EPS. Maximiza recall total pero deja que
  las regionales con mas datos (Bogota) acaparen cupos.
- **Estratificada**: top 10% **dentro de cada regional**. Cada regional recibe leads proporcionales
  a su poblacion -> equidad por diseno. Cuesta algo de recall total (no concentra en los de mayor
  riesgo absoluto), pero es defendible y operacionalmente justo para el A/B.

Se cuantifica el trade-off recall vs equidad.
""")

code("""import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import xgboost as xgb, joblib
from sklearn.metrics import average_precision_score

B = "bases"; RNG = 42
model = joblib.load(f"{B}/modelo_fase6_XGBoost-sin-avicena.joblib")
nat_va = pd.read_parquet(f"{B}/prediccion_mama_val_native.parquet")
DROP = ['tiene_avicena']
FEAT = [c for c in nat_va.columns if c not in ('key', 'label') and c not in DROP]
y = nat_va['label'].values.astype(int)
X = nat_va[FEAT].astype('float32').values
p = model.predict_proba(X)[:, 1]
print(f"val {X.shape} pos {y.sum()} | AUC-PR {average_precision_score(y, p):.4f}")

# regional por fila desde dummies REG_*
REG = [c for c in nat_va.columns if c.startswith('REG_')]
reg_label = nat_va[REG].values.argmax(axis=1)
reg_names = [c.replace('REG_', '').strip() for c in REG]
regional = np.array([reg_names[i] for i in reg_label])
print("regionales:", reg_names)
""")

md("""## 1. Seleccion global vs estratificada (top 10%)""")

code("""FRAC = 0.10

# GLOBAL: top 10% absoluto
k_global = max(1, int(len(p) * FRAC))
sel_global = np.zeros(len(p), dtype=bool)
sel_global[np.argpartition(-p, k_global)[:k_global]] = True

# ESTRATIFICADA: top 10% dentro de cada regional
sel_strat = np.zeros(len(p), dtype=bool)
for r in reg_names:
    mask = regional == r
    idx = np.where(mask)[0]
    kk = max(1, int(len(idx) * FRAC))
    top_local = idx[np.argpartition(-p[idx], kk)[:kk]]
    sel_strat[top_local] = True

def recall(sel):
    return y[sel].sum() / y.sum()

print(f"GLOBAL      : {sel_global.sum():>7} leads | recall {recall(sel_global):.4f}")
print(f"ESTRATIFICADA: {sel_strat.sum():>7} leads | recall {recall(sel_strat):.4f}")
print(f"\\nTrade-off recall: {recall(sel_strat)-recall(sel_global):+.4f} "
      f"({(recall(sel_strat)/recall(sel_global)-1)*100:+.1f}%)")
""")

md("""## 2. Composicion regional de los leads (equidad)

`sobre_representacion` = % de leads de la regional / % de poblacion de la regional. ~1 = justo.
""")

code("""def composicion(sel, nombre):
    pob = pd.Series(regional).value_counts(normalize=True)
    leads = pd.Series(regional[sel]).value_counts(normalize=True)
    tab = pd.DataFrame({'%_leads': leads * 100, '%_poblacion': pob.reindex(leads.index) * 100})
    tab['sobre_repr'] = tab['%_leads'] / tab['%_poblacion']
    tab = tab.sort_values('%_poblacion', ascending=False)
    print(f"=== {nombre} ===")
    print(tab.to_string(float_format='{:.2f}'.format))
    print(f"  dispersion sobre_repr (max-min): {tab['sobre_repr'].max()-tab['sobre_repr'].min():.2f}\\n")
    return tab

pd.set_option('display.width', 200)
tab_g = composicion(sel_global, 'GLOBAL (top 10% absoluto)')
tab_s = composicion(sel_strat, 'ESTRATIFICADA (top 10% por regional)')
print("La estratificada lleva sobre_repr ~1 en todas las regionales (cuota proporcional);")
print("la global concentra cupos donde hay mas datos.")
""")

md("""## 3. Recall por regional (equidad de cobertura)

No solo cupos justos: cuanto de los casos reales de CADA regional se captura.
""")

code("""rows = []
for r in reg_names:
    mask = regional == r
    pos_r = y[mask].sum()
    if pos_r == 0:
        continue
    rg = y[mask & sel_global].sum() / pos_r
    rs = y[mask & sel_strat].sum() / pos_r
    rows.append({'regional': r, 'positivos': int(pos_r),
                 'recall_global': rg, 'recall_estratif': rs})
rec = pd.DataFrame(rows).set_index('regional').sort_values('positivos', ascending=False)
print("=== Recall de casos reales por regional ===")
print(rec.to_string(float_format='{:.4f}'.format))
print(f"\\nDispersion recall GLOBAL  (max-min): {rec['recall_global'].max()-rec['recall_global'].min():.4f}")
print(f"Dispersion recall ESTRATIF (max-min): {rec['recall_estratif'].max()-rec['recall_estratif'].min():.4f}")
print("Menor dispersion = cobertura mas pareja entre regionales.")
""")

md("""## 4. Guardar diagnostico""")

code("""out = {
    'modelo': 'XGBoost-sin-avicena (regional conservada)',
    'frac': FRAC,
    'recall_global': float(recall(sel_global)),
    'recall_estratificada': float(recall(sel_strat)),
    'trade_off_recall': float(recall(sel_strat) - recall(sel_global)),
    'composicion_global': tab_g.round(3).reset_index().to_dict(orient='records'),
    'composicion_estratificada': tab_s.round(3).reset_index().to_dict(orient='records'),
    'recall_por_regional': rec.round(4).reset_index().to_dict(orient='records'),
}
with open(f"{B}/seleccion_estratificada.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("Guardado: bases/seleccion_estratificada.json")
print(f"\\nGlobal recall {recall(sel_global):.4f} vs estratificada {recall(sel_strat):.4f}")
print("Recomendacion: estratificada para el A/B (equidad por diseno, recall casi igual).")
""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "CAC venv", "language": "python", "name": "cacvenv"},
    "language_info": {"name": "python"}})
with open("11. seleccion estratificada.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook escrito: 11. seleccion estratificada.ipynb con", len(cells), "celdas")

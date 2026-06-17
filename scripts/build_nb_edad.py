# -*- coding: utf-8 -*-
"""Construye 13. operativo por edad.ipynb (Fase 6 - rol del modelo vs tamizaje bienal >=45).

Restriccion operativa (usuario 2026-06-17): el programa garantiza 1 tamizaje cada 2 anos al
menos en mayores de 45. Ese grupo ya esta cubierto por estandar de atencion -> el modelo NO
necesita racionarlo. El valor INCREMENTAL del modelo esta en alto riesgo <45 (que la tamizacion
por edad no capta) y, opcionalmente, densificar (anual) a las >=45 de muy alto riesgo.

Cuantifica: estructura por edad de los casos incidentes, cuanto cubre el bienal >=45, y el
'gap' <45 que el modelo podria llenar + su recall en ese subgrupo.
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

md("""# 13. Operativo por edad — rol del modelo frente al tamizaje bienal >=45

Restriccion: el programa garantiza **1 tamizaje cada 2 anos en >=45** (estandar). Ese grupo ya
queda cubierto. El modelo aporta donde el cribado por edad NO llega:

1. **Alto riesgo <45**: casos que ocurren antes de la edad de tamizaje rutinario -> tamizacion
   proactiva temprana. Aqui esta el valor incremental claro.
2. **Muy alto riesgo >=45**: candidatas a tamizaje **anual** en vez de bienal (intensificacion).

Se cuantifica la estructura por edad de los casos incidentes y el desempeno del modelo en <45.
""")

code("""import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import joblib

B = "bases"
model = joblib.load(f"{B}/modelo_fase6_XGBoost-sin-avicena.joblib")
nat_va = pd.read_parquet(f"{B}/prediccion_mama_val_native.parquet")
FEAT = [c for c in nat_va.columns if c not in ('key', 'label', 'tiene_avicena')]
y = nat_va['label'].values.astype(int)
p = model.predict_proba(nat_va[FEAT].astype('float32').values)[:, 1]
edad = nat_va['edad'].values
N, P = len(y), int(y.sum())
print(f"N={N:,} | positivos={P} | edad: min {edad.min():.0f} max {edad.max():.0f} mediana {np.median(edad):.0f}")
""")

md("""## 1. Estructura por edad de la cohorte y de los casos incidentes""")

code("""bandas = [(18,40,'18-39'),(40,45,'40-44'),(45,50,'45-49'),(50,60,'50-59'),(60,200,'60+')]
rows = []
for lo, hi, name in bandas:
    m = (edad >= lo) & (edad < hi)
    n = int(m.sum()); pos = int(y[m].sum())
    rows.append({'banda': name, 'mujeres': n, '%_pob': n/N*100, 'casos': pos,
                 '%_casos': pos/P*100, 'tasa_banda%': (pos/n*100) if n else 0})
tab = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print("=== Estructura por edad ===")
print(tab.to_string(index=False, float_format='{:.3f}'.format))

cas_lt45 = int(y[edad < 45].sum()); cas_ge45 = int(y[edad >= 45].sum())
print(f"\\nCasos <45: {cas_lt45} ({cas_lt45/P*100:.1f}%)  |  Casos >=45: {cas_ge45} ({cas_ge45/P*100:.1f}%)")
print(f"=> El bienal >=45 cubre por edad ~{cas_ge45/P*100:.0f}% de los casos; queda gap <45 de ~{cas_lt45/P*100:.0f}%.")
""")

md("""## 2. Valor incremental del modelo en <45 (gap del cribado por edad)

Dentro de la subpoblacion <45, recall del modelo al seleccionar top-k% de ESE subgrupo.
Estas serian tamizaciones proactivas tempranas que la regla por edad no haria.
""")

code("""m45 = edad < 45
idx45 = np.where(m45)[0]; p45 = p[idx45]; y45 = y[idx45]
N45, P45 = len(idx45), int(y45.sum())
print(f"Subpoblacion <45: {N45:,} mujeres | {P45} casos | tasa {P45/N45*100:.4f}%")
rows = []
for f in (0.005, 0.01, 0.02, 0.05, 0.10):
    k = max(1, int(N45*f)); sel = np.argpartition(-p45, k)[:k]
    tp = int(y45[sel].sum()); vpp = tp/k
    rows.append({'top_k%_<45': f*100, 'cupos': k, 'recall_<45': tp/P45 if P45 else 0,
                 'VPP%': vpp*100, 'NNS': (1/vpp) if vpp else np.inf})
print("=== Modelo dentro de <45 ===")
print(pd.DataFrame(rows).to_string(index=False, float_format='{:.3f}'.format))
""")

md("""## 3. Intensificacion en >=45: muy alto riesgo -> tamizaje anual

Entre las >=45 (ya en bienal), el top de riesgo del modelo son candidatas a **anual**. Recall
del modelo dentro de >=45 para dimensionar cuantas tamizaciones extra implicaria.
""")

code("""mge = edad >= 45
idxge = np.where(mge)[0]; pge = p[idxge]; yge = y[idxge]
Nge, Pge = len(idxge), int(yge.sum())
print(f"Subpoblacion >=45: {Nge:,} mujeres | {Pge} casos | tasa {Pge/Nge*100:.4f}%")
rows = []
for f in (0.01, 0.02, 0.05, 0.10):
    k = max(1, int(Nge*f)); sel = np.argpartition(-pge, k)[:k]
    tp = int(yge[sel].sum()); vpp = tp/k
    rows.append({'top_k%_>=45': f*100, 'cupos_anual_extra': k, 'recall_>=45': tp/Pge if Pge else 0,
                 'VPP%': vpp*100, 'NNS': (1/vpp) if vpp else np.inf})
print("=== Modelo dentro de >=45 (candidatas a anual) ===")
print(pd.DataFrame(rows).to_string(index=False, float_format='{:.3f}'.format))
""")

md("""## 4. Guardar diagnostico por edad""")

code("""out = {
    'restriccion': 'tamizaje bienal garantizado en >=45',
    'edad_corte_cribado': 45,
    'casos_total': P, 'casos_lt45': cas_lt45, 'casos_ge45': cas_ge45,
    'pct_casos_lt45': cas_lt45/P*100, 'pct_casos_ge45': cas_ge45/P*100,
    'estructura_edad': tab.round(4).to_dict(orient='records'),
    'subpob_lt45': {'N': N45, 'casos': P45},
    'subpob_ge45': {'N': Nge, 'casos': Pge},
}
with open(f"{B}/operativo_edad.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False, default=float)
print("Guardado: bases/operativo_edad.json")
""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "CAC venv", "language": "python", "name": "cacvenv"},
    "language_info": {"name": "python"}})
with open("13. operativo por edad.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook escrito: 13. operativo por edad.ipynb con", len(cells), "celdas")

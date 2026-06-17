# -*- coding: utf-8 -*-
"""Construye 12. umbral operativo.ipynb (Fase 6 - punto de operacion).

Define el corte operativo del programa de tamizacion sobre el modelo canonico
(modelo_fase6_XGBoost-sin-avicena, 58 feats, regional conservada). Barre top-k% y reporta
cupos, recall, VPP/precision, lift y NNS (number needed to screen) en GLOBAL y ESTRATIFICADO.
La eleccion final del corte depende de la capacidad del programa (cupos/ano).
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

md("""# 12. Umbral operativo del programa de tamizacion — Fase 6

Modelo canonico **XGBoost-sin-avicena** (58 feats, regional conservada). Asignacion elegida:
**estratificada por regional**. Aqui se elige el **punto de corte** (cuantas mujeres tamizar).

Para cada fraccion top-k% se reporta:
- **cupos**: leads seleccionados (mujeres a tamizar).
- **umbral_prob**: probabilidad minima del corte.
- **recall**: % de casos incidentes reales captados.
- **VPP (precision)**: % de leads que son caso real.
- **lift**: VPP / tasa base (cuanto mejor que tamizar al azar).
- **NNS**: number needed to screen = 1/VPP (mujeres a tamizar por cada caso hallado).

> Tasa base ~0,04% => VPP bajo y NNS alto por naturaleza (incidencia, no diagnostico).
> El valor del modelo es el **lift**: concentra los casos en una fraccion pequena.
""")

code("""import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import joblib
from sklearn.metrics import average_precision_score

B = "bases"
model = joblib.load(f"{B}/modelo_fase6_XGBoost-sin-avicena.joblib")
nat_va = pd.read_parquet(f"{B}/prediccion_mama_val_native.parquet")
FEAT = [c for c in nat_va.columns if c not in ('key', 'label', 'tiene_avicena')]
y = nat_va['label'].values.astype(int)
X = nat_va[FEAT].astype('float32').values
p = model.predict_proba(X)[:, 1]
N, P = len(y), int(y.sum())
base = P / N
REG = [c for c in nat_va.columns if c.startswith('REG_')]
reg_names = [c.replace('REG_', '').strip() for c in REG]
regional = np.array([reg_names[i] for i in nat_va[REG].values.argmax(axis=1)])
print(f"Validacion 2023->2024: N={N:,} | positivos={P} | tasa base={base*100:.4f}% | AUC-PR={average_precision_score(y,p):.4f}")
""")

md("""## 1. Curva de operacion — GLOBAL (top-k% absoluto)""")

code("""def sel_global(frac):
    k = max(1, int(N * frac))
    s = np.zeros(N, dtype=bool); s[np.argpartition(-p, k)[:k]] = True
    return s

def sel_strat(frac):
    s = np.zeros(N, dtype=bool)
    for r in reg_names:
        idx = np.where(regional == r)[0]
        kk = max(1, int(len(idx) * frac))
        s[idx[np.argpartition(-p[idx], kk)[:kk]]] = True
    return s

def fila(s, frac):
    n = int(s.sum()); tp = int(y[s].sum())
    vpp = tp / n if n else 0
    return {'top_k%': frac*100, 'cupos': n, 'umbral_prob': float(p[s].min()),
            'recall': tp / P, 'VPP%': vpp*100, 'lift': (vpp/base) if base else 0,
            'NNS': (1/vpp) if vpp else np.inf}

FRACS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20]
tab_g = pd.DataFrame([fila(sel_global(f), f) for f in FRACS])
pd.set_option('display.width', 200, 'display.max_columns', 20)
print("=== GLOBAL (top-k% absoluto) ===")
print(tab_g.to_string(index=False, float_format='{:.3f}'.format))
""")

md("""## 2. Curva de operacion — ESTRATIFICADO por regional (asignacion elegida)""")

code("""tab_s = pd.DataFrame([fila(sel_strat(f), f) for f in FRACS])
print("=== ESTRATIFICADO (top-k% por regional) ===")
print(tab_s.to_string(index=False, float_format='{:.3f}'.format))
print("\\nNota: recall estratificado ~1pp por debajo del global; cupos casi iguales; VPP/NNS similares.")
""")

md("""## 3. Escalado a poblacion de produccion 2024

La validacion (2023->2024) aproxima el tamano de produccion. Se muestran los cupos absolutos
que implicaria cada corte para dimensionar la capacidad del programa.
""")

code("""esc = tab_s[['top_k%', 'cupos', 'recall', 'VPP%', 'lift', 'NNS']].copy()
esc['casos_esperados_captados'] = (esc['recall'] * P).round(0).astype(int)
esc['casos_totales'] = P
print("=== Dimensionamiento (estratificado) ===")
print(esc.to_string(index=False, float_format='{:.2f}'.format))
print(f"\\nLectura: con top-5% se tamizan ~{int(N*0.05):,} mujeres y se captan "
      f"~{int(tab_s.iloc[4]['recall']*P)} de {P} casos ({tab_s.iloc[4]['recall']*100:.0f}%).")
""")

md("""## 4. Guardar curva de operacion""")

code("""out = {
    'modelo': 'XGBoost-sin-avicena (estratificado)',
    'N_val': N, 'positivos': P, 'tasa_base': base,
    'curva_global': tab_g.replace([np.inf], None).to_dict(orient='records'),
    'curva_estratificada': tab_s.replace([np.inf], None).to_dict(orient='records'),
}
with open(f"{B}/umbral_operativo.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False, default=float)
print("Guardado: bases/umbral_operativo.json")
print("\\nElegir corte segun capacidad del programa (cupos/ano). Tabla lista para decidir.")
""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "CAC venv", "language": "python", "name": "cacvenv"},
    "language_info": {"name": "python"}})
with open("12. umbral operativo.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook escrito: 12. umbral operativo.ipynb con", len(cells), "celdas")

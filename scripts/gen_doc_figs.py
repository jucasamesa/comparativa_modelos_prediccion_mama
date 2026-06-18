# -*- coding: utf-8 -*-
"""Genera las figuras del documento tecnico a partir de los artefactos JSON/CSV del proyecto.
Todas las cifras son trazables (regla de integridad de datos). Salida: _doc/figs/*.png
Tambien extrae los beeswarm SHAP embebidos en los notebooks 05 (LightGBM) y 08 (XGBoost).
"""
import os, json, base64
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat

B = "bases"
OUT = "_doc/figs"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'figure.dpi': 130, 'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3})

def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}.png", bbox_inches='tight'); plt.close(fig)
    print("fig:", name)

# ---------- 1. Embudo de construccion del dataset ----------
etapas = ['Pob. T\n(sin mama)', 'Continuidad\nT->T+1', 'Negativos\n(label=0)', 'Positivos\n(label=1)']
train = [2462241+3253092, 2446215+3025131, 5469489, 1857]
fig, ax = plt.subplots(figsize=(7, 4))
b = ax.bar(etapas, train, color=['#4C72B0','#55A868','#C44E52','#DD8452'])
ax.set_yscale('log'); ax.set_ylabel('Registros (escala log)')
ax.set_title('Embudo de construccion — conjunto de entrenamiento (2 pares)')
for r, v in zip(b, train):
    ax.text(r.get_x()+r.get_width()/2, v, f"{v:,}", ha='center', va='bottom', fontsize=8)
save(fig, '01_embudo_dataset')

# ---------- 2. Comparacion de modelos (AUC-PR validacion) ----------
modelos = [
    ('LightGBM-spw (roto, spw=2307)', 0.000503),
    ('LR-balanced', 0.002019),
    ('LR-balanced (tuneado)', 0.009425),
    ('LR-SMOTE', 0.009758),
    ('LR-SMOTE (tuneado)', 0.009914),
    ('XGBoost-spw (sin tunear)', 0.030894),
    ('LGB-SMOTE (tuneado F5)', 0.031309),
    ('LightGBM-SMOTE (F4)', 0.033451),
    ('Random Forest (tuneado)', 0.033671),
    ('LightGBM spw=1 (F5, ex-ganador)', 0.034631),
    ('XGBoost tuneado (F5b, GANADOR)', 0.0380),
]
nom = [m[0] for m in modelos]; val = [m[1] for m in modelos]
colors = ['#999999']*5 + ['#4C72B0']*3 + ['#DD8452','#55A868','#C44E52']
fig, ax = plt.subplots(figsize=(8, 5))
b = ax.barh(nom, val, color=colors); ax.invert_yaxis()
ax.set_xlabel('AUC-PR (validacion temporal 2023->2024)')
ax.set_title('Comparacion de modelos — AUC-PR (tasa base 0,041%)')
for r, v in zip(b, val):
    ax.text(v, r.get_y()+r.get_height()/2, f" {v:.4f}", va='center', fontsize=8)
save(fig, '02_comparacion_modelos')

# ---------- 3. Recall@top-k ----------
curvas = {
    'XGBoost tuneado (F5b)': [0.266, 0.325, 0.563, 0.704],
    'LightGBM spw=1 (F5)':   [0.2437, 0.3061, 0.5468, 0.6727],
    'LightGBM-SMOTE (F4)':   [0.2477, 0.3333, 0.5559, 0.6777],
    'Random Forest (tuneado)': [0.2508, 0.3031, 0.4935, 0.6113],
    'XGBoost-spw (F4)':      [0.2437, 0.3021, 0.5277, 0.6445],
    'LR-SMOTE (F4)':         [0.2024, 0.2679, 0.5015, 0.6516],
}
ks = [0.5, 1, 5, 10]
fig, ax = plt.subplots(figsize=(7, 4.5))
for nm, ys in curvas.items():
    ax.plot(ks, ys, marker='o', label=nm)
ax.set_xlabel('Top-k% de la poblacion'); ax.set_ylabel('Recall (casos captados)')
ax.set_title('Recall@top-k% — validacion 2023->2024'); ax.legend(fontsize=8)
save(fig, '03_recall_topk')

# ---------- 4. Barrido scale_pos_weight LightGBM ----------
f5 = json.load(open(f"{B}/tuning_fase5.json", encoding='utf-8'))
spw = f5['spw_sweep']
xs = [s[0] for s in spw]; ys = [s[1] for s in spw]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(xs, ys, marker='s', color='#C44E52')
ax.set_xscale('log'); ax.set_xlabel('scale_pos_weight (escala log)')
ax.set_ylabel('AUC-PR (CV)')
ax.set_title('LightGBM: el reponderado degrada el ranking (spw=1 es optimo)')
ax.axhline(ys[0], ls='--', color='gray', alpha=0.6)
for x, y in zip(xs, ys):
    ax.text(x, y, f" {y:.3f}", fontsize=8)
save(fig, '04_spw_sweep')

# ---------- 5 y 6. SHAP barras XGBoost y LightGBM ----------
sx = json.load(open(f"{B}/shap_xgboost.json", encoding='utf-8'))['mean_abs_shap']
sx = pd.Series(sx).sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(7, 6))
sx[::-1].plot(kind='barh', ax=ax, color='#C44E52')
ax.set_xlabel('mean|SHAP|'); ax.set_title('SHAP — XGBoost ganador (top 20)')
save(fig, '05_shap_xgboost')

sl = pd.Series(f5['shap_contribucion']).sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(7, 6))
sl[::-1].plot(kind='barh', ax=ax, color='#4C72B0')
ax.set_xlabel('mean|SHAP|'); ax.set_title('SHAP — LightGBM spw=1 (top 20)')
save(fig, '06_shap_lightgbm')

# ---------- 7. Flags / familiares con SHAP ~0 ----------
sx_all = pd.Series(json.load(open(f"{B}/shap_xgboost.json", encoding='utf-8'))['mean_abs_shap'])
bottom = sx_all.sort_values().head(14)
fig, ax = plt.subplots(figsize=(7, 5))
bottom.plot(kind='barh', ax=ax, color='#8172B3')
ax.set_xlabel('mean|SHAP|'); ax.set_title('Caracteristicas con contribucion ~0 (XGBoost)')
save(fig, '07_flags_cero')

# ---------- 8. Curva de operacion (estratificada) ----------
uo = json.load(open(f"{B}/umbral_operativo.json", encoding='utf-8'))['curva_estratificada']
k = [r['top_k%'] for r in uo]; rec = [r['recall'] for r in uo]; nns = [r['NNS'] for r in uo]
fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(k, rec, marker='o', color='#55A868', label='Recall')
ax1.set_xlabel('Top-k% (estratificado por regional)'); ax1.set_ylabel('Recall', color='#55A868')
ax2 = ax1.twinx(); ax2.plot(k, nns, marker='s', color='#C44E52', label='NNS')
ax2.set_ylabel('NNS (a tamizar por caso)', color='#C44E52'); ax2.grid(False)
ax1.set_title('Curva de operacion del programa de tamizacion')
save(fig, '08_curva_operacion')

# ---------- 9. Estructura por edad ----------
oe = json.load(open(f"{B}/operativo_edad.json", encoding='utf-8'))['estructura_edad']
bandas = [r['banda'] for r in oe]; pcas = [r['%_casos'] for r in oe]; tasa = [r['tasa_banda%'] for r in oe]
fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.bar(bandas, pcas, color='#4C72B0', alpha=0.8, label='% de casos')
ax1.set_ylabel('% de casos incidentes', color='#4C72B0'); ax1.set_xlabel('Banda de edad')
ax2 = ax1.twinx(); ax2.plot(bandas, tasa, marker='o', color='#C44E52', label='tasa banda')
ax2.set_ylabel('Tasa de incidencia banda (%)', color='#C44E52'); ax2.grid(False)
ax1.axvline(1.5, ls='--', color='gray'); ax1.text(1.55, max(pcas)*0.9, 'corte 45', fontsize=8)
ax1.set_title('Estructura por edad de los casos incidentes')
save(fig, '09_estructura_edad')

# ---------- 10. Poder A/B: casos por brazo vs efecto ----------
ab = json.load(open(f"{B}/diseno_abtest.json", encoding='utf-8'))['casos_requeridos']
delta = [r['temprano_interv%'] for r in ab]; ncasos = [r['casos_por_brazo'] for r in ab]
fig, ax = plt.subplots(figsize=(7, 4))
b = ax.bar([f"{d:.0f}%" for d in delta], ncasos, color='#DD8452')
ax.set_xlabel('% estadio temprano en intervencion (base control 54%)')
ax.set_ylabel('Casos requeridos por brazo'); ax.set_yscale('log')
ax.set_title('Dimensionamiento A/B (poder 80%, alpha 0,05)')
for r, v in zip(b, ncasos):
    ax.text(r.get_x()+r.get_width()/2, v, str(v), ha='center', va='bottom', fontsize=8)
save(fig, '10_poder_ab')

# ---------- Extraer beeswarm SHAP de notebooks ----------
def extract_png(nb_path, out_name, which=0):
    try:
        nb = nbformat.read(nb_path, 4); count = 0
        for c in nb.cells:
            if c.cell_type != 'code':
                continue
            for o in c.get('outputs', []):
                data = o.get('data', {})
                if 'image/png' in data:
                    if count == which:
                        png = base64.b64decode(data['image/png'])
                        open(f"{OUT}/{out_name}.png", 'wb').write(png)
                        print("beeswarm:", out_name); return True
                    count += 1
        print("NO png en", nb_path)
    except Exception as e:
        print("err", nb_path, e)
    return False

extract_png("08. shap xgboost.ipynb", "11_beeswarm_xgboost", which=0)
extract_png("05. tuning e interpretabilidad.ipynb", "12_beeswarm_lightgbm", which=0)

print("\\nFiguras generadas en", OUT)

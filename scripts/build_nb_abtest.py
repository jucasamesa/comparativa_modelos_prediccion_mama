# -*- coding: utf-8 -*-
"""Construye 14. diseno ab test.ipynb (Fase 6 - dimensionamiento del A/B).

Alcance (decision usuario 2026-06-17): brazo unico = <45 alto riesgo -> mamografia proactiva
vs estandar (cribado inicia a los 45). Modelo canonico XGBoost-sin-avicena.

Dimensiona el A/B: cuantos leads/casos por brazo se necesitan para detectar un desplazamiento
de estadio al diagnostico. Lineas base verificadas (CAC 2024): estadios I-IIa 54%, III-IV 28,16%.

NO inventa el tamano del efecto: barre un rango de efectos y reporta n requerido + factibilidad
segun el pool de leads <45 disponible por ano a cada corte top-k%.
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

md("""# 14. Diseno y dimensionamiento del A/B test — Fase 6

**Alcance:** brazo unico, mujeres **<45 alto riesgo** (top-k% del modelo) -> mamografia
proactiva (intervencion) vs atencion estandar (control: cribado inicia a los 45).

**Endpoint primario:** estadio al diagnostico entre casos confirmados (I-IIa temprano vs III-IV
tardio). La intervencion deberia **aumentar la fraccion temprana** / reducir la tardia.

**Lineas base (CAC 2024, verificadas):** estadios I-IIa = **54%**; estadios III-IV = **28,16%**.

Como la incidencia es rara, el cuello de botella es el **numero de casos acumulados** por brazo.
Se barre el tamano del efecto (no se inventa) y se reporta n de casos requerido, leads requeridos
(via VPP del corte) y anos de acumulacion necesarios.
""")

code("""import json, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")

B = "bases"
ALPHA = 0.05; POWER = 0.80
z_a = stats.norm.ppf(1 - ALPHA/2); z_b = stats.norm.ppf(POWER)

# VPP del modelo dentro de <45 por corte (de notebook 13, validacion 2023->2024, horizonte 1 ano)
# top_k% -> (leads_pool_anual, VPP, recall_<45)
VPP_LT45 = {
    1.0: {'leads': 13810, 'vpp': 0.00710, 'recall': 0.426},
    2.0: {'leads': 27620, 'vpp': 0.00456, 'recall': 0.548},
    5.0: {'leads': 69051, 'vpp': 0.00223, 'recall': 0.670},
}
BASE_TEMPRANO = 0.54   # I-IIa, CAC 2024 Fig 2.4
print(f"alpha={ALPHA}, power={POWER} | z_a={z_a:.3f} z_b={z_b:.3f}")
print(f"Linea base temprano (I-IIa): {BASE_TEMPRANO*100:.0f}%")
""")

md("""## 1. Casos requeridos por brazo segun el efecto (estadio temprano)

Test de dos proporciones (control 54% temprano vs intervencion mas alta). n = casos por brazo.
""")

code("""def n_por_brazo(p1, p2):
    pbar = (p1 + p2) / 2
    num = (z_a*np.sqrt(2*pbar*(1-pbar)) + z_b*np.sqrt(p1*(1-p1)+p2*(1-p2)))**2
    return int(np.ceil(num / (p1-p2)**2))

efectos = [0.62, 0.66, 0.70, 0.74, 0.78]  # % temprano en intervencion
rows = []
for p2 in efectos:
    n = n_por_brazo(BASE_TEMPRANO, p2)
    rows.append({'temprano_control%': BASE_TEMPRANO*100, 'temprano_interv%': p2*100,
                 'delta_pp': (p2-BASE_TEMPRANO)*100, 'casos_por_brazo': n, 'casos_total': 2*n})
casos = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print("=== Casos confirmados requeridos por brazo (power 80%, alpha 0.05) ===")
print(casos.to_string(index=False, float_format='{:.1f}'.format))
print("\\nEfecto mas pequeno = mas casos. Un +8pp (54->62%) exige muchos casos; +20pp (54->74%) pocos.")
""")

md("""## 2. Leads y anos de acumulacion requeridos por corte

Casos por brazo / VPP = leads por brazo. El pool anual de leads <45 a cada corte limita cuanto
se acumula por ano (asumiendo 1:1 intervencion:control sobre el pool elegible).
""")

code("""rows = []
for k, info in VPP_LT45.items():
    pool, vpp = info['leads'], info['vpp']
    # leads por brazo = pool/2 por ano (split 1:1). casos por brazo por ano = (pool/2)*vpp
    casos_brazo_ano = (pool/2) * vpp
    for p2 in [0.66, 0.70, 0.74]:
        n = n_por_brazo(BASE_TEMPRANO, p2)
        leads_brazo = int(np.ceil(n / vpp))
        anios = leads_brazo / (pool/2)
        rows.append({'top_k%_<45': k, 'VPP%': vpp*100, 'pool_anual': pool,
                     'efecto_interv%': p2*100, 'casos/brazo_req': n,
                     'leads/brazo_req': leads_brazo, 'casos/brazo_por_ano': round(casos_brazo_ano,1),
                     'anios_acum': round(anios,1)})
diseño = pd.DataFrame(rows)
print("=== Factibilidad por corte y efecto ===")
print(diseño.to_string(index=False, float_format='{:.2f}'.format))
print("\\nanios_acum = cuantos anos de reclutamiento al pool anual para juntar los casos.")
""")

md("""## 3. Lectura y recomendacion de diseno

- Incidencia rara => el endpoint de **estadio** necesita decenas-cientos de casos por brazo;
  con un solo ano de leads <45 puede no alcanzar salvo efectos grandes.
- Opciones para ganar poder: (a) **acumular varios anos** de reclutamiento; (b) **endpoint
  intermedio** mas frecuente (deteccion temprana por 1000 tamizadas, combina incidencia+estadio);
  (c) ampliar el corte top-k% (mas leads, menor VPP -> mas tamizaciones por caso).
- **Unidad de randomizacion:** individual (paciente) maximiza poder (sin efecto de diseno). Si se
  randomiza por regional (cluster), el n se infla por ICC -> generalmente peor; usar solo si hay
  contaminacion operativa entre brazos.
""")

code("""# Endpoint secundario mas potente: tasa de deteccion de cancer (cualquier estadio) por brazo.
# La intervencion adelanta el dx; a 1 ano puede subir la tasa detectada. Test de 2 proporciones
# sobre incidencia detectada entre leads (mucho mas frecuente que el subgrupo de casos-estadio).
print("Endpoint secundario (deteccion): comparar % de leads con dx confirmado interv vs control.")
print("Mas potente porque usa TODOS los leads como denominador, no solo los casos.")
for k, info in VPP_LT45.items():
    pool, vpp = info['leads'], info['vpp']
    # asume intervencion detecta vpp; control detecta menos a 1 ano (lead time). MDE ilustrativo:
    # con pool/2 por brazo, MDE de proporciones alrededor de vpp:
    n_brazo = pool//2
    se = np.sqrt(2*vpp*(1-vpp)/n_brazo)
    mde = (z_a + z_b)*se
    print(f"  top-{k:.0f}%: {n_brazo:,}/brazo, VPP base {vpp*100:.3f}% -> MDE deteccion ~{mde*100:.3f}pp "
          f"(detectable si interv-control >= {mde*100:.3f}pp)")
""")

md("""## 4. Guardar dimensionamiento""")

code("""out = {
    'alcance': '<45 alto riesgo -> mamografia proactiva vs estandar',
    'endpoint_primario': 'estadio al diagnostico (I-IIa temprano)',
    'base_temprano_pct': BASE_TEMPRANO*100,
    'fuente_base': 'CAC 2024 Fig 2.4 (I-IIa 54%)',
    'alpha': ALPHA, 'power': POWER,
    'casos_requeridos': casos.to_dict(orient='records'),
    'factibilidad_por_corte': diseño.to_dict(orient='records'),
    'unidad_randomizacion_recomendada': 'individual (paciente)',
    'notas': 'incidencia rara -> considerar acumular varios anos o endpoint de deteccion por 1000',
}
with open(f"{B}/diseno_abtest.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False, default=float)
print("Guardado: bases/diseno_abtest.json")
""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "CAC venv", "language": "python", "name": "cacvenv"},
    "language_info": {"name": "python"}})
with open("14. diseno ab test.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook escrito: 14. diseno ab test.ipynb con", len(cells), "celdas")

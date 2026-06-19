# Comparativa de modelos para la predicción prospectiva de cáncer de mama

Código fuente del Trabajo de Fin de Estudios (TFE) sobre la **predicción prospectiva de
incidencia de cáncer de mama** en una población femenina afiliada de escala poblacional.
El repositorio contiene el código de **modelado, comparación de algoritmos, interpretabilidad
y diseño operativo**, junto con los artefactos de resultados agregados y las figuras de la
memoria.

> **Autoría.** Este proyecto se desarrollo para la maestría de Inteligencia Artificial
> de la Universidad Internacional de la Rioja (UNIR). Y los autores fueron:
> - Juan Salazar: jucasamesa2@gmail.com,
> - John Alvarado: John.alvaradocristancho@gmail.com
> - Miguel Mojica: chapid99@gmail.com
**Director**: Joan Escamilla Fuster
---

## Objetivo

A diferencia de los modelos de *diagnóstico* (sobre imagen mamográfica) o de los modelos
clásicos de riesgo individual (Gail, Tyrer-Cuzick), este proyecto adopta un enfoque de
**incidencia**: identificar, entre las mujeres actualmente **sin diagnóstico**, a quienes
presentan mayor probabilidad de desarrollar un **primer** cáncer de mama en el **año
siguiente**, para orientar una intervención proactiva de tamización. La evaluación de impacto
de dicha intervención mediante un ensayo A/B se plantea como **trabajo futuro**, en una fase
posterior a este TFE.

El problema se formaliza como una **clasificación binaria con desbalanceo extremo**
(tasa de evento ≈ 0,04 %; razón negativo:positivo ≈ 2.300:1) sobre pares temporales
**T → T+1** construidos con control estricto de fugas de información.

## Resultados principales

- **Modelo ganador:** XGBoost tuneado con Optuna sobre datos con *NaN* nativo, **58
  características** (sin la señal de vigilancia `tiene_avicena`) — **AUC-PR 0,0388**,
  **recall@top10 % 0,699**, AUC-ROC 0,901 (validación temporal). Random Forest queda 2.º
  (AUC-PR 0,0351).
- **Hallazgo metodológico:** la **reponderación agresiva de clases degrada el ordenamiento
  de riesgo**; el óptimo es reponderación nula o leve, fijando el punto de corte por separado
  mediante el percentil de riesgo (top-*k*).
- **Interpretabilidad (SHAP):** dominan la edad y la intensidad de uso del sistema de salud;
  los antecedentes familiares no aportan señal por una limitación de construcción del *proxy*
  (no por irrelevancia biológica), y los indicadores de disponibilidad de laboratorio resultan
  redundantes con el manejo nativo del *NaN*.
- **Diseño operativo:** punto de operación (curva recall/NNS) y asignación estratificada por
  regional para equidad geográfica. Se delimita la subpoblación < 45 años como aquella donde el
  modelo aporta valor incremental frente al tamizaje bienal por edad; su evaluación mediante un
  ensayo A/B queda como trabajo futuro.

## Estructura del repositorio

```
notebooks/      Notebooks de modelado, comparación, interpretabilidad y diseño operativo
  03. feature engineering.ipynb         Ingeniería de características
  04. entrenamiento.ipynb               Entrenamiento y comparación base (LR / LightGBM / XGBoost)
  05. tuning e interpretabilidad.ipynb  Tuning + SHAP (LightGBM)
  07. tuning modelos perdedores.ipynb   Tuning de los modelos perdedores (XGBoost / LR)
  08. shap xgboost.ipynb                Interpretabilidad SHAP del modelo ganador
  09. sin tiene_avicena.ipynb           Ablación controlada de la característica de vigilancia
  10. sin regional.ipynb                Análisis de sensibilidad de la regional
  11. seleccion estratificada.ipynb     Asignación estratificada (equidad geográfica)
  12. umbral operativo.ipynb            Curva de operación del programa
  13. operativo por edad.ipynb          Estructura por edad y rol incremental del modelo
  14. diseno ab test.ipynb              Dimensionamiento preliminar del ensayo A/B (trabajo futuro)
  15. random forest.ipynb               Random Forest (ensamble por bagging) — cierre de la comparativa
scripts/
  variantes_modelo.py   Variantes de imputación (mediana vs NaN nativo)
utils/
  fe_qq.py              Utilidades de diagnóstico cuantil-cuantil
resultados/       Artefactos de resultados agregados (métricas, SHAP, curvas) — sin datos de paciente
figs/             Figuras de la memoria (visualizaciones agregadas)
```

Los notebooks se ejecutan con `jupyter nbconvert` sobre un entorno virtual dedicado y cargan los
artefactos agregados de `resultados/` (métricas, hiperparámetros, valores SHAP) para reproducir
las tablas y figuras de la memoria sin acceder a datos de paciente. `scripts/variantes_modelo.py`
construye las dos variantes de imputación (mediana vs *NaN* nativo) empleadas en el modelado.

## Metodología (resumen)

1. **Construcción de pares T→T+1** con definición de incidencia estricta y triple barrera
   anti-*leakage* (temporal, de prevalencia y `GroupKFold` por paciente).
2. **Ingeniería de características** con cohorte ≥ 18 años, *winsorización* a cotas de
   plausibilidad clínica y codificación; dos variantes de imputación (mediana + indicador para
   modelos lineales; *NaN* nativo + indicador para árboles, por la naturaleza MNAR del faltante).
3. **Comparación de modelos** (regresión logística, Random Forest, LightGBM, XGBoost) bajo
   distintas estrategias de desbalanceo, con validación cruzada `GroupKFold` y validación
   temporal final.
4. **Tuning** con Optuna optimizando AUC-PR; **interpretabilidad** con SHAP.
5. **Diseño operativo**: punto de operación y equidad geográfica (el ensayo A/B se plantea como
   trabajo futuro, fuera del alcance de este TFE).

## Reproducibilidad y datos

Las cifras de la memoria provienen de los artefactos de `resultados/` y de las figuras de
`figs/`, todos ellos **agregados** (métricas, valores SHAP, recuentos por banda de edad,
composición regional) y **sin información a nivel de paciente**.

> **Nota de confidencialidad.** Este TFE está asociado a un proyecto empresarial. Por razones
> de confidencialidad de la información clínica de las pacientes, **no se publican las bases de
> datos utilizadas** (radicados de la Cuenta de Alto Costo, población de afiliados e historia
> clínica electrónica) ni el *pipeline* de construcción de datos que accede a la infraestructura
> interna. En consecuencia, los scripts de modelado **no son ejecutables de extremo a extremo**
> sin dichos datos; se publican como evidencia del trabajo desarrollado y para la
> reproducibilidad de los análisis y figuras a partir de los resultados agregados incluidos.

### Entorno

Stack principal (ver `requirements.txt`): Python 3.12, `numpy 1.26.4` (fijado), `pandas`,
`scikit-learn`, `lightgbm`, `xgboost`, `imbalanced-learn`, `optuna`, `shap`, `python-docx`,
`matplotlib`.

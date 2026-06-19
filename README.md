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
(tasa de evento ≈ 0,03 %; razón negativo:positivo ≈ 3.000:1) sobre pares temporales
**T → T+1** construidos con control estricto de fugas de información.

## Resultados principales

- **Modelo ganador:** XGBoost tuneado con Optuna sobre datos con *NaN* nativo —
  **AUC-PR 0,0380**, **recall@top10 % 0,704**, AUC-ROC 0,900 (validación temporal).
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
scripts/        Generadores de notebooks y análisis (modelado, tuning, SHAP, operativo, A/B)
  build_nb_fe.py            Ingeniería de características
  build_nb_train.py        Entrenamiento y comparación base (LR / LightGBM / XGBoost)
  build_nb_tune.py         Tuning + SHAP (LightGBM)
  build_nb_losers.py       Tuning de los modelos perdedores (XGBoost / LR)
  build_nb_sin_avicena.py  Poda de característica de vigilancia
  build_nb_sin_regional.py Análisis de sensibilidad de la regional
  build_nb_estratificado.py Asignación estratificada (equidad geográfica)
  build_nb_umbral.py       Curva de operación del programa
  build_nb_edad.py         Estructura por edad y rol incremental del modelo
  build_nb_abtest.py       Dimensionamiento preliminar del ensayo A/B (trabajo futuro)
  build_nb_shap.py         Interpretabilidad SHAP del modelo ganador
  build_nb_rf.py           Random Forest (ensamble por bagging) — cierre de la comparativa
  variantes_modelo.py      Variantes de imputación (mediana vs NaN nativo)
  gen_doc_figs.py          Generación de las figuras de la memoria
utils/
  fe_qq.py        Utilidades de diagnóstico cuantil-cuantil
doc/
  gen_documento.py  Generador del documento técnico (.docx)
resultados/       Artefactos de resultados agregados (métricas, SHAP, curvas) — sin datos de paciente
figs/             Figuras de la memoria (visualizaciones agregadas)
```

Los scripts `build_nb_*.py` no contienen lógica de negocio embebida: cada uno **construye un
notebook** reproducible mediante `nbformat` y se ejecuta con `jupyter nbconvert` sobre un
entorno virtual dedicado.

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

# -*- coding: utf-8 -*-
"""
06. Variantes de imputación para Fase 4
=======================================

Genera DOS variantes de la base de modelado con **columnas idénticas**, que difieren
únicamente en cómo se tratan los continuos con faltantes (la decisión de imputación):

  - `*_impute.parquet`  → mediana(train ≥18) + `{var}_disponible`   (para LR / modelos lineales)
  - `*_native.parquet`  → NaN nativo + `{var}_disponible`           (para XGBoost/LightGBM)

Todo lo demás es idéntico en ambas (winsorización, drops, encoding, flags Avicena=0,
deltas=0, ZONA=moda). Así la comparación en Fase 4 aísla exactamente median vs NaN.

Fuente: base cruda `bases/prediccion_mama_base entrenamiento {train|validacion}.parquet`
(único punto del pipeline que aún conserva los NaN originales).

Decisiones replicadas de Fase 3 / 3.5:
  - Cohorte edad ≥ 18 (antes de imputar).
  - Drop fuga CAC + remisión mama/onco; drop duplicados/constantes; drop PESO (se conserva IMC).
  - Winsorización a cotas de plausibilidad (errores de captura).
  - Sin log1p (find_log1p_candidates no recomendó ninguna).
  - imc_disponible se CONSERVA (en la variante nativa ya no es constante).
"""
import json

import numpy as np
import pandas as pd

BASES = "bases"

# ------------------------------------------------------------------ constantes
CAC_LEAKAGE = [
    'incidentes', 'cancer_mama', 'mama',
    'v17nombredelaneoplasiacncerotumo', 'v18fechadediagnsticodelcncerrepo',
    'v42antecedentedeotrocncerprimari', 'v43fechadediagnsticodelotrocncer',
    'v44tiponombredeesecnceranteceden', 'v27histologadeltumorenmuestradeb',
    'v28gradodediferenciacindeltumors', 'v29primeraestadificacinbasadaent',
    'v33paracncerdemamaresultadodelap', 'v126resultadofinaldelmanejooncol',
    'v127estadovitalalfinalizaresteco', 'v130fechadedesafiliacindelaeps',
    'v131fechademuerte',
]
REMISION_LEAKAGE = ['remision_mama', 'remision_onco']
DROP_ALWAYS = CAC_LEAKAGE + REMISION_LEAKAGE + [
    'key2', 'mes',
    'DEJO_FUMAR_MENOS_10_ANOS', 'FUMA_MAS_10_CIGARRILLOS_SEMANA',
    'FUMA_ACTUALMENTE', 'PRUEBA_RAPIDA_HBA1C',
    'FECHA_ULTIMA_MENSTRUACION',
    'METODO_PLANIFICACION', 'TIPO_METODO_PLANIFICACION', 'TIEMPO_USO_METODO',
    'TABAQUISMO_PAQUETE_ANO',
]

LABS = ['COLESTEROL TOTAL', 'COLESTEROL_HDL', 'COLESTEROL_LDL',
        'TRIGLICERIDOS', 'HEMOGLOBINA_GLICOSILADA']
DELTA_LABS = ['delta_COLESTEROL TOTAL', 'delta_COLESTEROL_HDL',
              'delta_COLESTEROL_LDL', 'delta_HEMOGLOBINA_GLICOSILADA',
              'delta_TRIGLICERIDOS']
AVICENA_FLAGS = [
    'hta', 'dm', 'biopsia', 'cancer',
    'ant_hta', 'ant_dm', 'ant_biopsia', 'ant_cancer', 'ant_mama',
    'cie_hta', 'cie_dm', 'cie_biopsia', 'cie_cancer', 'cie_mama',
    'remision_gine', 'remision_endo', 'n_consultas_T',
]
ANTROPOMETRIA = ['TENSION_ARTERIAL_SISTOLE', 'TENSION_ARTERIAL_DIASTOLE', 'PESO', 'TALLA']
HIGH_MISS_NUMERIC = ['PERIMETRO_ABDOMINAL', 'MENARQUIA', 'riesgo_alto']
BOOL_STR_COLS = ['FUMA_O_HA_FUMADO', 'ES_HA_SIDO_FUMADOR_PASIVO', 'COCINA_RECINTO_CERRADO']

# duplicados 100% idénticos y constantes (de Fase 3.5). imc_disponible NO se dropea aquí.
DROP_REDUNDANTE = ['ant_hta', 'ant_dm', 'ant_biopsia', 'ant_cancer']
DROP_CONSTANTE = ['ant_mama', 'cie_hta', 'cie_dm', 'cie_biopsia', 'cie_cancer', 'cie_mama']
DROP_PESO = ['PESO', 'PESO_disponible']

# continuos imputables (median en LR / NaN en árboles). PESO excluido (se dropea).
IMPUTE_VARS = ['COLESTEROL_TOTAL', 'COLESTEROL_HDL', 'COLESTEROL_LDL', 'TRIGLICERIDOS',
               'HEMOGLOBINA_GLICOSILADA', 'TENSION_ARTERIAL_SISTOLE',
               'TENSION_ARTERIAL_DIASTOLE', 'TALLA', 'PERIMETRO_ABDOMINAL',
               'MENARQUIA', 'riesgo_alto', 'IMC']

CLIP_BOUNDS = {
    'edad': (18, 110), 'IMC': (12, 70), 'PESO': (30, 250), 'TALLA': (120, 210),
    'PERIMETRO_ABDOMINAL': (40, 200), 'MENARQUIA': (8, 20),
    'TENSION_ARTERIAL_SISTOLE': (70, 250), 'TENSION_ARTERIAL_DIASTOLE': (40, 150),
    'COLESTEROL_TOTAL': (50, 500), 'COLESTEROL_HDL': (10, 150),
    'COLESTEROL_LDL': (10, 400), 'TRIGLICERIDOS': (20, 2000),
    'HEMOGLOBINA_GLICOSILADA': (3, 20),
}

YES_VALS = {'si', 'sí', 's', 'yes', 'y', '1', 'true', 'verdadero'}


def bool_str_to_int(series):
    return (series.str.strip().str.lower()
            .map(lambda x: 1 if x in YES_VALS else 0)
            .fillna(0).astype(int))


def limpia_sin_imputar(df_in):
    """Limpieza común SIN imputar continuos: deja NaN en IMPUTE_VARS, winsoriza, dropea.
    Devuelve el DF a nivel de features (pre-encoding)."""
    df = df_in.copy()

    # cohorte ≥18
    nac = pd.to_datetime(df['fec_nacim'], format='%d/%m/%Y', errors='coerce')
    df = df[(df['anio'] - nac.dt.year) >= 18].copy()

    # drops de fuga / basura
    df.drop(columns=[c for c in DROP_ALWAYS if c in df.columns], inplace=True)

    # edad, proxy
    nac = pd.to_datetime(df['fec_nacim'], format='%d/%m/%Y', errors='coerce')
    df['edad'] = (df['anio'] - nac.dt.year).clip(0, 110)
    df.drop(columns=['fec_nacim'], inplace=True)
    df['proxy_menopausia'] = (df['edad'] >= 50).astype(int)
    # tiene_avicena ELIMINADA (2026-06-17): = n_consultas_T.notna(), señal de vigilancia
    # redundante con n_consultas_T. Quitarla no movió métricas (AUC-PR 0.0380→0.0375,
    # recall@top10% 0.704→0.703). Ver 09. sin tiene_avicena.ipynb.

    # flags Avicena → 0 (ausencia = sin visita)
    for col in AVICENA_FLAGS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # labs → numérico + flag (SIN imputar)
    for col in LABS:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce')
        safe = col.replace(' ', '_')
        df[f'{safe}_disponible'] = df[col].notna().astype(int)
        if col != safe:
            df.rename(columns={col: safe}, inplace=True)

    # delta labs → flag + 0 (semántico en ambas variantes)
    for col in DELTA_LABS:
        if col not in df.columns:
            continue
        safe = col.replace(' ', '_')
        df[f'{safe}_disponible'] = df[col].notna().astype(int)
        df[col] = df[col].fillna(0)
        if col != safe:
            df.rename(columns={col: safe}, inplace=True)

    # antropometría → flag (SIN imputar)
    for col in ANTROPOMETRIA:
        if col in df.columns:
            df[f'{col}_disponible'] = df[col].notna().astype(int)

    # IMC a partir de PESO/TALLA crudos → NaN donde falte cualquiera
    df['IMC'] = np.where((df['TALLA'] > 0) & (df['PESO'] > 0),
                         df['PESO'] / (df['TALLA'] / 100) ** 2, np.nan)
    df['imc_disponible'] = df['IMC'].notna().astype(int)  # ya NO es constante

    # clínicas alta miss → numérico + flag (SIN imputar)
    for col in HIGH_MISS_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[f'{col}_disponible'] = df[col].notna().astype(int)

    # booleanos string → 0/1
    for col in BOOL_STR_COLS:
        if col in df.columns:
            df[col] = bool_str_to_int(df[col])

    # winsorización (clip preserva NaN)
    for col, (lo, hi) in CLIP_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    # drops Fase 3.5
    df.drop(columns=[c for c in DROP_REDUNDANTE + DROP_CONSTANTE + DROP_PESO
                     if c in df.columns], inplace=True)
    return df


def main():
    print("Leyendo bases crudas...")
    tr = pd.read_parquet(f"{BASES}/prediccion_mama_base entrenamiento train.parquet")
    vl = pd.read_parquet(f"{BASES}/prediccion_mama_base entrenamiento validacion.parquet")

    tr = limpia_sin_imputar(tr)
    vl = limpia_sin_imputar(vl)
    print(f"Limpio (NaN-nativo)  train {tr.shape} | val {vl.shape}")

    # n_consultas_T: cap p99.9 train (conteo sin máximo clínico)
    n_cap = float(tr['n_consultas_T'].quantile(0.999))
    tr['n_consultas_T'] = tr['n_consultas_T'].clip(0, n_cap)
    vl['n_consultas_T'] = vl['n_consultas_T'].clip(0, n_cap)

    # ZONA → moda train (ambas variantes)
    zona_moda = tr['ZONA'].mode()[0]
    tr['ZONA'] = tr['ZONA'].fillna(zona_moda)
    vl['ZONA'] = vl['ZONA'].fillna(zona_moda)

    # medianas (sobre train winsorizado, valores reales) para la variante imputada
    medians = {c: float(tr[c].median()) for c in IMPUTE_VARS if c in tr.columns}

    # encoding (params de train) — idéntico en ambas variantes.
    # Geografía SOLO a nivel regional (DESCRIP_REGIONAL): con tan pocos positivos,
    # desagregar a municipio (508 cat) añade ruido. Se eliminan cod_municip y anio
    # (anio no generaliza: train={2021,2022}, val={2023}, prod={2024}).
    def encode(df):
        df = df.copy()
        df.drop(columns=['cod_municip', 'anio'], inplace=True)
        return pd.get_dummies(df, columns=['ZONA', 'DESCRIP_REGIONAL'],
                              prefix=['ZONA', 'REG'], dummy_na=False)

    tr_enc = encode(tr)
    vl_enc = encode(vl)
    # alinear val a columnas de train
    for c in tr_enc.columns:
        if c not in vl_enc.columns:
            vl_enc[c] = 0
    vl_enc = vl_enc[tr_enc.columns]

    # ---- variante NATIVE (NaN) ----
    tr_native, vl_native = tr_enc.copy(), vl_enc.copy()

    # ---- variante IMPUTE (mediana) ----
    tr_imp, vl_imp = tr_enc.copy(), vl_enc.copy()
    for c, m in medians.items():
        tr_imp[c] = tr_imp[c].fillna(m)
        vl_imp[c] = vl_imp[c].fillna(m)

    # guardar
    out = {
        'native': (tr_native, vl_native),
        'impute': (tr_imp, vl_imp),
    }
    for tag, (t, v) in out.items():
        t.to_parquet(f"{BASES}/prediccion_mama_train_{tag}.parquet", index=False)
        v.to_parquet(f"{BASES}/prediccion_mama_val_{tag}.parquet", index=False)

    params = {
        'cohorte_edad_min': 18,
        'impute_vars': IMPUTE_VARS,
        'medians_train_ge18': medians,
        'n_consultas_cap': n_cap,
        'zona_moda': zona_moda,
        'clip_bounds': CLIP_BOUNDS,
        'log_vars': [],
        'geografia': 'solo DESCRIP_REGIONAL (REG_*); cod_municip y anio eliminados',
        'feature_columns': [c for c in tr_native.columns if c not in ('key', 'label')],
    }
    with open(f"{BASES}/variantes_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    # reporte
    feat = params['feature_columns']
    print("\n=== Variantes generadas ===")
    print(f"  columnas idénticas: {tr_native.shape[1]} ({len(feat)} features)")
    print(f"  cols iguales native==impute: {list(tr_native.columns) == list(tr_imp.columns)}")
    for tag, (t, v) in out.items():
        print(f"  {tag:7} train {t.shape} pos {int(t.label.sum())} | "
              f"val {v.shape} pos {int(v.label.sum())} | nulos train {int(t.isnull().sum().sum())}")
    print("\n  NaN por variable imputable (solo native debe tener NaN):")
    for c in IMPUTE_VARS:
        if c in tr_native.columns:
            print(f"    {c:<28} native {int(tr_native[c].isnull().sum()):>9,} | "
                  f"impute {int(tr_imp[c].isnull().sum()):>9,}")


if __name__ == "__main__":
    main()

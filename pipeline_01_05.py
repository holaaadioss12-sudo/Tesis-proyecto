# -*- coding: utf-8 -*-
"""TESIS"""

1.- PREPROCESAMIENTO

# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 01 — PREPROCESAMIENTO
=============================================================================
Entrada:  Dataframe - CR010.csv   (el crudo del PI System, ~5M registros)
Salida:   CR010_limpio.csv        (~1,18M registros, 14 variables)
"""

# --- Conectar al Drive (Colab) -----------------------------------------
from google.colab import drive
drive.mount('/content/drive')
RUTA_BASE = '/content/drive/MyDrive/tesis_chancadores'

import os as _os
EQUIPO = _os.environ.get('EQUIPO', 'CR010')
RUTA_EQ = RUTA_BASE if EQUIPO == 'CR010' else f'{RUTA_BASE}/{EQUIPO}'
_os.makedirs(RUTA_EQ, exist_ok=True)
print(f'Equipo: {EQUIPO} | carpeta de trabajo: {RUTA_EQ}')

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
np.random.seed(SEED)

MOSTRAR_GRAFICOS = True

ARCHIVO_ENTRADA = f'{RUTA_BASE}/Dataframe - {EQUIPO}.csv'
ARCHIVO_SALIDA = f'{RUTA_EQ}/{EQUIPO}_limpio.csv'


# =============================================================================
# 0. CARGA
# =============================================================================
print('=' * 70)
print('0. CARGANDO EL CRUDO')
print('=' * 70)

with open(ARCHIVO_ENTRADA, encoding='utf-8-sig') as _f:
    _cab = _f.readline()
_SEP = ';' if _cab.count(';') > _cab.count(',') else ','
print(f'Separador detectado: {_SEP!r}')
df = pd.read_csv(ARCHIVO_ENTRADA, sep=_SEP, parse_dates=['Time'], low_memory=False)
df = df.set_index('Time').sort_index()
print(f'Dataset crudo: {df.shape[0]:,} filas × {df.shape[1]} columnas')
print(f'Rango temporal: {df.index.min()}  ->  {df.index.max()}')

print('\nConvirtiendo texto a numérico (coma decimal -> punto)...')
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.replace(',', '.')
        df[col] = pd.to_numeric(df[col], errors='coerce')

resumen_nulos = pd.DataFrame({
    'n_nulos': df.isnull().sum(),
    'pct_nulos': (df.isnull().sum() / len(df)) * 100,
})
resumen_nulos = resumen_nulos[resumen_nulos['n_nulos'] > 0].sort_values('pct_nulos', ascending=False)
print('\n--- Nulos por columna (solo las que tienen) ---')
print(resumen_nulos.round(2) if not resumen_nulos.empty else 'Sin nulos.')

if MOSTRAR_GRAFICOS:
    df.hist(figsize=(16, 12), bins=60)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Matriz de correlación — dataset crudo')
    plt.show()


# =============================================================================
# 1. RENOMBRAR COLUMNAS A NOMBRES CANÓNICOS
# =============================================================================
print('\n' + '=' * 70)
print('1. RENOMBRANDO COLUMNAS')
print('=' * 70)

CANONICAL_LOOKUP = {
    ('AI',   None, None): 'CM',   # Corriente motor
    ('LIT',  '54', None): 'PI',   # Nivel / plano inclinado alimentación
    ('PDIT', '63', None): 'PDF',  # Presión diferencial filtro
    ('PIT',  '60', None): 'PEL',  # Presión entrada aceite lubricación
    ('RUN',  None, None): 'RUN',
    ('TIT',  '57', None): 'T7',   # Temp. buje excéntrico
    ('TIT',  '58', None): 'T8',   # Temp. socket liner
    ('TIT',  '59', None): 'T9',   # Temp. contraeje
    ('TIT',  '61', None): 'T1',   # Temp. aceite retorno
    ('TIT',  '62', None): 'T2',   # Temp. aceite entrada
    ('TIT',  '65', None): 'T5',   # Temp. tanque de aceite
    ('VIT',  '55', 'A'):  'V1',   # Vibración A
    ('VIT',  '55', 'B'):  'V2',   # Vibración B
    ('VIT',  '55', 'C'):  'V3',   # Vibración C
    ('VIT',  '55', 'D'):  'V4',   # Vibración D
}


def parsear_tag(col):
    tag = col.split('-')[-1].strip().split('_')[-1]
    m = re.match(r'^([A-Z]+)(\d*)([A-Z]?)$', tag)
    if not m:
        return None
    prefijo, digitos, sufijo = m.groups()
    sufijo = sufijo or None
    if digitos:
        digitos = str(int(digitos) % 50 + 50)
    return (prefijo, digitos[-2:] if digitos else None, sufijo)


def renombrar_columnas(df):
    nuevos, sin_match = {}, []
    for col in df.columns:
        if col.strip().lower() == 'time':
            nuevos[col] = 'timestamp'
            continue
        canonico = CANONICAL_LOOKUP.get(parsear_tag(col))
        if canonico:
            nuevos[col] = canonico
        else:
            sin_match.append(col)
    if sin_match:
        print('Columnas sin mapeo (quedan con su nombre original):', sin_match)
    return df.rename(columns=nuevos)


df = renombrar_columnas(df)
print('Columnas resultantes:', df.columns.tolist())


# =============================================================================
# 2. ELIMINAR COLUMNAS DISPERSAS (> 90 % de nulos)
# =============================================================================
print('\n' + '=' * 70)
print('2. ELIMINANDO COLUMNAS DISPERSAS')
print('=' * 70)


def eliminar_columnas_dispersas(df, umbral=0.9, columnas_step=None):
    df = df.copy()
    for col in (columnas_step or []):
        if col in df.columns:
            df[col] = df[col].ffill()

    prop_nulos = df.isna().mean().sort_values(ascending=False)
    print('Proporción de nulos por columna:')
    print(prop_nulos.round(4))

    a_eliminar = prop_nulos[prop_nulos > umbral].index.tolist()
    print(f'\nEliminadas (> {umbral * 100:.0f} % nulos): {a_eliminar}')
    return df.drop(columns=a_eliminar), a_eliminar


df, columnas_eliminadas = eliminar_columnas_dispersas(df, umbral=0.9)


# =============================================================================
# 3. FILTRAR LOS "TICS" DE VIBRACIÓN DEL ARRANQUE
# =============================================================================
print('\n' + '=' * 70)
print('3. FILTRANDO TICS DE ARRANQUE')
print('=' * 70)


def filtrar_tics_arranque(df, col_corriente='CM', cols_vibracion=None, umbral_arranque=25):
    df = df.copy()
    cols_vibracion = [c for c in (cols_vibracion or ['V1', 'V2', 'V3', 'V4']) if c in df.columns]
    if col_corriente not in df.columns or not cols_vibracion:
        print('No se pudo aplicar (faltan columnas).')
        return df

    corriente_anterior = df[col_corriente].shift(1)
    arranque = (corriente_anterior <= umbral_arranque) & (df[col_corriente] > umbral_arranque)

    n_anulados = 0
    for col in cols_vibracion:
        afectados = arranque & df[col].notna()
        n_anulados += int(afectados.sum())
        df.loc[afectados, col] = np.nan

    print(f'{int(arranque.sum())} transiciones de arranque detectadas | '
          f'{n_anulados} valores de vibración anulados (se interpolan después)')
    return df


df = filtrar_tics_arranque(df, col_corriente='CM',
                           cols_vibracion=['V1', 'V2', 'V3', 'V4'],
                           umbral_arranque=25)


# =============================================================================
# 4. LIMPIEZA ETAPA 1 — FILTRO DE CORRIENTE (motor apagado o arrancando)
# =============================================================================
print('\n' + '=' * 70)
print('4. FILTRO DE CORRIENTE (CM > 25)')
print('=' * 70)


def limpieza_etapa1(df, col_corriente='CM', umbral_corriente=25):
    df = df.copy()
    df = df.dropna(how='all')
    df = df[df[col_corriente] > umbral_corriente]
    df = df[df[col_corriente].notna()]
    return df


filas_antes = len(df)
df_limpio1 = limpieza_etapa1(df)
print(f'Filas antes: {filas_antes:,} | después: {len(df_limpio1):,} '
      f'({(1 - len(df_limpio1) / filas_antes) * 100:.1f} % descartado)')


# =============================================================================
# EL CORTE DE ENTRENAMIENTO SE ELIGE ACÁ, Y LOS FILTROS SE AJUSTAN
# SOLO CON TRAIN
# =============================================================================
import json as _json

FRAC_TRAIN, FRAC_VAL = 0.70, 0.15
ARCHIVO_CORTE = f'{RUTA_EQ}/{EQUIPO}_corte_split.json'
VARIABLES_PARA_CORTE = [c for c in
                        ['CM', 'PI', 'PDF', 'PEL', 'T7', 'T8', 'T9',
                         'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4']
                        if c in df_limpio1.columns]

def _corte_en_borde_de_tramo(d, frac_train, frac_val):
    """Devuelve (T_train, T_val) en bordes de tramo, por fraccion de MUESTRAS."""
    _sal = d.index.to_series().diff() > pd.Timedelta(seconds=10)
    _blo = _sal.cumsum()
    _t = d.groupby(_blo).size()
    _ac = _t.cumsum() / _t.sum()
    _btr = int(_ac[_ac >= frac_train].index[0])
    _bva = int(_ac[_ac >= frac_train + frac_val].index[0])
    return (d.index[(_blo <= _btr).to_numpy()].max(),
            d.index[(_blo <= _bva).to_numpy()].max())


_completas = df_limpio1[VARIABLES_PARA_CORTE].dropna().index
print(f'\nFilas de df_limpio1: {len(df_limpio1):,} | con los 14 sensores '
      f'completos: {len(_completas):,} ({100*len(_completas)/len(df_limpio1):.1f} %)')
if len(_completas) < 0.5 * len(df_limpio1):
    print('  OJO: mas de la mitad de las filas tiene algun sensor en NaN. El')
    print('  dropna() del paso 6 las va a borrar, y si estan concentradas en')
    print('  un periodo el corte se desplaza. Por eso se calcula sobre estas.')

T_TRAIN, T_VAL = _corte_en_borde_de_tramo(
    df_limpio1.loc[_completas], FRAC_TRAIN, FRAC_VAL)

es_train_01 = (df_limpio1.index.isin(_completas)
               & (df_limpio1.index <= T_TRAIN))

_n_tr01 = int(es_train_01.sum())
_n_va01 = int(((_completas > T_TRAIN) & (_completas <= T_VAL)).sum())
_n_te01 = int((_completas > T_VAL).sum())
print(f'\nCORTE DE ENTRENAMIENTO (por muestras, en borde de tramo):')
print(f'  T_train = {T_TRAIN}')
print(f'  T_val   = {T_VAL}')
print(f'  filas completas: train {_n_tr01:,} '
      f'({100*_n_tr01/len(_completas):.1f} %) | '
      f'val {_n_va01:,} ({100*_n_va01/len(_completas):.1f} %) | '
      f'test {_n_te01:,} ({100*_n_te01/len(_completas):.1f} %)')
print('  Los porcentajes no van a dar 70/15/15 exacto: los pasos 5 y 6 botan')
print('  más filas. Se reportan los efectivos al final.')

_json.dump({'T_train': str(T_TRAIN), 'T_val': str(T_VAL),
            'frac_train': FRAC_TRAIN, 'frac_val': FRAC_VAL,
            'ajustado_en': 'paso 01, sobre el limpio de corriente'},
           open(ARCHIVO_CORTE, 'w'), indent=2)
print(f'  Guardado: {ARCHIVO_CORTE}')


# =============================================================================
# 5. FILTRO DE ATÍPICOS 4-SIGMA, ANCLADO A LOS UMBRALES DE LAS REGLAS
# =============================================================================
print('\n' + '=' * 70)
print('5. FILTRO DE ATÍPICOS')
print('=' * 70)

UMBRAL_REGLA_MAX = {
    'CM': 110, 'PI': 60, 'PDF': 400, 'PEL': 450,
    'T7': 116,
    'T8': 116,
    'T9': 60, 'T1': 60, 'T2': 50, 'T5': 50,
    'V1': 40, 'V2': 40, 'V3': 40, 'V4': 40,
}
MULTIPLICADOR_ANCLA = 2

print('Comprobación — tope 4σ vs. umbral de la regla más exigente:')
for col, umbral_regla in UMBRAL_REGLA_MAX.items():
    if col not in df_limpio1.columns:
        continue
    media, std = df_limpio1[col].mean(), df_limpio1[col].std()
    tope_4s = media + 4 * std
    estado = 'FILTRO MÁS ESTRICTO QUE LA REGLA -> se ancla' if tope_4s < umbral_regla else 'OK'
    print(f'  {col:4s}: tope 4σ={tope_4s:8.1f} | umbral regla={umbral_regla:5d} | '
          f'máx real={df_limpio1[col].max():8.1f} | {estado}')


def filtrar_atipicos(df, variables, n_std=4, umbrales_regla=None, multiplicador=2,
                     mask_train=None):
    """Media y sigma se ajustan SOLO con train."""
    df = df.copy()
    umbrales_regla = umbrales_regla or {}
    base = df if mask_train is None else df.loc[mask_train]
    if mask_train is not None:
        _vars_ok = [c for c in variables if c in df.columns]
        _n_comp = int(df[_vars_ok].notna().all(axis=1).sum())
        print(f'Estadísticos del filtro ajustados sobre {len(base):,} filas de '
              f'train ({100*len(base)/max(_n_comp, 1):.1f} % de las '
              f'{_n_comp:,} filas completas; '
              f'{100*len(base)/len(df):.1f} % del registro).')
    resumen = []
    for col in variables:
        if col not in df.columns:
            continue
        media, std = base[col].mean(), base[col].std()
        lim_inf = media - n_std * std
        lim_sup_estadistico = media + n_std * std

        if col in umbrales_regla:
            lim_sup = max(lim_sup_estadistico, multiplicador * umbrales_regla[col])
        else:
            lim_sup = lim_sup_estadistico

        atipicos = (df[col] < lim_inf) | (df[col] > lim_sup)
        resumen.append((col, int(atipicos.sum()), atipicos.mean() * 100, round(lim_sup, 1)))
        df.loc[atipicos, col] = np.nan

    print('\n' + pd.DataFrame(
        resumen, columns=['variable', 'n_atipicos', 'pct_atipicos', 'limite_superior']
    ).to_string(index=False))
    return df


VARIABLES_A_FILTRAR = ['CM', 'PI', 'PDF', 'PEL', 'T7', 'T8', 'T9',
                       'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4']

_pre_filtro_p5 = df_limpio1[[c for c in VARIABLES_A_FILTRAR
                             if c in df_limpio1.columns]].copy()

df_limpio1 = filtrar_atipicos(df_limpio1, VARIABLES_A_FILTRAR, n_std=4,
                              umbrales_regla=UMBRAL_REGLA_MAX,
                              multiplicador=MULTIPLICADOR_ANCLA,
                              mask_train=es_train_01)


# =============================================================================
# 6. LIMPIEZA ETAPA 2 — NEGATIVOS E INTERPOLACIÓN
# =============================================================================
print('\n' + '=' * 70)
print('6. NEGATIVOS E INTERPOLACIÓN')
print('=' * 70)


def duracion_huecos(serie):
    es_nulo = serie.isna()
    grupo = (es_nulo != es_nulo.shift()).cumsum()
    rachas = es_nulo.groupby(grupo).sum()
    return rachas[rachas > 0]


print('Duración de los huecos antes de interpolar:')
for col in ['PI', 'V1', 'V2', 'V3', 'V4', 'PEL']:
    if col not in df_limpio1.columns:
        continue
    rachas = duracion_huecos(df_limpio1[col])
    if len(rachas) == 0:
        print(f'  {col}: sin huecos')
        continue
    print(f'  {col:4s}: mediana={rachas.median():.0f} pasos (~{rachas.median() * 10:.0f}s) | '
          f'p90={rachas.quantile(0.9):.0f} pasos (~{rachas.quantile(0.9) * 10 / 60:.1f} min) | '
          f'máx={rachas.max():.0f} pasos (~{rachas.max() * 10 / 60:.1f} min)')


def limpieza_etapa2(df, vars_no_negativas=None, vars_interpolables=None, limites_interpolacion=None):
    df = df.copy()
    vars_no_negativas = vars_no_negativas or []
    vars_interpolables = vars_interpolables or []
    limites_interpolacion = limites_interpolacion or {}

    for col in vars_no_negativas:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    for col in vars_interpolables:
        if col in df.columns:
            limite = limites_interpolacion.get(col, 5)
            df[col] = df[col].interpolate(method='linear', limit=limite)

    return df.dropna()


VARS_NO_NEGATIVAS = ['PI', 'T9', 'T5', 'V1', 'V2', 'V3', 'V4', 'PEL', 'PDF']
VARS_INTERPOLABLES = ['T7', 'T8', 'T9', 'T1', 'T2', 'T5', 'PDF', 'PEL', 'PI', 'V1', 'V2', 'V3', 'V4']
LIMITES_INTERPOLACION = {'PI': 30}

df_limpio = limpieza_etapa2(df_limpio1, VARS_NO_NEGATIVAS, VARS_INTERPOLABLES, LIMITES_INTERPOLACION)
print(f'\nFilas después de la etapa 2: {len(df_limpio):,}')


# =============================================================================
# DIAGNÓSTICO DE A3: LA REGLA NO SE PUEDE ACTIVAR NUNCA
# =============================================================================
print('\n' + '=' * 70)
print('A3 — ¿POR QUÉ NO SE ACTIVA NUNCA?')
print('=' * 70)
_pi, _cm = df_limpio['PI'], df_limpio['CM']
print(f'PI: mínimo {_pi.min():.6f} | mediana {_pi.median():.2f} | '
      f'negativos {int((_pi < 0).sum()):,} | exactamente 0: {int((_pi == 0).sum()):,}')
print(f'\n{"tolerancia":>12}{"PI <= tol":>14}{"y además CM > 35":>20}{"% del limpio":>14}')
for _t in (0.0, 1e-6, 0.01, 0.05, 0.1, 0.5, 1.0):
    _n1 = int((_pi <= _t).sum())
    _n2 = int(((_pi <= _t) & (_cm > 35)).sum())
    print(f'{_t:>12g}{_n1:>14,}{_n2:>20,}{100*_n2/len(df_limpio):>13.2f}%')

_bajo = _pi[_pi <= 0.1]
if len(_bajo):
    print(f'\nDistribución de PI dentro de (0, 0,1]:  n = {len(_bajo):,}')
    print(f'  mediana {_bajo.median():.5f} | p99 {_bajo.quantile(0.99):.5f} | '
          f'máx {_bajo.max():.5f}')
    print('\nCÓMO LEERLO. Si hay DOS poblaciones separadas -- una pegada al piso')
    print('del sensor y el resto muy por encima -- cualquier corte entre medio')
    print('las separa igual de bien, y eso se declara en la memoria con esta')
    print('tabla al lado. Si están repartidas de forma continua, no hay corte')
    print('limpio y conviene decirlo como limitación en vez de elegir uno.')
print('\nLa constante vive en reglas_engine.py: TOL_CERO (hoy 0.0 = PI == 0).')


# =============================================================================
# 7. GUARDAR
# =============================================================================
print('\n' + '=' * 70)
print('7. RESULTADO FINAL')
print('=' * 70)

dias = (df_limpio.index.max() - df_limpio.index.min()).days
horas_reales = len(df_limpio) * 10 / 3600

print(f'Filas:            {len(df_limpio):,}   <- esperado ~1.300.000')
print(f'Columnas:         {df_limpio.shape[1]}  ->  {df_limpio.columns.tolist()}')
print(f'Rango temporal:   {df_limpio.index.min()}  ->  {df_limpio.index.max()}')
print(f'Calendario:       {dias} días')
print(f'Operación real:   {horas_reales / 24:.1f} días equivalentes de datos continuos')
print(f'Nulos restantes:  {df_limpio.isna().sum().sum()}  (debe ser 0)')

df_limpio.index.name = 'timestamp'
df_limpio.to_csv(ARCHIVO_SALIDA)
print(f'\nGuardado en: {ARCHIVO_SALIDA}')


# =============================================================================
# CORTE DEFINITIVO -- recalculado sobre el limpio FINAL
# =============================================================================
print('\n' + '=' * 70)
print('CORTE DEFINITIVO (sobre el limpio final)')
print('=' * 70)

T_TRAIN_PROV, T_VAL_PROV = T_TRAIN, T_VAL
T_TRAIN, T_VAL = _corte_en_borde_de_tramo(df_limpio, FRAC_TRAIN, FRAC_VAL)

_m_tr = (df_limpio.index <= T_TRAIN)
_m_va = (df_limpio.index > T_TRAIN) & (df_limpio.index <= T_VAL)
_m_te = df_limpio.index > T_VAL
_f_tr, _f_va, _f_te = _m_tr.mean(), _m_va.mean(), _m_te.mean()

print(f'  provisional (para el filtro): T_train = {T_TRAIN_PROV}')
print(f'  DEFINITIVO                  : T_train = {T_TRAIN}')
print(f'                                T_val   = {T_VAL}')
print(f'\n  train {int(_m_tr.sum()):>9,} ({100*_f_tr:5.1f} %)  <- objetivo '
      f'{100*FRAC_TRAIN:.0f} %')
print(f'  val   {int(_m_va.sum()):>9,} ({100*_f_va:5.1f} %)  <- objetivo '
      f'{100*FRAC_VAL:.0f} %')
print(f'  test  {int(_m_te.sum()):>9,} ({100*_f_te:5.1f} %)')

_desvio = abs(_f_tr - FRAC_TRAIN)
if _desvio > 0.05:
    print('\n' + '!' * 70)
    print(f'EL CORTE NO CAE DONDE DEBE: train quedo en {100*_f_tr:.1f} % y el')
    print(f'objetivo es {100*FRAC_TRAIN:.0f} %. Con una desviacion asi, todo lo')
    print('que venga despues se entrena sobre una fraccion equivocada del')
    print('registro y los resultados no significan nada.')
    print('Revisa la distribucion de filas en el tiempo antes de seguir.')
    print('!' * 70)
    raise SystemExit('Corte fuera de rango, ver arriba.')
print('\n  Corte verificado: la desviacion es de '
      f'{100*_desvio:.1f} puntos, dentro de lo que los bordes de tramo permiten.')

_f_prov = float((df_limpio.index <= T_TRAIN_PROV).mean())
print(f'  El filtro 4 sigma se ajusto hasta el T_train provisional, que en el')
print(f'  limpio final equivale al {100*_f_prov:.1f} % (definitivo: '
      f'{100*_f_tr:.1f} %).')
if T_TRAIN_PROV > T_TRAIN:
    print(f'  OJO: el provisional quedo DESPUES del definitivo. El filtro vio '
          f'{100*(_f_prov - _f_tr):.1f} puntos')
    print('  del periodo de validacion. Es chico, pero va declarado en la memoria.')

_cols_p5 = list(_pre_filtro_p5.columns)
_mask_def_p5 = (_pre_filtro_p5.index.isin(_completas)
                & (_pre_filtro_p5.index <= T_TRAIN))


def _limites_p5(mask):
    _b = _pre_filtro_p5.loc[mask]
    out = {}
    for c in _cols_p5:
        m, s = _b[c].mean(), _b[c].std()
        sup = m + 4 * s
        if c in UMBRAL_REGLA_MAX:
            sup = max(sup, MULTIPLICADOR_ANCLA * UMBRAL_REGLA_MAX[c])
        out[c] = (m - 4 * s, sup)
    return out


_lp, _ld = _limites_p5(es_train_01), _limites_p5(_mask_def_p5)
_filas_p5 = []
for c in _cols_p5:
    x = _pre_filtro_p5[c]
    at_p = (x < _lp[c][0]) | (x > _lp[c][1])
    at_d = (x < _ld[c][0]) | (x > _ld[c][1])
    _rango = max(_lp[c][1] - _lp[c][0], 1e-12)
    _filas_p5.append({
        'variable': c,
        'lim_inf_prov': round(_lp[c][0], 2), 'lim_inf_def': round(_ld[c][0], 2),
        'lim_sup_prov': round(_lp[c][1], 2), 'lim_sup_def': round(_ld[c][1], 2),
        'cambio_max_pct_rango': round(100 * max(abs(_lp[c][0] - _ld[c][0]),
                                                abs(_lp[c][1] - _ld[c][1]))
                                      / _rango, 2),
        'filas_que_cambian': int((at_p != at_d).sum())})
_tabla_p5 = pd.DataFrame(_filas_p5)
print('\nP5 -- limites del filtro 4 sigma: corte provisional vs definitivo')
print(_tabla_p5.to_string(index=False))
_p5_max = float(_tabla_p5.cambio_max_pct_rango.max())
_p5_filas = int(_tabla_p5.filas_que_cambian.sum())
print(f'  Cambio maximo de un limite: {_p5_max:.2f} % del rango del filtro | '
      f'filas que cambiarian de clasificacion: {_p5_filas:,} de '
      f'{len(_pre_filtro_p5):,} ({100*_p5_filas/max(len(_pre_filtro_p5),1):.3f} %)')
del _pre_filtro_p5

_json.dump({'T_train': str(T_TRAIN), 'T_val': str(T_VAL),
            'frac_train': FRAC_TRAIN, 'frac_val': FRAC_VAL,
            'frac_train_efectiva': round(float(_f_tr), 4),
            'frac_val_efectiva': round(float(_f_va), 4),
            'frac_test_efectiva': round(float(_f_te), 4),
            'muestras_train': int(_m_tr.sum()),
            'muestras_val': int(_m_va.sum()),
            'muestras_test': int(_m_te.sum()),
            'T_train_provisional_filtro': str(T_TRAIN_PROV),
            'frac_final_hasta_provisional': round(_f_prov, 4),
            'p5_cambio_max_limite_pct': round(_p5_max, 2),
            'p5_filas_que_cambian': _p5_filas,
            'ajustado_en': 'paso 01, sobre el limpio definitivo'},
           open(ARCHIVO_CORTE, 'w'), indent=2)
print(f'  Guardado: {ARCHIVO_CORTE}')

print('\n--- Primeras filas ---')
print(df_limpio.head())

print('\n--- Estadísticos ---')
print(df_limpio.describe().round(2).to_string())

print('\n' + '=' * 70)
print('LISTO. Siguiente paso: 02_reglas_y_ventanas.py')
print('=' * 70)

"""2.- REGLAS Y VENTANAS"""

# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 02 — REGLAS, VENTANAS Y ETIQUETADO   (W = 60, o sea 10 minutos)
=============================================================================
Entrada : CR010_limpio.csv                 (paso 01)
"""

from google.colab import drive
drive.mount('/content/drive')
RUTA_BASE = '/content/drive/MyDrive/tesis_chancadores'

import os as _os
EQUIPO = _os.environ.get('EQUIPO', 'CR010')
RUTA_EQ = RUTA_BASE if EQUIPO == 'CR010' else f'{RUTA_BASE}/{EQUIPO}'
if EQUIPO == 'CR010' and _os.path.isdir(f'{RUTA_BASE}/CR010'):
    RUTA_EQ = f'{RUTA_BASE}/CR010'
_os.makedirs(RUTA_EQ, exist_ok=True)
print(f'Equipo: {EQUIPO} | carpeta de trabajo: {RUTA_EQ}')

import sys
sys.path.append(RUTA_BASE)

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from reglas_engine import evaluar_reglas

SEED = 42
np.random.seed(SEED)

W = 60
PASO_S = 10

ARCHIVO_LIMPIO = f'{RUTA_EQ}/{EQUIPO}_limpio.csv'
ARCHIVO_VENTANAS = f'{RUTA_EQ}/{EQUIPO}_ventanas_W{W}.csv'
ARCHIVO_FEATURES = f'{RUTA_EQ}/{EQUIPO}_features_W{W}.csv'
ARCHIVO_EVENTOS_REGLA = f'{RUTA_EQ}/{EQUIPO}_eventos_por_regla.csv'
ARCHIVO_WASSERSTEIN = f'{RUTA_EQ}/{EQUIPO}_wasserstein_bandas_W{W}.csv'

VARIABLES = ['CM', 'PI', 'PDF', 'PEL', 'T7', 'T8', 'T9',
             'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4']


# =============================================================================
# 0. CARGA
# =============================================================================
print('=' * 70)
print(f'PASO 02 — VENTANAS DE {W * 10 / 60:.0f} MINUTOS (W = {W})')
print('=' * 70)

def cargar_limpio(ruta):
    """Acepta el CSV con la columna de tiempo llamada 'timestamp' o 'Time'."""
    cabecera = pd.read_csv(ruta, nrows=0)
    col_tiempo = 'timestamp' if 'timestamp' in cabecera.columns else 'Time'
    df = pd.read_csv(ruta, parse_dates=[col_tiempo], low_memory=False)
    return df.set_index(col_tiempo).sort_index().rename_axis('timestamp')


df = cargar_limpio(ARCHIVO_LIMPIO)
print(f'Dataset limpio: {df.shape[0]:,} filas × {df.shape[1]} columnas')

VARIABLES = [v for v in VARIABLES if v in df.columns]
print(f'Variables disponibles ({len(VARIABLES)}): {VARIABLES}')


# =============================================================================
# 1. SEGMENTAR EN TRAMOS CONTINUOS
# =============================================================================
print('\n' + '=' * 70)
print('1. SEGMENTANDO EN TRAMOS CONTINUOS')
print('=' * 70)


def segmentar_tramos_continuos(df, freq='10s', max_gap=1):
    idx = df.index.to_series()
    gaps = idx.diff() > pd.Timedelta(freq) * max_gap
    tramo_id = gaps.cumsum()
    df = df.copy()
    df['tramo_id'] = tramo_id
    return [g.drop(columns='tramo_id') for _, g in df.groupby('tramo_id')]


tramos = segmentar_tramos_continuos(df)
print(f'Tramos encontrados: {len(tramos):,}')

tramos = [t for t in tramos if len(t) >= W]
print(f'Tramos útiles (>= {W} registros): {len(tramos):,}   <- esperado: 1.929')


# =============================================================================
# UN SOLO CORTE DE ENTRENAMIENTO PARA TODO EL PIPELINE
# =============================================================================
import json as _json

ARCHIVO_CORTE = f'{RUTA_EQ}/{EQUIPO}_corte_split.json'

_corte01 = _json.load(open(ARCHIVO_CORTE))
T_TRAIN = pd.Timestamp(_corte01['T_train'])
T_VAL = pd.Timestamp(_corte01['T_val'])
FRAC_TRAIN = _corte01.get('frac_train', 0.70)
FRAC_VAL = _corte01.get('frac_val', 0.15)

_largos = np.array([len(t) for t in tramos])
_ini_tr = np.array([t.index[0] for t in tramos])
_i_tr = int(np.flatnonzero(_ini_tr <= T_TRAIN)[-1])
_i_va = int(np.flatnonzero(_ini_tr <= T_VAL)[-1])

_n_tr = int(_largos[:_i_tr + 1].sum())
_n_va = int(_largos[_i_tr + 1:_i_va + 1].sum())
_n_te = int(_largos[_i_va + 1:].sum())
_n = int(_largos.sum())
print(f'\nCORTE UNICO (por muestras, en borde de tramo):')
print(f'  T_train = {T_TRAIN}   (fin del tramo {_i_tr})')
print(f'  T_val   = {T_VAL}   (fin del tramo {_i_va})')
print(f'  muestras: train {_n_tr:,} ({100*_n_tr/_n:.1f} %) | '
      f'val {_n_va:,} ({100*_n_va/_n:.1f} %) | '
      f'test {_n_te:,} ({100*_n_te/_n:.1f} %)')
print(f'  tramos:   train {_i_tr+1:,} | val {_i_va-_i_tr:,} | '
      f'test {len(tramos)-_i_va-1:,}')
print('  Si los porcentajes de MUESTRAS dan cerca de 70/15/15 pero los de')
print('  TRAMOS no, eso es exactamente lo que el corte viejo confundia.')

_corte01.update({'muestras_train': _n_tr, 'muestras_val': _n_va,
                 'muestras_test': _n_te,
                 'frac_train_efectiva': round(_n_tr / _n, 4),
                 'frac_val_efectiva': round(_n_va / _n, 4)})
_json.dump(_corte01, open(ARCHIVO_CORTE, 'w'), indent=2)
print(f'  Cifras efectivas agregadas a: {ARCHIVO_CORTE}')


def _inicio_de(t):
    """Primer timestamp de un tramo, venga suelto o como (id, tramo)."""
    return (t[1] if isinstance(t, tuple) else t).index[0]


def split_por_fecha(lista_tramos, t_train=None, t_val=None):
    """Parte una lista de tramos por los DOS instantes del corte unico."""
    t_train = T_TRAIN if t_train is None else t_train
    t_val = T_VAL if t_val is None else t_val
    tr = [t for t in lista_tramos if _inicio_de(t) <= t_train]
    va = [t for t in lista_tramos if t_train < _inicio_de(t) <= t_val]
    te = [t for t in lista_tramos if _inicio_de(t) > t_val]
    return tr, va, te


def split_df_por_fecha(d, col='timestamp_inicio', t_train=None, t_val=None):
    """Lo mismo para un DataFrame de ventanas, por su columna de tiempo."""
    t_train = T_TRAIN if t_train is None else t_train
    t_val = T_VAL if t_val is None else t_val
    ts = pd.to_datetime(d[col])
    return (d[ts <= t_train], d[(ts > t_train) & (ts <= t_val)], d[ts > t_val])


# =============================================================================
# 2. EVALUAR LAS REGLAS SOBRE CADA TRAMO
# =============================================================================
print('\n' + '=' * 70)
print('2. EVALUANDO REGLAS')
print('=' * 70)

tramos_con_reglas = []
ids_reglas = None
reglas_omitidas = None

for t in tramos:
    resultado = evaluar_reglas(t, verbose=False)
    if ids_reglas is None:
        ids_reglas = [c for c in resultado.columns
                      if c not in ('n_reglas_activadas', 'estado')]
        reglas_omitidas = resultado.attrs.get('reglas_omitidas', [])
    tramos_con_reglas.append(t.join(resultado))

print(f'Reglas evaluadas: {len(ids_reglas)}   <- esperado: 31')
print(f'Orden de las reglas: {ids_reglas}')
if reglas_omitidas:
    print(f'Reglas omitidas por falta de sensor: {reglas_omitidas}')
    print('  (A23 necesita el tag ZI266 / columna Conf, que no está en el dataset)')


print('\n' + '=' * 70)
print('2b. CONVIRTIENDO MUESTRAS A EVENTOS')
print('=' * 70)

registros_eventos = []
muestras_totales = 0

for tramo_idx, t in enumerate(tramos_con_reglas):
    n_ventanas_tramo = len(t) // W          # ventanas que sí se construyen
    limite = n_ventanas_tramo * W           # lo que sobra del tramo se ignora
    for r in ids_reglas:
        s = t[r].fillna(False).to_numpy().astype(bool)
        muestras_totales += int(s.sum())
        if not s.any():
            continue
        si = s.astype(np.int8)
        subidas = np.flatnonzero(np.diff(np.concatenate(([0], si))) == 1)
        bajadas = np.flatnonzero(np.diff(np.concatenate((si, [0]))) == -1)
        for ini, fin in zip(subidas, bajadas):
            if ini >= limite:               # evento en la cola descartada
                continue
            registros_eventos.append({
                'tramo_id': tramo_idx,
                'ventana_id': int(ini // W),   # la ventana donde ARRANCA
                'ventana_fin': int(min(fin, limite - 1) // W),  # donde TERMINA
                'regla': r,
                'paso_inicio': int(ini),
                'muestras': int(fin - ini + 1),
                'duracion_s': int((fin - ini + 1) * PASO_S),
                'borde_tramo': bool(ini == 0),
                'paso_en_ventana': int(ini % W),
            })

eventos = pd.DataFrame(registros_eventos)
print(f'Muestras en estado de alarma (lo que se contaba antes): {muestras_totales:,}')
print(f'Eventos de alarma (flancos de subida)                 : {len(eventos):,}')
if len(eventos):
    print(f'Razón muestras/eventos: {muestras_totales / len(eventos):.1f}')
    print(f'Eventos que arrancan en el borde de un tramo: '
          f'{int(eventos.borde_tramo.sum()):,} '
          f'({eventos.borde_tramo.mean() * 100:.1f} %)')

resumen_eventos = (eventos.groupby('regla')
                   .agg(eventos=('muestras', 'size'),
                        muestras_totales=('muestras', 'sum'),
                        dur_mediana_s=('duracion_s', 'median'),
                        dur_p95_s=('duracion_s', lambda x: x.quantile(0.95)),
                        dur_max_s=('duracion_s', 'max'))
                   .sort_values('muestras_totales', ascending=False))
resumen_eventos['muestras_por_evento'] = (
    resumen_eventos.muestras_totales / resumen_eventos.eventos).round(1)
resumen_eventos['pct_de_la_ventana'] = (
    100 * resumen_eventos.muestras_por_evento / W).round(1)

print('\n--- Muestras vs. eventos por regla ---')
print(resumen_eventos.to_string())

print('\n--- Distribución de la duración de los episodios (segundos) ---')
print(eventos.duracion_s.describe(
    percentiles=[.25, .5, .75, .9, .95, .99]).round(1).to_string())
print('\nMirá estos percentiles ANTES de fijar los umbrales de abajo: los '
      'valores por defecto son una elección mía, no de la literatura.')

UMBRAL_ESTADO_S = 180
UMBRAL_CHATTER_S = 60    # 1 min: episodio típico más corto -> transitoria

resumen_eventos['tipo'] = np.select(
    [resumen_eventos.dur_mediana_s >= UMBRAL_ESTADO_S,
     resumen_eventos.dur_mediana_s <= UMBRAL_CHATTER_S],
    ['estado', 'transitoria'],
    default='intermedia')

print(f'\n--- Poblaciones (umbrales: estado >= {UMBRAL_ESTADO_S} s, '
      f'transitoria <= {UMBRAL_CHATTER_S} s) ---')
print(resumen_eventos[['eventos', 'dur_mediana_s', 'muestras_por_evento',
                       'pct_de_la_ventana', 'tipo']].to_string())
print('\nConteo por tipo:')
print(resumen_eventos.tipo.value_counts().to_string())

resumen_eventos.to_csv(ARCHIVO_EVENTOS_REGLA, sep=';', decimal=',')
print(f'\nGuardado: {ARCHIVO_EVENTOS_REGLA}')

conteo_iniciados = (eventos.groupby(['tramo_id', 'ventana_id', 'regla'])
                    .size().unstack(fill_value=0)
                    .reindex(columns=ids_reglas, fill_value=0))

filas_activos = []
for tramo_id, v0, v1, regla in eventos[['tramo_id', 'ventana_id',
                                        'ventana_fin', 'regla']].itertuples(index=False):
    for v in range(v0, v1 + 1):
        filas_activos.append((tramo_id, v, regla))

conteo_activos = (pd.DataFrame(filas_activos,
                               columns=['tramo_id', 'ventana_id', 'regla'])
                  .groupby(['tramo_id', 'ventana_id', 'regla']).size()
                  .unstack(fill_value=0)
                  .reindex(columns=ids_reglas, fill_value=0))

print(f'\nVentanas donde ARRANCA al menos un evento : {conteo_iniciados.shape[0]:,}')
print(f'Ventanas CUBIERTAS por al menos un evento : {conteo_activos.shape[0]:,}')
print(f'  diferencia (ventanas en medio de un episodio largo): '
      f'{conteo_activos.shape[0] - conteo_iniciados.shape[0]:,}')
print(f'Episodios que abarcan más de una ventana  : '
      f'{int((eventos.ventana_fin > eventos.ventana_id).sum()):,} '
      f'({(eventos.ventana_fin > eventos.ventana_id).mean() * 100:.1f} %)')


# =============================================================================
# 3. CONTAR ACTIVACIONES POR VENTANA
# =============================================================================
print('\n' + '=' * 70)
print('3. CONSTRUYENDO VENTANAS')
print('=' * 70)


def construir_conteo_por_ventana(tramos_con_reglas, reglas_ids, W):
    resumenes = []
    for tramo_idx, t in enumerate(tramos_con_reglas):
        for v in range(len(t) // W):
            ventana = t.iloc[v * W:(v + 1) * W]
            conteo = ventana[reglas_ids].sum()
            fila = {
                'tramo_id': tramo_idx,
                'ventana_id': v,
                'timestamp_inicio': ventana.index[0],
                'timestamp_fin': ventana.index[-1],
            }
            fila.update(conteo.to_dict())
            fila['total_alarmas_ventana'] = int(conteo.sum())
            resumenes.append(fila)
    return pd.DataFrame(resumenes)


ventanas = construir_conteo_por_ventana(tramos_con_reglas, ids_reglas, W)
print(f'Ventanas construidas: {len(ventanas):,}   <- esperado: 19.615')


# =============================================================================
# 4. ETIQUETAR CADA VENTANA
# =============================================================================
print('\n' + '=' * 70)
print('4. ETIQUETANDO')
print('=' * 70)

CRITICIDAD_REGLAS = {}


def etiquetar_ventana(ventanas, reglas_ids, criterio='mayor_frecuencia', criticidad=None):
    df_out = ventanas.copy()
    criticidad = criticidad or CRITICIDAD_REGLAS

    def elegir(fila):
        conteos = fila[reglas_ids]
        if conteos.sum() == 0:
            return 'normal'
        maximo = conteos.max()
        candidatas = [r for r in reglas_ids if conteos[r] == maximo]
        if criterio == 'criticidad' and criticidad:
            con_crit = [c for c in candidatas if c in criticidad]
            if con_crit:
                return max(con_crit, key=lambda c: criticidad[c])
        return candidatas[0]

    df_out['etiqueta_ventana'] = df_out.apply(elegir, axis=1)
    return df_out


ventanas = etiquetar_ventana(ventanas, ids_reglas)

n_normal = int((ventanas.etiqueta_ventana == 'normal').sum())
n_anomala = len(ventanas) - n_normal
print(f'Normales: {n_normal:,} ({n_normal / len(ventanas) * 100:.1f} %)   <- esperado: 6.323')
print(f'Anómalas: {n_anomala:,} ({n_anomala / len(ventanas) * 100:.1f} %)   <- esperado: 13.292')
print('\nDistribución de etiquetas (top 10):')
print(ventanas.etiqueta_ventana.value_counts().head(10).to_string())

# --- Diagnóstico de la MODA ----------------------------------------------
print('\n--- Diagnóstico de la moda ---')
M_todas = ventanas[ids_reglas].to_numpy(float)
mask_anom = M_todas.sum(axis=1) > 0
M_anom = M_todas[mask_anom]

max_por_ventana = M_anom.max(axis=1)
n_con_el_maximo = (M_anom == max_por_ventana[:, None]).sum(axis=1)
dominancia = max_por_ventana / M_anom.sum(axis=1)

ventanas['moda_empate'] = False
ventanas['moda_dominancia'] = 0.0
ventanas.loc[mask_anom, 'moda_empate'] = n_con_el_maximo > 1
ventanas.loc[mask_anom, 'moda_dominancia'] = dominancia

print(f'  Moda única : {(n_con_el_maximo == 1).sum():,} ({(n_con_el_maximo == 1).mean() * 100:.1f} %)')
print(f'  Con EMPATE : {(n_con_el_maximo > 1).sum():,} ({(n_con_el_maximo > 1).mean() * 100:.1f} %)'
      '  <- el desempate lo decide el orden de columnas')
print(f'  Dominancia mediana de la moda: {np.median(dominancia) * 100:.1f} % de las alarmas')
print(f'  Ventanas donde la moda es < 50 % de las alarmas: '
      f'{(dominancia < 0.5).sum():,} ({(dominancia < 0.5).mean() * 100:.1f} %)')


print('\n' + '=' * 70)
print('4b. ETIQUETA POR TF·IDF SOBRE EVENTOS')
print('=' * 70)

# alinear las dos matrices de eventos con las ventanas del paso 3
idx_ventanas = pd.MultiIndex.from_frame(ventanas[['tramo_id', 'ventana_id']])
iniciados = conteo_iniciados.reindex(idx_ventanas, fill_value=0)
activos = conteo_activos.reindex(idx_ventanas, fill_value=0)
iniciados.index = ventanas.index
activos.index = ventanas.index

reglas_activas = [r for r in ids_reglas if activos[r].sum() > 0]
nunca = sorted(set(ids_reglas) - set(reglas_activas))
print(f'Reglas que se activan alguna vez: {len(reglas_activas)} de {len(ids_reglas)}')
if nunca:
    print(f'Reglas que NUNCA se activan ({len(nunca)}): {nunca}')
    print('  -> el espacio de características real tiene '
          f'{len(reglas_activas)} dimensiones, no {len(ids_reglas)}.')

N = len(ventanas)
idf = {r: float(np.log(N / (activos[r] > 0).sum())) for r in reglas_activas}
vec_idf = np.array([idf[r] for r in reglas_activas])[None, :]

print('\n--- Peso IDF por regla (mayor = más específica) ---')
print(pd.DataFrame({'ventanas_presente': [int((activos[r] > 0).sum()) for r in reglas_activas],
                    'pct_ventanas': [round(100 * (activos[r] > 0).mean(), 2) for r in reglas_activas],
                    'IDF': [round(idf[r], 3) for r in reglas_activas]},
                   index=reglas_activas).sort_values('IDF').to_string())


def tfidf(matriz):
    """TF x IDF fila por fila."""
    M = matriz[reglas_activas].to_numpy(float)
    tot = M.sum(axis=1, keepdims=True)
    TF = np.divide(M, tot, out=np.zeros_like(M), where=tot > 0)
    return TF * vec_idf, tot.ravel()


W_ini, tot_ini = tfidf(iniciados)
W_act, tot_act = tfidf(activos)

ventanas['total_eventos_ventana'] = tot_ini.astype(int)   # episodios que ARRANCAN
ventanas['total_activos_ventana'] = tot_act.astype(int)   # episodios que CUBREN

etq_ini = np.array(reglas_activas)[W_ini.argmax(axis=1)]
etq_act = np.array(reglas_activas)[W_act.argmax(axis=1)]
ventanas['etiqueta_tfidf'] = np.where(tot_ini > 0, etq_ini,
                             np.where(tot_act > 0, etq_act, 'normal'))
ventanas['origen_etiqueta'] = np.where(tot_ini > 0, 'inicia',
                               np.where(tot_act > 0, 'continua', 'sin_alarma'))

print('\n--- Origen de la etiqueta ---')
print(ventanas.origen_etiqueta.value_counts().to_string())

huerfanas = (ventanas.total_alarmas_ventana > 0) & (ventanas.total_eventos_ventana == 0)
sin_etiqueta = huerfanas & (ventanas.etiqueta_tfidf == 'normal')
print(f'\nVentanas con alarmas pero sin evento que arranque: {int(huerfanas.sum()):,} '
      f'({huerfanas.mean() * 100:.1f} %)')
print(f'  de ésas, RECUPERADAS por los eventos activos: '
      f'{int(huerfanas.sum() - sin_etiqueta.sum()):,}')
print(f'  de ésas, que quedan sin etiquetar            : {int(sin_etiqueta.sum()):,}')
if sin_etiqueta.sum() > 0:
    print('  ⚠ Si este número no es 0, hay episodios que empiezan fuera de las')
    print('    ventanas construidas (en la cola descartada del tramo anterior).')

print('\n--- Distribución de la etiqueta TF·IDF (top 10) ---')
print(ventanas.etiqueta_tfidf.value_counts().head(10).to_string())

con_alarma = ventanas.total_alarmas_ventana > 0
cambian = (ventanas.etiqueta_ventana != ventanas.etiqueta_tfidf) & con_alarma
print(f'\n--- Moda vs. TF·IDF, sobre las {int(con_alarma.sum()):,} ventanas con alarmas ---')
print(f'Cambian de etiqueta: {int(cambian.sum()):,} ({cambian.sum() / con_alarma.sum() * 100:.1f} %)')

print('\nMatriz de cambio (filas = moda, columnas = TF·IDF), top 8×8:')
top_moda = ventanas.loc[con_alarma, 'etiqueta_ventana'].value_counts().head(8).index
top_tfidf = ventanas.loc[con_alarma, 'etiqueta_tfidf'].value_counts().head(8).index
print(pd.crosstab(ventanas.loc[con_alarma, 'etiqueta_ventana'],
                  ventanas.loc[con_alarma, 'etiqueta_tfidf'])
      .reindex(index=top_moda, columns=top_tfidf, fill_value=0).to_string())

print('\n--- ¿Se rompió el sesgo por persistencia? ---')
print(f'{"regla":>5} {"presente":>9} {"gana moda":>10} {"gana TF·IDF":>12}')
for regla in resumen_eventos.index[:6]:
    presente = ventanas[regla] > 0
    if presente.sum() == 0:
        continue
    antes = (ventanas.loc[presente, 'etiqueta_ventana'] == regla).mean() * 100
    despues = (ventanas.loc[presente, 'etiqueta_tfidf'] == regla).mean() * 100
    print(f'{regla:>5} {int(presente.sum()):>9,} {antes:>9.1f} % {despues:>11.1f} %')

print(f'\n  Empates de la moda: {int(ventanas.moda_empate.sum()):,} '
      f'({ventanas.moda_empate.mean() * 100:.1f} %). '
      'Con TF·IDF continuo el argmax no empata.')


# =============================================================================
# 5. CONCENTRACIÓN — la métrica de severidad por ventana
# =============================================================================
print('\n' + '=' * 70)
print('5. CONCENTRACIÓN POR VENTANA')
print('=' * 70)


def calcular_concentracion(ventanas, reglas_ids):
    df_out = ventanas.copy()
    entropia_max = np.log(len(reglas_ids))

    def fila_c(fila):
        c = fila[reglas_ids].to_numpy(dtype=float)
        total = c.sum()
        if total == 0:
            return 0.0
        p = c[c > 0] / total
        entropia = -(p * np.log(p)).sum()
        return 1.0 - entropia / entropia_max

    df_out['concentracion'] = df_out.apply(fila_c, axis=1)
    return df_out


ventanas = calcular_concentracion(ventanas, ids_reglas)

anom = ventanas[ventanas.total_alarmas_ventana > 0]
print('Resumen sobre las ventanas anómalas:')
print(anom[['concentracion']].describe().round(4).to_string())

M = ventanas[ids_reglas].to_numpy(float)
tot = M.sum(axis=1)
act = tot > 0
pos_media = (M[act] * np.arange(len(ids_reglas))).sum(axis=1) / tot[act]
v = ventanas.loc[act, 'concentracion'].to_numpy()
print(f'\nCorrelación de concentracion con la POSICIÓN de las reglas: '
      f'{np.corrcoef(pos_media, v)[0, 1]:+.3f}')
print('  (la métrica anterior daba -0,907 acá: dependía del orden de la lista)')


# =============================================================================
# 5b. BANDAS DE SEVERIDAD — normal / leve / media / grave
# =============================================================================
print('\n' + '=' * 70)
print('5b. BANDAS DE SEVERIDAD')
print('=' * 70)

ventanas['n_reglas_activas'] = (ventanas[ids_reglas] > 0).sum(axis=1)
ventanas['persistencia'] = ventanas[ids_reglas].max(axis=1) / W


def asignar_banda(n):
    if n == 0:
        return '0_normal'
    if n == 1:
        return '1_leve'
    if n == 2:
        return '2_media'
    return '3_grave'


ventanas['banda_severidad'] = ventanas.n_reglas_activas.apply(asignar_banda)

print('Criterio: número de reglas DISTINTAS activas en la ventana\n')
resumen_bandas = ventanas.groupby('banda_severidad').agg(
    ventanas=('n_reglas_activas', 'size'),
    alarmas_promedio=('total_alarmas_ventana', 'mean'),
    persistencia_mediana=('persistencia', 'median'),
).round(2)
resumen_bandas['pct'] = (resumen_bandas.ventanas / len(ventanas) * 100).round(1)
print(resumen_bandas.to_string())
print('\nEsperado:  0_normal 6.323 | 1_leve 4.957 | 2_media 5.434 | 3_grave 2.901')

print('\nDistribución de reglas distintas por ventana anómala:')
print(ventanas.loc[ventanas.n_reglas_activas > 0, 'n_reglas_activas']
      .value_counts().sort_index().head(8).to_string())


print('\n' + '=' * 70)
print('5d. WASSERSTEIN POR BANDA (vs. dataset completo)')
print('=' * 70)


def calcular_wasserstein_por_banda(ventanas, columna_grupo='banda_severidad',
                                     columna_valor='total_alarmas_ventana'):
    distribucion_total = ventanas[columna_valor].to_numpy(dtype=float)
    filas = []
    for grupo in sorted(ventanas[columna_grupo].unique()):
        valores_grupo = ventanas.loc[ventanas[columna_grupo] == grupo, columna_valor].to_numpy(dtype=float)
        wemd = wasserstein_distance(valores_grupo, distribucion_total)
        filas.append({'banda_severidad': grupo, 'n_ventanas': len(valores_grupo),
                      'alarmas_promedio': valores_grupo.mean(), 'wasserstein_vs_total': wemd})
    return pd.DataFrame(filas)


wasserstein_bandas = calcular_wasserstein_por_banda(ventanas)
print(wasserstein_bandas.round(4).to_string(index=False))

media_global = ventanas['total_alarmas_ventana'].mean()
wasserstein_bandas['tipo'] = np.where(
    wasserstein_bandas['alarmas_promedio'] > media_global, 'anómala', 'no anómala')
print(f'\nPromedio global de alarmas por ventana: {media_global:.2f}')
print(wasserstein_bandas[['banda_severidad', 'alarmas_promedio', 'tipo',
                          'wasserstein_vs_total']].round(4).to_string(index=False))
for tipo, texto in [('anómala', 'la más GRAVE'), ('no anómala', 'la más ESTABLE')]:
    sub = wasserstein_bandas[wasserstein_bandas.tipo == tipo]
    if len(sub):
        top = sub.loc[sub.wasserstein_vs_total.idxmax(), 'banda_severidad']
        print(f'  Entre las {tipo}s, {texto} (mayor Wemd): {top}')

wasserstein_bandas.to_csv(ARCHIVO_WASSERSTEIN, index=False)
print(f'Guardado: {ARCHIVO_WASSERSTEIN}')

# --- El mismo cálculo sobre EVENTOS, para comparar -----------------------
print('\n--- Wemd por banda calculado sobre EVENTOS en vez de muestras ---')
wass_eventos = calcular_wasserstein_por_banda(
    ventanas, columna_valor='total_eventos_ventana')
media_global_ev = ventanas['total_eventos_ventana'].mean()
wass_eventos['tipo'] = np.where(
    wass_eventos['alarmas_promedio'] > media_global_ev, 'anómala', 'no anómala')
print(f'Promedio global de eventos por ventana: {media_global_ev:.2f}')
print(wass_eventos.round(4).to_string(index=False))
wass_eventos.to_csv(ARCHIVO_WASSERSTEIN.replace('.csv', '_eventos.csv'), index=False)


# =============================================================================
# 5c. DIAGNÓSTICO DEL CRITERIO DEL QUINTIL
# =============================================================================
print('\n' + '=' * 70)
print('5c. ¿QUÉ PASA SI APLICAMOS EL CRITERIO DEL PRIMER QUINTIL?')
print('=' * 70)

pct_sin_alarmas = (ventanas.total_alarmas_ventana == 0).mean() * 100
print(f'Ventanas que YA están en cero alarmas: {pct_sin_alarmas:.1f} %')
if pct_sin_alarmas > 20:
    print('  >> El primer quintil (20 %) cae completo DENTRO de lo que ya es')
    print('     normal, así que aplicado sobre todas las ventanas NO reclasifica nada.')

anomalas = ventanas[ventanas.total_alarmas_ventana > 0]
print('\nSi se aplica solo sobre las anómalas:')
for indice in ['concentracion', 'total_alarmas_ventana', 'total_eventos_ventana']:
    umbral = np.percentile(anomalas[indice], 20)
    bajo_umbral = anomalas[anomalas[indice] <= umbral]
    sobre_umbral = anomalas[anomalas[indice] > umbral]
    print(f'\n  {indice}:  umbral Q1 = {umbral:.2f}')
    print(f'    pasarían a normal: {len(bajo_umbral):,} ventanas, '
          f'con {bajo_umbral.total_alarmas_ventana.mean():.1f} alarmas promedio')
    print(f'    quedan anómalas  : {len(sobre_umbral):,} ventanas, '
          f'con {sobre_umbral.total_alarmas_ventana.mean():.1f} alarmas promedio')
    if bajo_umbral.total_alarmas_ventana.mean() > sobre_umbral.total_alarmas_ventana.mean():
        print('    ⚠ INVERTIDO: las que pasarían a normal tienen MÁS alarmas que el resto.')


# =============================================================================
# 5e. ÍNDICE W POR SENSOR, CORREGIDO POR MULTIPLICIDAD
# =============================================================================
print('\n' + '=' * 70)
print('5e. ÍNDICE W POR SENSOR')
print('=' * 70)

import sys as _sys
if RUTA_BASE not in _sys.path:
    _sys.path.append(RUTA_BASE)
from indice_w import IndiceW

ARCHIVO_W_SENSORES = f'{RUTA_EQ}/{EQUIPO}_w_sensores_W{W}.csv'

_X = np.stack([t[VARIABLES].to_numpy(dtype=np.float32)[v * W:(v + 1) * W]
               for t in tramos_con_reglas
               for v in range(len(t) // W)])
print(f'Ventanas para el índice W: {len(_X):,} '
      f'(deben ser las mismas {len(ventanas):,} del bloque 3)')
assert len(_X) == len(ventanas), 'el recorrido de ventanas no coincide'

_es_train = (pd.to_datetime(ventanas.timestamp_inicio) <= T_TRAIN).to_numpy()
_es_normal = (ventanas.n_reglas_activas == 0).to_numpy()
_fit = _es_train & _es_normal
print(f'Ventanas normales de entrenamiento para ajustar: {int(_fit.sum()):,}')
if _fit.sum() < 200:
    print('  AVISO: con tan pocas, los cuantiles de la nula van a ser ruidosos.')

_mu = _X[_es_train].reshape(-1, len(VARIABLES)).mean(axis=0)
_sd = _X[_es_train].reshape(-1, len(VARIABLES)).std(axis=0)
_sd = np.where(_sd > 0, _sd, 1.0)
_Xn = (_X - _mu) / _sd

_iw = IndiceW(VARIABLES, w=W).ajustar(_Xn[_fit])
_idx_w = _iw.transformar(_Xn)
_perfil = _iw.resumen_por_ventana(_idx_w)

for _c in _perfil.columns:
    ventanas[_c] = _perfil[_c].to_numpy()

_perfil.insert(0, 'ventana_id', ventanas.ventana_id.to_numpy())
_perfil.insert(0, 'tramo_id', ventanas.tramo_id.to_numpy())
_perfil.insert(2, 'timestamp_inicio', ventanas.timestamp_inicio.to_numpy())
_perfil.to_csv(ARCHIVO_W_SENSORES, index=False)
print(f'\nGuardado: {ARCHIVO_W_SENSORES}')

print('\n--- Tres escalas de severidad sobre las mismas ventanas ---')
print(pd.crosstab(ventanas.banda_w, ventanas.banda_severidad).to_string())
print('\n  n_reglas_activas -> cuántos subsistemas están comprometidos')
print('  criticidad_max   -> juicio experto (paso 02c)')
print('  banda_w          -> empírica, independiente de la tabla de criticidad')
print('El grado de acuerdo entre las tres es un RESULTADO, no un paso')
print('intermedio: es lo que compensa que la criticidad sea elicitación de un')
print('solo experto, no conversada con los operadores.')
print('\nY el sensor que alcanza el máximo (columna sensor_origen) contesta')
print('directamente lo que pidió el profesor: "faltaría determinar cuál es el')
print('que está originando la condición".')

del _X, _Xn


# =============================================================================
# 6. FEATURES DE SENSORES POR VENTANA
# =============================================================================
print('\n' + '=' * 70)
print('6. FEATURES DE SENSORES')
print('=' * 70)


def construir_features_sensores(tramos_con_reglas, variables, W):
    resumenes = []
    for tramo_idx, t in enumerate(tramos_con_reglas):
        for v in range(len(t) // W):
            ventana = t.iloc[v * W:(v + 1) * W]
            fila = {'tramo_id': tramo_idx, 'ventana_id': v}
            for var in variables:
                serie = ventana[var]
                fila[f'{var}_media'] = serie.mean()
                fila[f'{var}_std'] = serie.std()
                fila[f'{var}_min'] = serie.min()
                fila[f'{var}_max'] = serie.max()
                fila[f'{var}_delta'] = serie.iloc[-1] - serie.iloc[0]
            resumenes.append(fila)
    return pd.DataFrame(resumenes)


features = construir_features_sensores(tramos_con_reglas, VARIABLES, W)
n_features = len([c for c in features.columns if c not in ('tramo_id', 'ventana_id')])
print(f'Features construidos: {n_features}   <- esperado: 70 (14 variables × 5)')

dataset_features = features.merge(
    ventanas[['tramo_id', 'ventana_id', 'timestamp_inicio', 'timestamp_fin',
              'etiqueta_ventana', 'etiqueta_tfidf', 'total_alarmas_ventana',
              'total_eventos_ventana', 'n_reglas_activas',
              'banda_severidad', 'concentracion']],
    on=['tramo_id', 'ventana_id'], how='inner',
)
print(f'Filas tras el merge: {len(dataset_features):,}  '
      f'(debe coincidir con las {len(ventanas):,} ventanas)')

if len(dataset_features) != len(ventanas):
    print('⚠ ALERTA: el merge perdió filas. Revisa que tramo_id/ventana_id calcen.')


# =============================================================================
# 7. GUARDAR
# =============================================================================
print('\n' + '=' * 70)
print('7. GUARDANDO')
print('=' * 70)

ventanas.to_csv(ARCHIVO_VENTANAS, index=False)
print(f'{ARCHIVO_VENTANAS}')
print(f'   {ventanas.shape[0]:,} filas × {ventanas.shape[1]} columnas')

dataset_features.to_csv(ARCHIVO_FEATURES, index=False)
print(f'{ARCHIVO_FEATURES}')
print(f'   {dataset_features.shape[0]:,} filas × {dataset_features.shape[1]} columnas')

ARCHIVO_EVENTOS_VENTANA = f'{RUTA_EQ}/{EQUIPO}_eventos_por_ventana_W{W}.csv'
_ev_ventana = iniciados.copy()
_ev_ventana.insert(0, 'ventana_id', ventanas['ventana_id'].to_numpy())
_ev_ventana.insert(0, 'tramo_id', ventanas['tramo_id'].to_numpy())
_ev_ventana.to_csv(ARCHIVO_EVENTOS_VENTANA, index=False)
print(f'{ARCHIVO_EVENTOS_VENTANA}   (PARCHE P11)')
print(f'   {_ev_ventana.shape[0]:,} filas × {_ev_ventana.shape[1]} columnas')

ARCHIVO_ACTIVOS_VENTANA = f'{RUTA_EQ}/{EQUIPO}_activos_por_ventana_W{W}.csv'
_act_ventana = activos.copy()
_act_ventana.insert(0, 'ventana_id', ventanas['ventana_id'].to_numpy())
_act_ventana.insert(0, 'tramo_id', ventanas['tramo_id'].to_numpy())
_act_ventana.to_csv(ARCHIVO_ACTIVOS_VENTANA, index=False)
print(f'{ARCHIVO_ACTIVOS_VENTANA}   (PARCHE P11 v2)')
print(f'   {_act_ventana.shape[0]:,} filas × {_act_ventana.shape[1]} columnas')

print('\n--- Primeras filas de las ventanas ---')
cols_muestra = ['tramo_id', 'ventana_id', 'total_alarmas_ventana', 'total_eventos_ventana',
                'n_reglas_activas', 'banda_severidad', 'etiqueta_ventana',
                'etiqueta_tfidf', 'concentracion']
print(ventanas[cols_muestra].head(10).to_string(index=False))

print('\n--- Comparación de las dos etiquetas ---')
cmp_etq = pd.DataFrame({
    'por moda (como antes)': ventanas.etiqueta_ventana.value_counts(),
    'por TF·IDF (papers)': ventanas.etiqueta_tfidf.value_counts(),
}).fillna(0).astype(int).sort_values('por moda (como antes)', ascending=False).head(8)
print(cmp_etq.to_string())
print('\nLa etiqueta oficial sigue siendo la moda. La de TF·IDF va al lado')
print('para poder mostrar cuánto cambia el diagnóstico al ponderar por')
print('especificidad en vez de por persistencia.')

print('\n' + '=' * 70)
print('LISTO. Siguiente paso: 03_random_forest.py')
print('=' * 70)

"""3.- RANDOM FOREST"""

# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 03 — VALIDACIÓN DE LA SEVERIDAD CON RANDOM FOREST
=============================================================================
Entrada:  CR010_features_W60.csv   (salida del paso 02)
"""

from google.colab import drive
drive.mount('/content/drive')
RUTA_BASE = '/content/drive/MyDrive/tesis_chancadores'

import os as _os
EQUIPO = _os.environ.get('EQUIPO', 'CR010')
RUTA_EQ = RUTA_BASE if EQUIPO == 'CR010' else f'{RUTA_BASE}/{EQUIPO}'
_os.makedirs(RUTA_EQ, exist_ok=True)
print(f'Equipo: {EQUIPO} | carpeta de trabajo: {RUTA_EQ}')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

SEED = 42
np.random.seed(SEED)

W = 60
ARCHIVO_FEATURES = f'{RUTA_EQ}/{EQUIPO}_features_W{W}.csv'

FEATURES_ABLACION = ['CM_max', 'PEL_max']

# Las métricas de severidad a validar
OBJETIVOS = ['n_reglas_activas', 'concentracion']

CONFIGURACIONES = [
    {'n_estimators': 300, 'max_depth': None},
    {'n_estimators': 300, 'max_depth': 8},
    {'n_estimators': 300, 'max_depth': 4},
]


# =============================================================================
# 1. CARGA Y SPLIT CRONOLÓGICO
# =============================================================================
print('=' * 70)
print(f'PASO 03 — RANDOM FOREST (W = {W})')
print('=' * 70)

df = pd.read_csv(ARCHIVO_FEATURES, parse_dates=['timestamp_inicio', 'timestamp_fin'])
df = df.sort_values(['tramo_id', 'ventana_id']).reset_index(drop=True)

NO_FEATURE = ['tramo_id', 'ventana_id', 'timestamp_inicio', 'timestamp_fin',
              'etiqueta_ventana', 'etiqueta_tfidf', 'banda_severidad',
              'total_alarmas_ventana', 'total_eventos_ventana'] + OBJETIVOS
FEATURES = [c for c in df.columns if c not in NO_FEATURE]

print(f'Ventanas: {len(df):,}')
print(f'Features de sensores: {len(FEATURES)}   <- esperado: 70')


import json as _json
_corte = _json.load(open(f'{RUTA_EQ}/{EQUIPO}_corte_split.json'))
_T_TRAIN, _T_VAL = pd.Timestamp(_corte['T_train']), pd.Timestamp(_corte['T_val'])
_col_t = ('timestamp_inicio' if 'timestamp_inicio' in df.columns
          else 'timestamp_fin')
_ts = pd.to_datetime(df[_col_t])
df_train = df[_ts <= _T_TRAIN]
df_val = df[(_ts > _T_TRAIN) & (_ts <= _T_VAL)]
df_test = df[_ts > _T_VAL]
print(f'Split cronológico (corte unico, por fecha): train={len(df_train):,} | '
      f'val={len(df_val):,} | test={len(df_test):,}')
print(f'  T_train = {_T_TRAIN} | T_val = {_T_VAL}')


# =============================================================================
# 2. FUNCIÓN DE ENTRENAMIENTO
# =============================================================================
def entrenar_y_evaluar(features, objetivo, etiqueta):
    """Elige la configuración sobre validación y evalúa test UNA sola vez."""
    X_tr, y_tr = df_train[features], df_train[objetivo]
    X_va, y_va = df_val[features], df_val[objetivo]
    X_te, y_te = df_test[features], df_test[objetivo]

    mejor, mejor_r2, mejor_cfg = None, -np.inf, None
    print(f'\n  --- {etiqueta} ({len(features)} features) ---')
    for cfg in CONFIGURACIONES:
        m = RandomForestRegressor(random_state=SEED, n_jobs=-1, **cfg).fit(X_tr, y_tr)
        r2 = r2_score(y_va, m.predict(X_va))
        print(f'    {cfg} -> R² val = {r2:.4f}')
        if r2 > mejor_r2:
            mejor, mejor_r2, mejor_cfg = m, r2, cfg

    pred = mejor.predict(X_te)
    r2_test = r2_score(y_te, pred)
    rmse = mean_squared_error(y_te, pred) ** 0.5
    mae = mean_absolute_error(y_te, pred)

    print(f'    Elegida: {mejor_cfg}')
    print(f'    >> R² TEST = {r2_test:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f}')
    return mejor, r2_test, pred, y_te


# =============================================================================
# 3. CORRER LAS DOS MÉTRICAS, CON Y SIN ABLACIÓN
# =============================================================================
FEATURES_SIN_ABLACION = [f for f in FEATURES if f not in FEATURES_ABLACION]
FEATURES_SIN_EXTREMOS = [f for f in FEATURES
                         if not f.endswith(('_max', '_min'))]

CONJUNTOS_ABLACION = {
    'completo': FEATURES,
    'sin_dominantes': FEATURES_SIN_ABLACION,
    'sin_extremos': FEATURES_SIN_EXTREMOS,
}
print('\nNiveles de ablación:')
for _k, _v in CONJUNTOS_ABLACION.items():
    print(f'  {_k:<16}{len(_v):>3} features')
if len(FEATURES_SIN_EXTREMOS) == len(FEATURES):
    print('\n  AVISO: sin_extremos no quitó nada -- tus features no terminan en')
    print('  _max / _min, así que el patrón no aplica. Revisá cómo se llaman')
    print('  los extremos en el archivo antes de interpretar la ablación.')

resultados = {}

for objetivo in OBJETIVOS:
    print('\n' + '=' * 70)
    print(f'OBJETIVO: {objetivo}')
    print('=' * 70)

    _r2_triv = r2_score(df_test[objetivo],
                        np.full(len(df_test), df_train[objetivo].mean()))
    print(f'Línea base (predecir siempre la media de train): R² = {_r2_triv:.4f}')
    print(f'Distribución del objetivo en test: media '
          f'{df_test[objetivo].mean():.3f} | sd {df_test[objetivo].std():.3f}')

    _r2 = {}
    modelo = None
    for _nombre, _feats in CONJUNTOS_ABLACION.items():
        _m, _r, _p, _y = entrenar_y_evaluar(_feats, objetivo, _nombre)
        _r2[_nombre] = _r
        if _nombre == 'completo':
            modelo, pred, y_real = _m, _p, _y

    imp = pd.Series(modelo.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print('\n  Top 8 features:')
    for k, v in imp.head(8).items():
        print(f'    {k:14s} {v * 100:5.1f} %')
    print(f'    Las dos de la ablación suman: '
          f'{imp[FEATURES_ABLACION].sum() * 100:.1f} %')
    _extremos = [f for f in FEATURES if f.endswith(('_max', '_min'))]
    if _extremos:
        print(f'    Los {len(_extremos)} extremos suman: '
              f'{imp[_extremos].sum() * 100:.1f} %')
        print('    Si ese porcentaje es alto, el bosque se apoya en los cruces')
        print('    de umbral y la fila sin_extremos es la única que dice algo')
        print('    sobre la física.')

    resultados[objetivo] = {
        'r2_completo': _r2['completo'],
        'r2_ablacion': _r2['sin_dominantes'],
        'r2_sin_extremos': _r2['sin_extremos'],
        'r2_linea_base': _r2_triv,
        'importancias': imp, 'pred': pred, 'real': y_real,
    }


# =============================================================================
# 4. RESUMEN COMPARATIVO
# =============================================================================
print('\n' + '=' * 70)
print('RESUMEN')
print('=' * 70)
print(f'{"métrica":<16} {"línea base":>11} {"completo":>10} {"sin_domin.":>11} '
      f'{"SIN EXTREMOS":>13}')
print('-' * 64)
for obj, r in resultados.items():
    print(f'{obj:<16} {r["r2_linea_base"]:>11.4f} {r["r2_completo"]:>10.4f} '
          f'{r["r2_ablacion"]:>11.4f} {r["r2_sin_extremos"]:>13.4f}')

print('\nCómo leerlo, en este orden:')
print('  1. Compará SIEMPRE contra la línea base de predecir la media. Un R²')
print('     de 0,6 no dice nada si la línea base ya daba 0,55.')
print('  2. R² completo alto -> el bosque puede predecir la severidad. Todavía')
print('     no dice CON QUÉ.')
print('  3. La columna que decide es SIN EXTREMOS:')
print('     se sostiene  -> la severidad tiene firma en la DINÁMICA (media,')
print('                     dispersión y tendencia bastan). Ésta es la')
print('                     conclusión que vale y la que se defiende.')
print('     se desploma  -> el bosque estaba reaprendiendo los umbrales de las')
print('                     reglas. Decirlo y acotar lo que se afirma; también')
print('                     es un resultado.')
print('  4. sin_dominantes queda como contraste con la versión anterior, pero')
print('     ya no es la prueba: dejaba adentro T1_max, T7_max y los V*_max,')
print('     que son las condiciones de la veintena de reglas restantes.')


print('\n' + '=' * 70)
print('LISTO. Siguiente paso: 04_autoencoder.py')
print('=' * 70)

"""4 AUTOENCODER"""

# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 04-B — FASE A: AUTOENCODER ENTRENADO **SOLO CON VENTANAS NORMALES**
=============================================================================
"""

from google.colab import drive
drive.mount('/content/drive')
RUTA_BASE = '/content/drive/MyDrive/tesis_chancadores'

import os as _os
EQUIPO = _os.environ.get('EQUIPO', 'CR010')
RUTA_EQ = RUTA_BASE if EQUIPO == 'CR010' else f'{RUTA_BASE}/{EQUIPO}'
_os.makedirs(RUTA_EQ, exist_ok=True)
print(f'Equipo: {EQUIPO} | carpeta de trabajo: {RUTA_EQ}')

import sys
sys.path.append(RUTA_BASE)

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from reglas_engine import evaluar_reglas

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Dispositivo:', device)
if device.type != 'cuda':
    print('\n⚠ NO HAY GPU. Entorno de ejecución -> Cambiar tipo de entorno -> GPU (T4).')

W = 60
STRIDE_TRAIN = 12    # ventanas cada 2 min -- traslape alto, más ejemplos
STRIDE_EVAL = W      # sin traslape en val/test
EPOCHS_MAX = 100
PACIENCIA = 15       # épocas sin mejorar la pérdida de validación

ARCHIVO_LIMPIO = f'{RUTA_EQ}/{EQUIPO}_limpio.csv'
RUTA_MODELO = f'{RUTA_EQ}/autoencoder_W{W}_normales.pt'          # menor pérdida
RUTA_MODELO_SEP = f'{RUTA_EQ}/autoencoder_W{W}_normales_sep.pt'  # mayor separación
RUTA_NORMALIZADOR = f'{RUTA_EQ}/normalizador_W{W}.npz'
RUTA_CURVA = f'{RUTA_EQ}/04b_curva_entrenamiento.png'
RUTA_EPOCAS = f'{RUTA_EQ}/04b_comparacion_epocas.csv'

VARIABLES = ['CM', 'PI', 'PDF', 'PEL', 'T7', 'T8', 'T9',
             'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4']

REF = {'val_mse': 0.03057, 'test_mse': 0.03707,
       'err_normal': 0.0087, 'err_anomalo': 0.0302, 'razon': 3.46, 'auc': 0.8358}


# =============================================================================
# 1. CARGA Y SEGMENTACIÓN
# =============================================================================
print('=' * 70)
print(f'PASO 04-B — AUTOENCODER SOLO CON NORMALES (W = {W})')
print('=' * 70)


def cargar_limpio(ruta):
    cabecera = pd.read_csv(ruta, nrows=0)
    col_tiempo = 'timestamp' if 'timestamp' in cabecera.columns else 'Time'
    df = pd.read_csv(ruta, parse_dates=[col_tiempo], low_memory=False)
    return df.set_index(col_tiempo).sort_index().rename_axis('timestamp')


df = cargar_limpio(ARCHIVO_LIMPIO)
VARIABLES = [v for v in VARIABLES if v in df.columns]
print(f'Dataset: {df.shape[0]:,} filas | {len(VARIABLES)} variables')


def segmentar_tramos_continuos(df, freq='10s', max_gap=1):
    idx = df.index.to_series()
    gaps = idx.diff() > pd.Timedelta(freq) * max_gap
    df = df.copy()
    df['tramo_id'] = gaps.cumsum()
    return [g.drop(columns='tramo_id') for _, g in df.groupby('tramo_id')]


tramos = segmentar_tramos_continuos(df)
tramos = [t for t in tramos if len(t) >= W]
print(f'Tramos útiles: {len(tramos):,}   <- esperado: 1.929')


# =============================================================================
# 2. EVALUAR LAS REGLAS — para saber qué ventana es normal
# =============================================================================
print('\n' + '=' * 70)
print('2. EVALUANDO REGLAS PARA IDENTIFICAR LAS VENTANAS NORMALES')
print('=' * 70)

mascaras_activas = []      # una por tramo: bool por paso
ids_reglas = None

for i, t in enumerate(tramos):
    resultado = evaluar_reglas(t, verbose=False)
    if ids_reglas is None:
        ids_reglas = [c for c in resultado.columns
                      if c not in ('n_reglas_activadas', 'estado')]
    mascaras_activas.append(resultado[ids_reglas].to_numpy().any(axis=1))
    if (i + 1) % 500 == 0:
        print(f'  {i + 1:,} / {len(tramos):,} tramos')

print(f'Reglas evaluadas: {len(ids_reglas)}')
pasos_en_alarma = sum(int(m.sum()) for m in mascaras_activas)
pasos_totales = sum(len(m) for m in mascaras_activas)
print(f'Pasos con alguna regla activa: {pasos_en_alarma:,} de {pasos_totales:,} '
      f'({100 * pasos_en_alarma / pasos_totales:.1f} %)')


# =============================================================================
# 3. SPLIT CRONOLÓGICO
# =============================================================================
import json as _json
_corte = _json.load(open(f'{RUTA_EQ}/{EQUIPO}_corte_split.json'))
_T_TRAIN, _T_VAL = pd.Timestamp(_corte['T_train']), pd.Timestamp(_corte['T_val'])

_ini = np.array([t.index[0] for t in tramos])
s_tr = _ini <= _T_TRAIN
s_va = (_ini > _T_TRAIN) & (_ini <= _T_VAL)
s_te = _ini > _T_VAL
tramos_train = [t for t, m in zip(tramos, s_tr) if m]
tramos_val = [t for t, m in zip(tramos, s_va) if m]
tramos_test = [t for t, m in zip(tramos, s_te) if m]
masc_train = [m for m, k in zip(mascaras_activas, s_tr) if k]
masc_val = [m for m, k in zip(mascaras_activas, s_va) if k]
masc_test = [m for m, k in zip(mascaras_activas, s_te) if k]
print(f'\nTramos: train={len(tramos_train)} | val={len(tramos_val)} | '
      f'test={len(tramos_test)}')
print(f'Muestras: train={sum(len(t) for t in tramos_train):,} | '
      f'val={sum(len(t) for t in tramos_val):,} | '
      f'test={sum(len(t) for t in tramos_test):,}')
print(f'  corte unico: T_train = {_T_TRAIN} | T_val = {_T_VAL}')


# =============================================================================
# 4. NORMALIZACIÓN Z-SCORE — ajustada solo con train
# =============================================================================
def ajustar_normalizador(tramos_train, variables):
    concat = pd.concat(tramos_train)[variables]
    return {'media': concat.mean(), 'std': concat.std()}


def aplicar_normalizador(tramo, variables, params):
    tramo = tramo.copy()
    tramo[variables] = (tramo[variables] - params['media']) / params['std']
    return tramo


params_norm = ajustar_normalizador(tramos_train, VARIABLES)
tramos_train = [aplicar_normalizador(t, VARIABLES, params_norm) for t in tramos_train]
tramos_val = [aplicar_normalizador(t, VARIABLES, params_norm) for t in tramos_val]
tramos_test = [aplicar_normalizador(t, VARIABLES, params_norm) for t in tramos_test]

np.savez(RUTA_NORMALIZADOR,
         media=params_norm['media'].to_numpy(),
         std=params_norm['std'].to_numpy(),
         variables=np.array(VARIABLES, dtype=object))
print(f'Normalizador guardado: {RUTA_NORMALIZADOR} (idéntico al del 04 original)')


# =============================================================================
# 5. VENTANAS, CADA UNA CON SU ETIQUETA DE NORMALIDAD
# =============================================================================
def construir_ventanas(tramos, mascaras, variables, W, stride):
    ventanas, es_normal = [], []
    for t, m in zip(tramos, mascaras):
        arr = t[variables].to_numpy(dtype=np.float32)
        acum = np.concatenate(([0], np.cumsum(m.astype(np.int32))))
        for ini in range(0, len(arr) - W + 1, stride):
            ventanas.append(arr[ini:ini + W])
            es_normal.append((acum[ini + W] - acum[ini]) == 0)
    if not ventanas:
        return (np.empty((0, W, len(variables)), dtype=np.float32),
                np.empty(0, dtype=bool))
    return np.stack(ventanas), np.array(es_normal, dtype=bool)


X_train, norm_train = construir_ventanas(tramos_train, masc_train, VARIABLES, W, STRIDE_TRAIN)
X_val, norm_val = construir_ventanas(tramos_val, masc_val, VARIABLES, W, STRIDE_EVAL)
X_test, norm_test = construir_ventanas(tramos_test, masc_test, VARIABLES, W, STRIDE_EVAL)

print('\n' + '=' * 70)
print('5. VENTANAS')
print('=' * 70)
for nombre, X, nm in [('train', X_train, norm_train), ('val', X_val, norm_val),
                      ('test', X_test, norm_test)]:
    print(f'  {nombre:>5}: {len(X):>7,} ventanas | normales {int(nm.sum()):>7,} '
          f'({100 * nm.mean():.1f} %) | anómalas {int((~nm).sum()):>7,}')

# --- EL FILTRO: entrenar solo con las normales ---------------------------
X_train_normal = X_train[norm_train]
X_val_normal = X_val[norm_val]

print(f'\nEntrenamiento ANTERIOR : {len(X_train):>7,} ventanas (todas)')
print(f'Entrenamiento AHORA    : {len(X_train_normal):>7,} ventanas (solo normales)')
print(f'Validación para parar  : {len(X_val_normal):>7,} ventanas normales')

if len(X_train_normal) < 3000:
    print('\n⚠ Quedan pocas ventanas normales para entrenar. Con 135.758 parámetros')
    print('  hay riesgo de sobreajuste. Si pasa, bajar STRIDE_TRAIN a 6 para')
    print('  generar más ventanas, o reducir d_model de 64 a 32.')

ratio = len(X_train_normal) * W * len(VARIABLES) / 135758
print(f'Relación datos/parámetros: {ratio:.0f} a 1  (antes era ~314 a 1)')


class VentanasDataset(Dataset):
    def __init__(self, X):
        self.X = torch.from_numpy(X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i]


train_loader = DataLoader(VentanasDataset(X_train_normal), batch_size=64, shuffle=True)
val_norm_loader = DataLoader(VentanasDataset(X_val_normal), batch_size=256, shuffle=False)
val_full_loader = DataLoader(VentanasDataset(X_val), batch_size=256, shuffle=False)
test_loader = DataLoader(VentanasDataset(X_test), batch_size=256, shuffle=False)


# =============================================================================
# 6. EL MODELO
# =============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class AutoencoderTransformer(nn.Module):
    def __init__(self, n_vars, W, d_model=64, nhead=4, num_layers=2,
                 dim_ff=128, dropout=0.1):
        super().__init__()
        self.W = W
        self.embed = nn.Linear(n_vars, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=W)
        capa_encoder = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(capa_encoder, num_layers=num_layers)
        capa_decoder = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerEncoder(capa_decoder, num_layers=num_layers)
        self.salida = nn.Linear(d_model, n_vars)

    def encode(self, x):
        h = self.embed(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        return h.mean(dim=1)

    def decode(self, latente):
        h = latente.unsqueeze(1).repeat(1, self.W, 1)
        h = self.pos_enc(h)
        h = self.decoder(h)
        return self.salida(h)

    def forward(self, x):
        return self.decode(self.encode(x))


n_vars = len(VARIABLES)
modelo = AutoencoderTransformer(n_vars=n_vars, W=W).to(device)
n_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
print(f'\nModelo: {n_params:,} parámetros entrenables')
print(f'Compresión: {W * n_vars} números -> 64 ({W * n_vars / 64:.1f}x)')


# =============================================================================
# 7. ENTRENAMIENTO
# =============================================================================
print('\n' + '=' * 70)
print('ENTRENANDO — solo con ventanas normales')
print('=' * 70)

optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)
criterio = nn.MSELoss()
criterio_sin_reducir = nn.MSELoss(reduction='none')


def error_por_ventana(loader):
    """Error de reconstrucción de cada ventana del loader."""
    modelo.eval()
    errores = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            e = criterio_sin_reducir(modelo(batch), batch).mean(dim=(1, 2))
            errores.append(e.cpu().numpy())
    return np.concatenate(errores)


mejor_val = float('inf')
mejor_separacion = -float('inf')
mejor_epoca = 0
mejor_epoca_sep = 0
espera = 0
historial = {'train': [], 'val_normal': [], 'err_norm': [], 'err_anom': [],
             'razon': [], 'auc': []}

for epoch in range(1, EPOCHS_MAX + 1):
    modelo.train()
    perdida_train = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizador.zero_grad()
        loss = criterio(modelo(batch), batch)
        loss.backward()
        optimizador.step()
        perdida_train += loss.item() * batch.size(0)
    perdida_train /= len(train_loader.dataset)

    # error en las normales de validación -> criterio de parada
    modelo.eval()
    perdida_val = 0.0
    with torch.no_grad():
        for batch in val_norm_loader:
            batch = batch.to(device)
            perdida_val += criterio(modelo(batch), batch).item() * batch.size(0)
    perdida_val /= len(val_norm_loader.dataset)

    # diagnóstico: separación sobre TODA la validación
    err_val = error_por_ventana(val_full_loader)
    e_norm = float(np.median(err_val[norm_val]))
    e_anom = float(np.median(err_val[~norm_val]))
    razon = e_anom / e_norm if e_norm > 0 else float('nan')
    orden = np.argsort(err_val)
    rangos = np.empty(len(err_val), dtype=float)
    rangos[orden] = np.arange(1, len(err_val) + 1)
    n_pos, n_neg = int((~norm_val).sum()), int(norm_val.sum())
    auc = ((rangos[~norm_val].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
           if n_pos and n_neg else float('nan'))

    for k, v in [('train', perdida_train), ('val_normal', perdida_val),
                 ('err_norm', e_norm), ('err_anom', e_anom),
                 ('razon', razon), ('auc', auc)]:
        historial[k].append(v)

    print(f'Epoch {epoch:3d} | train={perdida_train:.5f} | val_norm={perdida_val:.5f} '
          f'| normal={e_norm:.5f} anómala={e_anom:.5f} | razón={razon:5.2f}x | AUC={auc:.4f}')

    # checkpoint 1: menor pérdida de reconstrucción sobre las normales
    if perdida_val < mejor_val - 1e-5:
        mejor_val = perdida_val
        espera = 0
        torch.save(modelo.state_dict(), RUTA_MODELO)
        mejor_epoca = epoch
    else:
        espera += 1

    if razon > mejor_separacion:
        mejor_separacion = razon
        torch.save(modelo.state_dict(), RUTA_MODELO_SEP)
        mejor_epoca_sep = epoch

    if espera >= PACIENCIA:
        print(f'\nEarly stopping en la época {epoch} (sin mejora en {PACIENCIA})')
        break

pd.DataFrame(historial).assign(epoca=range(1, len(historial['train']) + 1)).to_csv(
    RUTA_EPOCAS, sep=';', decimal=',', index=False, encoding='utf-8-sig')
print(f'\nCurva por época guardada: {RUTA_EPOCAS}')

# --- La comprobación que faltaba ------------------------------------------
_ep_final = len(historial['train'])
if mejor_epoca >= _ep_final:
    print('\n' + '!' * 70)
    print('ATENCIÓN: el mínimo de pérdida de validación cayó en la ÚLTIMA época')
    print(f'corrida ({_ep_final}). La pérdida seguía bajando, así que el modelo')
    print('NO convergió y "mejor época" es en realidad "la última época". Con')
    print('eso, cualquier comparación entre los dos criterios de checkpoint')
    print('queda confundida con el techo del entrenamiento, y no se puede')
    print(f'responder por qué se entrenó {_ep_final} épocas y no otra cantidad.')
    print(f'Subí EPOCHS_MAX (hoy {EPOCHS_MAX}) y volvé a correr.')
    print('Esto es exactamente lo que pasaba con EPOCHS_MAX = 20.')
    print('!' * 70)
else:
    print(f'\nConvergió: mínimo de pérdida en la época {mejor_epoca} de '
          f'{_ep_final} corridas ({_ep_final - mejor_epoca} épocas sin mejorar).')
    print('Ahora sí la comparación entre criterios de checkpoint es limpia.')


# =============================================================================
# 8. RESULTADOS Y COMPARACIÓN CON EL MODELO ANTERIOR
# =============================================================================
def evaluar(ruta):
    modelo.load_state_dict(torch.load(ruta))
    modelo.eval()
    ev = error_por_ventana(val_full_loader)
    et = error_por_ventana(test_loader)
    en = float(np.median(ev[norm_val]))
    ea = float(np.median(ev[~norm_val]))
    orden = np.argsort(ev)
    rangos = np.empty(len(ev), dtype=float)
    rangos[orden] = np.arange(1, len(ev) + 1)
    npos, nneg = int((~norm_val).sum()), int(norm_val.sum())
    auc = (rangos[~norm_val].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    return {'err_norm': en, 'err_anom': ea, 'razon': ea / en,
            'auc': auc, 'mse_test': float(et.mean())}


m_perdida = evaluar(RUTA_MODELO)
m_separacion = evaluar(RUTA_MODELO_SEP)

print('\n' + '=' * 70)
print('RESULTADOS — TRES MODELOS')
print('=' * 70)
print(f'Checkpoint por menor pérdida  : época {mejor_epoca}')
print(f'Checkpoint por mayor separación: época {mejor_epoca_sep}')
print()
cab = f'{"":<34}{"ANTERIOR":>12}{"menor pérdida":>16}{"mayor separación":>19}'
print(cab)
print('-' * len(cab))
for nombre, clave, ref in [
        ('Error mediano, normales', 'err_norm', REF['err_normal']),
        ('Error mediano, anómalas', 'err_anom', REF['err_anomalo']),
        ('Razón anómala / normal', 'razon', REF['razon']),
        ('AUC normal vs. anómala', 'auc', REF['auc'])]:
    print(f'{nombre:<34}{ref:>12.4f}{m_perdida[clave]:>16.4f}{m_separacion[clave]:>19.4f}')
print(f'{"MSE en test":<34}{REF["test_mse"]:>12.4f}'
      f'{m_perdida["mse_test"]:>16.4f}{m_separacion["mse_test"]:>19.4f}')

print()
mejor = max([('menor pérdida', m_perdida), ('mayor separación', m_separacion)],
            key=lambda x: x[1]['razon'])
print(f'El mejor detector es el checkpoint por {mejor[0]}: '
      f'razón {mejor[1]["razon"]:.2f}x, AUC {mejor[1]["auc"]:.4f}')
if m_separacion['razon'] > REF['razon']:
    print(f'Frente al modelo entrenado con todo ({REF["razon"]:.2f}x), la mejora es de '
          f'{m_separacion["razon"] / REF["razon"]:.1f} veces.')

razon_v = m_separacion['razon']

# --- ¿hay efecto Ayala? ---------------------------------------------------
razones = np.array(historial['razon'])
pico = int(np.nanargmax(razones)) + 1
print()
print(f'Separación máxima durante el entrenamiento: {np.nanmax(razones):.2f}x '
      f'en la época {pico}')
if pico < mejor_epoca:
    print(f'  ⚠ El pico de separación ({pico}) llegó ANTES que el mínimo de pérdida '
          f'({mejor_epoca}).')
    print('    Es el efecto que describe Ayala: entrenar de más empeora la detección.')
    print('    Conviene reportar las dos curvas y discutir cuál checkpoint conviene.')
else:
    print('  El mínimo de pérdida y el máximo de separación coinciden o van juntos.')
    print('  No aparece el efecto que describe Ayala en este caso.')


# =============================================================================
# 9. CURVAS
# =============================================================================
fig, ejes = plt.subplots(1, 2, figsize=(14, 5))

ejes[0].plot(historial['train'], label='train (solo normales)', linewidth=1.8)
ejes[0].plot(historial['val_normal'], label='validación (solo normales)', linewidth=1.8)
ejes[0].axvline(mejor_epoca - 1, color='gray', linestyle=':', label=f'mejor época ({mejor_epoca})')
ejes[0].set_xlabel('Época'); ejes[0].set_ylabel('MSE')
ejes[0].set_title('Pérdida de reconstrucción')
ejes[0].legend(); ejes[0].grid(alpha=.3)

ejes[1].plot(historial['razon'], color='#E8833A', linewidth=1.8, label='razón anómala/normal')
ejes[1].axhline(REF['razon'], color='gray', linestyle='--', linewidth=1,
                label=f'modelo anterior ({REF["razon"]:.2f}x)')
ejes[1].axvline(pico - 1, color='#2E7D32', linestyle=':', label=f'pico ({pico})')
ejes[1].set_xlabel('Época'); ejes[1].set_ylabel('Razón')
ejes[1].set_title('Separación entre ventanas normales y anómalas')
ejes[1].legend(); ejes[1].grid(alpha=.3)

plt.tight_layout()
plt.savefig(RUTA_CURVA, dpi=150)
print(f'\nCurvas guardadas: {RUTA_CURVA}')

print('\n' + '=' * 70)
print('LISTO. El modelo anterior quedó intacto en autoencoder_W60.pt')
print(f'      Checkpoint por menor pérdida  : {RUTA_MODELO}')
print(f'      Checkpoint por mayor separación: {RUTA_MODELO_SEP}')
print('Siguiente paso: correr el 05 apuntando al modelo nuevo para regenerar')
print('el archivo de errores por ventana.')
print('=' * 70)

"""5.- EVALUACION DEL AUTOENCODER"""

# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 05 — FASE A: EVALUACIÓN Y VALIDACIÓN DE LAS BANDAS
=============================================================================
"""

from google.colab import drive
drive.mount('/content/drive')
RUTA_BASE = '/content/drive/MyDrive/tesis_chancadores'

import os as _os
EQUIPO = _os.environ.get('EQUIPO', 'CR010')
RUTA_EQ = RUTA_BASE if EQUIPO == 'CR010' else f'{RUTA_BASE}/{EQUIPO}'
_os.makedirs(RUTA_EQ, exist_ok=True)
print(f'Equipo: {EQUIPO} | carpeta de trabajo: {RUTA_EQ}')

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Dispositivo:', device)

W = 60
ARCHIVO_LIMPIO = f'{RUTA_EQ}/{EQUIPO}_limpio.csv'
ARCHIVO_VENTANAS = f'{RUTA_EQ}/{EQUIPO}_ventanas_W{W}.csv'
CANDIDATOS = {
    'AE_normales': [f'{RUTA_EQ}/autoencoder_W{W}_normales.pt',
                    f'{RUTA_EQ}/CAMBIOS/autoencoder_W{W}_normales.pt'],
    'AE_normales_sep': [f'{RUTA_EQ}/autoencoder_W{W}_normales_sep.pt',
                        f'{RUTA_EQ}/CAMBIOS/autoencoder_W{W}_normales_sep.pt'],
}
MODELO_PRINCIPAL = 'AE_normales'
import os as _os
MODELOS = {}
for _n, _rutas in CANDIDATOS.items():
    _hit = next((r for r in _rutas if _os.path.exists(r)), None)
    if _hit:
        MODELOS[_n] = _hit
    else:
        print(f'AVISO: {_n} no está en ninguna de estas rutas:')
        for _r in _rutas:
            print(f'        {_r}')
if MODELO_PRINCIPAL not in MODELOS:
    raise SystemExit(f'Falta el modelo principal ({MODELO_PRINCIPAL}). '
                     'Corré el paso 04 antes que este.')
RUTA_MODELO = MODELOS[MODELO_PRINCIPAL]
print(f'Modelos encontrados: {len(MODELOS)} de {len(CANDIDATOS)}')
for _n, _r in MODELOS.items():
    print(f'  {_n:<18} {_r}')
RUTA_NORMALIZADOR = f'{RUTA_EQ}/normalizador_W{W}.npz'
ARCHIVO_SALIDA = f'{RUTA_EQ}/{EQUIPO}_error_reconstruccion_W{W}.csv'

VARIABLES = ['CM', 'PI', 'PDF', 'PEL', 'T7', 'T8', 'T9',
             'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4']


# =============================================================================
# 1. RECONSTRUIR EXACTAMENTE EL MISMO PIPELINE DEL PASO 04
# =============================================================================
print('=' * 70)
print(f'PASO 05 — EVALUACIÓN DE LA FASE A (W = {W})')
print('=' * 70)

def cargar_limpio(ruta):
    """Acepta el CSV con la columna de tiempo llamada 'timestamp' o 'Time'."""
    cabecera = pd.read_csv(ruta, nrows=0)
    col_tiempo = 'timestamp' if 'timestamp' in cabecera.columns else 'Time'
    df = pd.read_csv(ruta, parse_dates=[col_tiempo], low_memory=False)
    return df.set_index(col_tiempo).sort_index().rename_axis('timestamp')


df = cargar_limpio(ARCHIVO_LIMPIO)
VARIABLES = [v for v in VARIABLES if v in df.columns]


def segmentar_tramos_continuos(df, freq='10s', max_gap=1):
    idx = df.index.to_series()
    gaps = idx.diff() > pd.Timedelta(freq) * max_gap
    tramo_id = gaps.cumsum()
    df = df.copy()
    df['tramo_id'] = tramo_id
    return [g.drop(columns='tramo_id') for _, g in df.groupby('tramo_id')]


tramos = segmentar_tramos_continuos(df)
tramos = [t for t in tramos if len(t) >= W]

# ---- LA RE-NUMERACIÓN QUE NO SE PUEDE OLVIDAR --------------------------
tramos_con_id = [(i, t) for i, t in enumerate(tramos)]
print(f'Tramos útiles: {len(tramos_con_id):,}   <- esperado: 1.929')


# El mismo corte por fecha que usan los pasos 03 y 04.
import json as _json
_corte = _json.load(open(f'{RUTA_EQ}/{EQUIPO}_corte_split.json'))
_T_TRAIN, _T_VAL = pd.Timestamp(_corte['T_train']), pd.Timestamp(_corte['T_val'])
_ini5 = np.array([t[1].index[0] for t in tramos_con_id])
tramos_train = [t for t, i in zip(tramos_con_id, _ini5) if i <= _T_TRAIN]
tramos_val = [t for t, i in zip(tramos_con_id, _ini5)
              if _T_TRAIN < i <= _T_VAL]
tramos_test = [t for t, i in zip(tramos_con_id, _ini5) if i > _T_VAL]
print(f'Tramos: train={len(tramos_train)} | val={len(tramos_val)} | '
      f'test={len(tramos_test)}')
print(f'  corte unico: T_train = {_T_TRAIN} | T_val = {_T_VAL}')

norm = np.load(RUTA_NORMALIZADOR, allow_pickle=True)
vars_guardadas = list(norm['variables'])
params_norm = {
    'media': pd.Series(norm['media'], index=vars_guardadas),
    'std': pd.Series(norm['std'], index=vars_guardadas),
}
print(f'Normalizador cargado del paso 04 ({len(vars_guardadas)} variables)')
if vars_guardadas != VARIABLES:
    print('⚠ Las variables no coinciden con las del paso 04. Revisa antes de seguir.')


def aplicar_normalizador(tid_tramo, variables, params):
    tid, tramo = tid_tramo
    tramo = tramo.copy()
    tramo[variables] = (tramo[variables] - params['media']) / params['std']
    return tid, tramo


tramos_val_n = [aplicar_normalizador(t, VARIABLES, params_norm) for t in tramos_val]
tramos_test_n = [aplicar_normalizador(t, VARIABLES, params_norm) for t in tramos_test]


def construir_ventanas_con_id(tramos, variables, W, stride):
    ventanas, ids = [], []
    for tid, t in tramos:
        arr = t[variables].to_numpy(dtype=np.float32)
        vid = 0
        for inicio in range(0, len(arr) - W + 1, stride):
            ventanas.append(arr[inicio:inicio + W])
            ids.append((tid, vid))
            vid += 1
    if not ventanas:
        return np.empty((0, W, len(variables)), dtype=np.float32), []
    return np.stack(ventanas), ids


# Sin traslape: una ventana por bloque, igual que en el paso 02
X_val, ids_val = construir_ventanas_con_id(tramos_val_n, VARIABLES, W, W)
X_test, ids_test = construir_ventanas_con_id(tramos_test_n, VARIABLES, W, W)
X_eval = np.concatenate([X_val, X_test], axis=0)
ids_eval = ids_val + ids_test
conjunto_eval = np.array(['val'] * len(X_val) + ['test'] * len(X_test))
print(f'Ventanas a evaluar: {len(X_eval):,}   '
      f'(val {len(X_val):,} | test {len(X_test):,})')


# =============================================================================
# 2. CARGAR EL MODELO Y CALCULAR EL ERROR POR VENTANA
# =============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class AutoencoderTransformer(nn.Module):
    def __init__(self, n_vars, W, d_model=64, nhead=4, num_layers=2,
                 dim_ff=128, dropout=0.1):
        super().__init__()
        self.W = W
        self.embed = nn.Linear(n_vars, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=W)
        capa_encoder = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(capa_encoder, num_layers=num_layers)
        capa_decoder = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerEncoder(capa_decoder, num_layers=num_layers)
        self.salida = nn.Linear(d_model, n_vars)

    def encode(self, x):
        return self.encoder(self.pos_enc(self.embed(x))).mean(dim=1)

    def decode(self, latente):
        h = latente.unsqueeze(1).repeat(1, self.W, 1)
        return self.salida(self.decoder(self.pos_enc(h)))

    def forward(self, x):
        return self.decode(self.encode(x))


modelo = AutoencoderTransformer(n_vars=len(VARIABLES), W=W).to(device)
X_eval_t = torch.from_numpy(X_eval)
df_err = pd.DataFrame(ids_eval, columns=['tramo_id', 'ventana_id'])
df_err['conjunto'] = conjunto_eval

for nombre, ruta in MODELOS.items():
    modelo.load_state_dict(torch.load(ruta, map_location=device))
    modelo.eval()
    errores = []
    with torch.no_grad():
        for i in range(0, len(X_eval_t), 256):
            batch = X_eval_t[i:i + 256].to(device)
            recon = modelo(batch)
            errores.extend(((recon - batch) ** 2).mean(dim=(1, 2)).cpu().tolist())
    df_err[f'err_{nombre}'] = errores
    print(f'  {nombre:<18} evaluado | error medio {np.mean(errores):.5f}')

if f'err_{MODELO_PRINCIPAL}' not in df_err.columns:
    raise SystemExit(f'Falta el modelo principal ({MODELO_PRINCIPAL}). '
                     'Corré el paso 04 antes que este.')
df_err['error_reconstruccion'] = df_err[f'err_{MODELO_PRINCIPAL}']
print(f'\nColumna error_reconstruccion = {MODELO_PRINCIPAL}')


# =============================================================================
# 3. CRUZAR CON LAS ETIQUETAS (que el modelo nunca vio)
# =============================================================================
print('\n' + '=' * 70)
print('CRUZANDO CON LAS ETIQUETAS DEL MOTOR DE REGLAS')
print('=' * 70)

ventanas = pd.read_csv(ARCHIVO_VENTANAS)
cols_cruce = ['tramo_id', 'ventana_id', 'etiqueta_ventana', 'etiqueta_tfidf',
              'total_alarmas_ventana', 'total_eventos_ventana',
              'n_reglas_activas', 'banda_severidad', 'concentracion', 'persistencia']
cols_cruce = [c for c in cols_cruce if c in ventanas.columns]

m = df_err.merge(ventanas[cols_cruce], on=['tramo_id', 'ventana_id'], how='inner')
print(f'Ventanas evaluadas: {len(df_err):,} | cruzadas con éxito: {len(m):,}')

# --- La separación, por conjunto -------------------------------------------
if 'n_reglas_activas' in m.columns:
    print('\n' + '=' * 70)
    print('SEPARACIÓN NORMAL vs ANÓMALA, POR CONJUNTO')
    print('=' * 70)
    _es_norm = (m.n_reglas_activas == 0).to_numpy()
    _err = m.error_reconstruccion.to_numpy()
    print(f'{"conjunto":<10}{"n_norm":>8}{"n_anom":>8}{"err_norm":>11}'
          f'{"err_anom":>11}{"razón":>9}{"AUC":>9}')
    for _c in ('val', 'test', 'val+test'):
        _sel = (m.conjunto == _c).to_numpy() if _c != 'val+test' else np.ones(len(m), bool)
        _n, _a = _sel & _es_norm, _sel & ~_es_norm
        if not _n.any() or not _a.any():
            continue
        _en, _ea = float(np.median(_err[_n])), float(np.median(_err[_a]))
        _r = pd.Series(_err[_sel]).rank().to_numpy()
        _mn = _es_norm[_sel]
        _np_, _nn = int((~_mn).sum()), int(_mn.sum())
        _auc = (_r[~_mn].sum() - _np_ * (_np_ + 1) / 2) / (_np_ * _nn)
        _marca = '   <- la que se defiende' if _c == 'test' else (
                 '   <- aquí se eligió el modelo' if _c == 'val' else '')
        print(f'{_c:<10}{int(_n.sum()):>8,}{int(_a.sum()):>8,}{_en:>11.5f}'
              f'{_ea:>11.5f}{_ea/_en:>8.2f}x{_auc:>9.4f}{_marca}')
    print('\nLa brecha entre val y test mide cuánto sesgó la selección del')
    print('checkpoint. Si el checkpoint por separación cae mucho más que el de')
    print('pérdida, eso confirma que elegir maximizando la razón en validación')
    print('sobreajusta esa métrica, y es un resultado para declarar.')

if len(m) == 0:
    raise SystemExit('\n0 coincidencias. Casi seguro es la re-numeración de tramo_id. '
                     'Revisa que el paso 02 y este usen el mismo criterio.')
if len(m) < len(df_err) * 0.99:
    print('⚠ Se perdieron filas en el merge. Revisa los tramo_id / ventana_id.')

m['es_normal'] = m['etiqueta_ventana'] == 'normal'


# =============================================================================
# 4. CRITERIO NORMAL VS. ANÓMALO
# =============================================================================
print('\n' + '=' * 70)
print('4. CRITERIO DEL PROFESOR: ¿el error es mayor en las ventanas malas?')
print('=' * 70)

resumen = m.groupby('es_normal')['error_reconstruccion'].agg(['count', 'mean', 'median', 'std'])
resumen.index = ['Anómala', 'Normal']
print(resumen.round(5).to_string())

err_normal = m.loc[m.es_normal, 'error_reconstruccion']
err_anomal = m.loc[~m.es_normal, 'error_reconstruccion']
razon_media = err_anomal.mean() / err_normal.mean()
razon_mediana = err_anomal.median() / err_normal.median()

print(f'\nRazón anómalo/normal en PROMEDIOS: {razon_media:.2f}x   <- referencia: 3,43x')
print(f'Razón anómalo/normal en MEDIANAS : {razon_mediana:.2f}x   <- referencia: 3,46x')
print('\nQue las dos razones sean parecidas es lo importante: significa que la')
print('separación NO la producen unos pocos casos extremos, sino que toda la')
print('distribución está corrida.')

print(f'\nVarianza recuperada — normales: {(1 - err_normal.mean()) * 100:.1f} % | '
      f'anómalas: {(1 - err_anomal.mean()) * 100:.1f} %')

# El umbral del percentil 90
p90 = err_normal.quantile(0.90)
capturadas = (err_anomal > p90).mean() * 100
falsas = (err_normal > p90).mean() * 100
print(f'\nUmbral en el percentil 90 de las normales = {p90:.5f}')
print(f'  captura el {capturadas:.1f} % de las ventanas anómalas')
print(f'  con {falsas:.1f} % de falsas alarmas')

corr = m['error_reconstruccion'].corr(m['total_alarmas_ventana'])
print(f'\nCorrelación error vs. total de alarmas: {corr:.3f}')
print('  (baja es lo esperado: el modelo detecta rareza, no cantidad)')


# =============================================================================
# 5. VALIDACIÓN DE LAS BANDAS DE SEVERIDAD
# =============================================================================
if 'banda_severidad' in m.columns:
    print('\n' + '=' * 70)
    print('5. VALIDACIÓN DE LAS BANDAS DE SEVERIDAD')
    print('=' * 70)

    g = m.groupby('banda_severidad').agg(
        ventanas=('error_reconstruccion', 'size'),
        err_mediano=('error_reconstruccion', 'median'),
        err_medio=('error_reconstruccion', 'mean'),
        alarmas_prom=('total_alarmas_ventana', 'mean'),
    ).round(5).sort_index()
    print(g.to_string())

    v = g.err_mediano.values
    monotono = all(v[i] <= v[i + 1] for i in range(len(v) - 1))
    print(f'\n¿El error crece de forma monótona entre bandas? -> '
          f'{"SÍ ✓" if monotono else "NO ✗"}')
    if monotono:
        salto = g.err_medio.iloc[-1] / g.err_medio.iloc[0]
        print(f'De la banda normal a la grave el error promedio se multiplica por {salto:.1f}x')
        print('Un modelo sin etiquetas confirma el criterio de severidad por su cuenta.')

    # --- comparación con los índices alternativos ------------------------
    print('\n--- ¿Y los otros índices logran el mismo orden? ---')
    anom = m[m.total_alarmas_ventana > 0]
    candidatos = {}
    c2 = np.percentile(anom.total_alarmas_ventana, [33.3, 66.6])
    candidatos['total_alarmas'] = pd.cut(
        m.total_alarmas_ventana, [-1, 0, c2[0], c2[1], 1e9],
        labels=['0 normal', 'bajo', 'medio', 'alto'])
    if 'total_eventos_ventana' in m.columns:
        c3 = np.percentile(anom.total_eventos_ventana, [33.3, 66.6])
        candidatos['total_eventos'] = pd.cut(
            m.total_eventos_ventana, [-1, 0, max(c3[0], 0.5), max(c3[1], 1.5), 1e9],
            labels=['0 normal', 'bajo', 'medio', 'alto'], duplicates='drop')

    for nombre, serie in candidatos.items():
        med = m.groupby(serie, observed=True).error_reconstruccion.median()
        vv = med.values
        ok = all(vv[i] <= vv[i + 1] for i in range(len(vv) - 1))
        print(f'  {nombre:16s}: {" -> ".join(f"{x:.4f}" for x in vv)}   '
              f'{"ordena ✓" if ok else "NO ordena ✗"}')


m.to_csv(ARCHIVO_SALIDA, index=False)
print(f'\nGuardado: {ARCHIVO_SALIDA}')

plt.figure(figsize=(6, 5))
plt.boxplot([err_normal, err_anomal], tick_labels=['Normal', 'Anómala'], showfliers=False)
plt.ylabel('Error de reconstrucción (MSE)')
plt.title(f'Separación normal vs. anómala — W={W}  ({razon_media:.2f}x)')
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{RUTA_EQ}/05_separacion_normal_anomalo.png', dpi=150)
plt.show()

if 'banda_severidad' in m.columns:
    orden = sorted(m.banda_severidad.unique())
    datos = [m.loc[m.banda_severidad == b, 'error_reconstruccion'] for b in orden]
    plt.figure(figsize=(8, 5))
    plt.boxplot(datos, tick_labels=orden, showfliers=False)
    plt.ylabel('Error de reconstrucción (MSE)')
    plt.xlabel('Banda de severidad (nº de reglas distintas activas)')
    plt.title('Validación de las bandas — el autoencoder nunca vio estas etiquetas')
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{RUTA_EQ}/05_validacion_bandas.png', dpi=150)
    plt.show()

print('\n' + '=' * 70)
print('FASE A COMPLETA. Siguiente: 06_fase_b.py (el clasificador)')
print('=' * 70)

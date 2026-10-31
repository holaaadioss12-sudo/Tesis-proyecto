# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 01 — PREPROCESAMIENTO
=============================================================================
Entrada:  Dataframe - CR010.csv   (el crudo del PI System, ~5M registros)
Salida:   CR010_limpio.csv        (~1,18M registros, 14 variables)
"""

# --- Conectar al Drive (Colab) -----------------------------------------
try:
    from google.colab import drive
    drive.mount('/content/drive')
    RUTA_BASE = '/content/drive/MyDrive/tesis_chancadores'
except ImportError:                  # fuera de Colab
    RUTA_BASE = __import__('os').environ.get('RUTA_BASE', '.')

import os as _os
EQUIPO = _os.environ.get('EQUIPO', 'CR010')
RUTA_EQ = RUTA_BASE if EQUIPO == 'CR010' else f'{RUTA_BASE}/{EQUIPO}'
if EQUIPO == 'CR010' and _os.path.isdir(f'{RUTA_BASE}/CR010'):
    RUTA_EQ = f'{RUTA_BASE}/CR010'     # CR010 en su subcarpeta, como en los demás pasos
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

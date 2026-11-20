# -*- coding: utf-8 -*-
"""
=============================================================================
limpieza_config.py  -  los criterios de limpieza, en un solo lugar (pasos 01 y 01b)
=============================================================================
Lo pedido en la reunión del 20 de noviembre: "metodológicamente, primero se elimina lo
atípico". Con el chancador andando (CM > 25), el paso 01 aplica, en este orden, seis criterios
contra los errores de medición, y recién después el filtro estadístico de N_DESV desviaciones:
  1. Límites físicos: lo imposible (una temperatura bajo 2 °C, un estanque de aceite a 160 °C).
  2. Sensor pegado: exactamente el mismo valor durante PEGADO_HORAS o más.
  3. Picos aislados: la lectura que "se dispara" y vuelve en 30 s o menos, y que pasa como
     alarma sin serlo (la presión que mencionó el profesor). Solo temperaturas y presiones.
  4. Consistencia física: el metal no puede quedar más frío que el aceite que lo enfría (T8 de
     CR009 en enero de 2023).
  5. Arranques: los primeros minutos después de cada partida, mientras se estabilizan el motor
     y la bomba de aceite.
  6. Períodos malos: los que se decidan a mano al revisar el paso 01b.
Lo anulado por 1 a 4 y 6 queda como hueco: si es corto (picos), el paso 01 lo interpola; si es
largo, se botan esas filas, porque el autoencoder necesita las 14 variables. Si se cambia algo
aquí, hay que volver a correr el paso 01 y todo lo que viene después.
"""
import re

import numpy as np
import pandas as pd

VARIABLES = ['CM', 'PI', 'PDF', 'PEL', 'T7', 'T8', 'T9', 'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4']

# --- filtro estadístico: media ± N_DESV desviaciones, ajustadas solo con entrenamiento -----
N_DESV = 6

# --- 1. límites físicos: fuera de ellos el valor es imposible y se anula ----------------------
# (mínimo, máximo). Temperaturas: 2 °C como mínimo, como propuso el profesor; el aceite no
# llega a 100 °C y el metal (buje, socket liner, contraeje) no pasa de 130 °C.
LIMITES_FISICOS = {
    'CM':  (0, 150),     # % de la corriente nominal (alarmas en 100 y 110)
    'PI':  (0, 100),     # % de nivel de la cámara
    'PDF': (0, 1000),    # kPa (alarma en 400)
    'PEL': (0, 1000),    # kPa (alarmas en 150, 350 y 450)
    'T7':  (2, 130), 'T8': (2, 130), 'T9': (2, 130),     # °C, metal
    'T1':  (2, 100), 'T2': (2, 100), 'T5': (2, 100),     # °C, aceite
    'V1':  (0, 100), 'V2': (0, 100), 'V3': (0, 100), 'V4': (0, 100),   # mm/s (alarmas hasta 40)
}

# --- 2. sensor pegado: el mismo valor exacto durante horas no es una medición -----------------
PEGADO_HORAS = 6.0

# --- 3. picos aislados (filtro de Hampel) -------------------------------------------------
# Cada valor se compara con la mediana de los 5 minutos que lo rodean. Si se aleja más de k
# desviaciones robustas locales (1,4826 · MAD), y por lo menos el mínimo de la variable, y el
# salto dura como máximo max_registros registros seguidos (30 s), es una lectura mala: se anula
# y el paso 01 la rellena con sus vecinos. Si dura más, es un cambio real y se deja. Solo en
# temperaturas y presiones, que tienen inercia; la corriente, el nivel y las vibraciones sí
# tienen picos reales (impactos, sobrecargas) y no se tocan.
PICOS = {'variables': ['PDF', 'PEL', 'T7', 'T8', 'T9', 'T1', 'T2', 'T5'], 'ventana': '5min', 'k': 6.0,
         'max_registros': 3,
         'minimo': {'PDF': 20.0, 'PEL': 20.0, 'T7': 2.0, 'T8': 2.0, 'T9': 2.0, 'T1': 2.0, 'T2': 2.0, 'T5': 2.0}}

# --- 4. consistencia física: el metal no puede estar más frío que el aceite que lo enfría ------
# El aceite entra al chancador (T2) y enfría el metal: el buje excéntrico (T7), el socket liner
# (T8) y el contraeje (T9) no pueden quedar, en la mediana de un día con el chancador andando,
# más de 'margen' °C bajo el aceite de entrada. Si además la variable se corrió de su propio
# nivel (z de la mediana del día bajo -z), es una falla del sensor, como T8 de CR009 en enero de
# 2023 (26,8 °C con el aceite entrando a 39 °C), y se anula esa variable ese día.
CONSISTENCIA = {'referencia': 'T2', 'variables': ['T7', 'T8', 'T9'], 'margen': 5.0, 'z': 3.0}

# --- 5. arranques: se descartan los primeros minutos después de cada partida -----------------
# Una partida es el primer registro con el chancador andando después de un hueco de más de
# HUECO_PARTIDA_S segundos (una parada, o datos que faltan).
ARRANQUE_MIN = 2.0
HUECO_PARTIDA_S = 60

# --- 6. períodos en que un sensor midió mal: se anula esa variable en ese período --------------
# (equipo, variable, desde, hasta, motivo), para lo que los criterios anteriores no ven y se
# decida cortar al revisar el paso 01b.
PERIODOS_MALOS = [
]


def dias_inconsistentes(df, cfg=None):
    """Días en que una temperatura de metal quedó bajo el aceite de entrada y corrida de su nivel."""
    cfg = CONSISTENCIA if cfg is None else cfg
    columnas = ['variable', 'dia', 'mediana', 'mediana_referencia', 'diferencia', 'z']
    if not cfg or cfg.get('referencia') not in df.columns:
        return pd.DataFrame(columns=columnas)
    ref = cfg['referencia']
    dia = df.index.normalize()
    med_ref = df[ref].groupby(dia).median()
    filas = []
    for v in cfg['variables']:
        if v not in df.columns:
            continue
        med = df[v].groupby(dia).median().dropna()
        centro = med.median()
        escala = 1.4826 * (med - centro).abs().median()
        escala = escala if escala > 0 else med.std()
        z = (med - centro) / escala
        dif = med - med_ref.reindex(med.index)
        malo = (dif < -cfg['margen']) & (z < -cfg['z'])
        for d in med.index[malo.fillna(False).to_numpy()]:
            filas.append({'variable': v, 'dia': d, 'mediana': med[d], 'mediana_referencia': med_ref[d],
                          'diferencia': dif[d], 'z': z[d]})
    return pd.DataFrame(filas, columns=columnas)


def sensores_pegados(df, horas=None, variables=None):
    """{variable: máscara} de los valores que repiten exactamente el anterior durante 'horas' o más."""
    horas = PEGADO_HORAS if horas is None else horas
    out = {}
    if not horas:
        return out
    t = pd.Series(df.index, index=df.index)
    for v in (VARIABLES if variables is None else variables):
        if v not in df.columns:
            continue
        x = df[v]
        grupo = (~(x.notna() & (x == x.shift()))).cumsum().to_numpy()
        dur = t.groupby(grupo).transform('max') - t.groupby(grupo).transform('min')
        out[v] = ((dur.dt.total_seconds() >= 3600 * horas) & x.notna()).to_numpy()
    return out


def picos_aislados(df, cfg=None):
    """{variable: máscara} de las lecturas que se disparan y vuelven en pocos registros (Hampel)."""
    cfg = PICOS if cfg is None else cfg
    out = {}
    for v in (cfg or {}).get('variables', []):
        if v not in df.columns:
            continue
        x = df[v]
        med = x.rolling(cfg['ventana'], center=True, min_periods=5).median()
        dev = (x - med).abs()
        mad = dev.rolling(cfg['ventana'], center=True, min_periods=5).median()
        pico = (dev > np.maximum(cfg['k'] * 1.4826 * mad, cfg['minimo'].get(v, 0.0))).to_numpy()
        s = pd.Series(pico)
        largo = s.groupby((s != s.shift()).cumsum()).transform('size').to_numpy()
        out[v] = pico & (largo <= cfg['max_registros'])
    return out


def tras_arranque(df, minutos=None, hueco_s=None):
    """Máscara de los registros de los primeros 'minutos' después de cada partida."""
    minutos = ARRANQUE_MIN if minutos is None else minutos
    hueco_s = HUECO_PARTIDA_S if hueco_s is None else hueco_s
    if not minutos:
        return np.zeros(len(df), dtype=bool)
    t = df.index.to_series()
    partida = t.diff().dt.total_seconds().fillna(np.inf) > hueco_s
    desde = (t - t.where(partida).ffill()).dt.total_seconds()
    return (desde < 60 * minutos).to_numpy()


def aplicar_limites(df, equipo, limites=None, periodos=None, consistencia=None, picos=None,
                    pegado_horas=None, arranque_min=None):
    """Aplica los seis criterios en orden. Devuelve el df corregido y una tabla con lo anulado
    (criterio, variable, anulados, % del registro y detalle); los días de la regla de consistencia
    quedan en tabla.attrs['dias_inconsistentes']."""
    limites = LIMITES_FISICOS if limites is None else limites
    periodos = PERIODOS_MALOS if periodos is None else periodos
    cfg_c = CONSISTENCIA if consistencia is None else consistencia
    cfg_p = PICOS if picos is None else picos
    horas = PEGADO_HORAS if pegado_horas is None else pegado_horas
    df = df.copy()
    filas = []

    def anular(criterio, v, mascara, detalle):
        mascara = np.asarray(mascara, dtype=bool)
        n = int((mascara & df[v].notna().to_numpy()).sum())
        if n:
            filas.append({'criterio': criterio, 'variable': v, 'anulados': n, 'pct_del_registro': 100 * n / len(df),
                          'detalle': detalle})
            df.loc[mascara, v] = np.nan

    for v, (lo, hi) in limites.items():
        if v in df.columns:
            anular('1. límite físico', v, (df[v] < lo) | (df[v] > hi), f'fuera de {lo} a {hi}')
    for v, m in sensores_pegados(df, horas).items():
        anular('2. sensor pegado', v, m, f'mismo valor exacto {horas:g} h o más')
    if cfg_p:
        for v, m in picos_aislados(df, cfg_p).items():
            anular('3. pico aislado', v, m, f'se aleja más de {cfg_p["k"]:g} desviaciones de la mediana de 5 min '
                                            f'y vuelve en {10 * cfg_p["max_registros"]} s o menos')
    malos = dias_inconsistentes(df, cfg_c)
    if len(malos):
        dia = df.index.normalize()
        for v, g in malos.groupby('variable'):
            anular('4. consistencia física', v, dia.isin(g.dia),
                   f'{len(g)} días entre {g.dia.min().date()} y {g.dia.max().date()}: más de {cfg_c["margen"]:g} °C '
                   f'bajo el aceite de entrada ({cfg_c["referencia"]}) y corrida de su nivel')
    arr = tras_arranque(df, arranque_min)
    if arr.any():
        n_part = int((pd.Series(arr).astype(int).diff() == 1).sum() + int(arr[0]))
        for v in [c for c in VARIABLES if c in df.columns]:
            anular('5. arranque', v, arr, f'primeros {ARRANQUE_MIN if arranque_min is None else arranque_min:g} min '
                                          f'de {n_part:,} partidas')
    for eq, v, desde, hasta, motivo in periodos:
        if eq == equipo and v in df.columns:
            anular('6. período malo', v, (df.index >= pd.Timestamp(desde)) & (df.index <= pd.Timestamp(hasta)),
                   f'{desde} a {hasta}: {motivo}')
    tabla = pd.DataFrame(filas, columns=['criterio', 'variable', 'anulados', 'pct_del_registro', 'detalle'])
    tabla.attrs['dias_inconsistentes'] = malos
    return df, tabla


# --- los nombres de las columnas del crudo (el mismo mapeo del paso 01) ---------------------
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
    """Renombra las columnas del crudo a CM, PI, ..., V4 y devuelve también el tag original."""
    nuevos, originales = {}, {}
    for col in df.columns:
        if col.strip().lower() == 'time':
            continue
        canonico = CANONICAL_LOOKUP.get(parsear_tag(col))
        if canonico:
            nuevos[col] = canonico
            originales[canonico] = col
    return df.rename(columns=nuevos), originales

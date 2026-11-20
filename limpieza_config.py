# -*- coding: utf-8 -*-
"""
=============================================================================
limpieza_config.py  -  los criterios de limpieza, en un solo lugar (pasos 01 y 01b)
=============================================================================
Lo pedido en la reunión del 20 de noviembre: primero se elimina lo atípico, con un filtro
de 6 desviaciones estándar y con límites físicos para lo que es imposible (una temperatura
bajo 2 °C o un estanque de aceite a 160 °C es un error de medición), y se revisa cada
variable antes de entrenar. Si se cambia algo aquí, hay que volver a correr el paso 01 y
todo lo que viene después.
"""
import re

import numpy as np
import pandas as pd

# --- filtro estadístico: media ± N_DESV desviaciones, ajustadas solo con entrenamiento -----
N_DESV = 6

# --- límites físicos: fuera de ellos el valor es imposible y se anula ----------------------
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

# --- consistencia física: el metal no puede estar más frío que el aceite que lo enfría ---------
# El aceite entra al chancador (T2) y enfría el metal: el buje excéntrico (T7), el socket liner
# (T8) y el contraeje (T9) no pueden quedar, en la mediana de un día con el chancador andando,
# más de MARGEN °C bajo el aceite de entrada. Si además la variable se corrió de su propio nivel
# (z de la mediana del día bajo -Z), es una falla del sensor, como T8 de CR009 en enero de 2023
# (26,8 °C con el aceite entrando a 39 °C), y se anula esa variable ese día.
CONSISTENCIA = {'referencia': 'T2', 'variables': ['T7', 'T8', 'T9'], 'margen': 5.0, 'z': 3.0}

# --- períodos en que un sensor midió mal: se anula esa variable en ese período --------------
# (equipo, variable, desde, hasta, motivo), para lo que la regla de consistencia no ve y se
# decida cortar al revisar el paso 01b. Las filas que quedan sin esa variable se botan en el
# paso 01, como cualquier hueco largo, porque el autoencoder necesita las 14 variables.
PERIODOS_MALOS = [
]


def dias_inconsistentes(df, cfg=None):
    """Días en que una temperatura de metal quedó bajo el aceite de entrada y corrida de su nivel."""
    cfg = CONSISTENCIA if cfg is None else cfg
    if not cfg or cfg.get('referencia') not in df.columns:
        return pd.DataFrame(columns=['variable', 'dia', 'mediana', 'mediana_referencia', 'diferencia', 'z'])
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
    return pd.DataFrame(filas, columns=['variable', 'dia', 'mediana', 'mediana_referencia', 'diferencia', 'z'])


def aplicar_limites(df, equipo, limites=None, periodos=None, consistencia=None):
    """Anula los valores fuera de los límites físicos, los días que fallan la consistencia física y
    los períodos malos del equipo. Devuelve el df corregido y una tabla con lo anulado."""
    limites = LIMITES_FISICOS if limites is None else limites
    periodos = PERIODOS_MALOS if periodos is None else periodos
    df = df.copy()
    filas = []
    for col, (lo, hi) in limites.items():
        if col not in df.columns:
            continue
        x = df[col]
        bajo, alto = x < lo, x > hi
        filas.append({'variable': col, 'limite_min': lo, 'limite_max': hi,
                      'bajo_el_minimo': int(bajo.sum()), 'sobre_el_maximo': int(alto.sum()),
                      'pct_anulado': 100 * float((bajo | alto).mean())})
        df.loc[bajo | alto, col] = np.nan
    cfg = CONSISTENCIA if consistencia is None else consistencia
    malos = dias_inconsistentes(df, cfg)
    if len(malos):
        dia = df.index.normalize()
        for v, g in malos.groupby('variable'):
            en = dia.isin(g.dia)
            filas.append({'variable': v, 'periodo': f'{len(g)} días entre {g.dia.min().date()} y {g.dia.max().date()}',
                          'motivo': f'más de {cfg["margen"]:.0f} °C bajo el aceite de entrada ({cfg["referencia"]}) y '
                                    'corrida de su nivel: falla del sensor',
                          'anulados_por_consistencia': int(df.loc[en, v].notna().sum())})
            df.loc[en, v] = np.nan
    for eq, col, desde, hasta, motivo in periodos:
        if eq != equipo or col not in df.columns:
            continue
        en = (df.index >= pd.Timestamp(desde)) & (df.index <= pd.Timestamp(hasta))
        filas.append({'variable': col, 'periodo': f'{desde} a {hasta}', 'motivo': motivo,
                      'anulados_por_periodo': int(df.loc[en, col].notna().sum())})
        df.loc[en, col] = np.nan
    return df, pd.DataFrame(filas)


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

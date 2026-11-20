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

# --- períodos en que un sensor midió mal: se anula esa variable en ese período --------------
# (equipo, variable, desde, hasta, motivo). Se llena después de revisar el paso 01b con el
# profesor. Las filas que quedan sin esa variable se botan en el paso 01, como cualquier hueco
# largo, porque el autoencoder necesita las 14 variables.
PERIODOS_MALOS = [
    # ('CR009', 'T8', '2023-01-01', '2023-01-31 23:59:59',
    #  'T8 unos 12 °C bajo el aceite de entrada: falla del sensor, por confirmar'),
]


def aplicar_limites(df, equipo, limites=None, periodos=None):
    """Anula los valores fuera de los límites físicos y los de los períodos malos del equipo.
    Devuelve el df corregido y una tabla con lo anulado por variable."""
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

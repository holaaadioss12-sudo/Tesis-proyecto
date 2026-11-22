# -*- coding: utf-8 -*-
"""
=============================================================================
limpieza_config.py  -  los criterios de limpieza, en un solo lugar (pasos 01 y 01b)
=============================================================================
Lo pedido por el profesor: "metodológicamente, primero se elimina lo atípico", con una buena
limpieza de todas las variables de los tres equipos. Con el chancador andando (CM > 25), el paso 01
revisa cada una de las 14 variables contra cada tipo de error de medición, y recién después aplica
el filtro estadístico de N_DESV desviaciones:
  1. Límites físicos: lo imposible para la física o para el sensor (una temperatura bajo 2 °C, un
     estanque de aceite a 160 °C, una vibración de 0 con el chancador andando).
  2. Sensor pegado: exactamente el mismo valor durante PEGADO_MIN minutos de operación continua.
  3. Picos aislados: la lectura que se dispara y vuelve en 30 s o menos (la presión que mencionó el
     profesor, el nivel que salta a 125 % y vuelve). En las vibraciones, solo si salta un sensor y
     los otros tres no se mueven: un golpe real se siente en más de uno.
     3b. Saltos imposibles: una temperatura que cambia más de 10 °C en un minuto (T8 de CR009 el
     18 de noviembre de 2022: de 64 a 95 °C y de vuelta, minuto a minuto).
  4. Consistencia física: el sensor que contradice a otro ligado por la física y además se corrió
     de su propio nivel (el metal más frío que el aceite que lo enfría, como T8 de CR009 en enero de
     2023; el aceite que vuelve más frío de lo que entró; el estanque lejos del aceite de retorno).
  5. Arranques: los primeros minutos después de cada partida.
  6. Períodos malos: los que se decidan a mano al revisar el paso 01b.
Qué NO se corta: la operación poco común pero real. Una partida en frío (el metal a 31 °C, el
aceite frío y la presión alta, todo coherente), los días de poca carga, el filtro que se va tapando
o las cuatro vibraciones subiendo juntas son justamente lo que el modelo debe aprender o detectar:
el 01b los muestra para revisarlos, pero no se borran.
Lo anulado queda como hueco: si es corto (picos), el paso 01 lo interpola; si es largo, se botan
esas filas, porque el autoencoder necesita las 14 variables. Si se cambia algo aquí, hay que volver
a correr el paso 01 y todo lo que viene después.

Qué criterio se aplica a cada variable (el 5 y el 6, a todas):
              1. límite  2. pegado  3. pico            3b. salto  4. consistencia
  CM          sí         sí         no (*)             no         no
  PI          sí         sí         sí                 no         no
  PDF, PEL    sí         sí         sí                 no         no
  T7, T8, T9  sí         sí         sí                 sí         con el aceite de entrada (T2)
  T1, T2, T5  sí         sí         sí                 sí         retorno con entrada y estanque
  V1 a V4     sí         sí         si salta uno solo  no         no
(*) Los picos de corriente son sobrecargas reales del motor. El salto (3b) es solo para las
temperaturas, que tienen inercia térmica: la corriente, el nivel, las presiones y las vibraciones
sí cambian rápido de verdad.
"""
import re

import numpy as np
import pandas as pd

VARIABLES = ['CM', 'PI', 'PDF', 'PEL', 'T7', 'T8', 'T9', 'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4']

# --- filtro estadístico: media ± N_DESV desviaciones, ajustadas solo con entrenamiento -----
N_DESV = 6

# --- 1. fuera de rango: fuera de estos límites el valor es imposible y se anula --------------
# (mínimo, máximo). Temperaturas: 2 °C como mínimo, como propuso el profesor; el aceite no
# llega a 100 °C y el metal (buje, socket liner, contraeje) no pasa de 130 °C.
LIMITES_FISICOS = {
    'CM':  (0, 150),     # % de la corriente nominal (alarmas en 100 y 110)
    'PI':  (-1, 126),    # % de nivel de la cámara (ver abajo)
    'PDF': (0, 1000),    # kPa (alarma en 400)
    'PEL': (0, 1000),    # kPa (alarmas en 150, 350 y 450)
    'T7':  (2, 130), 'T8': (2, 130), 'T9': (2, 130),     # °C, metal
    'T1':  (2, 100), 'T2': (2, 100), 'T5': (2, 100),     # °C, aceite
    'V1':  (0.2, 100), 'V2': (0.2, 100), 'V3': (0.2, 100), 'V4': (0.2, 100),   # mm/s (alarmas hasta 40)
}
# El nivel no se corta en 100 %: el sensor mide hasta 125 % (los tres equipos topan ahí) y sobre
# 100 % los valores se reparten hasta ese tope, no quedan pegados en uno solo. Es la cámara más
# llena que su 100 % de calibración, no un error, y cortarlo botaba el 4,4 % de CR010 justo en
# los tramos de nivel alto (los de la alarma de atollo). Bajo 0 hay solo ruido de centésimas
# con la cámara vacía.
# Vibraciones: con el chancador andando, el motor y el contraeje giran; el mínimo es de 0,4 a 0,7
# mm/s en CR010 y CR011. En CR009, el 26 de mayo de 2022 las cuatro vibraciones marcan cero todo el
# día (mediana entre -0,08 y 0,05 mm/s) con el chancador andando: es el sistema de vibraciones
# desconectado, no una máquina sin vibración.

# --- 2. sensor pegado: el mismo valor exacto durante media hora no es una medición -----------
# Con registros cada 10 s, 30 min son 180 lecturas idénticas seguidas. Con el chancador andando
# ninguna variable se queda así: las temperaturas fluctúan en centésimas y la corriente, el nivel,
# las presiones y las vibraciones cambian en cada registro. En CR010 y CR011 no hay ni una hora de
# operación en que alguna de las 14 variables repita el mismo valor de principio a fin. La racha se mide sin cruzar
# detenciones (un hueco de más de HUECO_PARTIDA_S la corta), para que el mismo valor antes y
# después de una parada larga no cuente como sensor pegado.
PEGADO_MIN = 30.0
PEGADO_HORAS = PEGADO_MIN / 60      # el nombre antiguo, por compatibilidad

# --- 3. picos aislados (filtro de Hampel) -------------------------------------------------
# Cada valor se compara con la mediana de los 5 minutos que lo rodean. Si se aleja más de k
# desviaciones robustas locales (1,4826 · MAD), y por lo menos el mínimo de la variable, y el
# salto dura como máximo max_registros registros seguidos (30 s), es una lectura mala: se anula
# y el paso 01 la rellena con sus vecinos. Si dura más, es un cambio real y se deja.
#   - Temperaturas y presiones: tienen inercia; no se disparan 30 s y vuelven.
#   - Nivel (PI): la cámara no se llena 30 puntos y se vacía en 30 s; eso es un eco falso del
#     radar (una roca o polvo que cruza el haz). En CR010 hay 96 horas con la cámara vacía en que el
#     nivel marca más de 120 % en algún momento; si dura 30 s o menos, se corta.
#   - Vibraciones: un golpe real (una roca dura, un fierro) sacude la máquina y se siente en más de
#     un sensor. Por eso el pico de una vibración solo se anula si ninguna de las otras tres se
#     aleja más de 'acompanante_max' mm/s de su mediana en ese momento: entonces es el sensor.
#   - Corriente: no se toca. Sus picos son sobrecargas reales del motor, y una caída bajo 25 %
#     ya se trata como una detención.
PICOS = {'variables': ['PI', 'PDF', 'PEL', 'T7', 'T8', 'T9', 'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4'],
         'ventana': '5min', 'k': 6.0, 'max_registros': 3,
         'minimo': {'PI': 30.0, 'PDF': 20.0, 'PEL': 20.0, 'T7': 2.0, 'T8': 2.0, 'T9': 2.0, 'T1': 2.0,
                    'T2': 2.0, 'T5': 2.0, 'V1': 10.0, 'V2': 10.0, 'V3': 10.0, 'V4': 10.0},
         'acompanantes': {'V1': ['V2', 'V3', 'V4'], 'V2': ['V1', 'V3', 'V4'], 'V3': ['V1', 'V2', 'V4'],
                          'V4': ['V1', 'V2', 'V3']},
         'acompanante_max': 3.0}

# --- 3b. saltos imposibles: una temperatura no cambia 10 °C en un minuto ------------------------
# El metal y el aceite tienen inercia térmica: con el chancador andando, una temperatura cambia
# décimas de grado por minuto (al partir en frío, el buje sube cerca de medio grado por minuto).
# Si en un minuto se mueve más de 'max_c_min' °C (lo más alto menos lo más bajo de ese minuto), el
# que falla es el sensor, aunque el salto dure más de 30 s y el criterio 3 no lo vea. Pasó con T8
# de CR009 el 18 de noviembre de 2022: durante 13 minutos osciló entre 64 y 95 °C de un minuto a
# otro, con el aceite de entrada (T2) y de retorno (T1) estables, y después de la detención marcaba
# 36 °C. También con T8 de CR011 el 4 y 5 de noviembre de 2021 (saltos a 80 y 93 °C durante horas,
# con las otras temperaturas quietas) y con T9 de CR011 (caídas a 6 °C y saltos a 227 °C). Se anula
# esa variable desde 'margen_min' minutos antes del minuto del primer salto hasta 'margen_min'
# minutos después del último: antes de un salto la lectura suele irse de a poco (el 15 de julio
# de 2022, T8 de CR009 ya marcaba 52 °C un par de minutos antes del salto que se detectó). Saltos
# separados por menos de 'union_min' minutos son un mismo episodio. Y si una variable salta
# 'dia_completo' veces o más en un mismo día, el sensor estaba fallando y no es confiable entre un
# salto y otro: se anula el día completo (T8 de CR009 el 25 de diciembre de 2022, con 4 saltos y
# lecturas de 54 y 27 °C entre ellos, la víspera de los 41 días bajo el aceite; T8 de CR011 del 4 al
# 6 de noviembre de 2021, con 35 saltos). Los minutos de arranque no se miran: los corta el 5.
SALTOS = {'variables': ['T7', 'T8', 'T9', 'T1', 'T2', 'T5'], 'max_c_min': 10.0, 'union_min': 5.0,
          'margen_min': 3.0, 'dia_completo': 3}

# --- 4. consistencia física: un sensor no puede contradecir a otro ligado por la física ------------
# El aceite sale del estanque (T5), se enfría, entra al chancador (T2), enfría el metal (el buje
# excéntrico T7, el socket liner T8 y el contraeje T9), vuelve más caliente (T1) y cae de nuevo al
# estanque. De ahí salen las reglas, en la mediana de cada día con el chancador andando:
#   - el metal no puede quedar más de 5 °C bajo el aceite que lo enfría;
#   - el aceite no puede volver más frío de lo que entró (en CR010 y CR011, el retorno nunca queda
#     más de 0,2 °C bajo la entrada en una hora, y en la mediana vuelve 6 a 7 °C más caliente);
#   - el estanque y el retorno son el mismo aceite: en CR010 y CR011, la mitad de las horas se
#     separan menos de 0,3 °C, y nunca más de 8,3 °C.
# Una regla sola no basta para saber cuál de los dos sensores falla: se anula la variable solo si
# además se corrió de su propio nivel (z de la mediana del día más allá de 'z', en el sentido de la
# contradicción). Así pasó con T8 de CR009 en enero de 2023 (26,8 °C con el aceite entrando a 39 °C).
# (variable, referencia, sentido, margen en °C, por qué): 'bajo' = la variable no puede quedar más
# de 'margen' bajo la referencia; 'sobre' = no puede quedar más de 'margen' sobre ella; 'ambos'.
CONSISTENCIA = {'z': 3.0, 'reglas': [
    ('T7', 'T2', 'bajo', 5.0, 'el metal más frío que el aceite que lo enfría (T2)'),
    ('T8', 'T2', 'bajo', 5.0, 'el metal más frío que el aceite que lo enfría (T2)'),
    ('T9', 'T2', 'bajo', 5.0, 'el metal más frío que el aceite que lo enfría (T2)'),
    ('T1', 'T2', 'bajo', 3.0, 'el aceite vuelve más frío de lo que entró (T2)'),
    ('T2', 'T1', 'sobre', 3.0, 'el aceite entra más caliente de lo que vuelve (T1)'),
    ('T5', 'T1', 'ambos', 10.0, 'el estanque lejos del aceite de retorno (T1)'),
    ('T1', 'T5', 'ambos', 10.0, 'el retorno lejos del aceite del estanque (T5)'),
]}

# --- 5. arranques: se descartan los primeros minutos después de cada partida -----------------
# Una partida es el primer registro con el chancador andando después de un hueco de más de
# HUECO_PARTIDA_S segundos (una parada, o datos que faltan). El margen se midió con el paso 01b
# (sección 1b, más de mil partidas por equipo). Con registros cada 10 s la corriente no muestra
# pico de partida, pero la presión de entrada del aceite parte entre 57 y 188 kPa sobre su nivel
# y demora 230, 350 y 180 s en volver (CR009, CR010 y CR011). Con 6 minutos, las 14 variables
# de los tres equipos ya están en su nivel: la más lenta demora 350 s.
ARRANQUE_MIN = 6.0
HUECO_PARTIDA_S = 60

# --- 6. períodos en que un sensor midió mal: se anula esa variable en ese período --------------
# (equipo, variable, desde, hasta, motivo), para lo que los criterios anteriores no ven y se
# decida cortar al revisar el paso 01b. Los períodos que sugiere el 01b se revisaron el 22 de
# noviembre: casi todos son operación real (partidas en frío, días de poca carga, el filtro que
# se tapa, las vibraciones subiendo juntas), y los errores que había en ellos ya los cortan los
# criterios 1 a 4.
PERIODOS_MALOS = [
]


def dias_inconsistentes(df, cfg=None):
    """Días en que un sensor contradice a otro ligado por la física y además se corrió de su nivel.
    Una fila por variable y día, con el motivo (si dos reglas marcan el mismo día, van juntas)."""
    cfg = CONSISTENCIA if cfg is None else cfg
    columnas = ['variable', 'dia', 'mediana', 'referencia', 'mediana_referencia', 'diferencia', 'z', 'motivo']
    if not cfg or len(df) == 0:
        return pd.DataFrame(columns=columnas)
    dia = df.index.normalize()
    medianas = {}

    def mediana_diaria(v):
        if v not in medianas:
            medianas[v] = df[v].groupby(dia).median()
        return medianas[v]

    filas = []
    for v, ref, sentido, margen, motivo in cfg['reglas']:
        if v not in df.columns or ref not in df.columns:
            continue
        med = mediana_diaria(v).dropna()
        if len(med) == 0:
            continue
        centro = med.median()
        escala = 1.4826 * (med - centro).abs().median()
        escala = escala if escala > 0 else med.std()
        if not np.isfinite(escala) or escala <= 0:
            continue
        z = (med - centro) / escala
        dif = med - mediana_diaria(ref).reindex(med.index)
        bajo = (dif < -margen) & (z < -cfg['z'])
        sobre = (dif > margen) & (z > cfg['z'])
        malo = {'bajo': bajo, 'sobre': sobre}.get(sentido, bajo | sobre)
        for d in med.index[malo.fillna(False).to_numpy()]:
            filas.append({'variable': v, 'dia': d, 'mediana': med[d], 'referencia': ref,
                          'mediana_referencia': mediana_diaria(ref).get(d, np.nan), 'diferencia': dif[d], 'z': z[d],
                          'motivo': motivo})
    out = pd.DataFrame(filas, columns=columnas)
    if len(out):
        out = (out.groupby(['variable', 'dia'], as_index=False, sort=False)
               .agg({'mediana': 'first', 'referencia': 'first', 'mediana_referencia': 'first',
                     'diferencia': 'first', 'z': 'first', 'motivo': lambda m: '; '.join(dict.fromkeys(m))}))
    return out[columnas]


def sensores_pegados(df, minutos=None, variables=None, hueco_s=None, horas=None):
    """{variable: máscara} de los valores que repiten exactamente el anterior durante 'minutos' o más
    de operación continua (un hueco de más de 'hueco_s' segundos corta la racha)."""
    minutos = (60 * horas if horas is not None else PEGADO_MIN) if minutos is None else minutos
    hueco_s = HUECO_PARTIDA_S if hueco_s is None else hueco_s
    out = {}
    if not minutos or len(df) == 0:
        return out
    ns = df.index.asi8
    corte = np.r_[True, np.diff(ns) > hueco_s * 1e9]
    for v in (VARIABLES if variables is None else variables):
        if v not in df.columns:
            continue
        x = df[v]
        igual = (x.notna() & (x == x.shift())).to_numpy() & ~corte
        grupo = np.cumsum(~igual)
        t = pd.Series(ns).groupby(grupo)
        dur = (t.transform('max') - t.transform('min')).to_numpy() / 1e9
        out[v] = (dur >= 60 * minutos) & x.notna().to_numpy()
    return out


def picos_aislados(df, cfg=None):
    """{variable: máscara} de las lecturas que se disparan y vuelven en pocos registros (Hampel).
    En las variables con 'acompanantes' (las vibraciones), el pico se anula solo si ninguna de las
    otras se movió en ese momento (un registro antes o después)."""
    cfg = PICOS if cfg is None else cfg
    out = {}
    if not cfg:
        return out
    desvio = {}

    def desvio_local(v):
        if v not in desvio:
            x = df[v]
            med = x.rolling(cfg['ventana'], center=True, min_periods=5).median()
            desvio[v] = (x - med).abs()
        return desvio[v]

    for v in cfg.get('variables', []):
        if v not in df.columns:
            continue
        dev = desvio_local(v)
        mad = dev.rolling(cfg['ventana'], center=True, min_periods=5).median()
        pico = (dev > np.maximum(cfg['k'] * 1.4826 * mad, cfg['minimo'].get(v, 0.0))).to_numpy()
        s = pd.Series(pico)
        largo = s.groupby((s != s.shift()).cumsum()).transform('size').to_numpy()
        m = pico & (largo <= cfg['max_registros'])
        otras = [c for c in cfg.get('acompanantes', {}).get(v, []) if c in df.columns]
        if otras and m.any():
            mueve = np.zeros(len(df), dtype=bool)
            for c in otras:
                mueve |= np.nan_to_num(desvio_local(c).to_numpy(), nan=0.0) > cfg['acompanante_max']
            mueve = mueve | np.r_[mueve[1:], False] | np.r_[False, mueve[:-1]]
            tramo = np.cumsum(np.r_[True, m[1:] != m[:-1]])
            acompanado = pd.Series(mueve & m).groupby(tramo).transform('any').to_numpy()
            m = m & ~acompanado
        out[v] = m
    return out


def saltos_imposibles(df, cfg=None, excluir=None):
    """Tramos en que una temperatura cambia más rápido de lo físicamente posible.
    Devuelve ({variable: máscara}, tabla de episodios: variable, desde, hasta, minutos, salto_max)."""
    cfg = SALTOS if cfg is None else cfg
    columnas = ['variable', 'desde', 'hasta', 'minutos', 'salto_max']
    out, episodios = {}, []
    if not cfg or len(df) == 0:
        return out, pd.DataFrame(columns=columnas)
    fuera = np.zeros(len(df), dtype=bool) if excluir is None else np.asarray(excluir, dtype=bool)
    orden = None
    if not df.index.is_monotonic_increasing:          # la ventana de 60 s necesita el tiempo ordenado
        orden = np.argsort(df.index.asi8, kind='stable')
        df, fuera = df.iloc[orden], fuera[orden]
    t = df.index
    antes = pd.Timedelta(minutes=1 + cfg['margen_min'])
    despues = pd.Timedelta(minutes=cfg['margen_min'])
    for v in cfg['variables']:
        if v not in df.columns:
            continue
        x = df[v].where(~fuera)
        rango = (x.rolling('60s').max() - x.rolling('60s').min()).to_numpy()
        salto = np.nan_to_num(rango, nan=0.0) > cfg['max_c_min']
        mascara = np.zeros(len(df), dtype=bool)
        idx = np.flatnonzero(salto)
        if len(idx):
            ts = t[idx]
            nuevo = np.r_[True, np.diff(ts.asi8) > cfg['union_min'] * 60e9]
            for g in np.split(np.arange(len(idx)), np.flatnonzero(nuevo)[1:]):
                a, b = ts[g[0]] - antes, ts[g[-1]] + despues
                mascara[t.searchsorted(a, 'left'):t.searchsorted(b, 'right')] = True
                episodios.append({'variable': v, 'desde': a, 'hasta': b,
                                  'minutos': (b - a).total_seconds() / 60,
                                  'salto_max': float(np.nanmax(rango[idx[g]]))})
        out[v] = mascara & ~fuera
    if orden is not None:                              # de vuelta al orden original de las filas
        inversa = np.empty_like(orden)
        inversa[orden] = np.arange(len(orden))
        out = {v: m[inversa] for v, m in out.items()}
    return out, pd.DataFrame(episodios, columns=columnas)


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
                    pegado_horas=None, arranque_min=None, saltos=None, pegado_min=None):
    """Aplica los criterios en orden (el 3b va junto al 3). Devuelve el df corregido y una tabla con
    lo anulado (criterio, variable, anulados, % del registro y detalle); los días de la regla de
    consistencia quedan en tabla.attrs['dias_inconsistentes'] y los episodios del 3b en
    tabla.attrs['saltos']."""
    limites = LIMITES_FISICOS if limites is None else limites
    periodos = PERIODOS_MALOS if periodos is None else periodos
    cfg_c = CONSISTENCIA if consistencia is None else consistencia
    cfg_p = PICOS if picos is None else picos
    cfg_s = SALTOS if saltos is None else saltos
    if pegado_min is None:
        pegado_min = PEGADO_MIN if pegado_horas is None else 60 * pegado_horas
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
    for v, m in sensores_pegados(df, pegado_min).items():
        anular('2. sensor pegado', v, m, f'mismo valor exacto {pegado_min:g} min o más, andando sin parar')
    if cfg_p:
        for v, m in picos_aislados(df, cfg_p).items():
            solo = ' (y las otras vibraciones no se movieron)' if v in cfg_p.get('acompanantes', {}) else ''
            anular('3. pico aislado', v, m, f'se aleja más de {cfg_p["k"]:g} desviaciones de la mediana de 5 min '
                                            f'y vuelve en {10 * cfg_p["max_registros"]} s o menos{solo}')
    arr = tras_arranque(df, arranque_min)
    episodios = pd.DataFrame(columns=['variable', 'desde', 'hasta', 'minutos', 'salto_max'])
    if cfg_s:
        mascaras, episodios = saltos_imposibles(df, cfg_s, excluir=arr)
        for v, m in mascaras.items():
            e = episodios[episodios.variable == v]
            if len(e):
                largo = e.loc[e.minutos.idxmax()]
                anular('3b. salto imposible', v, m,
                       f'cambia más de {cfg_s["max_c_min"]:g} °C en un minuto: {len(e)} episodio(s), el más largo de '
                       f'{largo.minutos:.0f} min desde {largo.desde:%Y-%m-%d %H:%M} (salto de {largo.salto_max:.0f} °C)')
        if cfg_s.get('dia_completo') and len(episodios):
            dia = df.index.normalize()
            inicio = episodios.desde + pd.Timedelta(minutes=1 + cfg_s['margen_min'])   # el minuto del primer salto
            for v, n in episodios.assign(dia=inicio.dt.normalize()).groupby(['variable', 'dia']).size().groupby(level=0):
                dias_malos = n[n >= cfg_s['dia_completo']].index.get_level_values('dia')
                if len(dias_malos):
                    anular('3b. salto imposible', v, dia.isin(dias_malos),
                           f'{cfg_s["dia_completo"]} saltos o más en el día: el sensor estaba fallando y se anula el día '
                           f'completo ({", ".join(str(d.date()) for d in dias_malos)})')
    malos = dias_inconsistentes(df, cfg_c)
    if len(malos):
        dia = df.index.normalize()
        for v, g in malos.groupby('variable', sort=False):
            motivos = '; '.join(dict.fromkeys(m for x in g.motivo for m in x.split('; ')))
            anular('4. consistencia física', v, dia.isin(g.dia),
                   f'{len(g)} día{"s" if len(g) != 1 else ""} entre {g.dia.min().date()} y {g.dia.max().date()}: {motivos}, '
                   f'y corrida de su propio nivel')
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
    tabla.attrs['saltos'] = episodios
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

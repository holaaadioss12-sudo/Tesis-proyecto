# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 01b — REVISIÓN DE LAS VARIABLES, UNA POR UNA (antes de limpiar y entrenar)
=============================================================================
Lo pedido en la reunión del 20 de noviembre: revisar cada variable de cada equipo para
encontrar errores de medición (temperaturas negativas o casi cero, un estanque de aceite a
160 °C, presiones que se disparan, sensores pegados) y decidir qué se corta antes de entrenar.
Sobre el crudo, con el chancador andando (CM > 25, como el paso 01):
  1. Por variable: mínimo, percentiles, máximo, valores fuera de los límites físicos, fuera de
     4 y de 6 desviaciones (ajustadas con entrenamiento) y la racha más larga con el mismo
     valor (sensor pegado).
  2. Por día: % fuera de los límites, % fuera de 6 desviaciones y cuánto se aleja la mediana
     del día de la de entrenamiento. Los días que fallan se juntan en períodos sugeridos.
  3. Si ya existe el limpio del paso 01: qué dejó pasar la limpieza actual.
  4. La serie horaria de cada variable (mínimo, mediana y máximo), para verla en Power BI.
Nada se borra aquí: lo que se decida cortar se anota en limpieza_config.py y se corre el 01.
Entrada:  Dataframe - {EQUIPO}.csv (el crudo); {EQUIPO}_corte_split.json y {EQUIPO}_limpio.csv
          (paso 01, opcionales); limpieza_config.py
Salida:   01b_resumen_variables.csv, 01b_periodos_sugeridos.csv, 01b_dias_sospechosos.csv,
          01b_inconsistencias.csv (los días que el 01 anula por consistencia física),
          01b_lo_que_anula_el_01.csv (todo lo que el 01 va a anular, por criterio y variable),
          01b_limpio_actual.csv, 01b_series_horarias.csv, 01b_revision.png
"""
import os
import sys
import json

try:
    from google.colab import drive
    drive.mount('/content/drive')
    RUTA_BASE = '/content/drive/MyDrive/tesis_chancadores'
except ImportError:
    RUTA_BASE = os.environ.get('RUTA_BASE', '.')

EQUIPO = os.environ.get('EQUIPO', 'CR010')
RUTA_EQ = RUTA_BASE if EQUIPO == 'CR010' else f'{RUTA_BASE}/{EQUIPO}'
if EQUIPO == 'CR010' and os.path.isdir(f'{RUTA_BASE}/CR010'):
    RUTA_EQ = f'{RUTA_BASE}/CR010'
os.makedirs(RUTA_EQ, exist_ok=True)
print(f'Equipo: {EQUIPO} | carpeta de trabajo: {RUTA_EQ}')
if RUTA_BASE not in sys.path:
    sys.path.append(RUTA_BASE)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import importlib
import limpieza_config as lc
lc = importlib.reload(lc)        # Colab guarda los módulos ya importados: así lee siempre el último

ARCHIVO_CRUDO = f'{RUTA_BASE}/Dataframe - {EQUIPO}.csv'
ARCHIVO_CORTE = f'{RUTA_EQ}/{EQUIPO}_corte_split.json'
ARCHIVO_LIMPIO = f'{RUTA_EQ}/{EQUIPO}_limpio.csv'
VARIABLES = ['CM', 'PI', 'PDF', 'PEL', 'T7', 'T8', 'T9', 'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4']
DESCRIPCION = {'CM': 'corriente del motor (%)', 'PI': 'nivel de la cámara (%)',
               'PDF': 'presión diferencial del filtro (kPa)', 'PEL': 'presión de entrada del aceite (kPa)',
               'T7': 'buje excéntrico (°C)', 'T8': 'socket liner (°C)', 'T9': 'contraeje (°C)',
               'T1': 'aceite de retorno (°C)', 'T2': 'aceite de entrada (°C)', 'T5': 'estanque de aceite (°C)',
               'V1': 'vibración 1 (mm/s)', 'V2': 'vibración 2 (mm/s)', 'V3': 'vibración 3 (mm/s)',
               'V4': 'vibración 4 (mm/s)'}
CORRIENTE_MIN = 25           # el chancador andando, como el paso 01
DIA_FUERA_FISICO = 1.0       # % del día fuera de los límites físicos para marcarlo
DIA_FUERA_DESV = 5.0         # % del día fuera de 6 desviaciones para marcarlo
DIA_Z = 4.0                  # |mediana del día - mediana de entrenamiento| en desviaciones robustas
if not os.path.exists(ARCHIVO_CRUDO):
    raise SystemExit(f'Falta {ARCHIVO_CRUDO}.')


def guardar(df, nombre):
    df.to_csv(f'{RUTA_EQ}/{nombre}', sep=';', decimal=',', index=False, encoding='utf-8-sig')


# =============================================================================
# 0. EL CRUDO, CON EL CHANCADOR ANDANDO
# =============================================================================
print('=' * 72)
print(f'PASO 01b — REVISIÓN DE LAS VARIABLES ({EQUIPO})')
print('=' * 72)
with open(ARCHIVO_CRUDO, encoding='utf-8-sig') as _f:
    _cab = _f.readline()
_sep = ';' if _cab.count(';') > _cab.count(',') else ','
df = pd.read_csv(ARCHIVO_CRUDO, sep=_sep, parse_dates=['Time'], low_memory=False).set_index('Time').sort_index()
for c in df.columns:
    if df[c].dtype == 'object':
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '.'), errors='coerce')
df, TAG = lc.renombrar_columnas(df)
VARS = [v for v in VARIABLES if v in df.columns]
print(f'Crudo: {len(df):,} registros, {df.index.min()} a {df.index.max()}')
df = df[df.CM > CORRIENTE_MIN]
print(f'Con el chancador andando (CM > {CORRIENTE_MIN}): {len(df):,} registros')

if os.path.exists(ARCHIVO_CORTE):
    _c = json.load(open(ARCHIVO_CORTE))
    T_TRAIN, T_VAL = pd.Timestamp(_c['T_train']), pd.Timestamp(_c['T_val'])
    print(f'Corte del paso 01: entrenamiento hasta {T_TRAIN}, validación hasta {T_VAL}')
else:
    T_TRAIN, T_VAL = df.index[int(0.70 * len(df))], df.index[int(0.85 * len(df))]
    print(f'Sin {os.path.basename(ARCHIVO_CORTE)}: corte aproximado 70/15/15 ({T_TRAIN}, {T_VAL})')


def conjunto_de(t):
    return np.where(t <= T_TRAIN, 'train', np.where(t <= T_VAL, 'val', 'test'))


tr = df.index <= T_TRAIN


# =============================================================================
# 1. RESUMEN POR VARIABLE
# =============================================================================
def racha_constante_h(s):
    """La racha más larga (en horas) con exactamente el mismo valor: sensor pegado."""
    s = s.dropna()
    if len(s) < 2:
        return 0.0
    grupo = s.ne(s.shift()).cumsum()
    t = s.index.to_series()
    dur = t.groupby(grupo.to_numpy()).agg(['min', 'max'])
    return float((dur['max'] - dur['min']).dt.total_seconds().max() / 3600)


EST = {}
filas = []
for v in VARS:
    x = df[v]
    b = x[tr].dropna()
    media, desv = b.mean(), b.std()
    med = b.median()
    desv_r = 1.4826 * (b - med).abs().median()
    lo_f, hi_f = lc.LIMITES_FISICOS.get(v, (-np.inf, np.inf))
    EST[v] = {'media': media, 'desv': desv, 'mediana': med, 'desv_robusta': desv_r if desv_r > 0 else desv,
              'lo_f': lo_f, 'hi_f': hi_f}
    fuera_f = (x < lo_f) | (x > hi_f)
    q = x.quantile([0.001, 0.01, 0.5, 0.99, 0.999])
    filas.append({
        'variable': v, 'descripcion': DESCRIPCION[v], 'tag_original': TAG.get(v, ''),
        'validos': int(x.notna().sum()), 'pct_nulos': 100 * float(x.isna().mean()),
        'minimo': x.min(), 'p0_1': q[0.001], 'p1': q[0.01], 'mediana': q[0.5], 'p99': q[0.99], 'p99_9': q[0.999],
        'maximo': x.max(), 'negativos': int((x < 0).sum()),
        'limite_fisico': f'{lo_f} a {hi_f}', 'bajo_el_minimo_fisico': int((x < lo_f).sum()),
        'sobre_el_maximo_fisico': int((x > hi_f).sum()), 'pct_fuera_fisico': 100 * float(fuera_f.mean()),
        'media_entrenamiento': media, 'desv_entrenamiento': desv,
        'lim_inf_4desv': media - 4 * desv, 'lim_sup_4desv': media + 4 * desv,
        'lim_inf_6desv': media - 6 * desv, 'lim_sup_6desv': media + 6 * desv,
        'pct_fuera_4desv': 100 * float(((x < media - 4 * desv) | (x > media + 4 * desv)).mean()),
        'pct_fuera_6desv': 100 * float(((x < media - 6 * desv) | (x > media + 6 * desv)).mean()),
        'racha_constante_h': racha_constante_h(x)})
res = pd.DataFrame(filas)
guardar(res.round(4), '01b_resumen_variables.csv')
pd.set_option('display.width', 230)
print('\n1. RESUMEN POR VARIABLE (crudo con el chancador andando)')
print(res[['variable', 'tag_original', 'minimo', 'p0_1', 'mediana', 'p99_9', 'maximo', 'negativos',
           'pct_fuera_fisico', 'lim_inf_4desv', 'lim_inf_6desv', 'pct_fuera_6desv', 'racha_constante_h']]
      .round(2).to_string(index=False))
for f in res.itertuples():
    avisos = []
    if f.negativos:
        avisos.append(f'{f.negativos:,} negativos')
    if f.bajo_el_minimo_fisico or f.sobre_el_maximo_fisico:
        avisos.append(f'{f.bajo_el_minimo_fisico:,} bajo {lc.LIMITES_FISICOS[f.variable][0]} y '
                      f'{f.sobre_el_maximo_fisico:,} sobre {lc.LIMITES_FISICOS[f.variable][1]}')
    if f.bajo_el_minimo_fisico and f.lim_inf_6desv < lc.LIMITES_FISICOS[f.variable][0]:
        avisos.append(f'6 desviaciones no alcanzan a cortar lo imposible (límite inferior {f.lim_inf_6desv:.1f}): '
                      'ahí manda el límite físico')
    if f.racha_constante_h >= 6:
        avisos.append(f'mismo valor durante {f.racha_constante_h:.0f} h seguidas (¿sensor pegado?)')
    if avisos:
        print(f'  {f.variable} ({f.tag_original}): ' + '; '.join(avisos))


# =============================================================================
# 2. DÍAS Y PERÍODOS SOSPECHOSOS
# =============================================================================
print('\n2. DÍAS Y PERÍODOS SOSPECHOSOS')
dias, periodos = [], []
for v in VARS:
    e = EST[v]
    x = df[v]
    g = pd.DataFrame({'x': x, 'fuera_f': (x < e['lo_f']) | (x > e['hi_f']),
                      'fuera_d': (x < e['media'] - 6 * e['desv']) | (x > e['media'] + 6 * e['desv'])})
    d = g.groupby(g.index.normalize()).agg(mediana=('x', 'median'), n=('x', 'count'),
                                           pct_fuera_fisico=('fuera_f', 'mean'), pct_fuera_6desv=('fuera_d', 'mean'))
    d[['pct_fuera_fisico', 'pct_fuera_6desv']] *= 100
    d['z_mediana'] = (d.mediana - e['mediana']) / e['desv_robusta']
    motivo = np.where(d.pct_fuera_fisico > DIA_FUERA_FISICO, 'fuera de límites físicos;', '')
    motivo = np.char.add(motivo, np.where(d.pct_fuera_6desv > DIA_FUERA_DESV, 'fuera de 6 desviaciones;', ''))
    motivo = np.char.add(motivo, np.where(d.z_mediana.abs() > DIA_Z, 'nivel corrido;', ''))
    d['motivo'] = motivo
    d = d[d.motivo != ''].reset_index(names='dia')
    if d.empty:
        continue
    d['dia'] = pd.to_datetime(d['dia'])
    d.insert(0, 'variable', v)
    d['conjunto'] = conjunto_de(d.dia)
    dias.append(d)
    # días seguidos (se toleran huecos de un día) forman un período
    corte = (d.dia.diff().dt.days > 2).cumsum()
    for _, p in d.groupby(corte):
        periodos.append({'variable': v, 'descripcion': DESCRIPCION[v], 'desde': pd.Timestamp(p.dia.min()),
                         'hasta': pd.Timestamp(p.dia.max()).replace(hour=23, minute=59, second=59), 'dias': len(p),
                         'conjunto': '/'.join(dict.fromkeys(p.conjunto)),
                         'motivos': '; '.join(sorted({x.strip() for m in p.motivo for x in m.split(';')} - {''})),
                         'mediana_del_periodo': p.mediana.median(), 'mediana_entrenamiento': e['mediana'],
                         'z_mediana': p.z_mediana.median(),
                         'pct_fuera_fisico': p.pct_fuera_fisico.mean(), 'pct_fuera_6desv': p.pct_fuera_6desv.mean()})
# lo que el paso 01 va a anular, con los mismos seis criterios de limpieza_config.py
_prev, ANULA = lc.aplicar_limites(df[VARS], EQUIPO)
inc = ANULA.attrs['dias_inconsistentes']
guardar(ANULA.round(4), '01b_lo_que_anula_el_01.csv')
guardar(inc.round(3), '01b_inconsistencias.csv')
for v, g in inc.groupby('variable'):
    g = g.sort_values('dia')
    for _, p in g.groupby((g.dia.diff().dt.days > 2).cumsum()):
        periodos.append({'variable': v, 'descripcion': DESCRIPCION[v], 'desde': pd.Timestamp(p.dia.min()),
                         'hasta': pd.Timestamp(p.dia.max()).replace(hour=23, minute=59, second=59), 'dias': len(p),
                         'conjunto': '/'.join(dict.fromkeys(conjunto_de(p.dia))),
                         'motivos': f'bajo el aceite de entrada ({lc.CONSISTENCIA["referencia"]}): se anula en el 01',
                         'mediana_del_periodo': p.mediana.median(), 'mediana_entrenamiento': EST[v]['mediana'],
                         'z_mediana': (p.mediana.median() - EST[v]['mediana']) / EST[v]['desv_robusta'],
                         'pct_fuera_fisico': np.nan, 'pct_fuera_6desv': np.nan})
dias = pd.concat(dias, ignore_index=True) if dias else pd.DataFrame()
per = pd.DataFrame(periodos)
if len(per):
    per = per.sort_values(['dias', 'variable'], ascending=[False, True])
guardar(dias.round(3), '01b_dias_sospechosos.csv')
guardar(per.round(3), '01b_periodos_sugeridos.csv')
if len(per):
    print(per.head(20)[['variable', 'desde', 'hasta', 'dias', 'conjunto', 'motivos', 'mediana_del_periodo',
                        'mediana_entrenamiento', 'z_mediana']].round(2).to_string(index=False))
    print('  Los que dicen "se anula en el 01" los corta solo la regla de consistencia. Para cortar otro,')
    print('  cópialo en PERIODOS_MALOS de limpieza_config.py, con su motivo, y corre el 01.')
else:
    print('  Ningún día fuera de lo normal con estos criterios.')
print('\n2b. LO QUE EL PASO 01 VA A ANULAR (límites físicos, sensores pegados, picos aislados, consistencia,')
print('    arranques y períodos malos de limpieza_config.py)')
print(ANULA.round(3).to_string(index=False) if len(ANULA) else '  nada.')


# =============================================================================
# 3. ¿QUÉ DEJÓ PASAR LA LIMPIEZA ACTUAL?
# =============================================================================
lim = None
if os.path.exists(ARCHIVO_LIMPIO):
    print('\n3. ¿QUÉ DEJÓ PASAR LA LIMPIEZA ACTUAL? (el limpio del paso 01)')
    _l = pd.read_csv(ARCHIVO_LIMPIO, low_memory=False)
    _t = 'timestamp' if 'timestamp' in _l.columns else _l.columns[0]
    _l = _l.set_index(pd.to_datetime(_l[_t])).sort_index()
    filas = []
    for v in [x for x in VARS if x in _l.columns]:
        e, x = EST[v], _l[v]
        dmed = x.groupby(x.index.normalize()).median()
        z = (dmed - e['mediana']) / e['desv_robusta']
        filas.append({'variable': v, 'minimo': x.min(), 'maximo': x.max(),
                      'fuera_fisico': int(((x < e['lo_f']) | (x > e['hi_f'])).sum()),
                      'picos_que_quedan': int(lc.picos_aislados(_l[[v]]).get(v, np.zeros(1, bool)).sum()),
                      'pegados_que_quedan': int(lc.sensores_pegados(_l[[v]], variables=[v]).get(v, np.zeros(1, bool)).sum()),
                      'fuera_6desv': int(((x < e['media'] - 6 * e['desv']) | (x > e['media'] + 6 * e['desv'])).sum()),
                      'dias_nivel_corrido': int((z.abs() > DIA_Z).sum()),
                      'peor_dia': z.abs().idxmax().date() if len(z) else None,
                      'z_peor_dia': z.loc[z.abs().idxmax()] if len(z) else np.nan})
    lim = pd.DataFrame(filas)
    guardar(lim.round(3), '01b_limpio_actual.csv')
    print(lim.round(2).to_string(index=False))
else:
    print(f'\n3. Sin {os.path.basename(ARCHIVO_LIMPIO)}: se omite la revisión del limpio actual.')


# =============================================================================
# 4. SERIE HORARIA PARA POWER BI Y FIGURA
# =============================================================================
h = df[VARS].resample('1h').agg(['min', 'median', 'max', 'count'])
partes = []
for v in VARS:
    p = h[v].rename(columns={'min': 'minimo', 'median': 'mediana', 'max': 'maximo', 'count': 'n'})
    p = p[p.n > 0].reset_index(names='hora')
    p.insert(0, 'variable', v)
    p['conjunto'] = conjunto_de(p.hora)
    p['limite_min'], p['limite_max'] = EST[v]['lo_f'], EST[v]['hi_f']
    p['lim_inf_6desv'] = EST[v]['media'] - 6 * EST[v]['desv']
    p['lim_sup_6desv'] = EST[v]['media'] + 6 * EST[v]['desv']
    partes.append(p)
serie = pd.concat(partes, ignore_index=True)
guardar(serie.round(3), '01b_series_horarias.csv')

dia = df[VARS].resample('1D').agg(['min', 'median', 'max'])
fig, ejes = plt.subplots(7, 2, figsize=(16, 24), sharex=True)
for ax, v in zip(ejes.ravel(), VARS):
    e = EST[v]
    ax.fill_between(dia.index, dia[v]['min'], dia[v]['max'], color='#BFD7EA', lw=0, label='mínimo y máximo del día')
    ax.plot(dia.index, dia[v]['median'], color='#156082', lw=1, label='mediana del día')
    for y in (e['lo_f'], e['hi_f']):
        if np.isfinite(y):
            ax.axhline(y, color='#C00000', ls='--', lw=0.9)
    for y in (e['media'] - 6 * e['desv'], e['media'] + 6 * e['desv']):
        ax.axhline(y, color='#7F7F7F', ls=':', lw=0.9)
    ax.axvspan(T_TRAIN, T_VAL, color='#FFF2CC', alpha=0.5, lw=0)
    ax.axvspan(T_VAL, df.index.max(), color='#FCE4D6', alpha=0.5, lw=0)
    if len(per):
        for p in per[per.variable == v].itertuples():
            ax.axvspan(p.desde, p.hasta, color='#FF0000', alpha=0.15, lw=0)
    datos = dia[v][['min', 'max']].stack()
    if len(datos):
        lo, hi = np.nanpercentile(datos, [0.5, 99.5])
        marg = 0.15 * (hi - lo) if hi > lo else 1
        ax.set_ylim(min(lo, e['media'] - 6 * e['desv']) - marg, max(hi, e['media'] + 6 * e['desv']) + marg)
    ax.set_title(f'{v} · {DESCRIPCION[v]}', fontsize=10)
    ax.tick_params(labelsize=8)
ejes[0, 0].legend(fontsize=8, loc='upper left')
fig.suptitle(f'{EQUIPO}: cada variable día a día (rojo discontinuo: límite físico; gris punteado: 6 desviaciones; '
             'fondo amarillo: validación; salmón: prueba; rojo: período sospechoso)', fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(f'{RUTA_EQ}/01b_revision.png', dpi=110)
plt.close(fig)
print('\nGuardados: 01b_resumen_variables, 01b_periodos_sugeridos, 01b_dias_sospechosos, 01b_limpio_actual, '
      '01b_series_horarias (csv) y 01b_revision.png')

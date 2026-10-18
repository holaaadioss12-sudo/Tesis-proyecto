# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 10c — ¿LOS NIVELES TIENEN SENTIDO FÍSICO EN EL CIRCUITO DE LUBRICACIÓN?
=============================================================================
Lo pidió el profesor en la reunión del 16-oct:

    "Me gustaría que tuviéramos súper claro la parte hidráulica ... si los
     resultados de leve, medio, alto, al mirar el comportamiento hidráulico de
     todas estas cosas, de las temperaturas, de todo el circuito, tiene
     sentido. ¿Qué pasa cuando las temperaturas del fluido suben, bajan? ¿Qué
     pasa con la viscosidad? ... ¿Cuál es la causa de la alarma? El fenómeno
     físico."

El nivel de cada ventana ya lo asigna el paso 10 (N = suma de peso x
activaciones, cortado en leve / media / grave). Este paso no clasifica nada:
describe, con las variables en sus unidades, cómo se ve el circuito en cada
nivel, para poder contrastarlo con la física. No entrena modelos.

  1. Perfil físico por nivel: mediana y rango intercuartil de presión de
     entrada de aceite (PEL), presión diferencial del filtro (PDF),
     temperaturas del aceite (entrada T2, estanque T5, retorno T1), calor que
     se lleva el aceite (T1 - T2), temperaturas de componentes (T7 buje,
     T8 socket liner, T9 contraeje), corriente (CM) y vibración máxima.
     Spearman de cada variable con el nivel (0 = normal ... 3 = grave): ¿la
     variable cambia de forma ordenada de nivel en nivel?
     Se calcula con las dos escalas de pesos (lineal 3/2/1 y empinada 9/3/1)
     en ENTRENAMIENTO + VALIDACIÓN, que es donde se decide; prueba se reporta
     aparte, como confirmación.
  2. A5 (PEL > 450, "línea muy obstruida", criticidad 1) contra D sin A5
     (PEL 350-450) y contra las ventanas sin alarma: temperaturas, presiones y
     cuánto ocurre en la primera media hora de cada tramo continuo de
     operación (después de una detención o de un corte en los datos).
  3. Viscosidad: PEL según la temperatura del aceite de entrada (T2). Con un
     caudal parecido, un aceite más frío es más viscoso y necesita más presión
     para pasar por las mismas líneas (flujo laminar: la caída de presión es
     proporcional a la viscosidad). Si PEL baja cuando T2 sube, el dato es
     coherente con esa explicación; no la prueba, porque no se mide el caudal.
  4. Tendencia mensual: % de ventanas con A5, y PEL, PDF y T2 en las ventanas
     sin ninguna alarma (la línea base del circuito, mes a mes).

Entradas: {EQUIPO}_features_W60.csv              (paso 02: 70 estadísticos)
          {EQUIPO}_activos_por_ventana_W60.csv   (paso 02, PARCHE P11 v2)
          10_nivel_ventanas.csv                  (paso 10)
Salidas:  10c_perfil_por_nivel.csv, 10c_a5.csv, 10c_pel_vs_t2.csv,
          10c_mensual.csv, 10c_perfil_niveles.png, 10c_pel_vs_temperatura.png
=============================================================================
"""
import os

try:
    from google.colab import drive
    drive.mount('/content/drive')
    RUTA_BASE = '/content/drive/MyDrive/tesis_chancadores'
except ImportError:
    RUTA_BASE = os.environ.get('RUTA_BASE', '.')

EQUIPO = os.environ.get('EQUIPO', 'CR010')
RUTA_EQ = RUTA_BASE if EQUIPO == 'CR010' else f'{RUTA_BASE}/{EQUIPO}'
if EQUIPO == 'CR010' and os.path.isdir(f'{RUTA_BASE}/CR010'):
    RUTA_EQ = f'{RUTA_BASE}/CR010'     # CR010 movido a su subcarpeta
print(f'Equipo: {EQUIPO} | carpeta de trabajo: {RUTA_EQ}')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =============================================================================
# 0. CONSTANTES
# =============================================================================
W = 60
LLAVES = ['tramo_id', 'ventana_id']
NIVELES = ['0_normal', '1_leve', '2_media', '3_grave']
NOMBRE_NIVEL = {'0_normal': 'Normal', '1_leve': 'Leve', '2_media': 'Media',
                '3_grave': 'Grave'}
ESCALAS = {'lineal (3/2/1)': 'banda_N_lineal_natural',
           'empinada (9/3/1)': 'banda_N_empinada_natural'}
MIN_ARRANQUE = 30               # "recién partido": primeros 30 min del tramo
N_BINS_T2 = 10                  # deciles de temperatura de entrada
UMBRAL_D, UMBRAL_A5 = 350, 450  # reglas D y A5 sobre PEL (reglas_engine.py)
MIN_RHO = 0.10                  # tendencia mínima para llamarla ordenada

# Variables del perfil: nombre en la tabla -> cómo se calcula desde features
PERFIL = {
    'PEL media (presión entrada aceite)': 'PEL_media',
    'PEL máxima': 'PEL_max',
    'PDF media (ΔP filtro)': 'PDF_media',
    'PDF máxima': 'PDF_max',
    'T2 media (aceite entrada)': 'T2_media',
    'T5 media (estanque)': 'T5_media',
    'T1 media (aceite retorno)': 'T1_media',
    'T1 - T2 (calor que se lleva el aceite)': None,     # derivada
    'T7 media (buje excéntrico)': 'T7_media',
    'T8 media (socket liner)': 'T8_media',
    'T9 media (contraeje)': 'T9_media',
    'CM media (corriente)': 'CM_media',
    'Vibración máxima (V1..V4)': None,                  # derivada
}

ARCHIVO_FEATURES = f'{RUTA_EQ}/{EQUIPO}_features_W{W}.csv'
ARCHIVO_ACTIVOS = f'{RUTA_EQ}/{EQUIPO}_activos_por_ventana_W{W}.csv'
ARCHIVO_NIVEL = f'{RUTA_EQ}/10_nivel_ventanas.csv'
for _f in [ARCHIVO_FEATURES, ARCHIVO_ACTIVOS, ARCHIVO_NIVEL]:
    if not os.path.exists(_f):
        raise SystemExit(f'Falta {_f}. Corre paso_02_eventos.py y el paso 10.')


def salida(nombre):
    return f'{RUTA_EQ}/{nombre}'


# =============================================================================
# 1. CARGA
# =============================================================================
print('=' * 72)
print(f'PASO 10c — SENTIDO FÍSICO DE LOS NIVELES (W = {W})')
print('=' * 72)

_cab = pd.read_csv(ARCHIVO_FEATURES, nrows=0).columns
_vars_f = [c for c in PERFIL.values() if c] + [
    f'{v}_max' for v in ['V1', 'V2', 'V3', 'V4']]
_falta = [c for c in _vars_f if c not in _cab]
if _falta:
    raise SystemExit(f'El archivo de features no trae {_falta}.')
f = pd.read_csv(ARCHIVO_FEATURES, usecols=LLAVES + _vars_f)
f['T1 - T2 (calor que se lleva el aceite)'] = f.T1_media - f.T2_media
f['Vibración máxima (V1..V4)'] = f[[f'{v}_max' for v in
                                     ['V1', 'V2', 'V3', 'V4']]].max(axis=1)
for nombre, col in PERFIL.items():
    if col:
        f[nombre] = f[col]

act = pd.read_csv(ARCHIVO_ACTIVOS)
_reglas = [r for r in ['A5', 'D', 'C', 'E'] if r in act.columns]
nivel = pd.read_csv(ARCHIVO_NIVEL, parse_dates=['timestamp_inicio'])
d = (nivel.merge(f, on=LLAVES, how='inner', validate='one_to_one')
     .merge(act[LLAVES + _reglas], on=LLAVES, how='left', validate='one_to_one'))
d[_reglas] = d[_reglas].fillna(0)
d['min_desde_partida'] = d.ventana_id * (W * 10 / 60)       # 10 s por muestra
d['recien_partido'] = d.min_desde_partida < MIN_ARRANQUE
d['decision'] = np.where(d.conjunto_fecha == 'test', 'prueba',
                         'entrenamiento + validación')
print(f'Ventanas: {len(d):,} (sin cruzar: {len(nivel) - len(d):,})')


# =============================================================================
# 2. PERFIL FÍSICO POR NIVEL
# =============================================================================
print('\n' + '=' * 72)
print('2. PERFIL FÍSICO POR NIVEL (mediana [P25 – P75])')
print('=' * 72)


def perfil(x, banda):
    filas = []
    orden = x[banda].map({nv: i for i, nv in enumerate(NIVELES)})
    for nombre in PERFIL:
        g = x.groupby(banda)[nombre]
        med = g.median().reindex(NIVELES)
        fila = {'variable': nombre}
        for nv in NIVELES:
            q1, q3 = g.quantile(.25).get(nv, np.nan), g.quantile(.75).get(nv, np.nan)
            fila[nv] = round(med[nv], 1)
            fila[f'{nv}_iqr'] = f'{q1:.1f} – {q3:.1f}'
        dif = np.diff(med.to_numpy())
        rho = spearmanr(orden, x[nombre], nan_policy='omit').correlation
        # ordenada: la mediana avanza en el mismo sentido en los tres pasos y
        # la tendencia no es despreciable (|Spearman| >= MIN_RHO)
        fila['ordenada'] = ('no' if abs(rho) < MIN_RHO else
                            'sube' if (dif > 0).all() else
                            'baja' if (dif < 0).all() else 'no')
        fila['spearman'] = round(rho, 2)
        filas.append(fila)
    return pd.DataFrame(filas)


tablas = []
for escala, banda in ESCALAS.items():
    for conj in ['entrenamiento + validación', 'prueba']:
        t = perfil(d[d.decision == conj], banda)
        t.insert(0, 'datos', conj)
        t.insert(0, 'escala', escala)
        tablas.append(t)
        n = d[d.decision == conj][banda].value_counts().reindex(NIVELES).fillna(0)
        print(f'\nEscala {escala} | {conj} | ventanas por nivel: '
              + ', '.join(f'{NOMBRE_NIVEL[k]} {int(v):,}' for k, v in n.items()))
        print(t[['variable'] + NIVELES + ['ordenada', 'spearman']]
              .rename(columns=NOMBRE_NIVEL).to_string(index=False))
perfil_tab = pd.concat(tablas, ignore_index=True)
print('\nCómo leerlo: "ordenada" dice si la mediana sube (o baja) en cada paso de')
print('normal a grave. Spearman resume la tendencia (-1 a 1; 0 = no cambia).')
print('La escala se decide con "entrenamiento + validación"; prueba confirma.')


# =============================================================================
# 3. A5 CONTRA D SIN A5 Y CONTRA LAS VENTANAS SIN ALARMA
# =============================================================================
print('\n' + '=' * 72)
print('3. A5 (PEL > 450) CONTRA D SIN A5 (PEL 350-450) Y SIN ALARMA')
print('=' * 72)
grupos = {'A5 (PEL > 450)': d.A5 > 0,
          'D sin A5 (PEL 350-450)': (d.D > 0) & (d.A5 == 0),
          'Sin ninguna alarma': d.N_lineal == 0}
cols_a5 = ['PEL media (presión entrada aceite)', 'PDF media (ΔP filtro)',
           'T2 media (aceite entrada)', 'T5 media (estanque)',
           'T1 media (aceite retorno)', 'T1 - T2 (calor que se lleva el aceite)',
           'T7 media (buje excéntrico)', 'CM media (corriente)']
filas = []
for g, m in grupos.items():
    fila = {'grupo': g, 'ventanas': int(m.sum()),
            '% en los primeros 30 min tras partir': round(100 * d.loc[m, 'recien_partido'].mean(), 1)}
    if 'C' in d.columns:
        fila['% con filtro saturado (C)'] = round(100 * (d.loc[m, 'C'] > 0).mean(), 1)
    for c in cols_a5:
        fila[c] = round(d.loc[m, c].median(), 1)
    filas.append(fila)
a5 = pd.DataFrame(filas)
print(a5.set_index('grupo').T.to_string())
print(f'\nReferencia: el {100 * d.recien_partido.mean():.1f} % de TODAS las ventanas '
      f'cae en los primeros {MIN_ARRANQUE} min tras una partida.')
print('Si A5 aparece con T2 más baja y más seguido al partir, es coherente con')
print('aceite frío y más viscoso. Si aparece con PDF alta, apunta al filtro.')


# =============================================================================
# 4. VISCOSIDAD: PEL SEGÚN LA TEMPERATURA DE ENTRADA DEL ACEITE
# =============================================================================
print('\n' + '=' * 72)
print('4. PEL SEGÚN LA TEMPERATURA DEL ACEITE DE ENTRADA (T2)')
print('=' * 72)
T2, PEL = 'T2 media (aceite entrada)', 'PEL media (presión entrada aceite)'
x = d[[T2, PEL, 'A5', 'N_lineal']].dropna()
x['tramo_T2'] = pd.qcut(x[T2], N_BINS_T2, duplicates='drop')
pv = (x.groupby('tramo_T2', observed=True)
      .agg(T2_mediana=(T2, 'median'), ventanas=(PEL, 'size'),
           PEL_P25=(PEL, lambda s: s.quantile(.25)), PEL_mediana=(PEL, 'median'),
           PEL_P75=(PEL, lambda s: s.quantile(.75)),
           pct_A5=('A5', lambda s: 100 * (s > 0).mean()))
      .round(1).reset_index(drop=True))
print(pv.to_string(index=False))
_rho_todas = spearmanr(x[T2], x[PEL]).correlation
_sin = x[x.N_lineal == 0]
_rho_sin = spearmanr(_sin[T2], _sin[PEL]).correlation if len(_sin) > 10 else np.nan
print(f'\nSpearman T2 contra PEL: todas las ventanas {_rho_todas:+.2f} | '
      f'sólo sin alarma {_rho_sin:+.2f}')
print('Negativo = más frío, más presión: coherente con el efecto de la viscosidad.')


# =============================================================================
# 5. TENDENCIA MENSUAL
# =============================================================================
print('\n' + '=' * 72)
print('5. MES A MES')
print('=' * 72)
d['mes'] = d.timestamp_inicio.dt.to_period('M').astype(str)
_sa = d.N_lineal == 0
mens = (d.groupby('mes')
        .agg(ventanas=('A5', 'size'), pct_A5=('A5', lambda s: 100 * (s > 0).mean()))
        .join(d[_sa].groupby('mes')[[PEL, 'PDF media (ΔP filtro)', T2]].median()
              .rename(columns=lambda c: f'{c.split(" (")[0]} sin alarma'))
        .round(1))
print(mens.to_string())
print('Las tres últimas columnas son la línea base del circuito (ventanas sin')
print('alarma). Si suben o bajan mes a mes, el circuito está cambiando aunque')
print('no salten alarmas: filtro cargándose, aceite envejeciendo, desgaste.')


# =============================================================================
# 6. FIGURAS
# =============================================================================
TINTA, TINTA2, MUTED, GRILLA, FONDO = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
COLOR = {'0_normal': MUTED, '1_leve': '#86b6ef', '2_media': '#2a78d6',
         '3_grave': '#104281'}
plt.rcParams.update({'font.size': 9.5, 'axes.edgecolor': '#c3c2b7',
                     'axes.labelcolor': TINTA2, 'xtick.color': TINTA2,
                     'ytick.color': TINTA2, 'axes.titlecolor': TINTA})


def ejes(a):
    a.set_facecolor(FONDO)
    a.grid(axis='y', color=GRILLA, linewidth=0.6)
    a.set_axisbelow(True)
    for s in ['top', 'right']:
        a.spines[s].set_visible(False)


# 6a. perfil por nivel: mediana y P25-P75, escala principal del paso 10
_ver = ['PEL media (presión entrada aceite)', 'PDF media (ΔP filtro)',
        'T2 media (aceite entrada)', 'T1 - T2 (calor que se lleva el aceite)',
        'T7 media (buje excéntrico)', 'CM media (corriente)']
_banda = ESCALAS['lineal (3/2/1)']
_dv = d[d.decision == 'entrenamiento + validación']
fig, axs = plt.subplots(2, 3, figsize=(12, 6.4), facecolor=FONDO)
for a, var in zip(axs.ravel(), _ver):
    ejes(a)
    for i, nv in enumerate(NIVELES):
        s = _dv.loc[_dv[_banda] == nv, var].dropna()
        if s.empty:
            continue
        q1, me, q3 = s.quantile([.25, .5, .75])
        a.plot([i, i], [q1, q3], color=COLOR[nv], linewidth=2.5,
               solid_capstyle='round')
        a.plot(i, me, 'o', color=COLOR[nv], markersize=8,
               markeredgecolor=FONDO, markeredgewidth=2)
    a.set_xticks(range(4), [NOMBRE_NIVEL[nv] for nv in NIVELES])
    a.set_xlim(-0.5, 3.5)
    a.set_title(var, loc='left', fontsize=10)
fig.suptitle(f'{EQUIPO} · cómo se ve el circuito en cada nivel '
             f'(mediana y P25–P75, entrenamiento + validación, escala lineal)',
             color=TINTA, x=0.01, ha='left')
fig.tight_layout()
fig.savefig(salida('10c_perfil_niveles.png'), dpi=160, facecolor=FONDO)
plt.close(fig)

# 6b. PEL contra T2, con los umbrales de D y A5, y % de ventanas con A5
fig, axs = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True, facecolor=FONDO,
                        gridspec_kw={'height_ratios': [2, 1]})
for a in axs:
    ejes(a)
axs[0].fill_between(pv.T2_mediana, pv.PEL_P25, pv.PEL_P75, color='#86b6ef',
                    alpha=0.35, linewidth=0)
axs[0].plot(pv.T2_mediana, pv.PEL_mediana, color='#2a78d6', linewidth=2,
            marker='o', markersize=5)
for y, txt in [(UMBRAL_D, 'D: PEL > 350'), (UMBRAL_A5, 'A5: PEL > 450')]:
    axs[0].axhline(y, color=MUTED, linewidth=1, linestyle='--')
    axs[0].text(pv.T2_mediana.max(), y, f' {txt}', color=TINTA2, va='bottom',
                ha='right', fontsize=9)
axs[0].set_ylim(top=max(axs[0].get_ylim()[1], UMBRAL_A5 * 1.05))
axs[0].set_ylabel('PEL media de la ventana')
axs[0].set_title('Presión de entrada del aceite según su temperatura '
                 '(mediana y P25–P75 por decil de T2)', loc='left', fontsize=10)
axs[1].bar(pv.T2_mediana, pv.pct_A5, width=(pv.T2_mediana.diff().median() or 1) * 0.7,
           color='#2a78d6')
axs[1].set_ylabel('% ventanas con A5')
axs[1].set_xlabel('T2 media de la ventana (temperatura del aceite de entrada)')
fig.suptitle(f'{EQUIPO} · ¿aceite más frío, más presión?', color=TINTA,
             x=0.01, ha='left')
fig.tight_layout()
fig.savefig(salida('10c_pel_vs_temperatura.png'), dpi=160, facecolor=FONDO)
plt.close(fig)


# =============================================================================
# 7. GUARDAR Y RESUMEN
# =============================================================================
perfil_tab.to_csv(salida('10c_perfil_por_nivel.csv'), index=False)
a5.to_csv(salida('10c_a5.csv'), index=False)
pv.to_csv(salida('10c_pel_vs_t2.csv'), index=False)
mens.to_csv(salida('10c_mensual.csv'))

print('\n' + '=' * 72)
print(f'RESUMEN — {EQUIPO}')
print('=' * 72)
for escala in ESCALAS:
    t = perfil_tab[(perfil_tab.escala == escala) &
                   (perfil_tab.datos == 'entrenamiento + validación')]
    print(f'Escala {escala}: variables que cambian ordenadas de normal a grave: '
          f'{int((t.ordenada != "no").sum())} de {len(t)}')
_r = a5.set_index('grupo')
print(f'A5 en los primeros {MIN_ARRANQUE} min tras partir: '
      f'{_r.loc["A5 (PEL > 450)", f"% en los primeros {MIN_ARRANQUE} min tras partir"]} % '
      f'(todas las ventanas: {100 * d.recien_partido.mean():.1f} %)')
print(f'T2 mediana: A5 {_r.loc["A5 (PEL > 450)", T2]} | sin alarma '
      f'{_r.loc["Sin ninguna alarma", T2]}')
print(f'Spearman T2 contra PEL (todas las ventanas): {_rho_todas:+.2f}')
print('Guardado: 10c_perfil_por_nivel.csv, 10c_a5.csv, 10c_pel_vs_t2.csv, '
      '10c_mensual.csv, 10c_perfil_niveles.png, 10c_pel_vs_temperatura.png')

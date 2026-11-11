# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 02b — BALANCE DE CALOR DEL ACEITE, INSTANTE A INSTANTE
=============================================================================
Lo pedido en la reunión del 6 de noviembre: el calor que saca el aceite en cada
registro, Q = ṁ · cp · (T1 − T2), con el caudal nominal de referencia del
fabricante (los datos no miden caudal); cómo se comporta en el tiempo, por nivel
y frente a la potencia del motor.
T1: aceite de retorno (sale del chancador). T2: aceite de entrada, después del
enfriador. T5: estanque, de donde la bomba toma el aceite antes del enfriador.
Entrada:  {EQUIPO}_limpio.csv y {EQUIPO}_corte_split.json (paso 01), {EQUIPO}_features_W60.csv y
          {EQUIPO}_ventanas_W60.csv (paso 02), 05a_nivel_ventanas.csv (opcional)
Salida:   02b_calor_resumen.csv, 02b_calor_mensual.csv, 02b_calor_tiempo.png, 02b_calor_nivel.png
Supuestos (variables de entorno): CAUDAL_L_MIN = 631 (referencia del fabricante, red de 50 Hz),
          DENSIDAD = 0,88 kg/L y CP = 1,95 kJ/(kg·°C) (aceite mineral típico), POTENCIA_KW = 750
          (potencia nominal del motor; CM es el % de la carga nominal).
"""
import os
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
print(f'Equipo: {EQUIPO} | carpeta de trabajo: {RUTA_EQ}')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =============================================================================
# 0. CONSTANTES Y SUPUESTOS
# =============================================================================
W = 60
LLAVES = ['tramo_id', 'ventana_id']
BANDA = 'banda_N_empinada_natural'
TEXTO_NIVEL = {'0_normal': 'sin alarma', '1_leve': 'leve', '2_media': 'media', '3_grave': 'grave'}
ESTADOS = ['sin alarma', 'leve', 'media', 'grave']
CAUDAL = float(os.environ.get('CAUDAL_L_MIN', 631))        # L/min, referencia del fabricante
DENSIDAD = float(os.environ.get('DENSIDAD', 0.88))         # kg/L
CP = float(os.environ.get('CP', 1.95))                     # kJ/(kg·°C)
POTENCIA = float(os.environ.get('POTENCIA_KW', 750))       # kW, potencia nominal del motor
K = DENSIDAD * CAUDAL / 60 * CP                            # kW por cada °C entre T1 y T2
CM_MIN = 10                                                # % de carga para considerar que opera
AZUL, NARANJO, VERDE, TINTA, SUAVE, CRITICO = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#52514e', '#d03b3b'
ARCHIVO_LIMPIO = f'{RUTA_EQ}/{EQUIPO}_limpio.csv'
ARCHIVO_CORTE = f'{RUTA_EQ}/{EQUIPO}_corte_split.json'
ARCHIVO_FEATURES = f'{RUTA_EQ}/{EQUIPO}_features_W{W}.csv'
ARCHIVO_VENTANAS = f'{RUTA_EQ}/{EQUIPO}_ventanas_W{W}.csv'
ARCHIVO_NIVEL = f'{RUTA_EQ}/05a_nivel_ventanas.csv'
for _f in [ARCHIVO_LIMPIO, ARCHIVO_CORTE, ARCHIVO_FEATURES, ARCHIVO_VENTANAS]:
    if not os.path.exists(_f):
        raise SystemExit(f'Falta {_f}. ¿Corriste los pasos 01 y 02?')


def salida(nombre):
    return f'{RUTA_EQ}/{nombre}'


def guardar(df, nombre):
    df.to_csv(salida(nombre), sep=';', decimal=',', index=False, encoding='utf-8-sig')


def resumen(d):
    """Mediana y rango intercuartil del calor, la diferencia de temperatura y la potencia."""
    q = d.Q
    return {'ventanas': len(d), 'Q_mediana_kW': q.median(), 'Q_p25_kW': q.quantile(0.25), 'Q_p75_kW': q.quantile(0.75),
            'dT_mediana_C': d.dT.median(), 'P_motor_mediana_kW': d.P.median(), 'Q_pct_P_mediana': d.Q_pct_P.median(),
            'Q_enfriador_mediana_kW': d.Q_enf.median(), 'perdidas_mediana_kW': (d.Q - d.Q_enf).median()}


# =============================================================================
# 1. REGISTRO A REGISTRO (cada 10 s)
# =============================================================================
print('=' * 72)
print(f'PASO 02b — BALANCE DE CALOR DEL ACEITE ({EQUIPO})')
print('=' * 72)
print(f'Supuestos: caudal {CAUDAL:.0f} L/min, densidad {DENSIDAD} kg/L, cp {CP} kJ/(kg·°C) -> '
      f'{K:.1f} kW por cada °C entre T1 y T2; potencia nominal {POTENCIA:.0f} kW.')
corte = json.load(open(ARCHIVO_CORTE))
T_TRAIN, T_VAL = pd.Timestamp(corte['T_train']), pd.Timestamp(corte['T_val'])
_cab = pd.read_csv(ARCHIVO_LIMPIO, nrows=0).columns
_tc = 'timestamp' if 'timestamp' in _cab else 'Time'
r = pd.read_csv(ARCHIVO_LIMPIO, usecols=[_tc, 'T1', 'T2', 'T5', 'CM'], parse_dates=[_tc]).set_index(_tc).sort_index()
r['conjunto'] = np.where(r.index <= T_TRAIN, 'train', np.where(r.index <= T_VAL, 'val', 'test'))
op = r.CM > CM_MIN
r['Q'] = K * (r.T1 - r.T2)
r['P'] = r.CM / 100 * POTENCIA
r['Q_pct_P'] = np.where(op, 100 * r.Q / r.P, np.nan)
print(f'Registros: {len(r):,} | operando (CM > {CM_MIN} %): {op.mean() * 100:.1f} %')
_rop = r[op]
print(f'Calor que saca el aceite, registro a registro (operando): mediana {_rop.Q.median():.0f} kW '
      f'(P25 {_rop.Q.quantile(0.25):.0f}, P75 {_rop.Q.quantile(0.75):.0f}); '
      f'{100 * (_rop.Q < 0).mean():.1f} % de los registros con T1 < T2.')


# =============================================================================
# 2. POR VENTANA: CONJUNTO, NIVEL Y POTENCIA
# =============================================================================
w = pd.read_csv(ARCHIVO_FEATURES, usecols=LLAVES + ['T1_media', 'T2_media', 'T5_media', 'CM_media']).drop_duplicates(LLAVES)
_vc = pd.read_csv(ARCHIVO_VENTANAS, nrows=0).columns
_cols_v = LLAVES + ['timestamp_inicio'] + (['total_alarmas_ventana'] if 'total_alarmas_ventana' in _vc else [])
w = w.merge(pd.read_csv(ARCHIVO_VENTANAS, usecols=_cols_v, parse_dates=['timestamp_inicio']).drop_duplicates(LLAVES),
            on=LLAVES, how='left', validate='one_to_one')
if os.path.exists(ARCHIVO_NIVEL):
    w = w.merge(pd.read_csv(ARCHIVO_NIVEL, usecols=LLAVES + [BANDA]).drop_duplicates(LLAVES), on=LLAVES, how='left',
                validate='one_to_one')
    w['estado'] = w[BANDA].map(TEXTO_NIVEL)
elif 'total_alarmas_ventana' in w.columns:
    w['estado'] = np.where(w.total_alarmas_ventana > 0, 'con alarma', 'sin alarma')
else:
    w['estado'] = 'sin dato'
w['conjunto'] = np.where(w.timestamp_inicio <= T_TRAIN, 'train', np.where(w.timestamp_inicio <= T_VAL, 'val', 'test'))
w['dT'] = w.T1_media - w.T2_media
w['Q'] = K * w.dT
w['Q_enf'] = K * (w.T5_media - w.T2_media)
w['P'] = w.CM_media / 100 * POTENCIA
w = w[w.CM_media > CM_MIN].copy()
w['Q_pct_P'] = 100 * w.Q / w.P
w['mes'] = w.timestamp_inicio.dt.to_period('M').astype(str)

print('\n' + '=' * 72)
print('2. CALOR QUE SACA EL ACEITE POR CONJUNTO Y POR NIVEL (ventanas de 10 min operando)')
print('=' * 72)
filas = []
for cj in ('train', 'val', 'test'):
    d = w[w.conjunto == cj]
    if len(d):
        filas.append({'conjunto': cj, 'estado': 'todas', **resumen(d)})
    for es in ESTADOS + ['con alarma']:
        e = d[d.estado == es]
        if len(e) >= 10:
            filas.append({'conjunto': cj, 'estado': es, **resumen(e)})
res = pd.DataFrame(filas)
guardar(res.round(3), '02b_calor_resumen.csv')
_ver = res[['conjunto', 'estado', 'ventanas', 'Q_mediana_kW', 'Q_p25_kW', 'Q_p75_kW', 'dT_mediana_C',
            'P_motor_mediana_kW', 'Q_pct_P_mediana']].round(1)
print(_ver.to_string(index=False))
print('Q: calor que saca el aceite (kW). dT: T1 − T2 (°C). P_motor: CM × potencia nominal. Q_pct_P: Q como % de P.')

# balance: lo que saca el aceite del chancador contra lo que saca el enfriador
_n = w[w.estado == 'sin alarma'] if (w.estado == 'sin alarma').any() else w
print(f'\nBalance en ventanas sin alarma: el aceite saca {_n.Q.median():.0f} kW del chancador (T1 − T2) y el '
      f'enfriador saca {_n.Q_enf.median():.0f} kW (T5 − T2); diferencia {(_n.Q - _n.Q_enf).median():.0f} kW '
      '(lo que se pierde entre el retorno y el estanque).')

# relación con la carga del motor
_ref = w[w.conjunto.isin(['train', 'val']) & (w.estado == 'sin alarma')]
if len(_ref) < 30:
    _ref = w[w.conjunto.isin(['train', 'val'])]
pend, orig = np.polyfit(_ref.P, _ref.Q, 1)
r2 = np.corrcoef(_ref.P, _ref.Q)[0, 1] ** 2
print(f'Ajuste en ventanas sin alarma de entrenamiento y validación: Q = {orig:.0f} + {pend:.3f} · P_motor '
      f'(R² {r2:.2f}). Por cada 100 kW más de motor, el aceite saca {100 * pend:.1f} kW más.')


# =============================================================================
# 3. MES A MES
# =============================================================================
print('\n' + '=' * 72)
print('3. MES A MES (ventanas operando)')
print('=' * 72)
filas_m = []
for me, d in w.groupby('mes'):
    filas_m.append({'mes': me, 'conjunto': '/'.join(sorted(set(d.conjunto))), 'ventanas': len(d),
                    'pct_con_alarma': 100 * (d.estado != 'sin alarma').mean(), 'Q_mediana_kW': d.Q.median(),
                    'dT_mediana_C': d.dT.median(), 'P_motor_mediana_kW': d.P.median(),
                    'Q_pct_P_mediana': d.Q_pct_P.median(), 'T2_mediana_C': d.T2_media.median(),
                    'T5_mediana_C': d.T5_media.median()})
mensual = pd.DataFrame(filas_m)
guardar(mensual.round(3), '02b_calor_mensual.csv')
print(mensual.round(1).to_string(index=False))
print('Con caudal constante, si T1 − T2 sube sin que suba la potencia, o sale más calor del chancador o bajó')
print('el caudal (por ejemplo, desgaste de la bomba); sin medir caudal no se pueden separar.')


# =============================================================================
# 4. FIGURAS
# =============================================================================
_h = r[op].resample('1h')[['Q', 'P', 'Q_pct_P']].median()
_d = _h.rolling('1D', min_periods=6).median()
fig, ejes = plt.subplots(3, 1, figsize=(13, 8.4), sharex=True)
for ax, col, color, tit in [(ejes[0], 'Q', AZUL, 'Calor que saca el aceite, Q (kW)'),
                            (ejes[1], 'P', NARANJO, 'Potencia del motor, CM × potencia nominal (kW)'),
                            (ejes[2], 'Q_pct_P', VERDE, 'Q como % de la potencia del motor')]:
    ax.plot(_h.index, _h[col], color=color, lw=0.8, alpha=0.35, label='mediana por hora')
    ax.plot(_d.index, _d[col], color=color, lw=2, label='mediana móvil de 1 día')
    ax.axvspan(T_TRAIN, T_VAL, color='#e7e6e2', lw=0, label='validación')
    ax.axvspan(T_VAL, r.index.max(), color='#f6e1e1', lw=0, label='prueba')
    ax.set_title(tit, loc='left', fontsize=10)
    ax.grid(alpha=0.25)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
ejes[0].legend(fontsize=8, frameon=False, ncol=4, loc='upper left')
fig.suptitle(f'{EQUIPO}: balance de calor del aceite en el tiempo (caudal de referencia {CAUDAL:.0f} L/min)',
             x=0.01, ha='left', fontsize=12)
fig.tight_layout()
fig.savefig(salida('02b_calor_tiempo.png'), dpi=150)
plt.close(fig)

fig, ejes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={'width_ratios': [1.2, 1]})
ax = ejes[0]
_est = [e for e in ESTADOS if (w.estado == e).any()] or sorted(w.estado.unique())
for k, (cj, col) in enumerate([('val', AZUL), ('test', NARANJO)]):
    datos = [w[(w.conjunto == cj) & (w.estado == e)].Q.dropna() for e in _est]
    pos = np.arange(len(_est)) * 3 + k
    bp = ax.boxplot([x if len(x) else [np.nan] for x in datos], positions=pos, widths=0.8, showfliers=False,
                    patch_artist=True)
    for caja in bp['boxes']:
        caja.set(facecolor=col, alpha=0.35, edgecolor=col)
    for parte in ('whiskers', 'caps', 'medians'):
        for linea in bp[parte]:
            linea.set(color=col, lw=1.5)
    ax.plot([], [], color=col, lw=6, alpha=0.5, label='validación' if cj == 'val' else 'prueba')
ax.set_xticks(np.arange(len(_est)) * 3 + 0.5)
ax.set_xticklabels(_est)
ax.set_ylabel('Q (kW)')
ax.set_title('Calor que saca el aceite por nivel', loc='left', fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax = ejes[1]
_m = w.sample(min(len(w), 20000), random_state=42)
ax.scatter(_m.P, _m.Q, s=4, alpha=0.15, color=AZUL, lw=0)
_x = np.linspace(w.P.quantile(0.01), w.P.quantile(0.99), 50)
ax.plot(_x, orig + pend * _x, color=TINTA, lw=2, label=f'ajuste: +{100 * pend:.1f} kW por cada 100 kW')
ax.set_xlabel('Potencia del motor (kW)')
ax.set_ylabel('Q (kW)')
ax.set_title('Calor contra carga del motor (ventanas de 10 min)', loc='left', fontsize=10)
ax.legend(fontsize=8, frameon=False)
for ax in ejes:
    ax.grid(alpha=0.25)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
fig.suptitle(f'{EQUIPO}: el calor del aceite por nivel y según la carga', x=0.01, ha='left', fontsize=12)
fig.tight_layout()
fig.savefig(salida('02b_calor_nivel.png'), dpi=150)
plt.close(fig)
print(f'\nGuardado en {RUTA_EQ}: 02b_calor_resumen.csv, 02b_calor_mensual.csv, 02b_calor_tiempo.png, '
      f'02b_calor_nivel.png')

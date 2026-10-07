# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 07 — FIGURAS PARA LA MEMORIA
=============================================================================
No entrena nada ni cambia ningún resultado: lee los CSV que ya produjeron
los pasos 04, 05, 06 y 04c, y dibuja las figuras de la memoria con los mismos
números. Corre en CPU en un par de minutos.

Entradas (todas en la carpeta del equipo; CR010 = carpeta raíz):
    {EQUIPO}_error_reconstruccion_W60.csv   paso 05
    {EQUIPO}_ventanas_W60.csv               paso 02
    {EQUIPO}_features_W60.csv               paso 02 (para la línea base)
    {EQUIPO}_corte_split.json               paso 01
    04b_comparacion_epocas.csv              paso 04   (opcional)
    06_episodios.csv                        paso 06   (opcional)
    04c_barrido_hiperparametros_v2.csv      paso 04c  (opcional)

Salida: carpeta figuras/ dentro de la carpeta del equipo, un PNG por figura
(200 dpi), con el número de figura en el nombre. Si falta una entrada
opcional, esa figura se salta y se avisa; el resto se dibuja igual.

Las figuras:
  F01  curva de entrenamiento: pérdida en train y en normales de validación,
       y el AUC de validación, por época
  F02  distribución del error de reconstrucción, normales contra anómalas,
       en validación y en prueba
  F03  curvas ROC en prueba: los dos autoencoders contra las líneas base
  F04  el error de las ventanas normales en el tiempo: la deriva
  F05  el error por banda de severidad (conteo de reglas) y por banda W
  F06  tres episodios reales: el error subiendo antes de la alarma
  F07  distribución del adelanto de los avisos
  F08  sensibilidad de hiperparámetros (04c): error en normales contra AUC
=============================================================================
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

if RUTA_BASE not in sys.path:
    sys.path.append(RUTA_BASE)

# Mismo criterio que el PARCHE EQUIPO del resto del pipeline.
EQUIPO = os.environ.get('EQUIPO', 'CR010')
RUTA_EQ = RUTA_BASE if EQUIPO == 'CR010' else f'{RUTA_BASE}/{EQUIPO}'
RUTA_FIG = f'{RUTA_EQ}/figuras'
os.makedirs(RUTA_FIG, exist_ok=True)
print(f'Equipo: {EQUIPO} | figuras en: {RUTA_FIG}')

import numpy as np
import pandas as pd
import matplotlib
if 'google.colab' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

W = 60
MINUTOS_VENTANA = 10
# PARCHE P8 -- el detector principal pasa al checkpoint de MÍNIMA PÉRDIDA de
# validación (autoencoder_W60_normales.pt), que ya estaba entrenado. Razón,
# sin mirar prueba: el criterio de máxima separación elige épocas inestables
# (04e, 3 semillas: épocas 7, 30 y 47, sd 20 sobre 100) porque maximiza una
# razón de medianas ruidosa, y además usa las etiquetas de las reglas en
# validación; la mínima pérdida elige 97-98 en las tres semillas y no usa
# etiquetas. La separación se sigue reportando al lado como comparación.
MODELO_AE = 'AE_normales'            # el principal, como en Fase B y paso 06
PERCENTIL_UMBRAL = 95                # el umbral de los pasos B y 06

ARCHIVO_ERROR = f'{RUTA_EQ}/{EQUIPO}_error_reconstruccion_W{W}.csv'
ARCHIVO_VENTANAS = f'{RUTA_EQ}/{EQUIPO}_ventanas_W{W}.csv'
ARCHIVO_FEATURES = f'{RUTA_EQ}/{EQUIPO}_features_W{W}.csv'
ARCHIVO_CORTE = f'{RUTA_EQ}/{EQUIPO}_corte_split.json'
ARCHIVO_EPOCAS = f'{RUTA_EQ}/04b_comparacion_epocas.csv'
ARCHIVO_EPISODIOS = f'{RUTA_EQ}/06_episodios.csv'
ARCHIVO_04C = f'{RUTA_EQ}/04c_barrido_hiperparametros_v2.csv'

VARIABLES = ['CM', 'PI', 'PDF', 'PEL', 'T7', 'T8', 'T9',
             'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4']

# Estilo sobrio, legible impreso en blanco y negro: un color de acento para
# lo que importa (el autoencoder principal, las anómalas) y grises para el
# resto.
AZUL, NARANJO, GRIS, GRIS_OSC = '#1f5fa8', '#d1642a', '#9a9a9a', '#4a4a4a'
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 200, 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.25})


def guardar(fig, nombre):
    ruta = f'{RUTA_FIG}/{nombre}.png'
    fig.tight_layout()
    fig.savefig(ruta, bbox_inches='tight')
    plt.close(fig)
    print(f'  guardada: {ruta}')


def leer_csv_pc(ruta):
    """CSV con ; y coma decimal (los que escriben 04, 04c y 06)."""
    return pd.read_csv(ruta, sep=';', decimal=',', encoding='utf-8-sig')


# =============================================================================
# DATOS COMUNES
# =============================================================================
print('\nCargando...')
corte = json.load(open(ARCHIVO_CORTE))
T_VAL = pd.Timestamp(corte['T_val'])
LLAVES = ['tramo_id', 'ventana_id']

err = pd.read_csv(ARCHIVO_ERROR)
ven = pd.read_csv(ARCHIVO_VENTANAS, parse_dates=['timestamp_inicio'],
                  usecols=lambda c: c in LLAVES + [
                      'timestamp_inicio', 'n_reglas_activas', 'banda_severidad',
                      'banda_w', 'w_max', 'sensor_origen', 'etiqueta_ventana'])
d = err.drop(columns=['n_reglas_activas'], errors='ignore').merge(
    ven, on=LLAVES, how='left', validate='one_to_one')
d['alarma'] = d.n_reglas_activas > 0
col_ae = f'err_{MODELO_AE}'
cols_err = [c for c in d.columns if c.startswith('err_')]

# z_medias: la línea base del paso 06, calculada igual (medias de ventana
# estandarizadas con las quietas de validación, máximo |z| sobre los 14).
try:
    medias = [f'{v}_media' for v in VARIABLES]
    feat = pd.read_csv(ARCHIVO_FEATURES, usecols=LLAVES + medias)
    d = d.merge(feat, on=LLAVES, how='left', validate='one_to_one')
    ref = d[(d.conjunto == 'val') & ~d.alarma]
    mu, sd = ref[medias].mean(), ref[medias].std().replace(0, 1.0)
    d['z_medias'] = ((d[medias] - mu) / sd).abs().max(axis=1)
except Exception as e:                                   # noqa: BLE001
    print(f'  (sin z_medias: {e})')

umbral = float(np.percentile(d.loc[(d.conjunto == 'val') & ~d.alarma, col_ae],
                             PERCENTIL_UMBRAL))
print(f'Ventanas de val + test: {len(d):,} | umbral P{PERCENTIL_UMBRAL} '
      f'({MODELO_AE}) = {umbral:.5f}')


# =============================================================================
# F01 — curva de entrenamiento
# =============================================================================
print('\nF01 curva de entrenamiento')
if os.path.exists(ARCHIVO_EPOCAS):
    h = leer_csv_pc(ARCHIVO_EPOCAS)
    ep = h['epoca'] if 'epoca' in h else np.arange(1, len(h) + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(ep, h['train'], color=GRIS_OSC, label='Pérdida en entrenamiento')
    ax.plot(ep, h['val_normal'], color=AZUL, label='Pérdida en normales de validación')
    i_min = int(np.argmin(h['val_normal']))
    ax.axvline(ep.iloc[i_min] if hasattr(ep, 'iloc') else ep[i_min],
               color=AZUL, ls='--', lw=1)
    ax.set_yscale('log')
    ax.set_xlabel('Época')
    ax.set_ylabel('Error cuadrático medio (log)')
    ax2 = ax.twinx()
    ax2.plot(ep, h['auc'], color=NARANJO, lw=1, alpha=0.8, label='AUC en validación')
    ax2.set_ylabel('AUC en validación', color=NARANJO)
    ax2.grid(False)
    lineas = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax.legend(lineas, [l.get_label() for l in lineas], loc='upper right', fontsize=8)
    ax.set_title(f'Curva de entrenamiento — mínimo de validación en la época '
                 f'{i_min + 1}', fontsize=10)
    guardar(fig, 'F01_curva_entrenamiento')
else:
    print('  (falta 04b_comparacion_epocas.csv: se salta)')


# =============================================================================
# F02 — distribución del error
# =============================================================================
print('\nF02 distribución del error')
fig, ejes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
bins = np.logspace(np.log10(d[col_ae].min()), np.log10(d[col_ae].max()), 60)
for ax, conj, nombre in zip(ejes, ['val', 'test'], ['Validación', 'Prueba']):
    s = d[d.conjunto == conj]
    ax.hist(s.loc[~s.alarma, col_ae], bins=bins, color=AZUL, alpha=0.65,
            density=True, label=f'Normales (n = {int((~s.alarma).sum()):,})')
    ax.hist(s.loc[s.alarma, col_ae], bins=bins, color=NARANJO, alpha=0.55,
            density=True, label=f'Anómalas (n = {int(s.alarma.sum()):,})')
    ax.axvline(umbral, color=GRIS_OSC, ls='--', lw=1, label=f'Umbral P{PERCENTIL_UMBRAL}')
    ax.set_xscale('log')
    auc = roc_auc_score(s.alarma, s[col_ae])
    ax.set_title(f'{nombre} — AUC = {auc:.3f}', fontsize=10)
    ax.set_xlabel('Error de reconstrucción (log)')
ejes[0].set_ylabel('Densidad')
ejes[1].legend(fontsize=8)
guardar(fig, 'F02_distribucion_error')


# =============================================================================
# F03 — ROC en prueba
# =============================================================================
print('\nF03 ROC en prueba')
te = d[d.conjunto == 'test']
puntajes = [(col_ae, f'Autoencoder ({MODELO_AE}, principal)', AZUL, 2.2)]
puntajes += [(c, f'Autoencoder ({c[4:]})', AZUL, 1.0)
             for c in cols_err if c != col_ae and c != 'err_' and c != 'error_reconstruccion']
if 'z_medias' in te:
    puntajes.append(('z_medias', 'Línea base: desviación de medias', NARANJO, 1.4))
puntajes.append(('w_max', 'Índice W (máximo por sensor)', GRIS_OSC, 1.4))
fig, ax = plt.subplots(figsize=(5.2, 5))
for col, nombre, color, lw in puntajes:
    fpr, tpr, _ = roc_curve(te.alarma, te[col])
    estilo = '-' if lw > 1.5 or col in ('z_medias', 'w_max') else ':'
    ax.plot(fpr, tpr, color=color, lw=lw, ls=estilo,
            label=f'{nombre}: AUC {roc_auc_score(te.alarma, te[col]):.3f}')
ax.plot([0, 1], [0, 1], color=GRIS, lw=0.8, ls='--', label='Azar: AUC 0,5')
ax.set_xlabel('Tasa de falsas alarmas (1 − especificidad)')
ax.set_ylabel('Tasa de detección (sensibilidad)')
ax.set_title('Detección en prueba (período no visto)', fontsize=10)
ax.legend(fontsize=7.5, loc='lower right')
guardar(fig, 'F03_roc_prueba')


# =============================================================================
# F04 — la deriva: error de las normales en el tiempo
# =============================================================================
print('\nF04 deriva en el tiempo')
nor = d[~d.alarma].copy()
nor['dia'] = nor.timestamp_inicio.dt.floor('D')
diario = nor.groupby('dia')[col_ae].agg(['median', 'size'])
diario = diario[diario['size'] >= 6]                      # al menos 1 h de normales
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(diario.index, diario['median'], marker='o', ms=3, lw=1, color=AZUL,
        label='Mediana diaria del error, ventanas normales')
ax.plot(diario.index, diario['median'].rolling(7, min_periods=3).median(),
        color=GRIS_OSC, lw=2, label='Mediana móvil de 7 días')
ax.axhline(umbral, color=NARANJO, ls='--', lw=1, label=f'Umbral P{PERCENTIL_UMBRAL} (calibrado en validación)')
ax.axvline(T_VAL, color=GRIS, lw=1.2)
ax.text(T_VAL, ax.get_ylim()[1], '  inicio de prueba', va='top', fontsize=8, color=GRIS_OSC)
ax.set_yscale('log')
ax.set_ylabel('Error de reconstrucción (log)')
ax.set_title('El error de la operación normal sube en el período de prueba: deriva',
             fontsize=10)
ax.legend(fontsize=8, loc='upper left')
fig.autofmt_xdate()
guardar(fig, 'F04_deriva_temporal')


# =============================================================================
# F05 — error por banda
# =============================================================================
print('\nF05 error por banda')
fig, ejes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, col, titulo in zip(ejes, ['banda_severidad', 'banda_w'],
                           ['Banda por número de reglas activas', 'Banda W (índice de Wasserstein)']):
    if col not in d:
        ax.set_visible(False)
        continue
    orden = sorted(d[col].dropna().unique())
    datos = [d.loc[d[col] == b, col_ae] for b in orden]
    # sin 'labels=': matplotlib 3.9 lo renombró; las etiquetas van aparte
    caja = ax.boxplot(datos, showfliers=False, patch_artist=True)
    ax.set_xticks(range(1, len(orden) + 1))
    ax.set_xticklabels([f'{b}\n(n = {len(x):,})' for b, x in zip(orden, datos)])
    for p in caja['boxes']:
        p.set_facecolor('#dbe6f3')
        p.set_edgecolor(AZUL)
    ax.set_yscale('log')
    ax.set_title(titulo, fontsize=10)
ejes[0].set_ylabel('Error de reconstrucción (log)')
guardar(fig, 'F05_error_por_banda')


# =============================================================================
# F06 — tres episodios reales
# =============================================================================
print('\nF06 casos')
if os.path.exists(ARCHIVO_EPISODIOS):
    eps = leer_csv_pc(ARCHIVO_EPISODIOS)
    col_ant = f'anticip_{MODELO_AE}_min'
    casos = (eps[eps[col_ant] > 0].sort_values(col_ant, ascending=False)
             .drop_duplicates('tramo_id').head(3))
    if len(casos):
        fig, ejes = plt.subplots(len(casos), 1, figsize=(9, 2.6 * len(casos)))
        ejes = np.atleast_1d(ejes)
        for ax, (_, c) in zip(ejes, casos.iterrows()):
            t = d[d.tramo_id == c.tramo_id].sort_values('ventana_id')
            x = t.ventana_id * MINUTOS_VENTANA
            for _, v in t[t.alarma].iterrows():
                ax.axvspan(v.ventana_id * MINUTOS_VENTANA,
                           (v.ventana_id + 1) * MINUTOS_VENTANA,
                           color=NARANJO, alpha=0.15, lw=0)
            ax.plot(x, t[col_ae], marker='o', ms=3, color=AZUL, lw=1.2)
            ax.axhline(umbral, color=GRIS_OSC, ls='--', lw=1)
            ax.axvline(c.ventana_id * MINUTOS_VENTANA, color=NARANJO, lw=1.5)
            ax.set_yscale('log')
            sensor = t.loc[t.ventana_id == c.ventana_id, 'sensor_origen']
            sensor = sensor.iloc[0] if len(sensor) else '?'
            ax.set_title(f'Tramo {c.tramo_id}: inicio de alarma {c.regla_mas_grave} '
                         f'({c.componente}, criticidad {c.criticidad}); '
                         f'aviso {int(c[col_ant])} min antes; sensor más corrido: {sensor}',
                         fontsize=9)
            ax.set_ylabel('Error (log)')
        ejes[-1].set_xlabel('Minutos desde el inicio del tramo '
                            '(sombreado = ventanas con alarma; línea naranja = inicio; '
                            'punteada = umbral)')
        guardar(fig, 'F06_casos_anticipacion')
    else:
        print('  (ningún inicio avisado: se salta)')
else:
    print('  (falta 06_episodios.csv: se salta)')


# =============================================================================
# F07 — adelanto de los avisos
# =============================================================================
print('\nF07 adelanto')
if os.path.exists(ARCHIVO_EPISODIOS):
    eps = leer_csv_pc(ARCHIVO_EPISODIOS)
    a = eps.loc[eps[f'anticip_{MODELO_AE}_min'] > 0, f'anticip_{MODELO_AE}_min']
    if len(a):
        conteo = a.value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(7, 3.6))
        colores = [AZUL if m == conteo.index.min() else GRIS for m in conteo.index]
        ax.bar(conteo.index.astype(int).astype(str), conteo.values, color=colores)
        for i, v in enumerate(conteo.values):
            ax.text(i, v, str(v), ha='center', va='bottom', fontsize=8)
        ax.set_xlabel('Minutos de adelanto')
        ax.set_ylabel('Inicios de alarma avisados')
        ax.set_title(f'{len(a)} de {len(eps)} inicios avisados '
                     f'({100 * len(a) / len(eps):.0f} %); adelanto mediano '
                     f'{a.median():.0f} min', fontsize=10)
        ax.grid(axis='x', visible=False)
        guardar(fig, 'F07_adelanto')
else:
    print('  (falta 06_episodios.csv: se salta)')


# =============================================================================
# F08 — sensibilidad de hiperparámetros
# =============================================================================
print('\nF08 sensibilidad (04c)')
if os.path.exists(ARCHIVO_04C):
    b = leer_csv_pc(ARCHIVO_04C)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    base = b.configuracion.str.startswith('base')
    ax.scatter(b.loc[~base, 'err_normal'], b.loc[~base, 'auc'], s=45, color=GRIS)
    ax.scatter(b.loc[base, 'err_normal'], b.loc[base, 'auc'], s=70, color=AZUL)
    for _, r in b.iterrows():
        ax.annotate(r.configuracion, (r.err_normal, r.auc), fontsize=8,
                    xytext=(5, 4), textcoords='offset points')
    ax.set_xlabel('Error en normales de validación (menor es mejor)')
    ax.set_ylabel('AUC en validación')
    ax.set_title('Reconstruir mejor lo normal no mejora la detección', fontsize=10)
    guardar(fig, 'F08_sensibilidad_hiperparametros')
else:
    print('  (falta 04c_barrido_hiperparametros_v2.csv: se salta)')

print('\nLISTO.')

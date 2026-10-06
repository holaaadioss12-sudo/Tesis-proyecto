"""
PASO 08 — UN SOLO EXCEL PARA POWER BI CON LOS TRES CHANCADORES
=============================================================================
Entrada:  las carpetas de resultados de cada equipo (CR010 en la raiz o en
          tesis_chancadores/CR010, CR009 y CR011 en su subcarpeta):
            06_resumen.csv, 06_precursores.csv, 06_episodios.csv
            04b_comparacion_epocas.csv, B1_resultados.csv, B2_resultados.csv,
            B2_variable_causante.csv, {EQUIPO}_eventos_por_regla.csv,
            {EQUIPO}_wasserstein_bandas_W60*.csv,
            {EQUIPO}_error_reconstruccion_W60.csv + {EQUIPO}_ventanas_W60.csv
Salida:   tesis_chancadores/powerbi/tesis_powerbi.xlsx, una hoja por tema,
          cada una con la columna 'equipo' y los numeros ya como numeros.

No calcula nada nuevo salvo la deriva diaria (mediana del error de las
ventanas normales por dia), que es la misma cuenta de la figura F04. Corre
en CPU en uno o dos minutos.

POR QUE EXISTE. Los archivos de resultados vienen en dos formatos (';' con
coma decimal y ',' con punto decimal) y varios se llaman igual en las tres
carpetas. Power BI los lee mal si no se configura cada uno. Aca se leen con
el formato correcto, se les agrega el equipo y se juntan.
"""
import os

import pandas as pd

RUTA_BASE = '/content/drive/MyDrive/tesis_chancadores'
RUTA_BASE = os.environ.get('RUTA_BASE', RUTA_BASE)
EQUIPOS = ['CR009', 'CR010', 'CR011']
W = 60
SALIDA_DIR = f'{RUTA_BASE}/powerbi'
SALIDA = f'{SALIDA_DIR}/tesis_powerbi.xlsx'


def carpeta(eq):
    """CR010 puede estar en la raiz (corrida original) o en su subcarpeta."""
    sub = f'{RUTA_BASE}/{eq}'
    if os.path.isdir(sub):
        return sub
    return RUTA_BASE if eq == 'CR010' else None


def leer(ruta):
    """Lee cualquiera de los dos formatos que generan los pasos."""
    with open(ruta, encoding='utf-8-sig') as f:
        cab = f.readline()
    if cab.count(';') > cab.count(','):
        return pd.read_csv(ruta, sep=';', decimal=',', encoding='utf-8-sig')
    return pd.read_csv(ruta, encoding='utf-8-sig')


TABLAS = {
    # hoja                  archivo (con {eq} si lleva el prefijo del equipo)
    'resumen_06':           '06_resumen.csv',
    'precursores_06':       '06_precursores.csv',
    'episodios_06':         '06_episodios.csv',
    'entrenamiento_04b':    '04b_comparacion_epocas.csv',
    'faseB_B1':             'B1_resultados.csv',
    'faseB_B2':             'B2_resultados.csv',
    'faseB_causante':       'B2_variable_causante.csv',
    'reglas_eventos':       '{eq}_eventos_por_regla.csv',
    'wasserstein_bandas':   '{eq}_wasserstein_bandas_W60.csv',
    'wasserstein_eventos':  '{eq}_wasserstein_bandas_W60_eventos.csv',
}

hojas = {k: [] for k in TABLAS}
hojas['error_ventanas'] = []
hojas['deriva_diaria'] = []
hojas['error_resumen'] = []

for eq in EQUIPOS:
    c = carpeta(eq)
    if c is None:
        print(f'{eq}: sin carpeta, se salta')
        continue
    print(f'{eq}: {c}')
    for hoja, nombre in TABLAS.items():
        ruta = f'{c}/{nombre.format(eq=eq)}'
        if not os.path.exists(ruta):
            print(f'   falta {os.path.basename(ruta)}')
            continue
        d = leer(ruta)
        d.insert(0, 'equipo', eq)
        hojas[hoja].append(d)

    # error por ventana + su fecha (la fecha esta en el archivo de ventanas)
    r_err = f'{c}/{eq}_error_reconstruccion_W{W}.csv'
    r_ven = f'{c}/{eq}_ventanas_W{W}.csv'
    if os.path.exists(r_err):
        e = leer(r_err)
        if os.path.exists(r_ven):
            v = pd.read_csv(r_ven, usecols=['tramo_id', 'ventana_id',
                                            'timestamp_inicio'])
            e = e.merge(v, on=['tramo_id', 'ventana_id'], how='left')
            e['timestamp_inicio'] = pd.to_datetime(e['timestamp_inicio'])
            # deriva: mediana diaria del error de las ventanas normales
            n = e[e['es_normal'].astype(str) == 'True'].copy()
            n['fecha'] = n['timestamp_inicio'].dt.floor('D')
            dd = (n.groupby(['fecha', 'conjunto'])
                   .agg(mediana_error_sep=('err_AE_normales_sep', 'median'),
                        mediana_error_min=('err_AE_normales', 'median'),
                        n_ventanas=('ventana_id', 'size'))
                   .reset_index())
            dd.insert(0, 'equipo', eq)
            hojas['deriva_diaria'].append(dd)
        else:
            print(f'   falta {os.path.basename(r_ven)}: error sin fecha')
        cols = [x for x in ['tramo_id', 'ventana_id', 'conjunto',
                            'timestamp_inicio', 'err_AE_normales',
                            'err_AE_normales_sep', 'etiqueta_ventana',
                            'n_reglas_activas', 'banda_severidad',
                            'es_normal'] if x in e.columns]
        e = e[cols]
        e.insert(0, 'equipo', eq)
        hojas['error_ventanas'].append(e)

        # resumen del error: por conjunto y tipo, y por banda de severidad
        e['tipo'] = e['es_normal'].astype(str).map({'True': 'normal',
                                                    'False': 'anomala'})
        for grupo in (['conjunto', 'tipo'], ['conjunto', 'banda_severidad']):
            for col, ck in (('err_AE_normales', 'perdida_minima'),
                            ('err_AE_normales_sep', 'separacion_maxima')):
                r = (e.groupby(grupo)[col]
                      .agg(n='size', media='mean', mediana='median',
                           p95=lambda s: s.quantile(0.95))
                      .reset_index()
                      .rename(columns={grupo[1]: 'grupo'}))
                r.insert(0, 'agrupado_por', grupo[1])
                r.insert(0, 'checkpoint', ck)
                r.insert(0, 'equipo', eq)
                hojas['error_resumen'].append(r)

os.makedirs(SALIDA_DIR, exist_ok=True)
with pd.ExcelWriter(SALIDA) as xw:
    for hoja, partes in hojas.items():
        if not partes:
            continue
        d = pd.concat(partes, ignore_index=True)
        d.to_excel(xw, sheet_name=hoja, index=False)
        print(f'  hoja {hoja:20s} {len(d):7,} filas')
print(f'\nGuardado: {SALIDA}')

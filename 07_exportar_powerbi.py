# -*- coding: utf-8 -*-
"""
=============================================================================
PASO 07 — TABLAS PARA POWER BI (LOS TRES EQUIPOS EN UN SOLO LUGAR)
=============================================================================
Entrada:  salidas de los pasos 02b, 03a, 03b, 04, 04c, 04d, 04f, 04g, 05a, 05b, 05c, 06 y 06b de cada equipo,
          y del 04e y el 03a (k común) en la carpeta base
Salida:   carpeta powerbi/ con un CSV por tema (separador ';' y coma decimal)
"""
import os
import sys

try:
    from google.colab import drive
    drive.mount('/content/drive')
    RUTA_BASE = '/content/drive/MyDrive/tesis_chancadores'
except ImportError:
    RUTA_BASE = os.environ.get('RUTA_BASE', '.')

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EQUIPOS = os.environ.get('EQUIPOS', 'CR009,CR010,CR011').split(',')
W = 60
PERCENTIL_UMBRAL = 95
VARIABLES = ['CM', 'PI', 'PDF', 'PEL', 'T7', 'T8', 'T9',
             'T1', 'T2', 'T5', 'V1', 'V2', 'V3', 'V4']
LLAVES = ['tramo_id', 'ventana_id']
SALIDA = f'{RUTA_BASE}/powerbi'
os.makedirs(SALIDA, exist_ok=True)


def carpeta(eq):
    if eq == 'CR010' and not os.path.isdir(f'{RUTA_BASE}/CR010'):
        return RUTA_BASE
    return f'{RUTA_BASE}/{eq}'


def leer(ruta):
    """Lee un CSV del pipeline, con ';' y coma decimal o con ',' y punto."""
    if not os.path.exists(ruta):
        return None
    with open(ruta, encoding='utf-8-sig') as f:
        cab = f.readline()
    if cab.count(';') > cab.count(','):
        return pd.read_csv(ruta, sep=';', decimal=',', encoding='utf-8-sig', low_memory=False)
    return pd.read_csv(ruta, encoding='utf-8-sig', low_memory=False)


def con_clave(df):
    """Columna única equipo-tramo-ventana para relacionar las tablas en Power BI."""
    df.insert(1, 'clave', df.equipo + '-' + df.tramo_id.astype(str) + '-' + df.ventana_id.astype(str))
    return df


def guardar(df, nombre):
    df.to_csv(f'{SALIDA}/{nombre}', sep=';', decimal=',', index=False, encoding='utf-8-sig')
    print(f'  {nombre:<32} {len(df):>9,} filas')


print('=' * 72)
print('PASO 07 — TABLAS PARA POWER BI')
print('=' * 72)

ventanas, por_variable, umbrales, nivel_n, curvas = [], [], [], [], []
resumen, agrup, perfil, proto, anticip, episodios = [], [], [], [], [], []
tabla_exp, delta_var, etiquetas, deriva, det_nivel = [], [], [], [], []
pron_met, pron_aviso, pron_ven, deriva_var, forma_err = [], [], [], [], []
calor_res, calor_mes, error_inicios, causa_raiz = [], [], [], []
eleccion_k, atribucion, metricas, consistencia, agr_det, escalas = [], [], [], [], [], []
eleccion_k_res, val_ind, val_dist, val_acu, val_conf, val_des, val_fir = [], [], [], [], [], [], []
pca_barr, pca_cargas, pca_det, pca_comp, pca_loc, pca_delta = [], [], [], [], [], []
inicios_tipo, prec_contexto = [], []

for eq in EQUIPOS:
    r = carpeta(eq)
    err = leer(f'{r}/{eq}_error_reconstruccion_W{W}.csv')
    if err is None:
        print(f'{eq}: falta {eq}_error_reconstruccion_W{W}.csv (paso 04); se salta.')
        continue
    print(f'{eq}: {r}')

    # --- ventanas de validación y prueba con su error --------------------
    sin_alarma = (err.n_reglas_activas == 0) if 'n_reglas_activas' in err.columns else err.es_normal
    base_umbral = err.loc[(err.conjunto == 'val') & sin_alarma, 'error_reconstruccion']
    umbral_ae = float(np.percentile(base_umbral, PERCENTIL_UMBRAL))
    v = err.copy()
    v.insert(0, 'equipo', eq)
    v = con_clave(v)
    v['estado_reglas'] = np.where(sin_alarma, 'normal', 'anomala')
    v['umbral_ae'] = umbral_ae
    parada = ('mayor separación' if 'err_AE_normales_sep' in err.columns
              and np.allclose(err.error_reconstruccion, err.err_AE_normales_sep) else 'menor pérdida')
    v['punto_parada'] = parada
    v['marcada_ae'] = (v.error_reconstruccion > umbral_ae).astype(int)

    n05a = leer(f'{r}/05a_nivel_ventanas.csv')
    n05b = leer(f'{r}/05b_nivel_revisado.csv')
    if n05a is not None:
        c = [x for x in ['timestamp_inicio', 'N_empinada', 'banda_N_empinada_natural', 'criticidad_min']
             if x in n05a.columns and x not in v.columns]
        v = v.merge(n05a[LLAVES + c], on=LLAVES, how='left', suffixes=('', '_05a'))
    if n05b is not None:
        c = [x for x in ['N_presencia', 'nivel_presencia', 'nivel_combinado', 'subsistemas', 'alarmas']
             if x in n05b.columns]
        v = v.merge(n05b[LLAVES + c], on=LLAVES, how='left')
    loc = leer(f'{r}/04c_localizacion_ventanas.csv')
    if loc is not None:                  # localización y etiqueta componente · nivel (paso 04c)
        c = [x for x in ['n_sobre_umbral', 'var_delta_1', 'pct_exceso_1', 'var_delta_2', 'var_delta_3',
                         'componente', 'nivel_texto', 'etiqueta'] if x in loc.columns]
        v = v.merge(loc[LLAVES + c], on=LLAVES, how='left')
        if 'etiqueta' in v.columns and v.etiqueta.notna().all():
            _no = (v.etiqueta == 'no marcada').astype(int)
            if (_no == v.marcada_ae).any():
                print(f'  OJO: la etiqueta del 04c no coincide con marcada_ae; ¿el 04c es de otra corrida del 04?')
    else:
        print(f'  {eq}: falta 04c_localizacion_ventanas.csv (paso 04c); sin etiqueta componente · nivel.')
    g = leer(f'{r}/04g_ventanas.csv')
    if g is not None:                    # lo que dicen las alarmas y si coincide (paso 04g)
        g = g.rename(columns={'reglas': 'reglas_que_sonaron'})
        c = [x for x in ['mes', 'reglas_que_sonaron', 'componente_alarma', 'componentes_alarma', 'acuerdo_variable',
                         'acuerdo_estricto', 'acuerdo_amplio', 'caso', 'lectura', 'tipo_lectura', 'sentido',
                         'sentido_vigilado', 'firma'] if x in g.columns and x not in v.columns]
        v = v.merge(g[LLAVES + c], on=LLAVES, how='left')
    else:
        print(f'  {eq}: falta 04g_ventanas.csv (paso 04g); sin la comparación con las alarmas.')
    ventanas.append(v)

    # --- error por variable, en formato largo ----------------------------
    u_var = leer(f'{r}/04_umbral_por_variable.csv')
    col_ev = [f'evar_{x}' for x in VARIABLES if f'evar_{x}' in v.columns]
    if col_ev:
        larga = v[['equipo', 'clave', 'conjunto', 'estado_reglas'] + col_ev].melt(
            id_vars=['equipo', 'clave', 'conjunto', 'estado_reglas'],
            var_name='variable', value_name='error_variable')
        larga['variable'] = larga.variable.str.replace('evar_', '', regex=False)
        if u_var is not None:
            larga = larga.merge(u_var.rename(columns={'umbral_P95': 'umbral_variable'}),
                                on='variable', how='left')
            larga['sobre_umbral'] = (larga.error_variable > larga.umbral_variable).astype(int)
            larga['veces_umbral'] = larga.error_variable / larga.umbral_variable
        por_variable.append(larga)

        # --- causa raíz de cada ventana marcada (tablas 3 y 4 del paper) ---
        if u_var is not None:
            cr = larga[larga.clave.isin(v.loc[v.marcada_ae == 1, 'clave'])].copy()
            cr['exceso'] = (cr.error_variable - cr.umbral_variable).clip(lower=0)
            tot = cr.groupby('clave')[['exceso', 'error_variable']].transform('sum')
            cr['pct_exceso'] = 100 * cr.exceso / tot.exceso.where(tot.exceso > 0)   # como el 04c
            cr['pct_error'] = 100 * cr.error_variable / tot.error_variable          # como el paper, sobre las 14
            cr = cr[cr.exceso > 0].copy()
            cr['orden'] = cr.groupby('clave').exceso.rank(ascending=False, method='first').astype(int)
            c = [x for x in ['timestamp_inicio', 'nivel_texto', 'etiqueta', 'error_reconstruccion', 'umbral_ae']
                 if x in v.columns]
            cr = cr.merge(v[['clave'] + c], on='clave', how='left')
            texp = leer(f'{r}/04c_tabla_experto.csv')
            if texp is not None:
                t = texp[['variable'] + [x for x in ['componente_principal', 'criticidad_mas_alta'] if x in texp.columns]].copy()
                if 'modos_de_falla' in texp.columns:   # el más crítico; la lista completa está en pbi_tabla_experto
                    t['modo_falla_principal'] = texp.modos_de_falla.fillna('').str.split(' | ', regex=False).str[0]
                cr = cr.merge(t, on='variable', how='left')
            else:
                print(f'  {eq}: falta 04c_tabla_experto.csv (paso 04c); la causa raíz va sin componente.')
            cr = cr.round({'error_variable': 6, 'umbral_variable': 6, 'veces_umbral': 3, 'exceso': 6,
                           'pct_exceso': 2, 'pct_error': 2, 'error_reconstruccion': 6, 'umbral_ae': 6})
            causa_raiz.append(cr.sort_values(['clave', 'orden']))
    u = pd.DataFrame({'equipo': eq, 'variable': ['TOTAL'], 'umbral_P95': [umbral_ae]})
    if u_var is not None:
        u = pd.concat([u, u_var.assign(equipo=eq)[['equipo', 'variable', 'umbral_P95']]])
    umbrales.append(u)

    # --- resumen normal contra anómala por conjunto ---------------------
    for conj in ['val', 'test']:
        s = v[v.conjunto == conj]
        y = (s.estado_reglas == 'anomala').astype(int)
        en = s.loc[y == 0, 'error_reconstruccion'].median()
        ea = s.loc[y == 1, 'error_reconstruccion'].median()
        resumen.append({'equipo': eq, 'conjunto': conj, 'punto_parada': parada,
                        'ventanas_normales': int((y == 0).sum()), 'ventanas_anomalas': int(y.sum()),
                        'error_mediano_normal': en, 'error_mediano_anomala': ea,
                        'razon_anomala_normal': ea / en if en else np.nan,
                        'AUC': roc_auc_score(y, s.error_reconstruccion) if y.nunique() == 2 else np.nan,
                        'umbral_ae': umbral_ae,
                        'pct_marcadas_normales': 100 * s.loc[y == 0, 'marcada_ae'].mean(),
                        'pct_marcadas_anomalas': 100 * s.loc[y == 1, 'marcada_ae'].mean()})

    # --- nivel N de todas las ventanas (entrenamiento incluido) ----------
    if n05a is not None:
        n = n05a.copy()
        if n05b is not None:
            c = [x for x in ['N_presencia', 'nivel_presencia', 'nivel_combinado', 'subsistemas', 'alarmas']
                 if x in n05b.columns]
            n = n.merge(n05b[LLAVES + c], on=LLAVES, how='left')
        n.insert(0, 'equipo', eq)
        nivel_n.append(con_clave(n))

    # --- resto de las tablas, con la columna equipo ----------------------
    for lista, archivo in [(curvas, '03b_comparacion_epocas.csv'), (agrup, '05c_agrupamiento.csv'),
                           (perfil, '05c_perfil_grupos.csv'), (proto, '05c_prototipos.csv'),
                           (anticip, '06_resumen.csv'), (episodios, '06_episodios.csv'),
                           (tabla_exp, '04c_tabla_experto.csv'), (delta_var, '04c_delta_variables.csv'),
                           (etiquetas, '04c_etiquetas_resumen.csv'), (deriva, '04d_mensual.csv'), (deriva_var, '04d_variables_mensual.csv'),
                           (forma_err, '04f_forma_error.csv'), (calor_res, '02b_calor_resumen.csv'),
                           (calor_mes, '02b_calor_mensual.csv'), (error_inicios, '04c_error_alrededor_inicios.csv'),
                           (det_nivel, '04c_deteccion_nivel.csv'), (pron_met, '06b_pronostico_metricas.csv'),
                           (pron_aviso, '06b_pronostico_aviso.csv'), (pron_ven, '06b_pronostico_ventanas.csv'),
                           (eleccion_k, '03a_latente_temporal.csv'), (atribucion, '04c_atribucion.csv'),
                           (metricas, '04c_metricas.csv'), (consistencia, '04c_consistencia.csv'),
                           (agr_det, '05e_deteccion.csv'), (escalas, '05b_comparacion.csv'),
                           (eleccion_k_res, '03a_eleccion.csv'), (val_ind, '04g_indicador_variable.csv'),
                           (val_dist, '04g_distribucion.csv'), (val_acu, '04g_acuerdo.csv'),
                           (val_conf, '04g_confusion.csv'), (val_des, '04g_desacuerdos.csv'),
                           (val_fir, '04g_firmas.csv'), (pca_barr, '04h_pca_barrido.csv'),
                           (pca_cargas, '04h_pca_cargas.csv'), (pca_det, '04h_pca_deteccion.csv'),
                           (pca_comp, '04h_pca_comparaciones.csv'), (pca_loc, '04h_pca_localizacion.csv'),
                           (pca_delta, '04h_pca_delta.csv'), (inicios_tipo, '06_inicios_por_tipo.csv'),
                           (prec_contexto, '06_precursores_contexto.csv')]:
        t = leer(f'{r}/{archivo}')
        if t is not None:
            t.insert(0, 'equipo', eq)
            lista.append(t)

# --- transferencia entre equipos (04e, un archivo para los tres) --------
transfer, transfer_var, medias_eq, k_comun = [], [], [], []
for lista, archivo in [(transfer, '04e_transferencia.csv'), (transfer_var, '04e_transferencia_variables.csv'),
                       (medias_eq, '04e_medias_desviaciones.csv'), (k_comun, '03a_k_comun.csv')]:
    t = leer(f'{RUTA_BASE}/{archivo}')
    if t is not None:
        lista.append(t)

print(f'\nGuardando en {SALIDA}:')
for lista, nombre in [(ventanas, 'pbi_ventanas.csv'), (por_variable, 'pbi_error_por_variable.csv'),
                      (umbrales, 'pbi_umbrales.csv'), (resumen, 'pbi_resumen_deteccion.csv'),
                      (nivel_n, 'pbi_nivel_N.csv'), (curvas, 'pbi_entrenamiento.csv'),
                      (agrup, 'pbi_agrupamiento.csv'), (perfil, 'pbi_perfil_grupos.csv'),
                      (proto, 'pbi_prototipos.csv'), (anticip, 'pbi_anticipacion.csv'),
                      (episodios, 'pbi_episodios.csv'), (tabla_exp, 'pbi_tabla_experto.csv'),
                      (delta_var, 'pbi_delta_variables.csv'), (etiquetas, 'pbi_etiquetas.csv'),
                      (deriva, 'pbi_deriva_mensual.csv'), (det_nivel, 'pbi_deteccion_nivel.csv'),
                      (pron_met, 'pbi_pronostico_metricas.csv'), (pron_aviso, 'pbi_pronostico_aviso.csv'),
                      (pron_ven, 'pbi_pronostico_ventanas.csv'), (deriva_var, 'pbi_deriva_variables.csv'),
                      (transfer, 'pbi_transferencia.csv'), (transfer_var, 'pbi_transferencia_variables.csv'),
                      (forma_err, 'pbi_forma_error.csv'), (calor_res, 'pbi_calor_resumen.csv'),
                      (calor_mes, 'pbi_calor_mensual.csv'), (error_inicios, 'pbi_error_alrededor_inicios.csv'),
                      (causa_raiz, 'pbi_causa_raiz.csv'), (medias_eq, 'pbi_medias_equipos.csv'),
                      (eleccion_k, 'pbi_eleccion_k.csv'), (atribucion, 'pbi_atribucion.csv'),
                      (metricas, 'pbi_metricas.csv'), (consistencia, 'pbi_consistencia.csv'),
                      (agr_det, 'pbi_agrupamiento_deteccion.csv'), (escalas, 'pbi_nivel_escalas.csv'),
                      (eleccion_k_res, 'pbi_eleccion_k_resumen.csv'), (k_comun, 'pbi_k_comun.csv'),
                      (val_ind, 'pbi_validacion_variables.csv'), (val_dist, 'pbi_validacion_distribucion.csv'),
                      (val_acu, 'pbi_validacion_acuerdo.csv'), (val_conf, 'pbi_validacion_confusion.csv'),
                      (val_des, 'pbi_validacion_desacuerdos.csv'), (val_fir, 'pbi_validacion_firmas.csv'),
                      (pca_barr, 'pbi_pca_barrido.csv'), (pca_cargas, 'pbi_pca_cargas.csv'),
                      (pca_det, 'pbi_pca_deteccion.csv'), (pca_comp, 'pbi_pca_comparaciones.csv'),
                      (pca_loc, 'pbi_pca_localizacion.csv'), (pca_delta, 'pbi_pca_delta.csv'),
                      (inicios_tipo, 'pbi_anticipacion_inicios_tipo.csv'),
                      (prec_contexto, 'pbi_anticipacion_contexto.csv')]:
    if lista:
        df = pd.DataFrame(lista) if isinstance(lista[0], dict) else pd.concat(lista, ignore_index=True)
        guardar(df, nombre)

if resumen:
    print('\nResumen de la detección (error mediano por conjunto):')
    pd.set_option('display.width', 200)
    print(pd.DataFrame(resumen).round(4).to_string(index=False))
print('\nLISTO.')

# -*- coding: utf-8 -*-
"""
=============================================================================
CORRER SÓLO EL PASO 02, CON LOS PARCHES P11 Y P12
=============================================================================
Para el paso 10 hace falta {EQUIPO}_activos_por_ventana_W60.csv, que sale del
paso 02 con el PARCHE P11 v2. Este archivo evita editar la celda a mano: toma
el paso 02 de pipeline_01_05.py (el que está en Drive), le aplica los parches
si no los trae y lo ejecuta. No corre el 01 ni reentrena nada.

    P11  guarda los eventos que ARRANCAN por regla y ventana (archivo aparte;
         ventanas_W60.csv y features_W60.csv salen idénticos).
    P11 v2  guarda además los episodios ACTIVOS por regla y ventana (los que
         arrancan y los que vienen de antes): es lo que usa el paso 10.
    P12  si CR010 ya está en su subcarpeta, lee y escribe ahí.

En Colab, una celda por equipo:
    import os; os.environ['EQUIPO'] = 'CR010'      # luego CR009 y CR011
    %run /content/drive/MyDrive/tesis_chancadores/paso_02_eventos.py
=============================================================================
"""
import os

try:
    from google.colab import drive
    drive.mount('/content/drive')
    RUTA_BASE = '/content/drive/MyDrive/tesis_chancadores'
except ImportError:
    RUTA_BASE = os.environ.get('RUTA_BASE', '.')

FUENTE = os.environ.get('PIPELINE', f'{RUTA_BASE}/pipeline_01_05.py')
texto = open(FUENTE, encoding='utf-8').read()

# --- el paso 02: del docstring que lo abre al que abre el paso 03 ----------
_i = texto.index('PASO 02 — REGLAS, VENTANAS Y ETIQUETADO')
_i = texto.rindex('"""', 0, _i)
_f = texto.index('PASO 03 — VALIDACIÓN DE LA SEVERIDAD', _i)
_f = texto.rindex('"""', 0, _f)
codigo = texto[_i:_f]

# --- PARCHE P12 -------------------------------------------------------------
_v12 = ("RUTA_EQ = RUTA_BASE if EQUIPO == 'CR010' else f'{RUTA_BASE}/{EQUIPO}'\n"
        "_os.makedirs(RUTA_EQ, exist_ok=True)\n")
_n12 = ("RUTA_EQ = RUTA_BASE if EQUIPO == 'CR010' else f'{RUTA_BASE}/{EQUIPO}'\n"
        "# PARCHE P12 -- si CR010 ya se movió a su subcarpeta (como asume el paso 09),\n"
        "# el paso 02 lee y escribe ahí. Sin la subcarpeta, todo sigue igual.\n"
        "if EQUIPO == 'CR010' and _os.path.isdir(f'{RUTA_BASE}/CR010'):\n"
        "    RUTA_EQ = f'{RUTA_BASE}/CR010'\n"
        "_os.makedirs(RUTA_EQ, exist_ok=True)\n")
if 'PARCHE P12' not in codigo:
    assert codigo.count(_v12) == 1, 'No encontré dónde va el PARCHE P12.'
    codigo = codigo.replace(_v12, _n12)

# --- PARCHE P11 -------------------------------------------------------------
_v11 = ("dataset_features.to_csv(ARCHIVO_FEATURES, index=False)\n"
        "print(f'{ARCHIVO_FEATURES}')\n"
        "print(f'   {dataset_features.shape[0]:,} filas × {dataset_features.shape[1]} columnas')\n")
_n11 = _v11 + """
# PARCHE P11 -- eventos por regla y ventana, para el paso 10 (nivel de
# anomalía del profesor: N = suma de peso x veces que se activa cada regla).
# La matriz 'iniciados' ya está calculada en el bloque 4b (episodios que
# ARRANCAN en la ventana); sólo no se guardaba. Va en un archivo aparte para
# no cambiar ni una columna de ventanas_W60.csv.
ARCHIVO_EVENTOS_VENTANA = f'{RUTA_EQ}/{EQUIPO}_eventos_por_ventana_W{W}.csv'
_ev_ventana = iniciados.copy()
_ev_ventana.insert(0, 'ventana_id', ventanas['ventana_id'].to_numpy())
_ev_ventana.insert(0, 'tramo_id', ventanas['tramo_id'].to_numpy())
_ev_ventana.to_csv(ARCHIVO_EVENTOS_VENTANA, index=False)
print(f'{ARCHIVO_EVENTOS_VENTANA}   (PARCHE P11)')
print(f'   {_ev_ventana.shape[0]:,} filas × {_ev_ventana.shape[1]} columnas')
"""
if 'PARCHE P11' not in codigo:
    assert codigo.count(_v11) == 1, 'No encontré dónde va el PARCHE P11.'
    codigo = codigo.replace(_v11, _n11)

# --- PARCHE P11 v2 ----------------------------------------------------------
_v11b = "print(f'   {_ev_ventana.shape[0]:,} filas × {_ev_ventana.shape[1]} columnas')\n"
_n11b = _v11b + """
# PARCHE P11 v2 -- también los episodios ACTIVOS por ventana (la matriz
# 'activos' del bloque 4b: los que arrancan en la ventana y los que vienen de
# antes). Con sólo los que arrancan, una ventana en medio de una alarma larga
# queda en cero, como advierte el mismo bloque 4b. El paso 10 usa este archivo.
ARCHIVO_ACTIVOS_VENTANA = f'{RUTA_EQ}/{EQUIPO}_activos_por_ventana_W{W}.csv'
_act_ventana = activos.copy()
_act_ventana.insert(0, 'ventana_id', ventanas['ventana_id'].to_numpy())
_act_ventana.insert(0, 'tramo_id', ventanas['tramo_id'].to_numpy())
_act_ventana.to_csv(ARCHIVO_ACTIVOS_VENTANA, index=False)
print(f'{ARCHIVO_ACTIVOS_VENTANA}   (PARCHE P11 v2)')
print(f'   {_act_ventana.shape[0]:,} filas × {_act_ventana.shape[1]} columnas')
"""
if 'PARCHE P11 v2' not in codigo:
    assert codigo.count(_v11b) == 1, 'No encontré dónde va el PARCHE P11 v2.'
    codigo = codigo.replace(_v11b, _n11b)

if os.environ.get('SOLO_EXTRAER'):          # para verificar sin datos
    print(codigo)
else:
    print(f'Corriendo el paso 02 de {FUENTE} con los PARCHES P11, P11 v2 y P12.')
    exec(compile(codigo, 'paso_02', 'exec'), {'__name__': '__main__'})

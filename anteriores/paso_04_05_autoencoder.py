# -*- coding: utf-8 -*-
"""
=============================================================================
PASOS 04 Y 05 — AUTOENCODER: ENTRENAR Y EVALUAR (desde pipeline_01_05.py)
=============================================================================
Entrada:  pipeline_01_05.py, {EQUIPO}_limpio.csv, {EQUIPO}_ventanas_W60.csv
Salida:   autoencoder_W60_*.pt, autoencoder_W60_config.json, {EQUIPO}_error_reconstruccion_W60.csv
"""
import os

try:
    from google.colab import drive
    drive.mount('/content/drive')
    RUTA_BASE = '/content/drive/MyDrive/tesis_chancadores'
    EN_COLAB = True
except ImportError:
    RUTA_BASE = os.environ.get('RUTA_BASE', '.')
    EN_COLAB = False

FUENTE = os.environ.get('PIPELINE', f'{RUTA_BASE}/pipeline_01_05.py')
PASOS = os.environ.get('PASOS', '04,05').split(',')
SEGMENTOS = {'04': ('PASO 04-B', 'PASO 05 —'), '05': ('PASO 05 —', None)}

texto = open(FUENTE, encoding='utf-8').read()
for p in PASOS:
    desde, hasta = SEGMENTOS[p.strip()]
    i = texto.rindex('"""', 0, texto.index(desde))
    f = texto.rindex('"""', 0, texto.index(hasta, i)) if hasta else len(texto)
    codigo = texto[i:f]
    if not EN_COLAB:                     # para correrlo fuera de Colab
        codigo = (codigo.replace("from google.colab import drive\n", "")
                  .replace("drive.mount('/content/drive')\n", "")
                  .replace("RUTA_BASE = '/content/drive/MyDrive/tesis_chancadores'",
                           f"RUTA_BASE = {RUTA_BASE!r}"))
    print(f'\n>>> Paso {p} de {FUENTE} | equipo {os.environ.get("EQUIPO", "CR010")} | '
          f'TIPO_AE = {os.environ.get("TIPO_AE", "temporal")}, '
          f'LATENT_DIM = {os.environ.get("LATENT_DIM", "4")}')
    exec(compile(codigo, f'paso_{p}', 'exec'), {'__name__': '__main__'})

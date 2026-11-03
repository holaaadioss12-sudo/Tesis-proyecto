# Tesis-proyecto

Detección de anomalías en chancadores con un autoencoder temporal. Los scripts están numerados según los pasos de la metodología y se corren en ese orden, una vez por equipo (`EQUIPO` = CR009, CR010 o CR011).

| Paso | Script | Qué hace | Lee | Escribe |
|---|---|---|---|---|
| 01 | `01_preprocesamiento.py` | Limpieza de los datos crudos | `Dataframe - {EQUIPO}.csv` | `{EQUIPO}_limpio.csv`, `{EQUIPO}_corte_split.json` |
| 02 | `02_reglas_ventanas.py` | Reglas, episodios, ventanas de 10 min e índice W | paso 01 | `{EQUIPO}_ventanas_W60.csv`, `{EQUIPO}_activos_por_ventana_W60.csv`, `{EQUIPO}_features_W60.csv`, `{EQUIPO}_w_sensores_W60.csv` |
| 03a | `03a_autoencoder_eleccion_k.py` | Barrido de `latent_dim` (k) y de arquitecturas temporales | paso 01 | `03a_latente_temporal.csv`, `03a_codo.png`, `03a_resumen.csv`, `03a_atribucion.csv`, `03a_modelos/` |
| 03b | `03b_autoencoder_entrenamiento.py` | Entrena el autoencoder oficial (latente 60 x `LATENT_DIM`) | paso 01 | `autoencoder_W60_normales.pt`, `autoencoder_W60_config.json`, `normalizador_W60.npz`, `03b_curva_entrenamiento.png` |
| 04 | `04_deteccion.py` | Error de cada ventana, error por variable, umbral por variable | pasos 02 y 03b | `{EQUIPO}_error_reconstruccion_W60.csv`, `04_umbral_por_variable.csv`, `04_atribucion_variables.csv` |
| 05a | `05a_nivel_N.py` | Nivel N, cortes leve / media / grave y agrupamiento de las variables | pasos 02 y 04 | `05a_nivel_ventanas.csv`, `05a_cortes.csv`, `05a_agrupamiento.csv` |
| 05b | `05b_nivel_revisado.py` | N por presencia y nivel combinado (criticidad x subsistemas) | pasos 02, 04 y 05a | `05b_nivel_revisado.csv`, `05b_comparacion.csv`, `05b_cruces.csv` |
| 05c | `05c_agrupamiento_silueta.py` | K-means, DBSCAN, HDBSCAN y GMM sobre el latente, también después de UMAP, con silueta | pasos 03b, 04 y 05a | `05c_agrupamiento.csv`, `05c_perfil_grupos.csv`, `05c_silueta.png`, `05c_umap.png` |
| 06 | `06_anticipacion.py` | Anticipación, deriva y comparación con z_medias e índice W | pasos 02 y 04 | `06_resumen.csv`, `06_precursores.csv`, `06_deriva.csv`, `06_episodios.csv` |
| 07 | `07_exportar_powerbi.py` | Junta los resultados de los tres equipos en tablas para Power BI | pasos 03b a 06 | carpeta `powerbi/` (`pbi_ventanas.csv`, `pbi_error_por_variable.csv`, `pbi_resumen_deteccion.csv`, `pbi_nivel_N.csv` y otras) |

Módulos que usan los scripts (van en la misma carpeta): `reglas_engine.py`, `criticidad.py`, `indice_w.py`.

Variables de entorno principales: `EQUIPO`; en 03b, `TIPO_AE` (`temporal` o `colapsado`), `LATENT_DIM` (4 por defecto) y `EPOCAS_MAX` (100 por defecto; si se sube, el entrenamiento sigue desde la última época); en 03a, `REDES`, `LATENTES` y `SEMILLAS`; en 05c, `FUENTE` (`oficial` o `03a`) y `MODELO_REFERENCIA`.

La carpeta `anteriores/` guarda los scripts que ya no se corren (el pipeline en un solo archivo, el Random Forest, la Fase B y los experimentos previos del autoencoder).

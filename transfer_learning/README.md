# Actividad 3: transferencia de aprendizaje con YOLO11n en VisDrone

Prototipo de detección de movilidad urbana desde dron. Se parte de YOLO11n preentrenado en COCO (`yolo11n.pt`) y se hace fine-tuning en dos etapas. El resultado se compara contra tres modelos base.

| Archivo | Contenido |
|---|---|
| `visdrone.py` | Conversión de anotaciones VisDrone a 6 clases YOLO y muestreo por secuencia (tiene pruebas en `tests/test_visdrone.py`). |
| `prepare_data.py` | Descarga los zips oficiales y arma `data/visdrone6` (train, val y test) y `data/visdrone6_coco` (test con ids COCO para la línea base). |
| `experiments.py` | Entrenamiento (Etapa 1 congelada, Etapa 2, desde cero, una etapa) y evaluación en test con métricas por clase y matriz de confusión. |
| `figures.py` | Tablas y figuras construidas solo a partir de los artefactos de `runs/act3`. |
| `act3_transfer_learning.ipynb` | Notebook que reproduce el flujo completo y muestra los resultados. |
| `report/` | Generador del informe en PDF (máximo 5 páginas) a partir de los mismos artefactos. |

## Reproducción

Se necesita Python 3.12. Todas las corridas usan `seed=0`, `imgsz=640`, `batch=16`, AdamW y el mismo subconjunto de datos.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt nbconvert ipykernel pypdf
.venv/bin/python -m transfer_learning.prepare_data --download --delete-zips   # ~1.9 GB de descarga, ~0.8 GB en disco
.venv/bin/python -m transfer_learning.experiments zeroshot      # línea base COCO sin ajuste
.venv/bin/python -m transfer_learning.experiments stage1        # 10 épocas, backbone (módulos 0-10) congelado
.venv/bin/python -m transfer_learning.experiments stage2        # 30 épocas, todo descongelado, LR 2e-4 con decaimiento coseno
.venv/bin/python -m transfer_learning.experiments scratch       # 40 épocas desde pesos aleatorios
.venv/bin/python -m transfer_learning.experiments full_ft       # 40 épocas de fine-tuning en una sola etapa
```

### Windows con GPU NVIDIA (CUDA)

```powershell
git clone https://github.com/AlejoMoncada/yolo-realtime-pipeline.git
cd yolo-realtime-pipeline
git switch AlejoMoncada/g9-act3-transfer-learning
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.venv\Scripts\python.exe -m transfer_learning.prepare_data --download --delete-zips
.venv\Scripts\python.exe -m transfer_learning.experiments scratch --device 0 --workers 6
.venv\Scripts\python.exe -m transfer_learning.experiments full_ft --device 0 --workers 6
```

Cada corrida deja en `runs/act3/<nombre>/` el archivo `results.csv` (curvas por época), `args.yaml`, `train_info.json` y `weights/best.pt`. La evaluación en test queda en `runs/act3/eval/<nombre>/metrics.json` y en `runs/act3/eval/<nombre>_cm/`.

## Decisiones de datos

- **Clases:** person (pedestrian + people), bicycle, car (car + van), motorcycle, bus y truck. Las seis existen en COCO, así que la línea base se puede evaluar sin reentrenar.
- **Regiones excluidas:** los triciclos, las *ignored regions* y la categoría *others* se rellenan con gris (114). Así no cuentan como positivos ni como falsos positivos.
- **Splits:** se usan los oficiales de VisDrone (train, val y test-dev), que vienen de secuencias de video disjuntas. Del split de train se toman 2.000 imágenes, muestreadas en *round-robin* por secuencia y con semilla fija.

## Estado (borrador en revisión)

| Pieza | Estado |
|---|---|
| Datos (`data/visdrone6`, `stats.json`) | ✅ Listos. Los conteos de test coinciden entre los dos equipos. |
| COCO sin ajuste, Etapa 1, Etapa 2 | ✅ Entrenadas y evaluadas en test (Apple M4, MPS). |
| Desde cero | ✅ Solo las métricas de test (`runs/act3/eval/scratch/metrics.json`). Se entrenó en una GTX 1660 Super y aquí faltan sus curvas y pesos. |
| Fine-tuning en una etapa (`full_ft`) | ⏳ Entrenando en la GTX 1660 Super. |
| Ablación de LR (`stage2_lr1e3`) | ⏳ Entrenando en el M4. Mientras no termine, la figura del notebook muestra curvas parciales. |
| Notebook `act3_transfer_learning.ipynb` | ✅ Se ejecuta sin errores con los artefactos actuales. Se vuelve a ejecutar al final. |
| Informe `report/Informe_Actividad3_G9.pdf` | 📝 Borrador de 4 páginas (límite: 5). Se regenera con `python -m transfer_learning.report.build_report`. |

La evidencia que respalda cada cifra del informe está en el repositorio: `runs/act3/eval/*/metrics.json` (métricas en test por modelo), `runs/act3/eval/{split_gap,module_drift,object_sizes_test}.json`, `runs/act3/{stage1,stage2}/{results.csv,args.yaml,train_info.json}` y `runs/act3/stage2/weights/best.pt` (modelo propuesto).

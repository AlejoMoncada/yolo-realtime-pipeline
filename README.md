# Pipeline de visión en tiempo real

Pipeline YOLO con un modo síncrono de referencia y un modo paralelo con colas acotadas y sin pérdida. `benchmark.py` compara ambos modos sobre el mismo video local y los mismos identificadores de fotograma. Los resultados son mediciones, no promesas de mejora ni de una tasa de fotogramas objetivo.

Consulte también el [README en inglés](README.en.md) y el [resumen detallado en español](docs/implementation-summary.es.md). El [resumen en inglés](docs/implementation-summary.en.md) ofrece el mismo contenido técnico.

## Instalación, ejecución, benchmark y pruebas

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python pipeline.py ./clip.mp4 --mode threaded --imgsz 640 --device auto --conf 0.25 --warmup 2 --max-frames 300 --queue-size 2 --output-dir results
.venv/bin/python benchmark.py ./clip.mp4 --weights yolo11n.pt --device auto --conf 0.25 --imgsz 640 --warmup 2 --max-frames 300 --queue-size 2 --repeats 3 --output-dir benchmark-results
.venv/bin/python -m unittest discover -s tests -v
```

Para usar pip en lugar de uv, cree un entorno virtual de Python 3.12 con `python3.12 -m venv .venv`. Después instale las dependencias con `.venv/bin/python -m pip install -r requirements.txt`.

Pesos como `yolo11n.pt` pueden descargarse durante la primera inferencia real. `--help` no importa Ultralytics ni carga pesos. Revise las [opciones de licencia de Ultralytics](https://www.ultralytics.com/license) y la [documentación oficial](https://docs.ultralytics.com/) para conocer las condiciones de licencia y las obligaciones de atribución aplicables a su uso.

## Benchmark controlado

El benchmark acepta solo un archivo de video local existente. No acepta un índice de webcam ni una fuente de red. Por ejemplo:

```sh
.venv/bin/python benchmark.py ./clip.mp4 --repeats 5 --output-dir benchmark-results
.venv/bin/python benchmark.py ./clip.mp4 --imgsz 640 --optimized-imgsz 480 --repeats 3
```

### Formato ONNX y ritmo de cámara

`export_model.py` exporta los pesos con un tamaño de entrada fijo y nombra el archivo según la resolución, por ejemplo `yolo11n-320.onnx`. `--optimized-weights` aplica esos pesos solo al caso optimizado y exige `--optimized-imgsz` con la misma resolución de exportación:

```sh
.venv/bin/python export_model.py yolo11n.pt --imgsz 320
.venv/bin/python benchmark.py ./clip.mp4 --device cpu --imgsz 640 --optimized-imgsz 320 --optimized-weights yolo11n-320.onnx --repeats 3
```

Sin `--realtime`, el benchmark lee el archivo tan rápido como puede: las colas permanecen llenas y `end_to_end_ms` incluye esa contrapresión. Con `--realtime` (en `benchmark.py` y `pipeline.py`), el fotograma `n` se libera en `origen + n / FPS de la fuente`, como si llegara de una cámara. Entonces `end_to_end_ms` se mide desde esa llegada programada e incluye cualquier atraso acumulado, y `capture_lag_ms` indica cuánto tarde se leyó cada fotograma respecto de su llegada. El rendimiento queda limitado por los FPS de la fuente, por lo que este modo mide latencia, no rendimiento máximo. `--realtime` requiere que la fuente informe FPS positivos.

Cada ejecución escribe en un subdirectorio único `benchmark-<timestamp>-<id>/`. Calcula el hash de la fuente antes de medir y crea un subdirectorio para cada ejecución. Cada ejecución contiene `frames.csv` y `summary.json`. `comparison.json` registra:

- Identidad de entrada: ruta resuelta, tamaño en bytes y SHA-256; cantidad e identificadores de fotogramas seleccionados.
- Parámetros, metadatos del entorno y orden exacto de las repeticiones.
- Para cada caso: rendimiento por ejecución, rendimiento mediano, p95 de E2E por ejecución y su mediana, speedup, cambio porcentual frente a sync y rutas relativas a los archivos.

El rendimiento es la cantidad de fotogramas únicos completados dividida por el tiempo medido de la ejecución. El valor informado es la mediana de las repeticiones. El p95 de E2E de cada ejecución cubre desde el inicio de la lectura hasta la anotación y la visualización, e incluye la espera en colas. La comparación informa la mediana de esos valores p95. El speedup es el rendimiento mediano del caso dividido por el rendimiento mediano de sync. El cambio porcentual es `(case / sync - 1) * 100`. La inicialización del modelo y el calentamiento con imágenes ficticias quedan fuera de cada medición. El orden de los casos rota entre repeticiones para evitar que un caso use siempre la misma posición de caché.

Los casos sync y threaded usan los mismos pesos, dispositivo, confianza, resolución de inferencia e identificadores de fotograma. La opción `--optimized-imgsz` agrega un caso threaded separado con menor resolución. Su rendimiento no mide solo el efecto de la concurrencia ni demuestra igualdad de calidad predictiva. No se afirma precisión o calidad sin datos de referencia. El benchmark rechaza resultados vacíos, parciales o con identificadores distintos. También rechaza fuentes que no informan una cantidad positiva de fotogramas.

## Comportamiento y tiempos del pipeline

- Sync ejecuta lectura, predicción, anotación y visualización opcional en serie. En threaded, un hilo captura, lee y libera la fuente; un hilo invoca la predicción; el hilo principal anota y muestra los resultados. Dos colas FIFO acotadas conservan el orden y no descartan fotogramas.
- Una mayor capacidad de cola puede mejorar el solapamiento cuando las etapas tienen cargas distintas. También puede aumentar la latencia de cola y el uso de memoria. La contrapresión FIFO conserva el orden y evita pérdidas, pero no garantiza más rendimiento.
- El fin del video y `--max-frames` detienen el proceso con normalidad. `q` cancela el trabajo pendiente después del resultado que se está mostrando. Los errores de los hilos se propagan. Una llamada nativa bloqueada a `capture.read()` no se puede interrumpir por la fuerza. El programa informa si sigue bloqueada y no afirma que se haya detenido.
- `capture_ms` mide la lectura de la fuente. `preprocess_ms`, `inference_ms` y `postprocess_ms` son valores en milisegundos por imagen de `Results.speed` de Ultralytics; pueden ser `null`. `predict_wall_ms` mide la llamada de predicción sincronizada completa, no solo la inferencia.
- `queue_wait_ms` incluye la espera desde captura hasta inferencia y desde el resultado hasta la anotación, incluida la contrapresión. `annotation_ms` y `display_ms` son medidas separadas. `display_ms` es `null` en modo sin pantalla. `end_to_end_ms` cubre lectura, esperas en cola, predicción, anotación y visualización opcional.
- El tiempo total comienza al iniciar la primera lectura medida. Excluye configuración del modelo, calentamiento, cierre de hilos y limpieza. Las filas CSV contienen IDs, etapas, cantidad de detecciones y CPU/RSS del proceso cuando están disponibles. Los resúmenes incluyen media, p50, p95 y p99, además de ventanas completas de estabilidad de un segundo. El rendimiento offline no es FPS de origen o reproducción. El rendimiento anotado sin pantalla tampoco es la frecuencia física de actualización del monitor.

## Video sintético de conexión

`examples/make_demo_video.py [output]` genera un video OpenCV breve con formas móviles etiquetadas. Por defecto, crea 48 fotogramas de 320×240 a 24 FPS. Se pueden configurar la cantidad de fotogramas, las dimensiones y los FPS. `--help` funciona sin OpenCV. La generación requiere OpenCV y NumPy instalados. Este video sintético solo permite comprobar la conexión entre captura y anotación. No demuestra precisión ni rendimiento realista de YOLO. No se afirma calidad a partir de datos sintéticos.

## Estado de validación local

Informe de la Actividad 4: [docs/Informe_Actividad4_Grupo9_Optimizacion_TiempoReal.pdf](docs/Informe_Actividad4_Grupo9_Optimizacion_TiempoReal.pdf).

Pruebas: `.venv/bin/python -m unittest discover -s tests` ejecuta 47 pruebas con backends falsos. Mediciones reales (macOS arm64, CPU, Python 3.12, Ultralytics 8.4, ONNX Runtime 1.31, `yolo11n`), con el mismo video `car-detection.mp4` (768×432, 12,5 FPS, 377 fotogramas, SHA-256 `d31e0ebf…cca34`) y 3 repeticiones por caso:

| Corrida | Caso | Rendimiento mediano | E2E p95 mediano |
|---|---|---|---|
| Offline `benchmark-20261012-171741-64736e26` | sync, `.pt`, 640 | 49,1 FPS | 27,3 ms |
| | threaded, `.pt`, 640 | 42,4 FPS | 126,9 ms |
| | threaded, ONNX, 320 | 138,3 FPS | 37,2 ms |
| `--realtime` `benchmark-20261012-172526-e682279c` | sync, `.pt`, 640 | 12,5 FPS | 34,4 ms |
| | threaded, `.pt`, 640 | 12,5 FPS | 45,3 ms |
| | threaded, ONNX, 320 | 12,5 FPS | 13,6 ms |

El salto por encima de 60 FPS proviene de bajar la resolución y usar ONNX, no de la concurrencia sola: con la inferencia dominando cada fotograma, los hilos a 640 no aumentan el rendimiento offline y su cola llena eleva la latencia. Al ritmo de la fuente, las colas quedan casi vacías y la latencia depende sobre todo de la inferencia. El rendimiento varía entre corridas. Sin pantalla, `display_fps` es `null`: estas cifras son rendimiento anotado, no FPS físicos de visualización. `detection_count` no reemplaza una evaluación de calidad con etiquetas.

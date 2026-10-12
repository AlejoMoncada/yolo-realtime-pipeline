# Resumen de implementación: pipeline de visión en tiempo real

Este documento explica la implementación, las mediciones disponibles y sus límites. Está dirigido a estudiantes que necesitan ejecutar el proyecto y entender qué demuestra cada resultado. Consulte también el [README en español](../README.md) y el [resumen en inglés](implementation-summary.en.md).

El texto sigue principios de claridad inspirados en ASD-STE100: frases directas, términos definidos y una idea principal por oración. La meta solicitada de simplificación del 80 % es aproximada. No es una puntuación medida ni una certificación. ASD-STE100 se orienta al inglés, por lo que el español adapta esos principios. No se validó el texto con un diccionario aprobado ni con una auditoría formal de una versión del estándar. El sitio oficial se incluye como contexto, no como prueba de conformidad: [ASD-STE100](https://www.asd-ste100.org/).

## 1. Qué se construyó

El proyecto parte de una implementación vacía. `pipeline.py` es la entrada de línea de comandos. `vision_pipeline.py` contiene `PipelineConfig`, las funciones `run_sync`, `run_threaded` y `run_real`, los cálculos de métricas y `export_run`. [`benchmark.py`](../benchmark.py) controla comparaciones. Las pruebas de comportamiento están en [`tests/`](../tests/). [`examples/make_demo_video.py`](../examples/make_demo_video.py) genera un video de conexión con formas sintéticas. [`requirements.txt`](../requirements.txt) declara cuatro grupos de dependencias: Ultralytics, OpenCV, NumPy y psutil.

El modo sync funciona como referencia sencilla. El modo threaded intenta solapar la lectura y la predicción. Ambos producen anotaciones y mediciones por fotograma. El diseño no promete una mejora universal. El rendimiento depende del video, el modelo, el dispositivo y el costo de cada etapa.

## 2. Cómo funciona cada modo

En sync, el mismo flujo lee un fotograma, llama al predictor, anota el resultado y, si se solicita, lo muestra. No hay solapamiento entre esas etapas.

En threaded, el flujo normal es:

1. El hilo de captura lee fotogramas y asigna IDs crecientes. Ese hilo es el único que usa y libera la captura después de transferir su propiedad.
2. Una cola FIFO acotada entrega cada pareja de ID e imagen al trabajador de inferencia.
3. Un único trabajador llama al predictor. No hay llamadas simultáneas al mismo modelo.
4. Una segunda cola FIFO acotada entrega el ID, la imagen original y el resultado al hilo principal.
5. El hilo principal anota la imagen que corresponde a ese resultado y, si se solicitó, muestra la imagen.

Las colas son acotadas y no descartan fotogramas. Conservan el orden y la asociación entre imagen y predicción. Si una etapa avanza más rápido que la siguiente, la contrapresión limita la acumulación. Una cola más grande puede permitir mayor solapamiento, pero también aumentar la latencia.

El tamaño de cola limita los fotogramas en espera, no toda la memoria del proceso. Las filas de medición se acumulan en memoria hasta la exportación. Por eso, el uso total de memoria puede crecer con la duración de la ejecución. No se midió ni se promete una aceleración por el GIL o por un hardware específico.

El calentamiento usa imágenes ficticias antes de la primera lectura medida. En threaded, ocurre antes de iniciar los hilos. Por lo tanto, no todas las llamadas al modelo ocurren siempre dentro del trabajador de inferencia. Configuración, calentamiento, cierre y limpieza quedan fuera del tiempo medido.

`--max-frames` limita la cantidad procesada. El fin del video termina normalmente. La tecla `q` solicita cancelación al mostrar. Los errores de los trabajadores se informan al hilo principal. Una llamada nativa bloqueada a `capture.read()` no se puede interrumpir de forma segura. Antes de iniciar los hilos, el hilo principal puede liberar una captura si falla la preparación. Después de iniciar la captura, no la libera en paralelo. Si la lectura nativa no vuelve, el programa informa el bloqueo.

## 3. Qué significan las mediciones

`capture_ms` mide el tiempo de lectura de la fuente. `preprocess_ms`, `inference_ms` y `postprocess_ms` proceden de `Results.speed` de Ultralytics y representan milisegundos por imagen. Estos valores son anulables. `predict_wall_ms` mide la llamada sincronizada completa al predictor; no equivale al tiempo puro de inferencia.

`queue_wait_ms` suma la espera entre captura e inferencia, y entre predicción y anotación. Incluye la espera causada por la contrapresión. `annotation_ms` y `display_ms` son medidas separadas. Sin pantalla, `display_ms` es `null`. `end_to_end_ms` empieza al iniciar la lectura y termina al completar la anotación o la visualización. Incluye espera en colas. No mide la exposición de la cámara ni los buffers previos del controlador.

Las estadísticas de latencia informan media, p50, p95 y p99. Los percentiles usan interpolación lineal. Un valor `null` significa que la medida no está disponible, no que sea cero. El rendimiento es la cantidad de fotogramas únicos completados dividida por el tiempo transcurrido medido. Los FPS de origen son metadatos y pueden diferir. `display_fps` solo se calcula con `--show`; representa el ciclo de software y no una medición de la frecuencia física del monitor.

Las ventanas de estabilidad cubren intervalos completos de un segundo. También incluyen intervalos con cero fotogramas. Se excluye la última ventana incompleta. Si psutil está disponible, el programa muestrea CPU y RSS del proceso. La primera lectura de CPU puede ser cero. El uso de CPU puede superar 100 % cuando hay varios núcleos. Estas medidas no indican uso de GPU. Las escrituras posteriores de CSV e informes quedan fuera del tiempo de ejecución medido.

## 4. Cómo funciona el benchmark

El benchmark acepta un archivo de video local existente con una cantidad positiva de fotogramas informada por OpenCV. Rechaza webcams, fuentes de red, resultados vacíos o parciales y IDs que no coincidan. Mantiene los pesos, dispositivo, confianza, resolución y rango de fotogramas iguales entre sync y threaded. El orden de casos rota entre repeticiones. Antes de medir, registra ruta, tamaño y SHA-256 del archivo. No vuelve a comprobar el contenido de cada fotograma durante la comparación. El operador debe evitar cambios en el video mientras se ejecutan las pruebas.

Cada invocación crea una carpeta de salida única. Cada ejecución incluye `frames.csv` y `summary.json`. `comparison.json` registra identidad, configuración, orden de las ejecuciones, rendimiento mediano, p95 E2E mediano, speedup y cambio porcentual. El benchmark informa los resultados observados, sin inventar valores ni declarar una mejora antes de medir.

`--optimized-imgsz` agrega un caso threaded con otra resolución. Ese caso cambia más que el grado de concurrencia y no demuestra igualdad de calidad. `--optimized-weights` permite que ese caso use otro formato, por ejemplo un ONNX exportado con `export_model.py` a la misma resolución. No se implementó entrenamiento, cuantización ni procesamiento por lotes. `--realtime` libera cada fotograma al ritmo de los FPS de la fuente para medir la latencia como con una cámara. La salida de `pipeline.py` puede sobrescribir `frames.csv` y `summary.json`; use directorios diferentes para ejecuciones independientes.

## 5. Uso básico y evidencia

Los comandos siguientes muestran la instalación, la ejecución, una comparación controlada y las pruebas.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python pipeline.py ./clip.mp4 --mode threaded --imgsz 640 --device auto --conf 0.25 --warmup 2 --max-frames 300 --queue-size 2 --output-dir results
.venv/bin/python benchmark.py ./clip.mp4 --weights yolo11n.pt --device auto --conf 0.25 --imgsz 640 --warmup 2 --max-frames 300 --queue-size 2 --repeats 3 --output-dir benchmark-results
.venv/bin/python -m unittest discover -s tests -v
```

La verificación independiente registrada ejecutó 32 pruebas con backends falsos. También usó OpenCV 4.11.0 para crear un video AVI sintético de 12 fotogramas y ejecutó cuatro casos con un predictor de checksum deliberadamente falso y `queue-size=1`. Comprobó hashes e IDs, asociación entre anotación e imagen, liberación de captura y lectura de artefactos. Esto comprueba conexiones de software. No comprueba YOLO real, GPU, GUI, cámara, precisión ni estabilidad prolongada.

Las correcciones documentadas separaron `capture_ms` de los tiempos de pantalla y anotación, repararon el reenvío de callbacks de `--show`, cerraron capturas cuando fallaba el calentamiento antes de iniciar los hilos y mejoraron el mensaje ante dependencias ausentes. Las pruebas con dobles no reemplazan una ejecución real.

## 6. Estado y siguientes pasos

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

Pendiente: una corrida con `--show` en pantalla física para medir `display_fps`, la validación de calidad con datos etiquetados y la estabilidad prolongada. No existe aprobación ni recibo de revisión nativa.

## Glosario

- **FIFO:** cola que entrega primero el elemento que recibió primero.
- **E2E:** tiempo desde el inicio de la lectura (o, con `--realtime`, desde la llegada programada del fotograma) hasta completar la anotación o visualización.
- **RSS:** memoria residente del proceso.
- **Speedup:** rendimiento mediano de un caso dividido por el rendimiento mediano de sync.
- **`null`:** dato no disponible; no equivale a cero.

## Referencias de implementación

- [Entrada CLI y `main`](../pipeline.py)
- [`PipelineConfig`, `run_sync`, `run_threaded`, `run_real` y `export_run`](../vision_pipeline.py)
- [`run_benchmark` y generación de informes](../benchmark.py)
- [Pruebas de pipeline](../tests/test_pipeline.py) y [pruebas de benchmark](../tests/test_benchmark.py)
- [Generador de video sintético](../examples/make_demo_video.py)
- [Documentación oficial de Ultralytics](https://docs.ultralytics.com/) y [licencias](https://www.ultralytics.com/license)
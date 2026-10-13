"""Report text. Every number comes from the run artifacts collected in build_report.collect()."""
from __future__ import annotations

from typing import Callable

REPO = "https://github.com/AlejoMoncada/yolo-realtime-pipeline/tree/AlejoMoncada/g9-act3-transfer-learning"
CLASS_ES = {"person": "persona", "bicycle": "bicicleta", "car": "carro", "motorcycle": "moto", "bus": "bus",
            "truck": "camión", "background": "fondo"}


def body(ctx: dict, pct: Callable, num: Callable, thousands: Callable) -> str:
    ev = ctx["evals"]
    rows = {r["key"]: r for r in ctx["rows"]}
    s2, s1, zs = ev["stage2"], ev["stage1"], ev["zeroshot"]
    sc, ff = ev.get("scratch"), ev.get("full_ft")
    st = ctx["stats"]
    classes = st["config"]["classes"]
    v = ctx["versions"]
    gap = ctx["gap"]
    cm = ctx["cm"]
    drift = ctx["drift"]
    frozen_ok = all(d["max_abs_diff"] == 0 for d in drift["coco_to_stage1"] if d["module"] <= 10)
    head_change = max(d["max_abs_diff"] for d in drift["coco_to_stage1"] if d["module"] > 10)
    bb_change = max(d["max_abs_diff"] for d in drift["stage1_to_stage2"] if d["module"] <= 10)
    c1, c2 = ctx["curves"]["stage1"], ctx["curves"]["stage2"]
    best2 = int(c2["epoch"][c2["metrics/mAP50-95(B)"].argmax()])
    hw = ctx.get("hardware", {})

    def rel(a: float, b: float) -> str:
        return f"{a / b:.1f}".replace(".", ",") if b else "—"

    # ---------- tables ----------
    split_rows = "".join(
        f"<tr><td>{name}</td><td class=n>{thousands(st[name]['images'])}</td>"
        + "".join(f"<td class=n>{thousands(st[name]['instances'][c])}</td>" for c in classes)
        + f"<td class=n>{thousands(sum(st[name]['instances'].values()))}</td><td>{use}</td></tr>"
        for name, use in (("train", "ajuste de pesos"), ("val", "selección de época y curvas"),
                          ("test", "reporte final únicamente")))
    head_classes = "".join(f"<th class=n>{CLASS_ES[c]}</th>" for c in classes)

    order = ["zeroshot", "scratch", "stage1", "full_ft", "stage2"]
    comp = ""
    for key in order:
        r = rows.get(key)
        if r is None:
            continue
        where = hw.get(key, "M4")
        comp += (f"<tr class={'hl' if key == 'stage2' else ''}><td>{r['model']}</td><td class=n>{pct(r['map50'])}</td>"
                 f"<td class=n>{pct(r['map50_95'])}</td><td class=n>{pct(r['precision'])}</td>"
                 f"<td class=n>{pct(r['recall'])}</td><td class=n>{r['epochs'] or '—'}</td>"
                 f"<td class=n>{r['train_min']:.0f} ({where})</td></tr>" if key != "zeroshot" else
                 f"<tr><td>{r['model']}</td><td class=n>{pct(r['map50'])}</td><td class=n>{pct(r['map50_95'])}</td>"
                 f"<td class=n>{pct(r['precision'])}</td><td class=n>{pct(r['recall'])}</td><td class=n>0</td>"
                 f"<td class=n>—</td></tr>")

    per_class = "".join(
        f"<tr><td>{CLASS_ES[c]}</td><td class=n>{num(m['precision'])}</td><td class=n>{num(m['recall'])}</td>"
        f"<td class=n>{num(m['f1'])}</td><td class=n>{num(m['map50'])}</td><td class=n>{num(m['map50_95'])}</td>"
        f"<td class=n>{num(zs['per_class'][c]['map50'])}</td><td class=n>{thousands(m['support'])}</td></tr>"
        for c, m in s2["per_class"].items())
    f1_macro = sum(m["f1"] for m in s2["per_class"].values()) / len(s2["per_class"])
    per_class += (f"<tr class=hl><td>macro</td><td class=n>{num(s2['precision'])}</td><td class=n>{num(s2['recall'])}</td>"
                  f"<td class=n>{num(f1_macro)}</td><td class=n>{num(s2['map50'])}</td><td class=n>{num(s2['map50_95'])}</td>"
                  f"<td class=n>{num(zs['map50'])}</td><td class=n>{thousands(sum(m['support'] for m in s2['per_class'].values()))}</td></tr>")

    gap_rows = "".join(
        f"<tr><td>{rows[n]['model'] if n in rows else n}</td><td class=n>{pct(g['train']['map50'])}</td>"
        f"<td class=n>{pct(g['val']['map50'])}</td><td class=n>{pct(g['test']['map50'])}</td></tr>"
        for n, g in gap.items())

    # ---------- confusion-matrix interpretation ----------
    miss = {c: cm["missed"][c] / cm["true_total"][c] for c in cm["missed"] if cm["true_total"][c]}
    labels = cm["labels"]

    def M(pred: str, true: str) -> int:
        return int(cm["matrix"][labels.index(pred), labels.index(true)])

    sizes = ctx["sizes"]
    fp_total = sum(cm["fp_bg"].values())
    fp_top = sorted(cm["fp_bg"].items(), key=lambda kv: kv[1], reverse=True)[:2]

    best_cls = max(s2["per_class"].items(), key=lambda kv: kv[1]["map50"])
    worst_cls = min(s2["per_class"].items(), key=lambda kv: kv[1]["map50"])

    baselines_txt = ""
    if sc is not None:
        baselines_txt += (f" Con el mismo presupuesto de {rows['scratch']['epochs']} épocas, el modelo entrenado desde cero "
                          f"llega a {pct(sc['map50'])} %, así que el preentrenamiento aporta {pct(s2['map50'] - sc['map50'])} "
                          "puntos porcentuales (pp) de mAP@50.")
    if ff is not None:
        diff = s2["map50"] - ff["map50"]
        verdict = ("supera" if diff > 0.005 else "queda por debajo de" if diff < -0.005 else "empata con")
        baselines_txt += (f" El esquema en dos etapas {verdict} el fine-tuning en una sola etapa "
                          f"({pct(s2['map50'])} % contra {pct(ff['map50'])} %; diferencia de {pct(diff)} pp).")

    return f"""
<header>
  <div class=inst>Universidad de La Sabana · Maestría en Inteligencia Artificial · Técnicas Avanzadas de Modelado en IA</div>
  <h1>Prototipo de detección de movilidad urbana desde dron mediante transferencia de aprendizaje (YOLO11n → VisDrone)</h1>
  <div class=meta>Grupo 9: Alejandro Moncada, Edgar Julián Méndez Ortegón, Noel Pérez · Actividad 3, Unidad 3 · Octubre de 2026 ·
  Código: <a href="{REPO}">github.com/AlejoMoncada/yolo-realtime-pipeline</a> (rama <code>g9-act3-transfer-learning</code>)</div>
</header>

<div class=box><b>Resumen.</b> Se adaptó un detector YOLO11n preentrenado en COCO a imágenes aéreas de dron (VisDrone2019-DET) con 6 clases de movilidad urbana.
Sin ajuste, el modelo COCO obtiene <b>{pct(zs['map50'])} %</b> de mAP@50 en test, porque el cambio de punto de vista y la escala de los objetos lo dejan casi ciego.
Con transferencia de aprendizaje en dos etapas (10 épocas con el backbone congelado y luego 30 de fine-tuning completo) se llega a <b>{pct(s2['map50'])} %</b> de mAP@50 y <b>{pct(s2['map50_95'])} %</b> de mAP@50-95, {rel(s2['map50'], zs['map50'])} veces el modelo base.{baselines_txt}
Todas las cifras de este informe salen de los artefactos de entrenamiento del repositorio. El conjunto de prueba (1.610 imágenes de test-dev) no se usó ni para entrenar ni para elegir modelos.</div>

<h2>1. Datos: carga, normalización, particiones y control de fuga</h2>
<p><b>Carga.</b> <code>prepare_data.py</code> lee las imágenes directamente de los zips oficiales de VisDrone2019-DET. Convierte las anotaciones (<code>x,y,w,h,score,categoría</code>) al formato YOLO normalizado (<code>clase xc yc w h</code> en [0, 1]) y agrupa las 10 categorías en 6 clases que también existen en COCO: <i>persona</i> (pedestrian + people), <i>bicicleta</i>, <i>carro</i> (car + van), <i>moto</i>, <i>bus</i> y <i>camión</i>. Gracias a esa correspondencia, el modelo COCO se puede evaluar sin reentrenar como línea base justa. Los triciclos, las <i>ignored regions</i> (score = 0) y <i>others</i> se <b>enmascaran en gris</b> (114) en lugar de borrarse; si se borraran, el modelo aprendería a tratar esos objetos como fondo y sus detecciones se contarían como falsos positivos. La conversión tiene pruebas unitarias (<code>tests/test_visdrone.py</code>).</p>
<p><b>Normalización.</b> Las imágenes se reducen a 1.024 px en el lado mayor al prepararlas. Durante el entrenamiento, Ultralytics aplica letterbox a 640×640 con relleno gris, convierte BGR→RGB y escala los píxeles a [0, 1]. No se resta la media ni se divide por la desviación de ImageNet, porque los pesos de YOLO11 se preentrenaron con esa misma escala [0, 1]; cambiarla rompería la transferencia.</p>
<p><b>Particiones y fuga de información.</b> Se usan los splits oficiales de VisDrone, que provienen de <b>secuencias de video disjuntas</b>. Esto equivale a un <i>group split</i>: dos cuadros casi idénticos del mismo clip nunca quedan uno en entrenamiento y otro en prueba. Por límites de disco y cómputo se tomó un subconjunto de 2.000 de las 6.471 imágenes de train, muestreado en <i>round-robin</i> por secuencia con semilla 0 para conservar la diversidad de escenas. Val y test se usan completos.</p>
<table><tr><th>Split</th><th class=n>Imágenes</th>{head_classes}<th class=n>Total</th><th>Uso</th></tr>{split_rows}</table>
<p><b>Aumentación aleatoria (solo en train).</b> Mosaic (p = 1; Ultralytics lo apaga en las últimas 10 épocas de cada corrida, así que la Etapa 1, de 10 épocas, entrena sin mosaic), flip horizontal (p = 0,5), variación HSV (h = 0,015, s = 0,7, v = 0,4), escala (±50 %) y traslación (±10 %). Val y test solo pasan por letterbox. El desbalance es marcado: <i>carro</i> y <i>persona</i> suman el {pct((st['train']['instances']['car'] + st['train']['instances']['person']) / sum(st['train']['instances'].values()), 0)} % de las instancias de train, y <i>bus</i> solo el {pct(st['train']['instances']['bus'] / sum(st['train']['instances'].values()))} %.</p>

<h2>2. Transferencia de aprendizaje: qué se congela, qué se entrena y por qué</h2>
<p><b>Modelo base.</b> YOLO11n ({num(rows['stage2']['params_m'], 2)} M de parámetros), cargado explícitamente desde los pesos COCO con <code>YOLO("yolo11n.pt")</code>. Tiene backbone (módulos 0-10: Conv/C3k2, SPPF y atención C2PSA), cuello PAN-FPN (11-22) y cabeza <code>Detect</code> desacoplada y sin anclas (23). <b>Capa de salida propia:</b> las convoluciones finales de clasificación de la cabeza se reemplazan por unas de 6 salidas inicializadas al azar, una por clase del problema. Las ramas de regresión de caja se conservan porque la geometría de las cajas es independiente de las clases.</p>
<p><b>Etapa 1, backbone congelado</b> (<code>freeze=11</code>, 10 épocas, LR 1e-3). Solo se entrenan el cuello y la cabeza. La cabeza nueva arranca al azar y produce gradientes grandes; si el backbone estuviera libre, esos gradientes degradarían los filtros preentrenados. <b>Evidencia:</b> al comparar los checkpoints módulo por módulo, el backbone tiene |Δw|<sub>máx</sub> = {'0 exacto' if frozen_ok else 'distinto de 0'} entre <code>yolo11n.pt</code> y la Etapa 1, mientras que el cuello y la cabeza cambian hasta {num(head_change, 2)}.
<b>Etapa 2, fine-tuning completo</b> (<code>freeze=0</code>, 30 épocas, LR 2e-4 → 1e-5 con decaimiento coseno y sin warmup). Se descongela desde la capa 0 porque el dominio aéreo cambia hasta los rasgos de bajo nivel: escala diminuta, vista cenital y texturas de techo y pavimento. En esta etapa el backbone sí se mueve (|Δw|<sub>máx</sub> = {num(bb_change, 2)}). El LR es 5 veces menor que en la Etapa 1 para adaptar sin olvido catastrófico.</p>
<table><tr><th>Hiperparámetro</th><th>Etapa 1</th><th>Etapa 2</th><th>Justificación</th></tr>
<tr><td>Optimizador</td><td colspan=2>AdamW (β<sub>1</sub> = 0,937, β<sub>2</sub> = 0,999), weight decay 5e-4</td><td>Adam con decaimiento desacoplado; converge con LR fijo sin ajuste fino de momentum.</td></tr>
<tr><td>Pérdida</td><td colspan=2>BCE (entropía cruzada binaria por clase) ×0,5 + CIoU ×7,5 + DFL ×1,5</td><td>La clasificación multietiqueta usa entropía cruzada; CIoU y DFL localizan.</td></tr>
<tr><td>LR inicial → final</td><td>1e-3 → 1e-5 lineal</td><td>2e-4 → 1e-5 coseno</td><td>Paso grande para la cabeza nueva; paso fino para no destruir el backbone.</td></tr>
<tr><td>Warmup</td><td>1 época</td><td>0</td><td>La Etapa 2 parte de pesos ya estables.</td></tr>
<tr><td>Capas entrenables</td><td>11-23</td><td>0-23</td><td>Congelar el backbone (0-10) estabiliza la cabeza.</td></tr>
<tr><td colspan=4>Comunes: imgsz 640, batch 16, semilla 0, <code>deterministic=True</code>, 2.000 imágenes de train, el mejor checkpoint se elige por mAP@50-95 en val, sin early stopping (paciencia igual al número de épocas, para que todas las corridas tengan el mismo presupuesto).</td></tr></table>

<h2>3. Monitoreo del entrenamiento</h2>
<figure><img src="figures/curves.png"><figcaption>Figura 1. Curvas por época: pérdidas de entrenamiento (discontinuas) y validación (continuas); precisión (discontinua) y mAP@50 (continua) en validación. La línea punteada marca el descongelamiento del backbone en la época 10.</figcaption></figure>
<p>En la Etapa 1, el mAP@50 en val sube de {pct(c1['metrics/mAP50(B)'][0])} % a {pct(c1['metrics/mAP50(B)'][-1])} % con el backbone fijo. Al descongelar, la curva sigue subiendo hasta {pct(c2['metrics/mAP50(B)'].max())} % (mejor época por mAP@50-95: {best2 + len(c1['epoch'])}), lo que confirma que adaptar el backbone aporta. Las pérdidas de validación bajan en paralelo a las de entrenamiento y no divergen. El salto de la pérdida de <i>entrenamiento</i> en la época 11 no es inestabilidad: la Etapa 1 corrió sin mosaic y la Etapa 2 lo reactiva, lo que vuelve más difíciles las imágenes de train. La pérdida de validación, que no usa aumentación, no salta. En la época {len(c1['epoch']) + 21} (la 21 de la Etapa 2) se apaga el mosaic: hay una caída puntual del mAP en val y la pérdida de entrenamiento cambia de escala, porque las imágenes dejan de ser collages de cuatro escenas. El modelo se recupera en 1-2 épocas. Con el LR casi nulo del final, el mAP se estabiliza: más épocas aportarían poco con este presupuesto de datos.</p>
<table><tr><th>Modelo (mejores pesos)</th><th class=n>mAP@50 train</th><th class=n>mAP@50 val</th><th class=n>mAP@50 test</th></tr>{gap_rows}</table>
<p class=cap>Tabla de brecha: los mismos pesos se evalúan sin aumentación en los tres splits. Que el desempeño en train sea mayor es esperable con 2.000 imágenes. Test-dev viene de secuencias y condiciones distintas a las de val; la caída de val a test mide cuánto generaliza cada modelo a escenas nuevas.</p>

<h2>4. Evaluación en el conjunto de prueba</h2>
<table><tr><th>Modelo</th><th class=n>mAP@50 (%)</th><th class=n>mAP@50-95 (%)</th><th class=n>Precisión (%)</th><th class=n>Recall (%)</th><th class=n>Épocas</th><th class=n>Entrenamiento (min, equipo)</th></tr>{comp}</table>
<p class=cap>Métricas macro sobre las 6 clases, con NMS IoU 0,7 y conf 0,001 para mAP. Precisión y recall están en el punto de máximo F1. Todas las corridas usan el mismo dataset, semilla, imgsz, batch, optimizador y presupuesto de épocas; los tiempos solo son comparables dentro del mismo equipo.</p>
<table><tr><th>Clase (modelo propuesto)</th><th class=n>Precisión</th><th class=n>Recall</th><th class=n>F1</th><th class=n>AP@50</th><th class=n>AP@50-95</th><th class=n>AP@50 COCO sin ajuste</th><th class=n>Instancias</th></tr>{per_class}</table>
<figure><img src="figures/confusion.png"><figcaption>Figura 2. Matrices de confusión en test (conf 0,25, IoU de emparejamiento 0,45), normalizadas por columna: cada columna es una clase real y suma 1. La fila "fondo" son objetos no detectados (falsos negativos); la columna "fondo" reparte los falsos positivos según la clase predicha.</figcaption></figure>
<figure><img src="figures/false_positives.jpg"><figcaption>Figura 3. Imágenes de test con más falsos positivos del modelo propuesto. Verde: verdad anotada. Rojo: predicciones (conf ≥ 0,25) sin caja anotada con IoU ≥ 0,45. Casi todas son vehículos y personas reales sin anotar.</figcaption></figure>
<p><b>Errores que se repiten.</b> (1) <b>Predomina la falta de detección, y depende del tamaño.</b> La mayor parte del error no es confundir una clase con otra: es no detectar el objeto (fila <i>fondo</i>). Se pierde el {pct(miss['person'], 0)} % de las personas, el {pct(miss['bicycle'], 0)} % de las bicicletas y el {pct(miss['motorcycle'], 0)} % de las motos, contra el {pct(miss['car'], 0)} % de los carros y el {pct(miss['bus'], 0)} % de los buses. El orden coincide con el tamaño aparente: a la entrada de 640 px, la mediana de una persona mide {num(sizes['person']['median_px'], 1)} px (√área) y el {pct(sizes['person']['pct_lt_16px'], 0)} % mide menos de 16 px; un bus mide {num(sizes['bus']['median_px'], 1)} px. Con objetos de 5 a 9 px, el mapa de mayor resolución de YOLO11n (P3, stride 8) asigna menos de una celda por objeto.
(2) <b>Confusiones entre vehículos grandes:</b> {thousands(M('car','truck'))} camiones se predicen como carro ({pct(M('car','truck') / cm['true_total']['truck'], 0)} % de los camiones) y {thousands(M('truck','car'))} carros como camión. También {thousands(M('car','bus'))} buses salen como carro y {thousands(M('truck','bus'))} como camión. Desde arriba, la diferencia entre una camioneta, un furgón (<i>van</i>, que se agrupó como carro) y un camión pequeño es casi solo de largo. Esta frontera la definió el agrupamiento de clases; no es un defecto del aprendizaje.
(3) <b>Vehículos de dos ruedas y personas:</b> {pct(M('motorcycle','bicycle') / cm['true_total']['bicycle'], 0)} % de las bicicletas se predicen como moto, {thousands(M('motorcycle','person'))} personas como moto y {thousands(M('person','motorcycle'))} motos como persona. VisDrone anota al conductor (<i>people</i>, que aquí es persona) y a su vehículo (<i>motor</i>) como dos cajas casi superpuestas. Desde arriba, el conductor tapa casi todo el vehículo, así que el modelo tiene que decidir cuál de las dos cajas es.
(4) <b>Falsos positivos sobre fondo:</b> {thousands(fp_total)} detecciones sin objeto real, sobre todo de {CLASS_ES[fp_top[0][0]]} ({thousands(fp_top[0][1])}) y {CLASS_ES[fp_top[1][0]]} ({thousands(fp_top[1][1])}). En la revisión visual de las imágenes de test con más falsos positivos (notebook, sección 4), la mayoría son <b>objetos reales sin anotar</b>: carros en vías lejanas del fondo y personas dentro de multitudes nocturnas. El resto cae sobre objetos con silueta parecida, como papeleras, troncos o postes. Por eso la precisión real es mayor que la medida, y el error de verdad es el recall de objetos pequeños.
La mejor clase en AP@50 es <i>{CLASS_ES[best_cls[0]]}</i> ({num(best_cls[1]['map50'])}), la que tiene más instancias y objetos más grandes, y la peor es <i>{CLASS_ES[worst_cls[0]]}</i> ({num(worst_cls[1]['map50'])}), pequeña y escasa.</p>

<h2>5. Conclusiones, limitaciones y reproducibilidad</h2>
<ul>
<li><b>La transferencia funciona, pero requiere fine-tuning.</b> Congelar el backbone multiplica el mAP@50 del modelo COCO por {rel(s1['map50'], zs['map50'])} ({pct(s1['map50'])} % en test). Descongelarlo lo lleva a {pct(s2['map50'])} %. El cambio de dominio llega hasta los filtros de bajo nivel, así que la cabeza sola no alcanza.</li>
<li><b>El cuello de botella es la escala, no la semántica.</b> La matriz muestra más falsos negativos que confusiones entre clases. La mejora más directa sería entrenar a mayor resolución (imgsz 960-1280) o con <i>tiling</i> (SAHI), agregar la cabeza P2 (stride 4) para objetos pequeños y usar el train completo (6.471 imágenes).</li>
<li><b>Limitaciones.</b> Se usó un subconjunto de train y una sola semilla, sin intervalos de confianza. En MPS hay operaciones no deterministas aunque <code>deterministic=True</code> esté activo. Los triciclos quedan fuera del problema.</li>
<li><b>Reproducibilidad.</b> Python {v['python']}, PyTorch {v['torch']}, Ultralytics {v['ultralytics']}, semilla 0. Equipos: {v['machine']}{hw.get('_note', '')}. Para regenerar todo: <code>python -m transfer_learning.prepare_data --download</code>, <code>python -m transfer_learning.experiments all</code>, el notebook <code>transfer_learning/act3_transfer_learning.ipynb</code> y <code>python -m transfer_learning.report.build_report</code>.</li>
</ul>
<ol class=refs>
<li>Zhu, P. et al. (2021). Detection and Tracking Meet Drones Challenge. <i>IEEE TPAMI</i>, 44(11), 7380-7399 (VisDrone).</li>
<li>Jocher, G., Qiu, J. (2024). <i>Ultralytics YOLO11</i>. https://docs.ultralytics.com</li>
<li>Yosinski, J. et al. (2014). How transferable are features in deep neural networks? <i>NeurIPS</i>.</li>
<li>Loshchilov, I., Hutter, F. (2019). Decoupled Weight Decay Regularization. <i>ICLR</i> (AdamW).</li>
<li>Akyon, F. C. et al. (2022). Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection. <i>ICIP</i> (SAHI).</li>
</ol>
"""

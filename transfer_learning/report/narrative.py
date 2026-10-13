"""Report text. Every number comes from the run artifacts collected in build_report.collect()."""
from __future__ import annotations

from typing import Callable

REPO = "https://github.com/AlejoMoncada/yolo-realtime-pipeline/tree/AlejoMoncada/g9-act3-transfer-learning"
CLASS_ES = {"person": "persona", "bicycle": "bicicleta", "car": "carro", "motorcycle": "moto", "bus": "bus",
            "truck": "camión", "background": "fondo"}
AUTHORS = "Alejandro Moncada, Edgar Julián Méndez Ortegón, Noel Eduardo Pérez Barrios"


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
    sizes = ctx["sizes"]
    hw = ctx.get("hardware", {})
    frozen_ok = all(d["max_abs_diff"] == 0 for d in drift["coco_to_stage1"] if d["module"] <= 10)
    c1, c2 = ctx["curves"]["stage1"], ctx["curves"]["stage2"]
    n1 = len(c1["epoch"])
    best2 = int(c2["epoch"][c2["metrics/mAP50-95(B)"].argmax()])
    train_total = sum(st["train"]["instances"].values())

    def rel(a: float, b: float) -> str:
        return f"{a / b:.1f}".replace(".", ",") if b else "—"

    def verdict(d: float, worse: str, same: str, better: str) -> str:
        return worse if d > 0.005 else better if d < -0.005 else same

    # ---------- figure numbering (the ablation figure exists only once its run finished) ----------
    labels_abl = {"stage2": "LR 2e-4, todo descongelado (elegido)", "stage2_lr1e3": "LR 1e-3, todo descongelado",
                  "stage2_freeze6": "LR 2e-4, módulos 0-5 congelados"}
    abl = [(k, ev.get(k), ctx["curves"].get(k)) for k in labels_abl
           if ev.get(k) is not None and ctx["curves"].get(k) is not None]
    has_abl = len(abl) >= 2
    fig_curves, fig_abl = 1, 2
    fig_cm = 3 if has_abl else 2
    fig_fp = fig_cm + 1

    # ---------- tables ----------
    split_rows = "".join(
        f"<tr><td>{name}</td><td class=n>{thousands(st[name]['images'])}</td>"
        + "".join(f"<td class=n>{thousands(st[name]['instances'][c])}</td>" for c in classes)
        + f"<td class=n>{thousands(sum(st[name]['instances'].values()))}</td><td>{use}</td></tr>"
        for name, use in (("train", "ajuste de pesos"), ("val", "selección de época y curvas"),
                          ("test", "solo el reporte final")))
    head_classes = "".join(f"<th class=n>{CLASS_ES[c]}</th>" for c in classes)

    comp = ""
    for key in ["zeroshot", "scratch", "stage1", "full_ft", "stage2"]:
        r = rows.get(key)
        if r is None:
            continue
        epochs = 0 if key == "zeroshot" else (r["epochs"] or "—")
        minutes = f"{r['train_min']:.0f}" if key != "zeroshot" and r["train_min"] else "—"
        comp += (f"<tr class={'hl' if key == 'stage2' else ''}><td>{r['model']}</td><td class=n>{pct(r['map50'])}</td>"
                 f"<td class=n>{pct(r['map50_95'])}</td><td class=n>{pct(r['precision'])}</td>"
                 f"<td class=n>{pct(r['recall'])}</td><td class=n>{epochs}</td><td class=n>{minutes}</td></tr>")
    gain_rows = (f"<tr class=gain><td>Mejora frente a COCO sin ajuste</td><td class=n>×{rel(s2['map50'], zs['map50'])} (+{pct(s2['map50'] - zs['map50'])} pp)</td>"
                 f"<td class=n>×{rel(s2['map50_95'], zs['map50_95'])}</td><td colspan=4></td></tr>")
    if sc is not None:
        gain_rows += (f"<tr class=gain><td>Mejora frente a desde cero</td><td class=n>×{rel(s2['map50'], sc['map50'])} (+{pct(s2['map50'] - sc['map50'])} pp)</td>"
                      f"<td class=n>×{rel(s2['map50_95'], sc['map50_95'])}</td><td colspan=4></td></tr>")

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

    def row_at(curve, i):
        return {k: float(val[i]) for k, val in curve.items()}

    milestones = [("Inicio de la Etapa 1 (backbone congelado)", 1, row_at(c1, 0)),
                  ("Fin de la Etapa 1", n1, row_at(c1, n1 - 1)),
                  ("Descongelamiento (inicio de la Etapa 2)", n1 + 1, row_at(c2, 0)),
                  ("Se apaga el mosaic", n1 + 21, row_at(c2, 20)),
                  ("Mejores pesos (máx. mAP@50-95 en val)", n1 + best2, row_at(c2, best2 - 1)),
                  ("Cierre", n1 + len(c2["epoch"]), row_at(c2, len(c2["epoch"]) - 1))]
    milestone_rows = "".join(
        f"<tr class={'hl' if 'Mejores' in name else ''}><td>{name}</td><td class=n>{ep}</td>"
        f"<td class=n>{num(r['train/box_loss'], 2)}</td><td class=n>{num(r['val/box_loss'], 2)}</td>"
        f"<td class=n>{num(r['train/cls_loss'], 2)}</td><td class=n>{num(r['val/cls_loss'], 2)}</td>"
        f"<td class=n>{pct(r['metrics/precision(B)'])}</td><td class=n>{pct(r['metrics/mAP50(B)'])}</td></tr>"
        for name, ep, r in milestones)

    # ---------- confusion matrix ----------
    labels = cm["labels"]
    miss = {c: cm["missed"][c] / cm["true_total"][c] for c in cm["missed"] if cm["true_total"][c]}

    def M(pred: str, true: str) -> int:
        return int(cm["matrix"][labels.index(pred), labels.index(true)])

    fp_total = sum(cm["fp_bg"].values())
    fp_top = sorted(cm["fp_bg"].items(), key=lambda kv: kv[1], reverse=True)[:2]
    best_cls = max(s2["per_class"].items(), key=lambda kv: kv[1]["map50"])
    worst_cls = min(s2["per_class"].items(), key=lambda kv: kv[1]["map50"])

    # ---------- comparison sentences ----------
    scratch_sentence = ""
    if sc is not None:
        scratch_sentence = (f" Entrenado desde cero con el mismo presupuesto de {rows['scratch']['epochs']} épocas, el mismo modelo solo llega a "
                            f"{pct(sc['map50'])} %, así que el preentrenamiento aporta {pct(s2['map50'] - sc['map50'])} puntos porcentuales (pp).")
    ff_sentence = ""
    if ff is not None:
        d = s2["map50"] - ff["map50"]
        ff_sentence = " " + verdict(
            d, f"Ajustar todo desde el inicio, en una sola etapa, quedó por debajo ({pct(ff['map50'])} %).",
            f"Ajustar todo desde el inicio, en una sola etapa, dio prácticamente lo mismo ({pct(ff['map50'])} %).",
            f"Nos sorprendió que ajustar todo desde el inicio, en una sola etapa con LR 1e-3, llegara a {pct(ff['map50'])} %, por encima de nuestro esquema; lo discutimos en la sección 3.3.")

    scratch_par = ""
    if sc is not None:
        gains = {c: (s2["per_class"][c]["map50"], sc["per_class"][c]["map50"]) for c in s2["per_class"]}
        ratio = sorted(((a / b if b else float("inf"), c) for c, (a, b) in gains.items()), reverse=True)
        scratch_par = (f"<p>Al comparar clase por clase con el modelo entrenado desde cero, el preentrenamiento mejora las seis. "
                       f"Donde más pesa es en las clases con pocos ejemplos: <i>{CLASS_ES[ratio[0][1]]}</i> multiplica su AP@50 por {rel(*gains[ratio[0][1]])} "
                       f"y <i>{CLASS_ES[ratio[1][1]]}</i> por {rel(*gains[ratio[1][1]])}. En <i>{CLASS_ES[ratio[-1][1]]}</i>, la clase más abundante, "
                       f"la ganancia es de solo {rel(*gains[ratio[-1][1]])} veces. Cuando faltan datos, lo aprendido en COCO reemplaza ejemplos que el modelo no tiene.</p>")

    single_html = ""
    if ff is not None:
        cff = ctx["curves"].get("full_ft")
        d = s2["map50"] - ff["map50"]
        better = sum(ff["per_class"][c]["map50"] > s2["per_class"][c]["map50"] for c in s2["per_class"])
        srows = "".join(
            f"<tr class={'hl' if k == 'stage2' else ''}><td>{label}</td><td class=n>{pct(e['map50'])}</td><td class=n>{pct(e['map50_95'])}</td>"
            f"<td class=n>{pct(c['metrics/mAP50(B)'].max()) if c is not None else '—'}</td><td class=n>{pct(gap[k]['train']['map50']) if k in gap else '—'}</td></tr>"
            for k, e, c, label in (("stage2", s2, c2, "Dos etapas: 10 congeladas + 30 con LR 2e-4"),
                                   ("full_ft", ff, cff, "Una etapa: 40 con todo libre y LR 1e-3")))
        if d < -0.005:
            body_txt = (f"<p>Para saber si congelar primero valía la pena, entrenamos el mismo modelo en una sola etapa: todo descongelado desde el inicio, LR 1e-3 y las mismas 40 épocas en total. "
                        f"Esperábamos que quedara por debajo y pasó lo contrario: llegó a {pct(ff['map50'])} % de mAP@50 en test, {pct(-d)} pp más que nuestro esquema, y fue mejor en {better} de las 6 clases. "
                        f"En la Figura {fig_curves} se ve que su curva de validación va por encima casi desde el principio.</p>"
                        "<p>Creemos que hay dos razones. La primera es el LR: en la Etapa 2 ajustamos el backbone con 2e-4, cinco veces menos que la corrida de una etapa, y la vista aérea pide cambios grandes en los filtros, así que fuimos demasiado conservadores. "
                        "La segunda es el presupuesto: nuestro backbone solo entrenó 30 de las 40 épocas, y la Etapa 1 corrió sin mosaic. "
                        "El daño que temíamos, que los gradientes de la cabeza nueva arruinaran los filtros de COCO, no se vio con una sola época de warmup. "
                        f"Hay que leer la diferencia con cuidado: es una sola semilla y la corrida de una etapa se hizo en otra GPU. Al reevaluar sus pesos en el M4, el mAP@50 en test dio {pct(gap['full_ft']['test']['map50'])} %, casi lo mismo que en la GPU donde se entrenó, así que el equipo de evaluación no explica la diferencia, aunque la variación entre semillas sí podría explicar una parte. "
                        "Aun así, que gane en casi todas las clases sugiere que el efecto es real.</p>")
        else:
            body_txt = (f"<p>Para saber si congelar primero valía la pena, entrenamos el mismo modelo en una sola etapa: todo descongelado desde el inicio, LR 1e-3 y las mismas 40 épocas. "
                        f"Llegó a {pct(ff['map50'])} % de mAP@50 en test, frente a {pct(s2['map50'])} % de nuestro esquema, así que congelar primero " + verdict(d, "sí aportó.", "no cambió el resultado.", "") + "</p>")
        single_html = f"""<h3>3.3 ¿Una etapa o dos?</h3>
{body_txt}
<table><tr><th>Esquema</th><th class=n>mAP@50 test</th><th class=n>mAP@50-95 test</th><th class=n>máx. mAP@50 val</th><th class=n>mAP@50 train</th></tr>{srows}</table>"""

    ablation_html = ""
    if has_abl:
        trows = "".join(
            f"<tr class={'hl' if k == 'stage2' else ''}><td>{labels_abl[k]}</td><td class=n>{pct(e['map50'])}</td>"
            f"<td class=n>{pct(e['map50_95'])}</td><td class=n>{pct(c['metrics/mAP50(B)'].max())}</td></tr>" for k, e, c in abl)
        parts = []
        if ev.get("stage2_lr1e3") is not None:
            lr = ev["stage2_lr1e3"]
            d = s2["map50"] - lr["map50"]
            parts.append(f"Con 1e-3 el modelo termina en {pct(lr['map50'])} % de mAP@50 en test, contra {pct(s2['map50'])} % con 2e-4. " + verdict(
                d, "El paso grande desordena los filtros preentrenados al comienzo de la Etapa 2, como se ve en su pérdida inicial más alta, y lo que se recupera después no compensa lo perdido. Eso respalda la elección de 2e-4.",
                "En este rango el LR no resultó crítico. Nos quedamos con 2e-4 porque perturba menos los pesos preentrenados.",
                "El LR mayor rindió más con este presupuesto de épocas, así que 2e-4 fue una elección conservadora y subirlo sería la primera mejora a probar."))
        if ev.get("stage2_freeze6") is not None:
            fz = ev["stage2_freeze6"]
            d = s2["map50"] - fz["map50"]
            parts.append(f"Si además dejamos congelados los módulos 0 a 5, el resultado es {pct(fz['map50'])} %. " + verdict(
                d, "Las primeras capas también necesitan adaptarse a la vista aérea, y por eso descongelamos desde la capa 0.",
                "Las primeras capas aportan poco al adaptarlas; descongelar solo desde el módulo 6 sería una opción más barata.",
                "Mantener congeladas las primeras capas generalizó mejor, así que convendría descongelar solo desde el módulo 6."))
        ablation_html = f"""<h3>3.3 Por qué una tasa de aprendizaje de 2e-4</h3>
<p>Para no justificar el LR solo con argumentos, repetimos la Etapa 2 desde los mismos pesos y con las mismas 30 épocas, cambiando únicamente el LR a 1e-3, cinco veces más alto. {' '.join(parts)}</p>
<div class=two><figure><img src="figures/ablation.png"><figcaption>Figura {fig_abl}. Curvas de validación de las variantes de la Etapa 2.</figcaption></figure>
<table><tr><th>Variante de la Etapa 2</th><th class=n>mAP@50 test</th><th class=n>mAP@50-95 test</th><th class=n>máx. mAP@50 val</th></tr>{trows}</table></div>"""

    return f"""
<header>
  <img class=logo src="figures/logo_unisabana.jpg" alt="Universidad de La Sabana">
  <div class=inst>UNIVERSIDAD DE LA SABANA<br>MAESTRÍA EN INTELIGENCIA ARTIFICIAL</div>
  <h1>REPORTE TÉCNICO: PROTOTIPO DE DETECCIÓN DE MOVILIDAD URBANA DESDE DRON MEDIANTE TRANSFERENCIA DE APRENDIZAJE Y FINE-TUNING (YOLO11n)</h1>
  <div class=meta>Autores: {AUTHORS} (Grupo 9)<br>
  Asignatura: Técnicas Avanzadas de Modelado en IA · Actividad 3, Unidad 3 · Octubre de 2026<br>
  Código: <a href="{REPO}">{REPO.removeprefix("https://")}</a></div>
</header>

<h2>1. Resumen ejecutivo y preparación de los datos</h2>
<div class=box><b>1.1 Resumen ejecutivo.</b> Tomamos YOLO11n, preentrenado en COCO, y lo adaptamos para detectar personas y vehículos en imágenes de dron (VisDrone2019-DET).
Esperábamos que el modelo COCO sirviera como punto de partida razonable, porque ya conoce personas, carros y buses. No fue así: sin ajuste obtuvo apenas <b>{pct(zs['map50'])} %</b> de mAP@50 en test.
Desde arriba y con pocos píxeles de tamaño, los objetos casi no se parecen a los de COCO.
Con transferencia de aprendizaje en dos etapas (10 épocas con el backbone congelado y 30 de ajuste completo) llegamos a <b>{pct(s2['map50'])} %</b> de mAP@50 y <b>{pct(s2['map50_95'])} %</b> de mAP@50-95, unas {rel(s2['map50'], zs['map50'])} veces más.{scratch_sentence}{ff_sentence}
Todas las cifras salen de los artefactos de entrenamiento del repositorio, y no usamos el conjunto de prueba ni para entrenar ni para elegir modelos.</div>

<h3>1.2 Carga del dataset, particiones y control de fuga</h3>
<p>Leemos las imágenes directamente de los zips oficiales de VisDrone2019-DET y convertimos sus anotaciones al formato YOLO normalizado (<code>clase xc yc w h</code> en [0, 1]).
Agrupamos las 10 categorías originales en 6 clases que también existen en COCO: <i>persona</i> (pedestrian y people), <i>bicicleta</i>, <i>carro</i> (car y van), <i>moto</i>, <i>bus</i> y <i>camión</i>.
Elegimos esa agrupación para poder evaluar el modelo COCO sin reentrenarlo y así tener una línea base justa.
Los triciclos y las regiones que VisDrone marca como ignoradas no los borramos: los pintamos de gris. Si solo quitáramos sus etiquetas, el modelo aprendería a tratarlos como fondo, y cada detección sobre ellos contaría como falso positivo.</p>
<p>Para las particiones usamos los splits oficiales, que vienen de <b>secuencias de video distintas</b>. En la práctica es un <i>group split</i>: dos cuadros casi iguales del mismo video nunca quedan uno en train y otro en test.
De train usamos 2.000 de las 6.471 imágenes. Entrenamos en un computador con chip Apple M4 y menos de 5 GB libres en disco, donde cada época sobre esas 2.000 imágenes tardaba entre dos y cuatro minutos; con el conjunto completo, las corridas no habrían cabido en el tiempo de la actividad.
Para no perder variedad de escenas, tomamos las imágenes de forma alternada entre secuencias, con semilla fija. Val y test los usamos completos.</p>
<table><tr><th>Split</th><th class=n>Imágenes</th>{head_classes}<th class=n>Total</th><th>Uso</th></tr>{split_rows}</table>

<h3>1.3 Normalización y aumentación aleatoria</h3>
<p>Reducimos las imágenes a 1.024 px en el lado mayor. En el entrenamiento, Ultralytics las lleva a 640×640 con relleno gris (letterbox), pasa de BGR a RGB y escala los píxeles a [0, 1].
No aplicamos la media y la desviación de ImageNet, porque YOLO11 se preentrenó con la escala [0, 1] y cambiarla rompería la transferencia.
La aumentación aleatoria va solo en train: mosaic, volteo horizontal (p = 0,5), variación de color HSV, escala (±50 %) y traslación (±10 %). Ultralytics apaga el mosaic en las últimas 10 épocas de cada corrida, así que la Etapa 1, de 10 épocas, se entrenó sin él.
El desbalance es fuerte: <i>carro</i> y <i>persona</i> suman el {pct((st['train']['instances']['car'] + st['train']['instances']['person']) / train_total, 0)} % de las instancias de train, y <i>bus</i> apenas el {pct(st['train']['instances']['bus'] / train_total)} %.</p>

<h2>2. Arquitectura y estrategia de transferencia de aprendizaje</h2>
<h3>2.1 Modelo preentrenado y capa de salida propia</h3>
<p>YOLO11n tiene {num(rows['stage2']['params_m'], 2)} millones de parámetros. Lo cargamos explícitamente desde los pesos de COCO con <code>YOLO("yolo11n.pt")</code>.
Su backbone ocupa los módulos 0 a 10, el cuello PAN-FPN va del 11 al 22 y la cabeza de detección es el módulo 23.
Nuestra capa de salida propia es la parte de clasificación de esa cabeza: reemplazamos las 80 salidas de COCO por 6 nuevas, inicializadas al azar, una por clase. La parte que predice la posición de las cajas la conservamos, porque la geometría de una caja no depende de la clase.</p>
<h3>2.2 Función de pérdida</h3>
<p class=eq><i>L</i><sub>total</sub> = 7,5·<i>L</i><sub>CIoU</sub> + 0,5·<i>L</i><sub>BCE</sub> + 1,5·<i>L</i><sub>DFL</sub>, &nbsp;&nbsp; <i>L</i><sub>BCE</sub> = −Σ<sub>c</sub> [<i>y</i><sub>c</sub> log <i>p</i><sub>c</sub> + (1 − <i>y</i><sub>c</sub>) log(1 − <i>p</i><sub>c</sub>)]</p>
<p>La clasificación se entrena con entropía cruzada binaria por clase, y CIoU y DFL se encargan de la localización. Optimizamos con AdamW, la versión de Adam con <i>weight decay</i> desacoplado.</p>
<h3>2.3 Esquema de transferencia en dos etapas</h3>
<p>En la <b>Etapa 1</b> congelamos todo el backbone (<code>freeze=11</code>) y entrenamos solo el cuello y la cabeza durante 10 épocas con LR 1e-3.
Es la práctica habitual: la cabeza nueva arranca al azar y al principio produce gradientes grandes que, con el backbone libre, podrían dañar los filtros que trae de COCO. En la sección 3.3 comprobamos si ese riesgo se da en nuestro caso.
Verificamos el congelamiento comparando los pesos módulo por módulo: el backbone sale de la Etapa 1 {'exactamente igual' if frozen_ok else 'con cambios'} a como entró, mientras que el cuello y la cabeza sí cambian.</p>
<p>En la <b>Etapa 2</b> descongelamos todo (<code>freeze=0</code>) y entrenamos 30 épocas más con un LR cinco veces menor, 2e-4, que baja hasta 1e-5 con decaimiento coseno.
Descongelamos desde la capa 0 porque la vista aérea cambia incluso los rasgos más básicos que aprenden las primeras capas: objetos diminutos, vista desde arriba y texturas de techos y pavimento.</p>
<h3>2.4 Configuración de hiperparámetros</h3>
<table><tr><th>Hiperparámetro</th><th>Etapa 1</th><th>Etapa 2</th><th>Justificación</th></tr>
<tr><td>Optimizador</td><td colspan=2>AdamW (β<sub>1</sub> = 0,937; β<sub>2</sub> = 0,999), weight decay 5e-4</td><td>Adam con decaimiento desacoplado.</td></tr>
<tr><td>LR inicial → final</td><td>1e-3 → 1e-5, lineal</td><td>2e-4 → 1e-5, coseno</td><td>Paso grande para la cabeza nueva; paso fino para no dañar el backbone.</td></tr>
<tr><td>Warmup</td><td>1 época</td><td>ninguno</td><td>La Etapa 2 parte de pesos ya estables.</td></tr>
<tr><td>Capas entrenables</td><td>11 a 23</td><td>0 a 23</td><td>Congelar el backbone (0 a 10) estabiliza la cabeza nueva.</td></tr>
<tr><td colspan=4>Comunes a todas las corridas: imagen de 640, batch 16, semilla 0, <code>deterministic=True</code> y 2.000 imágenes de train. Guardamos los pesos con mejor mAP@50-95 en val y no usamos early stopping, para que todas las corridas tengan el mismo presupuesto.</td></tr></table>
<div class=box><b>Carga explícita de pesos</b> (<code>experiments.py</code>): <code>YOLO("yolo11n.pt").train(freeze=11, epochs=10, lr0=1e-3, optimizer="AdamW", seed=0, ...)</code> para la Etapa 1 y <code>YOLO("runs/act3/stage1/weights/best.pt").train(freeze=0, epochs=30, lr0=2e-4, cos_lr=True, ...)</code> para la Etapa 2. Ultralytics copia del modelo COCO todos los pesos que tienen la misma forma, es decir, todo menos la salida de clases.</div>

<h2>3. Entrenamiento y curvas de aprendizaje</h2>
<h3>3.1 Evolución de pérdidas y precisión (train contra validación)</h3>
<figure><img src="figures/curves.png"><figcaption>Figura {fig_curves}. Curvas por época del esquema en dos etapas y de la corrida en una etapa. En las pérdidas, la línea discontinua es train y la continua es validación; en el panel derecho, la discontinua es precisión y la continua es mAP@50, ambas en validación. La línea punteada marca el descongelamiento.</figcaption></figure>
<p>Durante la Etapa 1, con el backbone fijo, el mAP@50 en val sube de {pct(c1['metrics/mAP50(B)'][0])} % a {pct(c1['metrics/mAP50(B)'][-1])} %. Al descongelar sigue subiendo hasta {pct(c2['metrics/mAP50(B)'].max())} %, lo que muestra que ajustar el backbone sí aporta.
Las pérdidas de validación bajan junto con las de train y no se separan, así que no vemos sobreajuste fuerte.
En la época 11 la pérdida de train da un salto. No es inestabilidad: la Etapa 1 corrió sin mosaic y la Etapa 2 lo vuelve a activar, lo que hace más difíciles las imágenes de train. La pérdida de validación, que no usa aumentación, no salta.
Cuando el mosaic se apaga de nuevo (época {n1 + 21}) hay una caída breve del mAP que se recupera en una o dos épocas.</p>
<p>Ultralytics no calcula precisión ni mAP sobre train en cada época, solo las pérdidas. Por eso la curva de precisión de la Figura {fig_curves} es de validación. Para comparar train con validación medimos el mAP en los tres conjuntos con los mejores pesos (segunda tabla de 3.2).</p>
<h3>3.2 Hitos del entrenamiento y control de sobreajuste</h3>
<table><tr><th>Hito</th><th class=n>Época</th><th class=n>box train</th><th class=n>box val</th><th class=n>cls train</th><th class=n>cls val</th><th class=n>Precisión val (%)</th><th class=n>mAP@50 val (%)</th></tr>{milestone_rows}</table>
<table><tr><th>Modelo (mejores pesos)</th><th class=n>mAP@50 train</th><th class=n>mAP@50 val</th><th class=n>mAP@50 test</th></tr>{gap_rows}</table>
<p class=cap>Con 2.000 imágenes es normal que train quede por encima. La caída de val a test indica cuánto generaliza el modelo a secuencias que nunca vio.</p>
{single_html}{ablation_html}
<h2>4. Evaluación en el conjunto de prueba</h2>
<h3>4.1 Comparación contra los modelos base</h3>
<table><tr><th>Modelo</th><th class=n>mAP@50 (%)</th><th class=n>mAP@50-95 (%)</th><th class=n>Precisión (%)</th><th class=n>Recall (%)</th><th class=n>Épocas</th><th class=n>Entrenamiento M4 (min)</th></tr>{comp}{gain_rows}</table>
<p class=cap>Promedios sobre las 6 clases. Precisión y recall se toman en el punto de máximo F1. Todas las corridas comparten datos, semilla, tamaño de imagen, batch, optimizador y número de épocas. No reportamos <i>accuracy</i> porque en detección no hay verdaderos negativos que contar; su equivalente son la precisión, el recall y el F1 por clase.</p>
{scratch_par}
<h3>4.2 Métricas por clase del modelo propuesto</h3>
<table><tr><th>Clase</th><th class=n>Precisión</th><th class=n>Recall</th><th class=n>F1</th><th class=n>AP@50</th><th class=n>AP@50-95</th><th class=n>AP@50 COCO sin ajuste</th><th class=n>Instancias</th></tr>{per_class}</table>
<h3>4.3 Matriz de confusión y análisis de errores</h3>
<p>La matriz del modelo propuesto (Figura {fig_cm}) muestra que el error dominante no es confundir clases sino no detectar el objeto.
Perdemos el {pct(miss['person'], 0)} % de las personas, el {pct(miss['bicycle'], 0)} % de las bicicletas y el {pct(miss['motorcycle'], 0)} % de las motos, frente al {pct(miss['car'], 0)} % de los carros.
Ese orden coincide con el tamaño de los objetos. A la entrada de 640 px, la mitad de las personas mide menos de {num(sizes['person']['median_px'], 1)} px de lado, mientras que un bus típico mide {num(sizes['bus']['median_px'], 1)} px.
Objetos tan pequeños ocupan menos de una celda en el mapa más fino que tiene YOLO11n.</p>
<p>Las confusiones entre clases se concentran en los vehículos grandes: {thousands(M('car','truck'))} camiones salen como carro, el {pct(M('car','truck') / cm['true_total']['truck'], 0)} % del total, y {thousands(M('truck','car'))} carros salen como camión.
Visto desde arriba, un furgón, que agrupamos como carro, y un camión pequeño casi solo se diferencian por el largo; esa frontera la pusimos nosotros al agrupar las clases.
Algo parecido pasa con los vehículos de dos ruedas. El {pct(M('motorcycle','bicycle') / cm['true_total']['bicycle'], 0)} % de las bicicletas se predice como moto, y hay cruces entre persona y moto en ambos sentidos. VisDrone anota por separado al conductor y a su vehículo, con cajas casi superpuestas, y desde el dron el conductor tapa casi toda la moto.</p>
<figure class=cm><img src="figures/confusion.png"><figcaption>Figura {fig_cm}. Matrices de confusión en test (confianza 0,25), normalizadas por columna: cada columna es una clase real. La fila "fondo" son objetos no detectados; la columna "fondo", detecciones sin objeto real.</figcaption></figure>
<p>Los falsos positivos suman {thousands(fp_total)}, sobre todo de {CLASS_ES[fp_top[0][0]]} y {CLASS_ES[fp_top[1][0]]}. Al revisar las imágenes con más falsos positivos (Figura {fig_fp}) encontramos que la mayoría son objetos reales que nadie anotó: carros en vías del fondo y personas en multitudes de noche. El resto cae sobre cosas con silueta parecida, como papeleras, troncos o postes.
Por eso creemos que la precisión real es más alta que la medida, y que el problema de fondo es el recall de objetos pequeños.
La mejor clase es <i>{CLASS_ES[best_cls[0]]}</i> (AP@50 de {num(best_cls[1]['map50'])}), la más abundante y la de objetos más grandes; la peor es <i>{CLASS_ES[worst_cls[0]]}</i> ({num(worst_cls[1]['map50'])}), pequeña y escasa.</p>
<figure class=fp><img src="figures/false_positives.jpg"><figcaption>Figura {fig_fp}. Imágenes de test con más falsos positivos. Verde: anotaciones. Rojo: detecciones sin anotación que coincida.</figcaption></figure>

<h2>5. Conclusiones, limitaciones y reproducibilidad</h2>
<p>La transferencia funciona, pero en este dominio no basta con entrenar la cabeza. Con el backbone congelado pasamos de {pct(zs['map50'])} % a {pct(s1['map50'])} % de mAP@50 en test, y solo al ajustar también el backbone llegamos a {pct(s2['map50'])} %. La vista aérea cambia los rasgos desde las primeras capas.{(' El esquema en dos etapas nos sirvió para separar lo que aporta la cabeza de lo que aporta el backbone, pero como receta final recomendaríamos una sola etapa con LR 1e-3, que dio ' + pct(ff['map50']) + ' %.') if ff is not None and ff['map50'] > s2['map50'] + 0.005 else ''}</p>
<p>El límite actual es la escala, no la semántica: la matriz tiene muchos más objetos perdidos que confundidos. Lo siguiente que probaríamos es entrenar con imágenes más grandes (960 a 1.280 px) o recortadas en mosaicos, agregar una salida de mayor resolución para objetos pequeños y usar las 6.471 imágenes de train.</p>
<p>Este trabajo tiene límites claros. Usamos un subconjunto de train y una sola semilla, sin intervalos de confianza. En la GPU del M4 algunas operaciones no son deterministas aunque fijemos la semilla. Además, los triciclos quedaron fuera del problema.</p>
<p class=small>Reproducibilidad: Python {v['python']}, PyTorch {v['torch']}, Ultralytics {v['ultralytics']} y semilla 0, en {v['machine']}{hw.get('_note', '')}. Todo se regenera con <code>python -m transfer_learning.prepare_data --download</code>, <code>python -m transfer_learning.experiments all</code>, el notebook <code>transfer_learning/act3_transfer_learning.ipynb</code> y <code>python -m transfer_learning.report.build_report</code>.</p>

<h2>6. Referencias</h2>
<ol class=refs>
<li>Zhu, P. et al. (2021). Detection and Tracking Meet Drones Challenge. <i>IEEE TPAMI</i>, 44(11), 7380-7399 (VisDrone).</li>
<li>Jocher, G., Qiu, J. (2024). <i>Ultralytics YOLO11</i>. https://docs.ultralytics.com</li>
<li>Yosinski, J. et al. (2014). How transferable are features in deep neural networks? <i>NeurIPS</i>.</li>
<li>Loshchilov, I., Hutter, F. (2019). Decoupled Weight Decay Regularization. <i>ICLR</i> (AdamW).</li>
<li>Akyon, F. C. et al. (2022). Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection. <i>ICIP</i> (SAHI).</li>
</ol>
"""

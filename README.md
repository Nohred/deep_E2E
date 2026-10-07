# Clasificación y localización con ConvDetector

Este proyecto entrena un `ConvDetector` para **un objeto por imagen** con
Pascal VOC 2012. No es un detector multiobjeto: no usa anchors, NMS ni
recortes diferenciables. Los objetos no seleccionados no se evalúan en este
baseline.

## Datos y selección del objeto

Se usan las anotaciones de detección XML en
`data/VOCdevkit/VOC2012/Annotations` y los splits oficiales
`ImageSets/Main/train.txt` y `val.txt`. Se excluyen objetos `difficult`, nombres
fuera de las 20 categorías VOC y cajas degeneradas. En imágenes con varios
objetos se selecciona el objeto elegible de mayor área; los empates conservan
el orden del XML. El dataset informa los conteos incluidos y excluidos de cada
split.

Las clases, en orden estable e índices `0..19`, son:
`aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train,
tvmonitor`.

VOC almacena cajas 1-based e inclusivas. Se convierten una sola vez a
coordenadas 0-based semiabiertas `[xmin, ymin, xmax, ymax]`, se redimensionan
y se normalizan por `(width, height)`. Las imágenes se convierten con
`to_tensor`, por lo que son `float32` en `[0, 1]`; no se aplica normalización
ImageNet.

## Flujo y contrato

```text
VOC XML + JPEG
    -> dataset selecciona un objeto
    -> [B,3,H,W] float32, [B] long, [B,4] float32 normalizado
    -> extractor convolucional compartido
       -> cabeza lineal de caja [B,4]
       -> cabeza lineal de clasificación [B,20] (logits)
```

`model(images)` devuelve `(pred_boxes, class_logits)`. La cabeza de caja es
lineal y no garantiza cajas válidas. `CrossEntropyLoss` recibe logits
directamente; `softmax` y `argmax` sólo se usan para scores y métricas.

La pérdida es:

```text
loss_total = CrossEntropyLoss(class_logits, labels)
             + LAMBDA_BOX * SmoothL1Loss(pred_boxes, target_boxes)
```

`LAMBDA_BOX` comienza en `1.0` y está declarado en `train.py`. Ambas pérdidas
actualizan la cabeza correspondiente y el extractor compartido. Scheduler y
early stopping monitorizan la pérdida total de validación. El mejor estado se
restaura antes de guardarlo.

El IoU de entrenamiento/validación recorta predicciones a `[0,1]` sólo para la
métrica, no para la pérdida. No intercambia esquinas: una caja invertida o
degenerada obtiene IoU cero. También se registra `invalid_box_rate`. El IoU
evalúa la localización del objeto seleccionado independientemente de acertar
su clase.

## Ejecución

Desde la raíz, con PyTorch, torchvision, NumPy y Matplotlib instalados:

```bash
python train.py
```

Variables editables en `train.py`: `batch_size`, `image_size`,
`learning_rate`, `num_epochs`, `patience`, `min_delta`, `weight_decay`,
`lambda_box` y `data_dir`. No se ejecuta un split aleatorio adicional.

Se generan:

- `artifacts/best_detector.pth`: pesos y metadatos (`num_classes`, clases,
  formato/normalización de cajas, tamaño y rango de imagen, historial y mejor
  pérdida).
- `artifacts/detector_metadata.json`: metadatos legibles.
- `artifacts/history.png`: pérdidas total, clasificación y localización.
- `artifacts/validation_detection.png`: cajas reales verdes y predichas rojas.

Para inferencia, reconstruir `ConvDetector(num_classes=checkpoint["num_classes"])`
y cargar `checkpoint["model_state_dict"]` con `torch.load(..., map_location=...)`.
El checkpoint es distinto de cualquier peso heredado de U-Net.

## Módulos relevantes

- `datasets/voc.py`: lectura XML, política de selección, transformación y loaders.
- `models/dect.py`: extractor convolucional y dos cabezas paralelas.
- `engine/trainer.py`: entrenamiento, validación, pérdidas y métricas.
- `callbacks/early_stopping.py`: guarda/restaura el mejor estado.
- `utils/plotting.py`: historial y visualización de detección.
- `utils/device.py`: selección de CPU/GPU.

Los módulos heredados de MNIST, ONNX, Triton y GUI no forman parte de este
flujo y no fueron adaptados en este cambio.

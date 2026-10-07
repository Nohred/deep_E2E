# MNIST Deep Learning

Proyecto de reconocimiento de digitos escritos a mano del dataset MNIST usando PyTorch. Incluye entrenamiento de una CNN, exportacion a ONNX y despliegue del modelo con NVIDIA Triton Inference Server para realizar inferencias por lotes.

## Flujo del proyecto

1. Descargar/cargar MNIST y entrenar la CNN.
2. Guardar los pesos entrenados en `artifacts/best_model.pth`.
3. Exportar y validar el modelo en formato ONNX.
4. Servir el modelo con Triton mediante Docker Compose.
5. Enviar un batch de imagenes y validar las predicciones.

## Estructura

```text
.
├── artifacts/
│   └── best_model.pth              # Pesos del modelo entrenado
├── callbacks/
│   └── early_stopping.py           # Early stopping
├── data/MNIST/                     # Datos de MNIST
├── datasets/main.py                # DataLoaders y preprocesamiento
├── engine/trainer.py               # Entrenamiento y evaluacion
├── models/
│   ├── base.py                     # Clase base
│   ├── cnn.py                      # CNN usada en el despliegue
│   ├── mlp.py                      # Modelo MLP alternativo
│   ├── vgg.py                      # Modelo VGG alternativo
│   └── factory.py                  # Fabrica de modelos
├── serving/
│   ├── export_cnn_onnx.py          # Exportacion y validacion ONNX
│   └── model_repository/cnn/       # Repositorio de modelos de Triton
├── test_inference.py               # Prueba de inferencia batch
├── test.bash                       # Pruebas simples de disponibilidad
├── train.py                        # Punto de entrada del entrenamiento
├── docker-compose.yml              # Servicio de Triton
└── README.md
```

## Requisitos

- Python 3.11 o superior
- Docker y Docker Compose
- Al menos 2 GB de espacio para MNIST, dependencias y la imagen de Triton
- GPU NVIDIA y NVIDIA Container Toolkit son opcionales. Sin ellos, Triton funciona usando CPU.

## Instalacion

Se recomienda usar un entorno virtual o Conda. Desde la raiz del proyecto:

```bash
conda create -n deeplearning python=3.11 -y
conda activate deeplearning
python -m pip install torch torchvision numpy matplotlib onnx onnxruntime "tritonclient[http]"
```

El extra `http` es necesario porque el cliente de Triton HTTP usa dependencias adicionales como `gevent`.

Para comprobar el cliente:

```bash
python -c "import torch, torchvision, onnx, onnxruntime, tritonclient.http; print('Dependencias OK')"
```

## Entrenamiento

El script entrena la CNN, descarga MNIST si es necesario y guarda los pesos en `artifacts/best_model.pth`:

```bash
python train.py
```

El entrenamiento usa normalizacion MNIST con media `0.1307` y desviacion `0.3081`. Si se cambia el modelo o sus pesos, se debe volver a ejecutar la exportacion ONNX antes de iniciar Triton.

## Exportar a ONNX

Con los pesos disponibles en `artifacts/best_model.pth`:

```bash
python serving/export_cnn_onnx.py
```

El archivo generado es:

```text
serving/model_repository/cnn/1/model.onnx
```

El script tambien compara la salida de PyTorch con la salida de ONNX. La exportacion ajusta la version IR del archivo para que sea compatible con Triton 23.06.

## Levantar Triton

Inicia el servidor desde la raiz del proyecto:

```bash
docker compose up -d
```

Comprobar el estado del servidor y del modelo:

```bash
curl http://localhost:8000/v2/health/ready
curl http://localhost:8000/v2/models/cnn/ready
```

Ambos comandos deben responder con HTTP `200`. Para ver los logs:

```bash
docker compose logs -f triton
```

Para detener el servidor:

```bash
docker compose down
```

El servicio expone:

- HTTP: `localhost:8000`
- gRPC: `localhost:8001`
- Metricas: `localhost:8002`

## Inferencia por batches

Con Triton ejecutandose, envia 10 imagenes de MNIST en una sola peticion HTTP:

```bash
python test_inference.py
```

El resultado muestra la forma del batch, la forma de salida, predicciones, etiquetas, confianza y accuracy. La forma esperada es:

```text
Sent batch: (10, 1, 28, 28)
Received output: (10, 10)
```

Cambiar el tamano del batch:

```bash
python test_inference.py --batch-size 32
```

El modelo acepta hasta 32 imagenes por batch segun `config.pbtxt`. Tambien se pueden cambiar la URL, el nombre del modelo y la carpeta de datos:

```bash
python test_inference.py \
  --url localhost:8000 \
  --model-name cnn \
  --data-dir data/MNIST \
  --batch-size 10
```



# Modelo generativo CNN
Este proyecto implementa un pipeline de visión por computadora + NLP para convertir imágenes de diagramas UML en estructuras JSON semánticas y narrativas en lenguaje natural. Combina un encoder CNN para imágenes y un decoder Transformer entrenado con un tokenizer GPT-2 byte-level.

🚀 Características principales
Tokenización GPT-2 byte-level con tokens especiales (<start>, <end>, <unk>, <pad>).

Dataset personalizado que enlaza imágenes .png con anotaciones .json.

Modelo híbrido:

Encoder CNN para extracción de características visuales.

Decoder Transformer para generación de secuencias JSON.

Entrenamiento completo con PyTorch (optimización con Adam, pérdida CrossEntropy).

Inferencia paso a paso con sampling controlado (temperature, top-k).

Post-procesamiento heurístico para limpiar texto y reconstruir JSON válido.

Narrativa automática que describe el sistema, actores, casos de uso y relaciones.

📂 Estructura del proyecto
Code
├── dataset/
│   ├── diagrama_0001.png
│   ├── diagrama_0001.json
│   └── ...
├── diagram_image2json.pth   # Modelo entrenado
├── tokenizer/               # Tokenizer guardado
├── import os.txt            # Script principal
└── README.md
⚙️ Instalación
Clona el repositorio:

bash
git clone https://github.com/tuusuario/image2json.git
cd image2json
Instala dependencias:

bash
pip install torch torchvision transformers pillow
(Opcional) Instala soporte GPU con CUDA para PyTorch siguiendo la guía oficial: PyTorch Get Started.

🏋️‍♂️ Entrenamiento
Ejecuta el script principal para entrenar el modelo:

bash
python import\ os.txt
Entrena durante 100 épocas.

Guarda el modelo en diagram_image2json.pth.

Guarda el tokenizer en ./tokenizer.

🔎 Inferencia
Ejemplo de uso:

python
from import_os import infer_image, flujo_a_texto
import torch, json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_test = "dataset/diagrama_0001.png"

flujo = infer_image(img_test, model, tokenizer, device)
print(json.dumps(flujo, indent=2, ensure_ascii=False))
print(flujo_a_texto(flujo))
Salida esperada:

json
{
  "sistema": "Sistema GestiónClientes",
  "actores": ["Cliente", "Administrador"],
  "casos_uso": ["Registrar usuario", "Consultar datos"],
  "relaciones": [
    {"actor": "Cliente", "caso_uso": "Registrar usuario"}
  ]
}
Narrativa:

Code
El sistema es Sistema GestiónClientes. 
Los actores principales son: Cliente, Administrador. 
Los casos de uso incluyen: Registrar usuario, Consultar datos. 
El actor Cliente participa en el caso de uso Registrar usuario.
📖 Aplicaciones
Interpretación automática de diagramas UML.

Generación de documentación técnica a partir de imágenes.

Integración en pipelines de FastAPI para endpoints de OCR semántico.

Base para proyectos de ingeniería de software asistida por IA.

🛠️ Tecnologías utilizadas
Python 3.10+

PyTorch (CNN + Transformer)

Transformers (Hugging Face)

PIL / torchvision para procesamiento de imágenes

Regex + heurísticas para limpieza y reconstrucción de JSON

📌 Próximos pasos
Mejorar dataset con más variaciones de diagramas UML.

Implementar curriculum learning para robustez en inferencia.

Exportar resultados en formatos adicionales (Markdown, HTML).

Integrar modelos de lenguaje más avanzados (LLaMA, GPT-NeoX).

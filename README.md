# Modelo OpenCV-OCR

Este repositorio contiene un script para la **detección de actores y casos de uso en diagramas de casos de uso del sistema** y extracción de texto asociado mediante **OCR (EasyOCR)**. El pipeline del primer script permite identificar la posición de cada actor, verificar la existencia de la cabeza, definir regiones de interés (ROI) hacia arriba y hacia abajo, y finalmente extraer el texto contenido debajo del actor.

---

## Archivos principales

| Archivo                | Descripción                                                                                                                                                |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `script_definitivo.py` | Script principal para la detección de actores y extracción de texto. Implementa todo el pipeline de procesamiento de imágenes, detección de cabezas y OCR. |

---

## Detección de actores

El script sigue un flujo de trabajo estructurado en varias funciones, con estrategias de procesamiento y detección adaptadas a diagramas de Draw.io:

### 1. Inicialización (`__init__`)

* Carga la imagen desde disco.
* Maneja imágenes con transparencia (canal alfa), rellenando el fondo con blanco si es necesario.
* Obtiene dimensiones de la imagen para calcular las ROIs.

### 2. Preprocesamiento (`preprocess`)

* Convierte la imagen a escala de grises.
* Invierte colores si el fondo es claro para facilitar la detección.
* Aplica un umbral binario para resaltar las figuras de los actores.

### 3. Detección de actores por plantilla (`find_actors_by_template`)

* Genera plantillas de actores (cabeza, cuerpo, brazos) de varios tamaños.
* Aplica **template matching** (`cv2.matchTemplate`) para localizar coincidencias en la imagen.
* Filtra duplicados cercanos para evitar contar el mismo actor varias veces.

### 4. Verificación de cabeza (`verify_head_circle`)

* Define una ROI por encima del actor donde debería encontrarse la cabeza.
* Aplica **HoughCircles** para detectar círculos que representen la cabeza.
* Filtra círculos no alineados geométricamente con la posición del actor.
* Permite ajustar el **ancho y alto de la ROI** para mayor precisión.
* Devuelve la posición de la cabeza si existe.

### 5. Extracción de texto debajo del actor (`extract_text_below`)

* Define una ROI hacia abajo del actor (ancho fijo, altura configurable).
* Utiliza **EasyOCR** para reconocer texto dentro de la ROI.
* Devuelve el texto detectado junto con la posición del ROI.
* Ideal para diagramas de Draw.io donde el texto es legible y bien definido.

### 6. Pipeline principal (`detect_actors`)

* Combina detección por plantilla y verificación de cabeza.
* Genera un listado de actores validados.
* Para cada actor, extrae el texto debajo usando OCR.
* Produce resultados finales y dibuja una imagen de salida con:

  * Actor detectado
  * Cabeza detectada
  * ROI superior e inferior
  * Texto detectado

### 7. Salida y visualización (`draw_results`)

* Dibuja círculos sobre actores y cabezas detectadas.
* Dibuja rectángulos para ROIs superiores e inferiores.
* Inserta etiquetas de texto indicando "HEAD" o "NO HEAD" y el texto detectado debajo.
* Guarda la imagen final como `actors_debug_output.png`.

---

### Estrategias y Consideraciones

* **ROI adaptables:** permite ajustar alto y ancho para mejorar precisión en distintos diagramas.
* **Filtro de duplicados:** evita detectar múltiples veces el mismo actor en áreas cercanas.
* **OCR confiable:** EasyOCR se configura con los idiomas `['en', 'es']` y no requiere GPU.
* **Pipeline modular:** cada función tiene responsabilidades claras (detección, verificación, OCR, dibujo), facilitando mantenimiento y extensión.
* **Debug opcional:** permite guardar imágenes intermedias de ROIs y detección de círculos para depuración.

---

¿Querés que haga eso también?

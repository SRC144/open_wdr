# Codificador de Imágenes Wavelet Difference Reduction (WDR) Híbrido Python/C++

[English](README.md) | [Español](README.es.md)

## Resumen

Wavelet Difference Reduction (WDR) combina transformadas wavelet discretas, bit-plane coding progresivo y adaptive arithmetic coding para lograr altos ratios de compresión y opcionalmente una reconstruccion lossless. La API en Python mantiene los flujos de trabajo simples, mientras que el engine en C++ (construido con pybind11 + CMake) se encarga del rendimiento.

La librería utiliza una arquitectura por bloques para procesar de forma eficiente en memoria imágenes grandes y gigapixel. Las imágenes se procesan en bloques de tamaño fijo (por defecto 512×512), lo que permite comprimir imágenes que superan la RAM disponible, manteniendo una calidad uniforme en los límites entre bloques gracias a un mecanismo de umbral global (global T).

## Instalación y Compilación (todas las plataformas)

1. **Clonar**
   ```bash
   git clone <repo>
   cd wdr_compression_pipeline
   ```
2. **(Opcional) Crear entorno virtual**
   ```bash
   python3 -m venv .venv && source .venv/bin/activate   # Linux/macOS
   python -m venv .venv && .venv\Scripts\activate       # Windows
   ```
3. **Instalar prerequisitos**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```
   Esto incluye las librerias de OpenSlide (`openslide-python` + `openslide-bin`) para soporte de formatos WSI de microscopía (NDPI, SVS, Philips TIFF).

4. **Compilar e instalar el paquete**
   ```bash
   pip install -e .
   ```

`pip install -e .` configura CMake, compila el módulo nativo `wdr.coder` y lo expone como paquete Python. Si la compilación falla, verifique que tiene Python ≥3.8, CMake ≥3.15 y un compilador C++17. El archivo `TROUBLESHOOTING.md` contiene diagnósticos detallados, notas por plataforma y métodos alternativos.


## Configuración de desarrollo y pruebas (opcional)

### Pruebas de Python

```bash
python -m pytest tests/
```

### Compilación nativa + pruebas C++

```bash
pip install pybind11
PYBIND11_DIR=$(python3 -c 'import pybind11, pathlib; print(pathlib.Path(pybind11.get_cmake_dir()))')
cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="$PYBIND11_DIR"
cmake --build build -j && ctest --test-dir build --output-on-failure
```

¿Solo necesita revisar el output de CMake? Ejecute la primera línea sin `-DBUILD_TESTING=ON` para compilar el módulo, o agregue `-DCMAKE_VERBOSE_MAKEFILE=ON` para más registro. El artefacto aparece dentro del paquete como `wdr/coder.cpython-<ver>-<platform>.so|pyd`.

Para ver qué cubre cada suite y cómo extenderla, consulte `tests/README.md`.

## API Principal

### `wdr.io.compress()`

Comprime un array numpy 2D en un archivo `.wdr` usando un enfoque por bloques. La función divide la imagen en bloques, aplica DWT a cada bloque, comprime cada uno con WDR y escribe al archivo, procesando los bloques de forma secuencial para mantener un uso de memoria O(1). Esto permite comprimir imágenes gigapixel sin cargar todos los coeficientes en RAM (Podrían ser billones para una imágen en gigapixeles.). El parámetro de umbral global (Global T) garantiza una alineación consistente de los planos de bits entre bloques, evitando artefactos visuales que aparecerían si cada bloque usara un nivel de cuantización distinto.

**Parámetros:**
- `image_source`: Array numpy 2D (canal único, dtype float64)
- `output_path`: Ruta del archivo `.wdr` de destino
- `global_T`: Umbral global calculado desde toda la imagen, asegurando alineación de planos de bits entre bloques
- `tile_size`: Tamaño de los bloques de compresión (por defecto 512)
- `scales`: Niveles de descomposición DWT
- `wavelet`: Tipo de wavelet (por defecto `'bior4.4'`)
- `num_passes`: Número de passes. A más passes mayor precisión pero mayor cantidad de bits.
- `quant_step`: Tamaño de paso de cuantización opcional (None o 0 para compresión lossless)

**Retorna:** None (escribe archivo `.wdr` en disco)

### `wdr.io.decompress()`

Transmite bloques descomprimidos desde un archivo `.wdr`, retornando un generador que produce arrays numpy 2D. La función lee los metadatos del archivo, descomprime los bloques de uno en uno, aplica DWT inversa y produce bloques en orden row-major. Este enfoque eficiente en memoria mantiene solo un bloque en RAM a la vez, lo que permite reconstruir imágenes más grandes que la memoria disponible.

**Parámetros:**
- `wdr_path`: Ruta al archivo `.wdr`
- `progress_callback`: Callback opcional que acepta un float (0.0-1.0) para seguimiento de progreso (ej. Indicador de progreso en cli)

**Retorna:** Generador que produce arrays numpy 2D (bloques reconstruidos, dtype float64)

### Funciones Auxiliares Principales (`wdr.utils.helpers`)

- **`scan_for_max_coefficient()`**: Escanea todos los bloques para encontrar el valor máximo global de coeficientes. Es necesario ejecutarla antes de la compresión para calcular `global_T`.
- **`calculate_global_T()`**: Calcula el umbral global a partir del coeficiente máximo. Garantiza que todos los bloques usen la misma alineación de planos de bits durante la compresión.
- **`do_dwt()` / `do_idwt()`**: Transformada wavelet discreta directa e inversa. Convierte bloques de imagen a coeficientes wavelet y viceversa.
- **`flatten_coeffs()` / `unflatten_coeffs()`**: Convierte tuplas de coeficientes DWT (subbandas multi-nivel) a arrays planos y viceversa para la codificación WDR.
- **`quantize_coeffs()` / `dequantize_coeffs()`**: Cuantización opcional para compresión con pérdidas. Use `quant_step=0` para flujos sin pérdidas.
- **`yield_tiles()`**: Generador que produce bloques de imagen con relleno en los bordes. Maneja tanto arrays en RAM como archivos mapeados en memoria.

## Arquitectura por Bloques

La librería procesa imágenes en bloques de tamaño fijo (por defecto 512×512) en lugar de procesar toda la imagen de una vez. Este enfoque ofrece varias ventajas:

1. **Eficiencia de Memoria**: Solo se procesa un bloque a la vez, manteniendo un uso de memoria O(1) independientemente del tamaño de la imagen. Esto permite manejar imágenes gigapixel que exceden la RAM disponible.

2. **Umbral Global**: Todos los bloques comparten el mismo umbral global (`global_T`) calculado a partir de toda la imagen. Esto garantiza la alineación de planos de bits entre los límites de los bloques, evitando artefactos visuales que aparecerían si cada bloque usara un nivel de cuantización distinto.

3. **Manejo de Bordes**: Los bloques en los bordes se rellenan para mantener un tamaño de bloque consistente; el relleno se elimina durante la reconstrucción. El relleno utiliza replicación de bordes para minimizar artefactos en los límites.

4. **Streaming**: Tanto la compresión como la descompresión transmiten bloques de forma secuencial, lo que hace la librería adecuada para imágenes almacenadas en disco (p. ej., archivos TIFF mapeados en memoria) y conjuntos de datos grandes.

## Ejemplos de Uso

### API de Python

Flujo completo que muestra compresión y descompresión con reensamblaje de bloques:

```python
import numpy as np
from PIL import Image
from wdr import io as wdr_io
from wdr.utils import helpers as hlp

# Cargar imagen (debe ser canal único)
img = np.array(Image.open("input.png").convert("L"), dtype=np.float64)
height, width = img.shape

# Paso 1: Calcular umbral global
# Esto escanea todos los bloques para encontrar el coeficiente máximo, asegurando
# alineación consistente de planos de bits entre bloques durante la compresión.
global_max = hlp.scan_for_max_coefficient(img, tile_size=512, scales=2, wavelet="bior4.4")
global_T = hlp.calculate_global_T(global_max)

# Paso 2: Comprimir (los bloques se procesan internamente y se transmiten a disco)
wdr_io.compress(
    image_source=img,
    output_path="output.wdr",
    global_T=global_T,
    tile_size=512,
    scales=2,
    wavelet="bior4.4",
    num_passes=16,
    quant_step=0  # 0 = sin pérdidas
)

# Paso 3: Descomprimir y reensamblar bloques
tiles = wdr_io.decompress("output.wdr")
reconstructed = np.zeros((height, width), dtype=np.float64)

tile_size = 512
tile_idx = 0
for r in range((height + tile_size - 1) // tile_size):
    for c in range((width + tile_size - 1) // tile_size):
        tile = next(tiles)
        
        # Calcular región válida (recortar relleno de bordes)
        y_start = r * tile_size
        x_start = c * tile_size
        y_end = min(y_start + tile_size, height)
        x_end = min(x_start + tile_size, width)
        tile_h = y_end - y_start
        tile_w = x_end - x_start
        
        # Colocar bloque en imagen reconstruida
        reconstructed[y_start:y_end, x_start:x_end] = tile[:tile_h, :tile_w]
        tile_idx += 1

# Guardar imagen reconstruida
Image.fromarray(np.clip(reconstructed, 0, 255).astype(np.uint8)).save("reconstructed.png")
```

### Aplicación de Ejemplo: Imágenes Médicas de Microscopía

El directorio `scripts/` incluye una implementación completa del flujo Color Wavelet Difference Reduction (CWDR) para imágenes médicas, según lo descrito en Zerva et al. (2023). Este ejemplo demuestra cómo aplicar la librería WDR a imágenes completas de microscopía (WSI) de escáneres de patología, manejando la conversión del espacio de color RGB a YUV, compresión WDR independiente por canal (Y, U, V), reconstrucción y evaluación de calidad. Las herramientas integran OpenSlide para leer formatos propietarios (NDPI, SVS, Philips TIFF).

#### Pipeline de Compresión/Extracción WSI

`scripts/wdr_wsi_pipeline.py` es la herramienta principal para comprimir y extraer imágenes de microscopía completas. Maneja la detección de formato, extracción de bloques mediante OpenSlide, conversión automática al espacio de color YCbCr y limpieza de archivos intermedios.

```bash
# Comprimir una imagen de microscopía
python scripts/wdr_wsi_pipeline.py compress CMU-1.svs cmu1 \
  --tile-size 512 \
  --scales 2 \
  --wavelet bior4.4 \
  --passes 16 \
  --qstep 0

# Extraer de vuelta a RGB BigTIFF
python scripts/wdr_wsi_pipeline.py extract results/ cmu1 reconstructed.tiff
```

El pipeline genera tres archivos de canal (`_Y.wdr`, `_U.wdr`, `_V.wdr`) para almacenamiento eficiente. Use `--keep-temp` para conservar los archivos TIFF intermedios durante depuración.

#### Inspección de Metadatos

`scripts/wsi_info.py` inspecciona rápidamente los metadatos de la lámina sin cargar la imagen completa. Útil para verificar dimensiones y formato antes del procesamiento.

```bash
python scripts/wsi_info.py CMU-1.svs
```

#### Evaluación de Métricas de Calidad

`scripts/wsi_metrics.py` calcula PSNR y SSIM entre láminas reconstruidas y originales usando procesamiento por bloques para evitar agotamiento de memoria en imágenes gigapixel.

```bash
python scripts/wsi_metrics.py reconstructed.tiff CMU-1.svs --tile-size 2048
```

Estas herramientas sirven como implementaciones de referencia que muestran flujos de trabajo prácticos con WSI. Adáptelas según sus necesidades específicas o construya pipelines personalizados usando la API principal directamente.

## Referencia

Esta implementación se basa en:

**Algoritmo Wavelet Difference Reduction (WDR):**  
Tian, J., Wells, R.O. (2002). Embedded Image Coding Using Wavelet Difference Reduction. In: Topiwala, P.N. (eds) Wavelet Image and Video Compression. The International Series in Engineering and Computer Science, vol 450. Springer, Boston, MA. https://doi.org/10.1007/0-306-47043-8_17

**Color WDR (CWDR) para Imágenes Médicas:**  
Zerva, M.C.H., Christou, V., Giannakeas, N., Tzallas, A.T., & Kondi, L.P. (2023). "An Improved Medical Image Compression Method Based on Wavelet Difference Reduction." IEEE Access, vol. 11, pp. 18026-18037. https://doi.org/10.1109/ACCESS.2023.3246948

**Codificación Aritmética Adaptativa:**  
Witten, I.H., Neal, R.M., & Cleary, J.G. (1987). "Arithmetic coding for data compression." Communications of the ACM, 30(6), 520-540.

## Documentación

- `TROUBLESHOOTING.md`: notas por plataforma, recetas Docker/Conda y matriz de problemas.
- `tests/README.md`: describe qué cubre cada suite de pruebas y cómo ampliarlas.
- `README.md`: versión en inglés de esta guía rápida.
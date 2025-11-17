# Refactor del Pipeline Basado en Disco con Tiles

## 1. Resumen de Alto Nivel

Se refactorizó el pipeline de compresión WDR para procesar imágenes grandes tile por tile con memoria O(tile) mientras se escriben datos intermedios a disco. Python ahora orquesta un flujo de dos pasos:

1. **Paso 1 – Escaneo y caché de tiles**: el orquestador divide la imagen fuente en tiles, ejecuta DWT/flattening por tile y escribe arrays aplanados a disco (`.npy`). Este paso también rastrea el máximo global de coeficientes para calcular el umbral inicial compartido `T`.

2. **Paso 2 – Compresión de tiles**: los tiles cacheados se recargan secuencialmente, se cuantizan opcionalmente y se envían al worker C++ mediante la nueva API `compress_tile`. El payload de cada tile se escribe con prefijo de longitud y se anexa a un archivo `.wdrt` junto con un header tiled y metadatos por tile. La descompresión invierte este proceso tile por tile, llamando a `decompress_tile`.

El diseño separa responsabilidades:

- **Python (main.py)** maneja I/O de archivos, DWT/IDWT, caché de tiles, formato de framing y ergonomía CLI.

- **C++ (`WDRCompressor`)** se convierte en un worker sin estado que comprime/descomprime un solo tile completamente en memoria, ya no asume propiedad del array completo de coeficientes, y expone APIs de tile vía pybind11.

## 2. Cambios Archivo por Archivo y Razonamiento

### `src/wdr_compressor.hpp` / `src/wdr_compressor.cpp`

- **Qué cambió**: La clase perdió estado global de miembros (`original_coeffs_ptr_`, vectores ICS/SCS/TPS, buffers de reconstrucción) y ahora mantiene estado con alcance de tile mediante structs `EncoderState`/`DecoderState` pasados a través de llamadas helper. Se agregaron métodos públicos `compress_tile`/`decompress_tile` que retornan/aceptan payloads raw, mientras que los legacy `compress`/`decompress` simplemente envuelven el path de tile para preservar compatibilidad hacia atrás. Las funciones helper (pases de sorting/refinement, reducciones binarias) ganaron corrección `const` y ahora aceptan parámetros de estado explícitos.

- **Por qué**: La API antigua asumía que todo el array de coeficientes permanecía residente, lo cual era inviable para imágenes gigapixel. Helpers sin estado permiten que Python alimente tiles de tamaño arbitrario sin filtrar memoria entre tiles y coinciden con la arquitectura de streaming a disco.

- **Teoría**: WDR es un codificador wavelet embebido impulsado por división de umbral (`T -> T/2` cada paso). Al compartir `initial_T` entre tiles, mantenemos orden determinístico de planos de bits a través de toda la imagen, asegurando que descomprimir tiles en aislamiento produzca la misma reconstrucción que compresión monolítica siempre que los escaneos de tile ordenen coeficientes idénticamente.

### `src/wdr_file_format.hpp`

- **Qué cambió**: Se introdujo metadatos de formato tiled—`TiledFileHeader`, `TileChunkHeader`, constantes magic/versión (`WDRT`), y flags describiendo cuantización. Cada chunk registra origen de tile, tamaño de pixel, conteo de coeficientes, y longitud de payload.

- **Por qué**: Se requiere un contenedor streaming para que la descompresión pueda iterar tiles sin leer todo el archivo. Los headers codifican parámetros de reconstrucción (scales, nombre de wavelet, T global, paso de cuantización) para que el decodificador no dependa de metadatos externos.

### `src/bindings.cpp`

- **Qué cambió**: Se agregaron bindings pybind11 para `compress_tile` y `decompress_tile`, exponiéndolos como funciones orientadas a bytes que reflejan las necesidades del orquestador Python. La validación de entrada es paralela a las funciones legacy.

- **Por qué**: El orquestador debe invocar el worker C++ por tile, así que los bindings ahora aceptan arrays NumPy/bytes y retornan `py::bytes`/arrays sin tocar el filesystem.

### `main.py`

- **Qué cambió**: CLI reescrito para implementar el flujo de trabajo tiled de dos pasos:

  - Nuevos helpers `_cache_tiles`, `_encode_tiles`, `_read_tiled_header`, `_decompress_tiled_file` gestionan caché respaldado en disco e iteración de tiles.

  - CLI gana flags de configuración de tile, control de directorio de caché, y logging mejorado.

  - Compresión/descompresión ahora producen archivos `.wdrt` con headers personalizados y framing de chunks.

- **Por qué**: Python orquesta DWT/IDWT e I/O de disco más convenientemente que C++. Puede aprovechar NumPy/Pillow para tiling y caches persistidos mientras reutiliza utilidades helper existentes (cuantización, flattening). El CLI necesitaba guiar al usuario a través del pipeline tiled, incluyendo advertencias sobre cuantización y ratios de compresión.

- **Teoría**: Separar DWT y codificación entrópica permite un pipeline estilo "map-reduce": DWT (map) por tile, codificación WDR (reduce) por tile. El `initial_T` compartido asegura que cada tile comience en el mismo umbral de significancia, evitando costuras al reconstruir. El paso de cuantización es especificado por el usuario o derivado de `initial_T` (método threshold-based) para balancear compresión vs. distorsión.

### `tests/test_tiled_pipeline.py`

- **Qué cambió**: Se agregaron tests unitarios Python cubriendo la estructura de header/chunk tiled, helper `calculate_initial_T`, y serialización de metadatos de caché.

- **Por qué**: Asegura que la lógica de framing del lado Python permanezca estable, especialmente porque el formato es nuevo y debe interoperar con descompresión.

### `tests/test_cpp/test_wdr_compressor_roundtrip.cpp`

- **Qué cambió**: Se agregó `TileCompressDecompressRoundTrip` (con tolerancia derivada de `initial_T / 2^passes`) y `TileCompressMatchesFullCompressor` para validar las nuevas APIs worker. Suite de tests actualizada ejecutada vía CMake/ctest.

- **Por qué**: Proporciona cobertura nativa de que las APIs de tile se comportan consistentemente con el compresor legacy y que la fidelidad de reconstrucción coincide con límites de error esperados impulsados por umbral.

### Documentación (`README.md`, `TROUBLESHOOTING.md`)

- **Qué cambió**: Se documentó el uso CLI tiled, comportamiento respaldado en disco, y se agregó guía de build offline (`pip install -e . --no-build-isolation`) para que desarrolladores puedan reconstruir el módulo sin acceso a red.

- **Por qué**: Los usuarios necesitan conocer el nuevo flujo de trabajo y cómo reconstruir la extensión nativa después de cambiar los bindings.

## 3. Flujo de Extremo a Extremo (Paso a Paso)

1. **Invocación CLI** (`main.py`):

   - Parsea argumentos tile/scales/cuantización, resuelve paths de salida.

   - Llama a `run_tiled_compression`.

2. **Paso 1 – `_cache_tiles`**:

   - Divide la imagen PIL en tiles de `(tile_width, tile_height)`.

   - Convierte cada tile a `np.float64`, realiza DWT vía `hlp.do_dwt`, aplana coeficientes con `hlp.flatten_coeffs`, los guarda como `.npy`.

   - Rastrea el máximo absoluto global de coeficientes para cálculo posterior de umbral.

3. **Calcular umbral compartido y cuantización**:

   - `_calculate_initial_threshold_from_max` refleja la lógica C++ para calcular `initial_T`.

   - `_resolve_quantization_step` usa el paso proporcionado por el usuario, deshabilita cuantización, o deriva un paso threshold-based (enfoque en compresión) para crear redundancia.

4. **Paso 2 – `_encode_tiles`**:

   - Escribe `TiledFileHeader` con metadatos de imagen/tile, número de pasos, info de cuantización, y el `initial_T` compartido.

   - Carga arrays de coeficientes cacheados tile por tile, cuantiza si está habilitado, y llama a `wdr_coder.compress_tile`.

   - Escribe `TileChunkHeader` + payload al archivo `.wdrt`; mantiene estadísticas corrientes para ratios de compresión algoritmo/sistema.

5. **Descompresión (opcional)**:

   - `_read_tiled_header` valida el archivo tiled, `_decompress_tiled_file` itera sobre chunks, llama a `wdr_coder.decompress_tile`, desquantiza si es necesario, reconstruye imágenes de tile vía `hlp.unflatten_coeffs` + `hlp.do_idwt`, y las une en el array final.

6. **Métricas y artefactos**:

   - CLI reporta total de tiles, tamaño de archivo, ratios de compresión algoritmo/sistema, paso de cuantización, y PSNR/MSE/RMSE si se solicitó una imagen reconstruida.

## 4. Cálculo de Métricas por Lotes (Batched Metrics)

### Problema Identificado

Después de implementar el pipeline tiled, el cálculo de métricas (PSNR/MSE) para imágenes gigapixel fallaba con `OSError: decoder error -9` cuando se intentaba cargar la imagen original completa en memoria usando `hlp.load_image()` o reconstruirla completamente con `_reconstruct_full_image()`. Esto ocurría porque:

- Pillow/libtiff intentaba decodificar toda la imagen TIFF gigapixel de una vez
- Incluso con el guard de decompression bomb deshabilitado, operaciones como `img.convert("L")` o `img.crop()` en imágenes muy grandes causaban agotamiento de memoria o recursos
- El enfoque anterior requería mantener ambas imágenes (original y reconstruida) completas en RAM simultáneamente

### Solución: Métricas por Lotes

**Qué cambió**: Se implementó un sistema de cálculo de métricas que procesa imágenes tile por tile, acumulando estadísticas sin cargar imágenes completas en memoria.

**Dónde cambió**:

1. **Nuevo archivo: `wdr/utils/batched_metrics.py`**
   - Clase `BatchedMetrics`: Acumula suma de errores cuadrados y conteo de píxeles a través de múltiples batches
   - Función `compute_batched_metrics()`: Orquesta el procesamiento tile por tile usando lectores de tile
   - Protocolo `TileReader`: Interfaz tipada para lectores que soportan acceso por bloques

2. **Modificado: `main.py` (líneas 719-752)**
   - Reemplazó la reconstrucción completa de imagen para métricas con cálculo por lotes
   - Crea lectores de tile para ambas imágenes (original y reconstruida)
   - Llama a `compute_batched_metrics()` en lugar de cargar imágenes completas

3. **Modificado: `wdr/utils/__init__.py`**
   - Exporta `BatchedMetrics` y `compute_batched_metrics` para fácil importación

4. **Nuevo archivo: `tests/test_batched_metrics.py`**
   - 8 casos de prueba cubriendo casos edge, acumulación, clipping, e integración con lectores reales

**Cómo funciona**:

1. **Acumulación incremental**:
   - `BatchedMetrics.add_batch()` procesa un tile a la vez:
     - Recorta ambos arrays (original y reconstruido) al rango [0, 255]
     - Calcula errores cuadrados: `(original - reconstructed)²`
     - Acumula `sum_squared_errors` y `total_pixels`
   - Múltiples llamadas a `add_batch()` acumulan estadísticas sin mantener tiles anteriores en memoria

2. **Cálculo final**:
   - `BatchedMetrics.finalize()` computa:
     - `MSE = sum_squared_errors / total_pixels`
     - `RMSE = sqrt(MSE)`
     - `PSNR = 20 * log10(255 / RMSE)` (o `inf` si MSE=0)

3. **Procesamiento streaming**:
   - `compute_batched_metrics()` itera sobre la grilla de tiles (misma que compresión)
   - Lee tiles correspondientes de ambos lectores simultáneamente
   - Maneja conversión RGB→grayscale si es necesario
   - Valida dimensiones y formas antes de agregar a métricas

**Por qué este enfoque**:

- **Memoria O(tile)**: Solo mantiene un tile de cada imagen en memoria a la vez
- **Compatible con lectores existentes**: Usa el mismo protocolo `TileReader` que el pipeline de compresión
- **Matemáticamente equivalente**: MSE acumulado por lotes es idéntico a MSE calculado sobre la imagen completa
- **Evita errores de decodificación**: No intenta cargar imágenes completas que exceden límites de memoria/recursos
- **Reutiliza infraestructura**: Aprovecha `PillowTileReader` y `TifffileTileReader` ya implementados

**Teoría matemática**:

El MSE se puede calcular incrementalmente porque es una suma separable:

```
MSE = (1/N) * Σ(original[i] - reconstructed[i])²
    = (1/N) * Σ_tiles Σ_pixels_in_tile (orig[p] - recon[p])²
    = (1/N) * Σ_tiles sum_squared_errors_tile
```

Donde `N = total_pixels = Σ_tiles pixels_in_tile`. Por lo tanto, acumular `sum_squared_errors` y `total_pixels` por lotes produce el mismo resultado que calcular sobre la imagen completa, pero con memoria constante.

**Detalles de implementación**:

- **Protocolo `TileReader`**: Define una interfaz mínima (`size()`, `read_block()`, `close()`) que permite que `compute_batched_metrics()` funcione con cualquier implementación de lector (Pillow, tifffile, o mocks para testing)

- **Manejo de tipos**: La función convierte automáticamente tiles a `float64` y maneja conversión RGB→grayscale usando los mismos coeficientes estándar (0.2989, 0.5870, 0.1140) que el resto del pipeline

- **Validación robusta**: Verifica que dimensiones coincidan entre imágenes, que shapes de tiles coincidan dentro de cada batch, y que se hayan agregado píxeles antes de finalizar

- **Clipping automático**: Ambos arrays se recortan a [0, 255] antes de calcular errores, asegurando que valores fuera de rango no distorsionen las métricas

## 5. Pruebas y Validación

- **Python**: `pytest` ahora incluye tests del pipeline tiled, asegurando que packing/unpacking de headers y lógica de cuantización permanezcan consistentes. También incluye 8 tests para métricas por lotes cubriendo:
  - Imágenes idénticas (MSE=0, PSNR=inf)
  - Diferencias conocidas con validación matemática
  - Acumulación de múltiples batches
  - Manejo de errores (shape mismatches, acumuladores vacíos)
  - Clipping de valores fuera de rango
  - Integración con lectores reales (PillowTileReader)

- **C++**: `ctest` ejercita tanto round trips WDR legacy como las nuevas APIs de tile (con tolerancia derivada de `initial_T / 2^passes` para límites de error predecibles).

- **Ejecución manual**: Comando de muestra  

  `python main.py assets/lenna.png compressed/demo-tiled.wdrt --tile-width 128 --tile-height 128 --scales 2 --num-passes 8 --reconstructed compressed/demo-recon.png`

  verifica que el pipeline complete y reporte los ratios de compresión/PSNR esperados.

- **Validación con imágenes gigapixel**: El pipeline ahora puede procesar imágenes TIFF de 51,200 × 38,144 píxeles (CMU-1.tiff) sin errores de decodificación, calculando métricas en lotes sin cargar la imagen completa.

## 6. Puntos Clave

- El pipeline ahora maneja imágenes grandes sin cargar todos los coeficientes simultáneamente, aprovechando disco como buffer de spill mientras mantiene el estado del codificador aritmético determinístico vía un umbral inicial compartido.

- El worker C++ es sin estado y reutilizable desde Python a través de nuevos bindings, lo cual simplifica testing e integraciones futuras (ej., workers multiproceso).

- El formato de archivo tiled (`.wdrt`) está explícitamente documentado y validado, haciendo el nuevo flujo de trabajo auditable y más fácil de depurar.

- El cálculo de métricas por lotes permite evaluar calidad de reconstrucción para imágenes gigapixel sin cargar imágenes completas, resolviendo errores de decodificación y limitaciones de memoria. La implementación es matemáticamente equivalente al cálculo tradicional pero con complejidad de memoria O(tile) en lugar de O(imagen completa).


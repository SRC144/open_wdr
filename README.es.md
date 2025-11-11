# Codificador de Imágenes Wavelet Difference Reduction (WDR) Híbrido Python/C++

[English](README.md) | [Español](README.es.md)

## Introducción

Este proyecto proporciona una implementación completa del algoritmo Wavelet Difference Reduction (WDR) para compresión de imágenes embebida. WDR es una técnica eficiente de compresión de imágenes que combina transformadas wavelets discretas con transmisión progresiva y codificación aritmética adaptativa para lograr altas tasas de compresión mientras soporta reconstrucción sin pérdidas.

### Características Principales

- **Flujo de Bits Embebido**: La compresión completamente embebida permite detenerse en cualquier punto para cumplir con una tasa de bits objetivo o nivel de distorsión
- **Transmisión Progresiva**: Las imágenes pueden transmitirse y mostrarse progresivamente con calidad mejorada
- **Compresión Sin Pérdidas**: Con suficientes pasadas, logra compresión sin pérdidas
- **Arquitectura Híbrida Python/C++**: Interfaz Python para facilitar el uso, núcleo C++ para rendimiento
- **Sin Entrenamiento Requerido**: La codificación aritmética adaptativa no requiere entrenamiento previo

## Características Principales

- **Algoritmo de Compresión WDR**: Implementación completa del algoritmo Wavelet Difference Reduction
- **Codificación Aritmética Adaptativa**: Implementación del algoritmo de codificación aritmética Witten-Neal-Cleary (1987)
- **Arquitectura Híbrida Python/C++**: Python para E/S de imágenes y DWT, C++ para el núcleo de compresión
- **Soporte para Transmisión Progresiva de Imágenes**: El flujo de bits embebido soporta decodificación progresiva
- **Capacidad de Compresión Sin Pérdidas**: Puede lograr compresión sin pérdidas con suficientes pasadas

## Cómo Funciona

El pipeline de compresión WDR consiste en tres etapas principales:

1. **Transformada Wavelet Discreta (DWT)**: Transforma la imagen al dominio de frecuencia usando wavelets
2. **Compresión WDR**: Codifica coeficientes significativos usando codificación diferencial, reducción binaria y transmisión por planos de bits
3. **Codificación Aritmética Adaptativa**: Comprime el flujo de símbolos usando codificación aritmética adaptativa

### Resumen del Algoritmo

```
Imagen de Entrada
    ↓
DWT (Transformada Wavelet Discreta)
    ↓
Aplanar Coeficientes (Orden de Escaneo WDR)
    ↓
Compresión WDR
    ├─ Pasa de Ordenamiento: Encuentra y codifica coeficientes significativos
    ├─ Pasa de Refinamiento: Refina coeficientes existentes
    └─ Codificación Aritmética Adaptativa: Compresión final
    ↓
Archivo .wdr
```

### Orden de Escaneo

El algoritmo procesa coeficientes wavelet en un orden "coarse-to-fine" (de grueso a fino):

- **Orden**: LL_N → HL_N → LH_N → HH_N → HL_{N-1} → ... → HH_1
- **Subbandas HL**: Escaneadas columna por columna (verticalmente)
- **Subbandas LL, LH, HH**: Escaneadas fila por fila (horizontalmente)

Para una explicación teórica detallada, consulte [docs/theory.md](docs/theory.md).

## Instalación

### Prerrequisitos

- **Python**: 3.8 o superior
- **CMake**: 3.15 o superior
- **Compilador C++**: Compilador compatible con C++17 (GCC, Clang, o MSVC)
- **Paquetes Python**: NumPy, PyWavelets, Pillow, pybind11

### Instalación Paso a Paso

1. **Clonar el repositorio** (o descargar el código fuente)

2. **Instalar dependencias de Python**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Compilar e instalar el paquete**:
   ```bash
   pip install -e .
   ```

Esto:
- Compilará el módulo de extensión C++ usando CMake
- Compilará el núcleo de compresión WDR
- Instalará el paquete Python con el módulo `wdr_coder`

### Compilar desde el Código Fuente

Si necesita compilar manualmente:

```bash
# Crear directorio de compilación
mkdir build
cd build

# Configurar con CMake
cmake ..

# Compilar
cmake --build .

# El módulo Python se compilará en la raíz del proyecto
```

### Solución de Problemas

**Problema**: CMake no encontrado
- **Solución**: Instalar CMake desde https://cmake.org/download/

**Problema**: Headers de desarrollo de Python no encontrados
- **Solución**: Instalar paquetes de desarrollo de Python:
  - Ubuntu/Debian: `sudo apt-get install python3-dev`
  - macOS: Los headers de desarrollo de Python están incluidos con las herramientas de línea de comandos de Xcode
  - Windows: Instalar Python desde python.org con herramientas de desarrollo

**Problema**: NumPy no encontrado durante la compilación
- **Solución**: Asegurarse de que NumPy esté instalado: `pip install numpy`

## Inicio Rápido

### API de Python

```python
import numpy as np
import wdr_coder
import wdr_helpers as hlp

# Cargar imagen
original_img = hlp.load_image("input.png")

# Realizar DWT (por defecto: 2 escalas, recomendado: 2-3)
# Nota: scales=6+ introduce artefactos de borde como advierte PyWavelets
wavelet_coeffs = hlp.do_dwt(original_img, scales=2, wavelet='bior4.4')

# Aplanar coeficientes
flat_coeffs, shape_data = hlp.flatten_coeffs(wavelet_coeffs)

# Comprimir (por defecto: 26 pasadas para alta precisión)
wdr_coder.compress(flat_coeffs, "compressed.wdr", num_passes=26)

# Descomprimir
decompressed_flat_coeffs = wdr_coder.decompress("compressed.wdr")

# Desaplanar coeficientes
decompressed_coeffs = hlp.unflatten_coeffs(decompressed_flat_coeffs, shape_data)

# Realizar IDWT
reconstructed_img = hlp.do_idwt(decompressed_coeffs)

# Guardar imagen reconstruida
hlp.save_image("reconstructed.png", reconstructed_img)
```

### Uso desde Línea de Comandos

```bash
# Comprimir una imagen
python main.py input.png output.wdr

# Comprimir con escalas personalizadas (recomendado: 2-3)
python main.py input.png output.wdr --scales 2

# Comprimir y guardar imagen reconstruida
python main.py input.png output.wdr --scales 2 --reconstructed recon.png
```

**Nota**: El valor por defecto de escalas es 2 (2-3 recomendado). Usar scales=6+ introduce artefactos de borde como advierte PyWavelets, por lo que no se recomienda para uso práctico.

## Estructura del Proyecto

```
wdr_compression_pipeline/
├── src/                    # Código fuente C++
│   ├── arithmetic_coder.*  # Codificador aritmético adaptativo (Witten-Neal-Cleary)
│   ├── adaptive_model.*    # Modelo de probabilidad adaptativo
│   ├── bit_stream.*        # E/S a nivel de bits
│   ├── wdr_compressor.*    # Núcleo de compresión WDR
│   ├── wdr_file_format.*   # Definiciones de formato de archivo
│   └── bindings.cpp        # Enlaces Python (pybind11)
├── wdr_helpers.py          # Funciones auxiliares Python (DWT, E/S, aplanado)
├── main.py                 # Script de ejemplo de línea de comandos
├── tests/                  # Archivos de prueba
│   ├── test_wdr_helpers.py # Pruebas de funciones auxiliares Python
│   ├── test_wdr_coder.py   # Pruebas de integración
│   └── test_cpp/           # Pruebas unitarias C++ (GTest)
├── docs/                   # Documentación
│   ├── theory.md           # Explicación teórica
│   └── theory.es.md        # Explicación teórica (Español)
├── CMakeLists.txt          # Configuración de compilación CMake
├── setup.py                # Configuración del paquete Python
└── requirements.txt        # Dependencias de Python
```

### Archivos Clave

- **`src/wdr_compressor.*`**: Implementación del algoritmo de compresión WDR
- **`src/arithmetic_coder.*`**: Implementación de codificación aritmética adaptativa (Witten-Neal-Cleary 1987)
- **`wdr_helpers.py`**: Funciones Python para E/S de imágenes, DWT y aplanado de coeficientes
- **`main.py`**: Script de ejemplo que demuestra el pipeline de compresión
- **`docs/theory.md`**: Explicación teórica exhaustiva de los algoritmos

## Pruebas

### Pruebas de Python

Ejecutar la suite de pruebas de Python:

```bash
python -m pytest tests/
```

O ejecutar archivos de prueba específicos:

```bash
python -m pytest tests/test_wdr_helpers.py
python -m pytest tests/test_wdr_coder.py
```

### Pruebas de C++

Compilar y ejecutar pruebas unitarias de C++:

```bash
cd build
cmake ..
cmake --build .
ctest
```

O ejecutar pruebas con salida detallada:

```bash
cd build
ctest --verbose
```

## Documentación

### Documentación Teórica

Para una explicación teórica exhaustiva del algoritmo WDR y la codificación aritmética adaptativa, consulte:

- **[docs/theory.md](docs/theory.md)**: Explicación detallada del algoritmo (Inglés)
- **[docs/theory.es.md](docs/theory.es.md)**: Explicación detallada del algoritmo (Español)

### Documentación de API

La API de Python está documentada en el código fuente. Funciones clave:

- **`wdr_coder.compress(coeffs, output_file, num_passes=26)`**: Comprimir coeficientes a un archivo .wdr
- **`wdr_coder.decompress(input_file)`**: Descomprimir coeficientes desde un archivo .wdr
- **`wdr_helpers.load_image(filepath)`**: Cargar un archivo de imagen
- **`wdr_helpers.save_image(filepath, img_array)`**: Guardar un archivo de imagen
- **`wdr_helpers.do_dwt(img_array, scales=2, wavelet='bior4.4')`**: Realizar DWT
- **`wdr_helpers.do_idwt(coeffs, wavelet='bior4.4')`**: Realizar IDWT
- **`wdr_helpers.flatten_coeffs(coeffs)`**: Aplanar coeficientes wavelet
- **`wdr_helpers.unflatten_coeffs(flat_coeffs, shape_data)`**: Desaplanar coeficientes wavelet

## Créditos y Referencias

### Atribución Apropiada

Esta implementación está basada en las siguientes fuentes teóricas:

#### Algoritmo WDR

Esta implementación está basada en el algoritmo Wavelet Difference Reduction para compresión de imágenes embebida. El algoritmo combina transformadas wavelets discretas con codificación eficiente de índices y transmisión progresiva.

**Cita Completa:**
[Cita del artículo WDR - por completar con detalles del artículo]

#### Codificación Aritmética Adaptativa

La implementación de codificación aritmética adaptativa está basada en el algoritmo de:

**Witten, I.H., Neal, R.M., & Cleary, J.G. (1987).** "Arithmetic coding for data compression." *Communications of the ACM*, 30(6), 520-540.

Este artículo presenta el algoritmo de codificación aritmética adaptativa utilizado en la etapa final de compresión de WDR. La implementación en C++ mantiene equivalencia matemática con el algoritmo original mientras usa características modernas de C++17.

### Declaración de Implementación

Esta implementación está basada en las fuentes teóricas citadas anteriormente. Se otorga crédito completo a los autores e investigadores originales que desarrollaron estos algoritmos. Los comentarios del código y la documentación incluyen atribución apropiada para asegurar integridad académica y otorgar crédito donde corresponde.

### Recursos Adicionales

- **PyWavelets**: Biblioteca Python para transformadas wavelets discretas (https://pywavelets.readthedocs.io/)
- **NumPy**: Biblioteca de computación numérica para Python (https://numpy.org/)
- **Pillow**: Biblioteca de imágenes Python para E/S de imágenes (https://pillow.readthedocs.io/)
- **pybind11**: Operabilidad fluida entre C++11 y Python (https://pybind11.readthedocs.io/)

## Contribuir

¡Las contribuciones son bienvenidas! Por favor siga estas pautas:

### Estilo de Código

- **C++**: Seguir mejores prácticas de C++17 moderno, usar nombres de variables significativos, agregar comentarios para lógica compleja
- **Python**: Seguir la guía de estilo PEP 8, usar anotaciones de tipo donde sea apropiado, agregar docstrings a todas las funciones

### Pruebas

- Agregar pruebas para nuevas características
- Asegurar que todas las pruebas pasen antes de enviar
- Probar con ambas suites de pruebas Python y C++

### Documentación

- Actualizar documentación para nuevas características
- Agregar ejemplos para nueva funcionalidad
- Mantener documentación teórica precisa

## Licencia

[Información de licencia por agregar]

## Selección de Idioma

- **[English](README.md)**: Este documento
- **[Español](README.es.md)**: Documentación en español

---

Para explicaciones teóricas detalladas, consulte [docs/theory.md](docs/theory.md).


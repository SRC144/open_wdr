# Codificador de Imágenes Wavelet Difference Reduction (WDR) Híbrido Python/C++

[English](README.md) | [Español](README.es.md)

## Resumen

Wavelet Difference Reduction (WDR) combina transformadas wavelet discretas, codificación progresiva por planos de bits y codificación aritmética adaptativa para obtener alta compresión con opción de reconstrucción sin pérdidas. Python ofrece la interfaz sencilla; C++ (via pybind11 + CMake) entrega el rendimiento.

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
3. **Instalar herramientas**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```
4. **Compilar e instalar**
   ```bash
   pip install -e .
   ```

`pip install -e .` configura CMake, compila el módulo nativo `wdr.coder` y lo expone como paquete Python. Si algo falla, confirme que tiene Python ≥3.8, CMake ≥3.15 y un compilador C++17. El archivo `TROUBLESHOOTING.md` reúne diagnósticos detallados, notas por plataforma y métodos alternativos.

## Activos y salidas

- Los insumos de ejemplo viven en `assets/` (por ejemplo, `assets/lenna.png`, `assets/pattern.png`) para ejecutar demos rápidas.
- Las salidas generadas se guardan en `compressed/` (ejecuciones del CLI) y `compressed/tests/` (suite de pruebas). Si solo indica un nombre de archivo en `main.py`, el script escribe el `.wdr` y la reconstrucción dentro de `compressed/`.

## Configuración de desarrollo y pruebas (opcional)

### Pruebas de Python

```bash
python -m pytest tests/
```

### Compilación nativa + pruebas C++

```bash
cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

¿Solo necesita revisar la salida de CMake? Ejecute la primera línea sin `-DBUILD_TESTING=ON` para compilar el módulo, o agregue `-DCMAKE_VERBOSE_MAKEFILE=ON` para más registro. El artefacto aparece dentro del paquete como `wdr/coder.cpython-<ver>-<platform>.so|pyd`.

Para conocer qué cubre cada suite y cómo extenderla, consulte `tests/README.md`.

## Ejemplos de Uso

### API de Python

```python
from wdr import coder as wdr_coder
from wdr.utils import helpers as hlp

img = hlp.load_image("input.png")
coeffs = hlp.do_dwt(img, scales=2, wavelet="bior4.4")
flat_coeffs, shape_data = hlp.flatten_coeffs(coeffs)

wdr_coder.compress(flat_coeffs, "output.wdr", num_passes=26)
decoded = wdr_coder.decompress("output.wdr")
unflat = hlp.unflatten_coeffs(decoded, shape_data)
reconstructed = hlp.do_idwt(unflat)
hlp.save_image("reconstructed.png", reconstructed)
```

Las utilidades de cuantización (`calculate_quantization_step`, `quantize_coeffs`, `dequantize_coeffs`) viven en `wdr.utils.helpers` y siguen siendo opcionales; defina `--quantization-step 0` (o sáltelas) para obtener un flujo sin pérdidas.

### CLI

```bash
python main.py input.png output.wdr \
  --scales 2 \
  --wavelet bior4.4 \
  --num-passes 26 \
  --reconstructed recon.png  # archivo opcional con la imagen decodificada
```

Si omite el directorio en `output.wdr` o `--reconstructed`, el script los guarda automáticamente en `compressed/`; indique rutas completas si desea otra ubicación.

## Documentación

- `TROUBLESHOOTING.md`: notas por plataforma, recetas Docker/Conda y matriz de problemas.
- `tests/README.md`: describe qué cubre cada suite de pruebas y cómo ampliarlas.
- `README.md`: versión en inglés de esta guía rápida.

Disfrute del pipeline y abra un issue si alguno de estos pasos queda desactualizado.


# Teoría del Algoritmo WDR

[English](theory.md) | [Español](theory.es.md) | [Volver a README](../README.md)

## Tabla de Contenidos

1. [Introducción](#introducción)
2. [Fundamentos Matemáticos](#fundamentos-matemáticos)
3. [Detalles del Algoritmo WDR](#detalles-del-algoritmo-wdr)
4. [Codificación Aritmética Adaptativa](#codificación-aritmética-adaptativa)
5. [Formato de Archivo](#formato-de-archivo)
6. [Referencias y Atribución](#referencias-y-atribución)

## Introducción

El método Wavelet Difference Reduction (WDR) es una técnica de compresión de imágenes embebida que combina el poder de las transformadas wavelets con codificación eficiente de índices y transmisión progresiva. A diferencia de métodos basados en árboles como EZW (Embedded Zerotree Wavelet) o SPIHT (Set Partitioning in Hierarchical Trees), WDR utiliza un enfoque directo para encontrar y codificar las posiciones de coeficientes wavelet significativos.

### Ventajas Clave

- **Flujo de Bits Embebido**: Los datos comprimidos están completamente embebidos, lo que significa que el decodificador puede detenerse en cualquier punto para cumplir con una tasa de bits objetivo o nivel de distorsión
- **Transmisión Progresiva**: Las imágenes pueden transmitirse y mostrarse progresivamente, con la calidad mejorando a medida que se reciben más bits
- **Compresión Sin Pérdidas**: Con suficientes pasadas, el algoritmo puede lograr compresión sin pérdidas
- **Sin Entrenamiento Requerido**: La etapa de codificación aritmética adaptativa no requiere entrenamiento previo ni parámetros del modelo

### Resumen del Algoritmo

El algoritmo WDR consiste en tres etapas principales:

1. **Transformada Wavelet Discreta (DWT)**: Transforma la imagen al dominio de frecuencia usando wavelets
2. **Codificación de Índices**: Codifica las posiciones de coeficientes significativos usando codificación diferencial y reducción binaria
3. **Codificación Aritmética Adaptativa**: Comprime el flujo de símbolos resultante usando codificación aritmética adaptativa

## Fundamentos Matemáticos

### Transformada Wavelet Discreta (DWT)

La Transformada Wavelet Discreta descompone una imagen en múltiples bandas de frecuencia en diferentes escalas. Para una imagen 2D, la DWT produce:

- **LL (Low-Low)**: Coeficientes de aproximación (baja frecuencia en ambas direcciones)
- **HL (High-Low)**: Coeficientes de detalle horizontal (alta frecuencia horizontalmente, baja frecuencia verticalmente)
- **LH (Low-High)**: Coeficientes de detalle vertical (baja frecuencia horizontalmente, alta frecuencia verticalmente)
- **HH (High-High)**: Coeficientes de detalle diagonal (alta frecuencia en ambas direcciones)

Esta descomposición puede repetirse en la banda LL para crear múltiples escalas (niveles) de descomposición.

### Funciones Base Wavelet

Las wavelets son funciones matemáticas que localizan energía en los dominios de tiempo (o espacio) y frecuencia. Esta localización dual las hace ideales para representar señales transitorias como bordes y texturas en imágenes.

### Análisis Multi-Resolución

La DWT proporciona una representación multi-resolución de la imagen, donde:
- Las escalas gruesas capturan características a gran escala y regiones suaves
- Las escalas finas capturan detalles, bordes y texturas

Esta estructura multi-resolución es explotada por WDR a través de su orden de escaneo "coarse-to-fine" (de grueso a fino).

## Detalles del Algoritmo WDR

### Orden de Escaneo

El algoritmo WDR procesa coeficientes wavelet en un orden de escaneo "coarse-to-fine" específico. Este orden asegura que los coeficientes importantes (aquellos en bandas de frecuencia más bajas) se procesen y transmitan antes que los menos importantes.

#### Secuencia de Escaneo

Para una descomposición DWT de N niveles, el orden de escaneo es:

```
LL_N → HL_N → LH_N → HH_N → HL_{N-1} → LH_{N-1} → HH_{N-1} → ... → HL_1 → LH_1 → HH_1
```

#### Heurísticas de Escaneo

- **Subbandas HL**: Escaneadas **columna por columna** (verticalmente)
- **Subbandas LL, LH, HH**: Escaneadas **fila por fila** (horizontalmente)

Este orden de escaneo es crítico para el rendimiento del algoritmo, ya que asegura que los coeficientes se procesen en orden de importancia.

#### Diagrama: Orden de Escaneo WDR

```
3-Level DWT Decomposition:

┌─────────┬─────────┬─────────┬─────────┐
│   LL_3  │   HL_3  │         │         │
│         │    ↓    │         │         │
│         │    ↓    │   LH_3  │   HH_3  │
├─────────┼─────────┤    →    │    →    │
│   HL_2  │         │         │         │
│    ↓    │         │         │         │
│    ↓    │         │         │         │
├─────────┼─────────┼─────────┼─────────┤
│         │   LH_2  │   HH_2  │         │
│         │    →    │    →    │         │
│         │         │         │         │
├─────────┴─────────┴─────────┴─────────┤
│   HL_1  │   LH_1  │   HH_1  │         │
│    ↓    │    →    │    →    │         │
└─────────┴─────────┴─────────┴─────────┘

Scanning Order: 
  1. LL_3 (row-by-row: →)
  2. HL_3 (column-by-column: ↓)
  3. LH_3 (row-by-row: →)
  4. HH_3 (row-by-row: →)
  5. HL_2 (column-by-column: ↓)
  6. LH_2 (row-by-row: →)
  7. HH_2 (row-by-row: →)
  8. HL_1 (column-by-column: ↓)
  9. LH_1 (row-by-row: →)
 10. HH_1 (row-by-row: →)

Legend: 
  ↓ = column-by-column (vertical scan)
  → = row-by-row (horizontal scan)
```

### Pasa de Ordenamiento

La pasa de ordenamiento identifica coeficientes que se vuelven "significativos" (su valor absoluto excede el umbral actual T) y codifica sus posiciones.

#### Proceso

1. **Encontrar Coeficientes Significativos**: Iterar a través del ICS (Conjunto de Coeficientes Insignificantes) e identificar coeficientes donde |x| ≥ T
2. **Almacenar Índices**: Almacenar los índices de coeficientes significativos en la lista P
3. **Almacenar Signos**: Almacenar los signos de los coeficientes significativos
4. **Codificación Diferencial**: Codificar los índices usando codificación diferencial
5. **Reducción Binaria**: Aplicar reducción binaria a los índices diferenciales
6. **Codificar**: Codificar los índices reducidos binariamente y los signos usando codificación aritmética
7. **Actualizar ICS**: Remover coeficientes significativos del ICS y re-enumerar

#### Diagrama: Diagrama de Flujo de la Pasa de Ordenamiento

```mermaid
flowchart TD
    A[Start: ICS with coefficients] --> B[Iterate through ICS]
    B --> C{Is |x| >= T?}
    C -->|Yes| D[Add index to P]
    C -->|No| E[Keep in ICS]
    D --> F[Store sign]
    F --> G[Move to TPS]
    E --> H{More coefficients?}
    G --> H
    H -->|Yes| B
    H -->|No| I[Apply Differential Coding to P]
    I --> J[Apply Binary Reduction]
    J --> K[Encode with Arithmetic Coding]
    K --> L[Remove from ICS]
    L --> M[Move TPS to SCS]
    M --> N[End]
```

#### Ejemplo de Pasa de Ordenamiento

**Entrada:**
- ICS: `[10, -5, 35, 8, -42, 3]`
- T = 32

**Proceso:**
1. Encontrar significativos: índices 2 (35) y 4 (-42)
2. P = `[2, 4]`, signos = `[1, 0]` (positivo, negativo)
3. Codificación diferencial: P' = `[2, 2]` (4 - 2 = 2)
4. Reducción binaria: 
   - 2 = `10` → `0` (remover MSB)
   - 2 = `10` → `0` (remover MSB)
5. Codificar: `0`, `1`, `0`, `0` (índices y signos entrelazados)
6. Actualizar ICS: Remover índices 2 y 4, re-enumerar → `[10, -5, 8, 3]`

#### Codificación Diferencial

La codificación diferencial codifica las diferencias entre valores adyacentes en una secuencia monótonamente creciente.

**Ejemplo:**
- Índices originales: `P = {1, 2, 5, 36, 42}`
- Codificación diferencial: `P' = {1, 1, 3, 31, 6}`

El primer valor permanece sin cambios, y cada valor subsecuente es la diferencia del anterior.

**Proceso de Codificación:**
```
P[0] = 1  → P'[0] = 1          (first value unchanged)
P[1] = 2  → P'[1] = 2 - 1 = 1  (difference from previous)
P[2] = 5  → P'[2] = 5 - 2 = 3  (difference from previous)
P[3] = 36 → P'[3] = 36 - 5 = 31 (difference from previous)
P[4] = 42 → P'[4] = 42 - 36 = 6 (difference from previous)
```

**Proceso de Decodificación:** Revertir tomando la suma parcial:
```
P'[0] = 1  → P[0] = 1
P'[1] = 1  → P[1] = 1 + 1 = 2
P'[2] = 3  → P[2] = 2 + 3 = 5
P'[3] = 31 → P[3] = 5 + 31 = 36
P'[4] = 6  → P[4] = 36 + 6 = 42
```

Esta codificación es eficiente cuando los índices están agrupados, ya que las diferencias son típicamente pequeñas.

#### Reducción Binaria

La reducción binaria representa un entero binario positivo removiendo el Bit Más Significativo (MSB). Esto reduce el número de bits necesarios para representar el número, ya que el MSB siempre es '1' para números positivos.

**Ejemplo:**
- Número: `19` en binario es `10011`
- Reducción binaria: Remover MSB → `0011`

**Proceso de Codificación:**
```
Value: 19
Binary: 10011
        ^
        MSB (always 1 for positive numbers)

Remove MSB: 0011
```

**Proceso de Decodificación:** Revertir anteponiendo un '1' como MSB:
```
Reduced: 0011
Prepend '1': 10011
Convert to decimal: 19
```

**Ejemplo Visual:**
```
Original:  19 = 10011 (5 bits)
            ^
            MSB
Reduced:   0011 (4 bits)
            ^
            Prepend '1' to decode

This saves 1 bit per number (20% reduction for 5-bit numbers).
```

En WDR, el signo del coeficiente se usa como delimitador entre índices reducidos en el flujo de bits, permitiendo que el decodificador sepa dónde termina un índice y comienza el siguiente.

### Pasa de Refinamiento

La pasa de refinamiento agrega un bit de precisión a coeficientes que ya fueron encontrados significativos en pasadas anteriores.

#### Proceso

Para cada coeficiente en el SCS (Conjunto de Coeficientes Significativos):

1. **Calcular Intervalo**: 
   - `low = center - T`
   - `high = center + T`

2. **Determinar Bit**:
   - Si el valor verdadero está en la mitad superior `[center, high)`: salida bit '1'
   - Si el valor verdadero está en la mitad inferior `[low, center)`: salida bit '0'

3. **Actualizar Centro**:
   - Si bit es '1': `center = (center + high) / 2`
   - Si bit es '0': `center = (low + center) / 2`

4. **Codificar Bit**: Codificar el bit usando codificación aritmética con el modelo de refinamiento

Este proceso estrecha el intervalo que contiene el valor verdadero del coeficiente, agregando un bit de precisión por pasada.

#### Diagrama: Pasa de Refinamiento

```
Pass 0: Coefficient found significant at T = 32
  True value: x = 49
  Interval: [32, 64)
  Initial center: 48 (1.5*T)
  
Pass 1: T = 16
  Current center: 48
  Interval: [32, 64) = [48-16, 48+16)
  ┌─────────────────────────────────────┐
  │ [32)        [48)        [64)        │
  │   │──────────┼──────────│           │
  │   │  lower   │  upper   │           │
  │   │  half    │  half    │           │
  │   └──────────┴──────────┘           │
  │           x=49 is here → output '1' │
  └─────────────────────────────────────┘
  New center: (48 + 64) / 2 = 56
  
Pass 2: T = 8
  Current center: 56
  Interval: [48, 64) = [56-8, 56+8)
  ┌─────────────────────────────────────┐
  │ [48)        [56)        [64)        │
  │   │──────────┼──────────│           │
  │   │  lower   │  upper   │           │
  │   └──────────┴──────────┘           │
  │      x=49 is here → output '0'      │
  └─────────────────────────────────────┘
  New center: (48 + 56) / 2 = 52
  
And so on... The interval narrows with each pass.
```

### Gestión de Listas

El algoritmo WDR mantiene tres estructuras de datos clave que rastrean coeficientes a lo largo del proceso de compresión:

#### ICS (Conjunto de Coeficientes Insignificantes)
- Contiene coeficientes que aún no son significativos (|x| < T)
- Inicialmente contiene todos los coeficientes en orden de escaneo
- Se reduce a medida que los coeficientes se vuelven significativos y se mueven a SCS
- Los coeficientes se re-enumeran después de cada pasada para mantener indexación secuencial

#### SCS (Conjunto de Coeficientes Significativos)
- Contiene coeficientes que son significativos (|x| ≥ T)
- Almacena tuplas de `(value, center)` donde:
  - `value`: El valor original del coeficiente (o aproximación actual)
  - `center`: El centro de reconstrucción actual (usado para refinamiento)
- Crece a medida que los coeficientes se vuelven significativos
- Los coeficientes en SCS se refinan en cada pasada para agregar precisión
- El valor center se actualiza durante el refinamiento para estrechar el intervalo

#### TPS (Conjunto Temporal de Pasada)
- Contiene coeficientes que se volvieron significativos en la pasada actual
- Se usa para transferir coeficientes de ICS a SCS
- Se limpia después de cada pasada
- Almacena el valor de reconstrucción inicial (center = T + T/2) para nuevos coeficientes significativos

#### Diagrama: Transiciones de Estado de Listas

```
Initial State:
  ICS = [all coefficients]
  SCS = []
  TPS = []

After Sorting Pass (T = 32):
  ICS = [insignificant coefficients]
  SCS = []
  TPS = [new significant coefficients]

After Moving TPS to SCS:
  ICS = [insignificant coefficients]
  SCS = [significant coefficients with initial center]
  TPS = []

After Refinement Pass:
  ICS = [insignificant coefficients]
  SCS = [significant coefficients with refined center]
  TPS = []

After Next Sorting Pass (T = 16):
  ICS = [remaining insignificant coefficients]
  SCS = [previous significant coefficients]
  TPS = [new significant coefficients at T=16]

And so on...
```

### Bucle Principal

El algoritmo WDR ejecuta el siguiente bucle hasta que se alcanza la precisión deseada:

1. **Pasa de Ordenamiento**: Encontrar y codificar nuevos coeficientes significativos
2. **Pasa de Refinamiento**: Refinar coeficientes significativos existentes
3. **Actualizar Umbral**: Reducir a la mitad el umbral T (T = T / 2)
4. **Mover a SCS**: Mover coeficientes de TPS a SCS
5. **Repetir**: Continuar con la siguiente pasada

El umbral T se reduce a la mitad en cada pasada, permitiendo que el algoritmo identifique y refine progresivamente coeficientes en escalas más finas.

## Codificación Aritmética Adaptativa

### Resumen

La etapa final de la compresión WDR usa codificación aritmética adaptativa para comprimir el flujo de símbolos generado por las pasas de ordenamiento y refinamiento. Esta implementación usa el algoritmo de Witten, Neal y Cleary (1987).

### Descripción del Algoritmo

La codificación aritmética representa un mensaje completo como una fracción única dentro del intervalo [0, 1). Cada símbolo estrecha este intervalo basado en su probabilidad. El aspecto "adaptativo" significa que el modelo de probabilidad actualiza sus estimaciones a medida que procesa cada símbolo.

### Características Clave

- **Aritmética Entera**: Usa matemáticas de enteros de precisión fija para representar intervalos, evitando operaciones de punto flotante
- **Operación Incremental**: Emite bits tan pronto como los bits más significativos del intervalo coinciden
- **Manejo de Underflow**: Incluye un mecanismo para prevenir pérdida de precisión cuando el intervalo se vuelve muy pequeño

### Nota de Implementación

La implementación en C++ en este proyecto está basada en el algoritmo matemático de Witten, Neal, & Cleary (1987). Se proporciona crédito completo y referencias en los comentarios del código y la documentación. La implementación mantiene equivalencia matemática con el algoritmo original mientras usa características modernas de C++17 para claridad y seguridad de tipos.

### Manejo de Underflow

Cuando el intervalo de codificación se vuelve muy pequeño pero cruza el punto medio, el algoritmo no puede emitir un bit inmediatamente. En su lugar:
1. Rastrea el número de "bits opuestos" para emitir después
2. Escala el intervalo para expandirlo
3. Emite los bits opuestos cuando finalmente se puede emitir un bit

Este mecanismo asegura que la precisión se mantenga incluso cuando el intervalo es muy pequeño.

## Formato de Archivo

### Estructura del Archivo .wdr

Un archivo `.wdr` consiste en:

1. **Encabezado**: Contiene metadatos sobre los datos comprimidos
   - Umbral inicial T
   - Número de pasadas
   - Número de coeficientes
   - Tamaño de datos comprimidos

2. **Datos Comprimidos**: El flujo de bits codificado aritméticamente
   - Contiene datos de pasa de ordenamiento codificados (índices y signos)
   - Contiene datos de pasa de refinamiento codificados (bits de refinamiento)

### Formato del Encabezado

El encabezado es una estructura binaria que contiene:
- `double initial_T`: Valor del umbral inicial
- `uint32_t num_passes`: Número de pasadas de planos de bits
- `uint64_t num_coeffs`: Número de coeficientes en el arreglo original
- `uint64_t data_size`: Tamaño de datos comprimidos en bytes

### Organización del Flujo de Bits

El flujo de bits está organizado como una secuencia de pasadas:

```
Pass 0:
  - Sorting pass: count, indices, signs
  - Refinement pass: (empty, no coefficients in SCS yet)

Pass 1:
  - Sorting pass: count, indices, signs
  - Refinement pass: refinement bits for Pass 0 coefficients

Pass 2:
  - Sorting pass: count, indices, signs
  - Refinement pass: refinement bits for all previous coefficients

... and so on
```

## Referencias y Atribución

### Algoritmo WDR

Esta implementación está basada en el algoritmo Wavelet Difference Reduction para compresión de imágenes embebida. El algoritmo combina transformadas wavelets discretas con codificación eficiente de índices y transmisión progresiva.

**Cita Completa:**
[Cita del artículo WDR - por completar con detalles del artículo]

### Codificación Aritmética Adaptativa

La implementación de codificación aritmética adaptativa está basada en el algoritmo de:

**Witten, I.H., Neal, R.M., & Cleary, J.G. (1987).** "Arithmetic coding for data compression." *Communications of the ACM*, 30(6), 520-540.

Este artículo presenta el algoritmo de codificación aritmética adaptativa usado en la etapa final de compresión de WDR. La implementación en C++ mantiene equivalencia matemática con el algoritmo original mientras usa características modernas de C++17.

### Declaración de Implementación

Esta implementación está basada en las fuentes teóricas citadas anteriormente. Se otorga crédito completo a los autores e investigadores originales que desarrollaron estos algoritmos. Los comentarios del código y la documentación incluyen atribución apropiada para asegurar integridad académica y otorgar crédito donde corresponde.

### Recursos Adicionales

- PyWavelets: Biblioteca Python para transformadas wavelets discretas
- NumPy: Biblioteca de computación numérica para Python
- Pillow: Biblioteca de imágenes Python para E/S de imágenes

---

[English](theory.md) | [Español](theory.es.md) | [Volver a README](../README.md)


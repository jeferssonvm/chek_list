

**Última edición:** 2025-08-29 23:56

# Fase 2: Limpieza y Transformación de Datos
✅ Orden Limpieza y Transformación de Datos (para Análisis y IA)
# 1. Revisión inicial

* Identificar dimensiones (filas, columnas, tipos de variables).
```
import pandas as pd

    # 1. Dimensiones del dataset
    print("📊 Dimensiones del dataset:")
    print(f"Filas: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}\n")

    # 2. Tipos de variables
    print("📝 Tipos de variables:")
    print(df.dtypes)

    # 3. Opcional: Resumen rápido con valores no nulos
    print("\n🔎 Resumen con valores no nulos:")
    print(df.info())


```

* Verificar nombres de columnas.
```
    print("📌 Nombres originales de columnas:")
    print(df.columns.tolist())
```


* Detectar duplicados en filas o IDs.

```
    duplicados_filas = df.duplicated().sum()
    print(f"📌 Número de filas duplicadas: {duplicados_filas}")
    
    # Elimina filas duplicadas manteniendo la primera ocurrencia
    df_sin_dups = df.drop_duplicates()

```
* Explorar valores únicos en variables categóricas.
```
# 1. Seleccionar solo columnas categóricas (tipo 'object' o 'category')
categoricas = df.select_dtypes(include=["object", "category"]).columns

 2. Explorar valores únicos en cada columna categórica
for col in categoricas:
    print(f"\n🔎 Columna: {col}")
    print(f"- Número de categorías únicas: {df[col].nunique()}")
    print(f"- Categorías: {df[col].unique()[:10]}")  # Muestra solo las 10 primeras
    
```

# 2. Tratamiento de valores faltantes

MCAR (Missing Completely At Random): faltan “al azar total”. Imputaciones simples suelen ser seguras.

MAR (Missing At Random): la ausencia depende de otras variables observadas (p. ej., los ingresos faltan más en gente joven). Métodos multivariados (KNN, MICE, modelos) funcionan mejor.

MNAR (Missing Not At Random): la ausencia depende del valor en sí (p. ej., personas con ingresos muy altos no reportan). Es el más difícil; requiere supuestos y, si es posible, recolectar más datos o usar modelado explícito de la ausencia (indicadores, modelos de selección).

* Calcular porcentaje de NaN/Null.
```
# 1. Calcular porcentaje de NaN/Null por columna
porcentaje_nulos = (df.isnull().sum() / len(df)) * 100

# 2. Crear un DataFrame ordenado con el resultado
reporte_nulos = porcentaje_nulos.reset_index()
reporte_nulos.columns = ["columna", "porcentaje_nulos"]
reporte_nulos = reporte_nulos.sort_values(by="porcentaje_nulos", ascending=False)

print("📊 Porcentaje de valores NaN/Null por columna:")
print(reporte_nulos)


```
visualizar valores faltantes 
```
    # df es tu DataFrame de Pandas con datos faltantes
    msno.bar(df)           # Gráfico de barras de missing
    msno.matrix(df)        # Matriz de valores faltantes
    msno.heatmap(df)       # Heatmap de correlaciones de missing
    msno.dendrogram(df)    # Dendrograma de patrones de valores faltantes

```
* Decidir: eliminar o imputar (media, mediana, moda, interpolación, KNN, modelos, etc.).

* Convertir faltantes implícitos → explícitos (NaN)

¿Qué es “implícito”? Celdas vacías, "NA", "N/A", "None", "?", "—", "Sin dato", 0 ó 9999 usados como “faltante”, etc.

```
Métodos de imputación

import numpy as np

placeholders = ["NA","N/A","na","n/a","None","none","-","—","?", "", "Sin dato", "sd"]
df = df.replace(placeholders, np.nan)

# Si hay sentinelas numéricos:
sentinelas = { "edad": [0, 999], "ingreso": [0, 9999999] }
for col, vals in sentinelas.items():
    df[col] = df[col].replace(vals, np.nan)

```
### Métodos de imputación





| Método                   | ¿Cuándo usar?                             | Ventajas                                         | Desventajas                                                                            | Notas/Implementación                                        |        |        |
| ------------------------ | ----------------------------------------- | ------------------------------------------------ | -------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ------ | ------ |
| Media/Mediana/Moda       | MCAR o MAR leve; baseline rápido          | Simple, rápido, reproducible                     | Distorsiona varianza; media sensible a outliers; moda puede crear sesgo en categóricas | `SimpleImputer`; mediana para sesgo/outliers                |        |        |
| ffill/bfill              | Series temporales con continuidad         | Respeta tendencia local                          | Propaga errores; malo con huecos largos                                                | Ordenar por tiempo; combinar `ffill().bfill()`              |        |        |
| Interpolación            | Señal “suave” (temperatura, sensores)     | Coherente con continuidad                        | Puede sobreajustar (spline); no sirve en categóricas                                   | \`interpolate(method="linear"                               | "time" | ...)\` |
| KNN Imputer              | MAR; relaciones multivariadas             | Captura relaciones entre features                | Costoso; sensible a escala y ruido                                                     | Estandariza antes; elegir `k` y distancia                   |        |        |
| Modelos (RF, GBM, etc.)  | MAR fuerte; señal predictiva alta         | Flexible, no lineal                              | Complejo; riesgo de leakage si no se cuida                                             | Entrenar **solo con train**; repetir por variable objetivo  |        |        |
| MICE (IterativeImputer)  | MAR multivariado; datasets medianos       | Usa todas las relaciones; múltiples imputaciones | Lento; tuning/convergencia                                                             | `IterativeImputer` (experimental); puede samplear posterior |        |        |
| Eliminar filas/columnas  | Alto % de faltantes y/o poca importancia  | Sencillo, evita supuestos                        | Pierdes información; posible sesgo                                                     | Usa umbral y criterio de valor analítico                    |        |        |
| Transformación + inversa | Variables muy sesgadas (p. ej., ingresos) | Mejora supuestos de métodos                      | Añade complejidad                                                                      | `log1p` → imputar → `expm1` para revertir                   |        |        |



#### Media / Mediana / Moda (rápidos)

* ¿Cómo funciona?

    *   Media: sustituye valores faltantes por la media de la columna.
       
    *   Mediana: sustituye por el valor central (más robusta a outliers).
       
    *   Moda: sustituye con la categoría más frecuente (para variables categóricas).

* 📈 Cuándo usar

    * Datos MCAR (faltan completamente al azar).

    *   Columnas con baja proporción de faltantes (< 20%).
       
    *   Variables numéricas con distribución no muy sesgada (media) o sesgada/outliers (mediana).

* ✅ Ventajas

    * Muy rápido y fácil de aplicar.
    
    * Baseline ideal antes de probar métodos más complejos.

* ❌ Desventajas

    *Reduce la varianza → subestima la dispersión.
    
    *Puede introducir sesgo si los datos no son MCAR.
    
    *No captura relaciones entre variables.

```
from sklearn.impute import SimpleImputer

# Numéricas → mediana
imp_mediana = SimpleImputer(strategy="median")
df["edad"] = imp_mediana.fit_transform(df[["edad"]])

# Categóricas → moda
imp_moda = SimpleImputer(strategy="most_frequent")
df["pais"] = imp_moda.fit_transform(df[["pais"]])
```

#### Llenado hacia atrás / hacia adelante (series de tiempo)

* ¿Cómo funciona?

ffill (forward fill): copia el último valor válido hacia adelante.

bfill (backward fill): copia el siguiente valor válido hacia atrás.

* 📈 Cuándo usar

Series de tiempo o datos ordenados.

Variables que cambian lentamente en el tiempo (ej. temperatura, saldos).

* ✅ Ventajas

Conserva la tendencia local.

Muy eficiente en datos secuenciales.

* ❌ Desventajas

Propaga errores si el último dato es incorrecto.

No funciona con huecos muy largos.

```
df = df.sort_values("fecha")
df["temperatura"] = df["temperatura"].ffill().bfill()
```
#### Interpolación

* 🔧 ¿Cómo funciona?

Estima los valores faltantes en base a valores vecinos.

Métodos comunes: linear, time, spline, polynomial.

 `method="linear"` (por defecto)
- Rellena los valores faltantes con una línea recta entre los puntos conocidos.  
- Ejemplo: si tienes `2` y `6` con un `NaN` en medio → lo rellena con `4`.  
- Es el más usado y rápido.  

---

 `method="time"`
- Se usa si el índice es de tipo fecha (`DatetimeIndex`).  
- Hace interpolación lineal tomando en cuenta el tiempo transcurrido.  
- Útil en series temporales.  

---

 `method="index"`
- Similar a `linear`, pero usa los valores del índice numérico como referencia para calcular la interpolación.  

---

 `method="nearest"`
- Rellena usando el valor del punto más cercano.  
- No genera valores intermedios, solo copia.  

---

`method="spline", order=n`
- Usa polinomios (curvas suaves) para estimar valores.  
- `order=2` → cuadrática.  
- `order=3` → cúbica.  
- Más preciso para datos con curvatura, pero más lento.  

---
 `method="polynomial", order=n`
- Similar a `spline`, pero ajusta un polinomio global a los datos.  
- Puede sobreajustar si los datos son muy ruidosos.  

---

 `method="pad"` o `method="ffill"` (forward fill)
- Copia el último valor válido hacia adelante.  
- Útil en datos categóricos o de sensores.  

---

`method="bfill"` (backward fill)
- Copia el siguiente valor válido hacia atrás.  
- También se usa en datos categóricos.  

---


* 📈 Cuándo usar

Series de tiempo o procesos continuos/suaves.

Variables físicas (ej. sensores, clima).

* ✅ Ventajas

Más realista que ffill/bfill.

Admite interpolaciones avanzadas (polinomiales, spline).

* ❌ Desventajas

Puede sobreajustar si los datos no son lineales.

No sirve para categóricas.

#### KNN Imputer

🔧 ¿Cómo funciona?

Para cada valor faltante, busca sus k vecinos más cercanos (por distancia en otras variables) y asigna:

La media (numéricas).

La moda (categóricas).

📈 Cuándo usar

Datos MAR (faltan en función de otras variables).

Variables con correlación fuerte con otras.

✅ Ventajas

Captura relaciones multivariadas.

Puede ser más preciso que métodos univariados.

❌ Desventajas

Muy costoso en datasets grandes.

Sensible al escalado (normalizar antes).

Si hay mucho ruido, puede imputar mal.

```
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5, weights="distance")
df_imputed = imputer.fit_transform(df)
```
####  Imputación basada en Modelos predictivos
🔧 ¿Cómo funciona?

Entrena un modelo (ej. regresión lineal, Random Forest, XGBoost) para predecir la variable con faltantes en función de las demás.

📈 Cuándo usar

Variables con mucha relación con otras.

Cuando los datos faltan bajo MAR.

✅ Ventajas

Captura relaciones complejas.

Puede mejorar la calidad de la imputación.

❌ Desventajas

Más costoso en tiempo y recursos.

Riesgo de data leakage si no se divide correctamente.

Puede introducir sobreajuste.

```
from sklearn.ensemble import RandomForestRegressor

train = df[df["ingreso"].notna()]
test  = df[df["ingreso"].isna()]

X_train = train.drop("ingreso", axis=1)
y_train = train["ingreso"]

model = RandomForestRegressor()
model.fit(X_train, y_train)

df.loc[df["ingreso"].isna(), "ingreso"] = model.predict(test.drop("ingreso", axis=1))
```

####   MICE (Multiple Imputation by Chained Equations)
🔧 ¿Cómo funciona?

Imputa iterativamente cada variable faltante en función de las demás.

Repite varias rondas → convergencia.

Permite generar múltiples datasets imputados para tener en cuenta la incertidumbre.

📈 Cuándo usar

Datos con varios campos faltantes.

MAR y correlaciones complejas entre variables.

✅ Ventajas

Considera relaciones multivariadas.

Puede mejorar modelos finales.

Permite estimar incertidumbre (estadística inferencial).

❌ Desventajas

Costoso y lento en grandes datasets.

Implementación más compleja.

```
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imp = IterativeImputer(max_iter=10, random_state=42)
df_imputed = imp.fit_transform(df)

``` 

####  Transformación inversa de datos
🔧 ¿Cómo funciona?

Se transforma la variable (ej. logaritmo, Box-Cox) para estabilizar la distribución.

Se imputa en el espacio transformado.

Se aplica la inversa para recuperar la escala original.

📈 Cuándo usar

Variables muy sesgadas (ej. ingresos, precios, ventas).

✅ Ventajas

Hace que imputaciones simples (media/mediana) sean más realistas.

Reduce sesgos por distribuciones muy largas.

❌ Desventajas

Requiere aplicar correctamente la inversa.

No todas las transformaciones son adecuadas.

```
import numpy as np

df["ingreso_log"] = np.log1p(df["ingreso"])
df["ingreso_log"] = df["ingreso_log"].fillna(df["ingreso_log"].median())
df["ingreso"] = np.expm1(df["ingreso_log"])
```





# 3. Estandarización de tipos de datos

* Convertir fechas a datetime.
```
pd.to_datetime(["2024-05-10", "10/05/2024"], dayfirst=True)
gumentos más útiles:

errors: qué hacer si no puede convertir.

"raise" → error.

"coerce" → pone NaT (Not a Time).

"ignore" → deja el valor como estaba.

dayfirst=True: interpreta formato DD/MM/YYYY.

format: le dices el formato exacto ("%d-%m-%Y", "%Y/%m/%d %H:%M").

```
| Conjunto       | % típico | Uso principal                                 | ¿Se entrena con él? | ¿Se ajusta con él? |
| -------------- | -------- | --------------------------------------------- | ------------------- | ------------------ |
| **Train**      | 60–80%   | Aprender patrones, entrenar modelo            | ✅ Sí                | ❌ No               |
| **Validation** | 10–20%   | Ajustar hiperparámetros, prevenir overfitting | ❌ No                | ✅ Sí               |
| **Test**       | 10–20%   | Evaluación final del modelo                   | ❌ No                | ❌ No               |


* Variables categóricas → tipo category.
    Regla general
    Primero se divide el dataset en entrenamiento y prueba.
    Después se ajusta (fit) la codificación solo con el conjunto de entrenamiento.
    Luego se aplica la misma transformación al conjunto de prueba (transform).

* Variables numéricas → asegurar int o float.
```
    df["edad"] = pd.to_numeric(df["edad"], errors="coerce").astype("Int64")

```

* Normalizar texto (minúsculas, sin espacios extra, sin caracteres raros).
 ```
    str(x).lower().strip()
 ```
* Outliers evidentes (errores de captura) → limpiar aquí.

# 4. División en conjuntos (opcional si se entrena/valida en el mismo dataset)

Train / Test (y validación si aplica).
``` 
# Dividir en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# opcional 
# Luego dividir el temporal en Validation/Test (50%-50%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

```


A partir de aquí, todas las transformaciones que dependen de cálculos se hacen con fit en train, transform en test.

# 5. Tratamiento de valores atípicos (outliers genuinos)

Métodos: IQR, Z-score, boxplots.

Opciones: winsorización, escalamiento robusto, mantenerlos si son válidos.

# 6. Transformaciones dependientes de estadísticos

Normalización / Escalado (MinMax, StandardScaler, RobustScaler).

Codificación de categóricas (OneHot, Label, Target Encoding → con fit en train).

Reducción de dimensionalidad (PCA, UMAP, etc.).

# 7. Feature Engineering (creación de nuevas variables)

Derivadas de tiempo (año, mes, día, hora).

Interacciones (ratios, logaritmos, diferencias).

Texto: TF-IDF, embeddings.

Imágenes: normalización de píxeles, extracción de features.

# 8. Balanceo de clases (si es clasificación)

Oversampling (SMOTE), undersampling o pesos de clase.

Siempre solo en el conjunto de entrenamiento.

# 9. Validación de consistencia

Revisar correlaciones entre variables.

Revisar coherencia temporal (fechas de inicio < fechas de fin).

Verificar que train/test tienen distribuciones similares (no hay “drift”).

# 10. Guardar dataset limpio

data/raw → dataset original.

data/processed → datasets limpios (train y test transformados).

Documentar decisiones (log de transformaciones).
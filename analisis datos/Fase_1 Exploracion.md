# exploracion 
## Estructura de carpetas( recomendado )
📁 Proyecto_Analisis_Datos/
│
├── 📂 0_Documentacion/
│   ├── README.md                 # Descripción general del proyecto
│   ├── planteamiento_problema.md # Objetivos, hipótesis, preguntas de análisis
│   ├── diccionario_datos.md      # Explicación de cada variable
│   ├── referencias.md            # Bibliografía, enlaces, papers
│   └── requisitos.md             # Dependencias, librerías, versiones
│
├── 📂 1_Datos/
│   ├── 📂 brutos/                 # Datos originales (sin modificar)
│   ├── 📂 procesados/             # Datos limpios y transformados
│   ├── 📂 externos/               # Datos de APIs, scraping, encuestas, etc.
│   └── 📂 diccionarios/           # Diccionarios de variables y catálogos
│
├── 📂 2_Notebooks/                # Exploración, limpieza y pruebas rápidas
│   ├── 01_exploracion.ipynb
│   ├── 02_limpieza.ipynb
│   └── 03_modelado.ipynb
│
├── 📂 3_Scripts/                  # Código reutilizable en Python/R/etc.
│   ├── limpieza.py
│   ├── visualizacion.py
│   ├── feature_engineering.py
│   └── modelos.py
│
├── 📂 4_Resultados/
│   ├── 📂 graficos/               # Visualizaciones (png, jpg, svg)
│   ├── 📂 tablas/                 # Tablas exportadas (csv, xlsx)
│   └── 📂 reportes/               # Reportes parciales en PDF/MD
│
├── 📂 5_Modelos/ (opcional IA/ML) 
│   ├── 📂 entrenados/             # Modelos entrenados (.pkl, .h5, .onnx)
│   ├── 📂 evaluaciones/           # Resultados de métricas, validaciones
│   └── 📂 tuning/                 # Configuración de hiperparámetros
│
├── 📂 6_Implementacion/ (opcional IA/Producción)
│   ├── api_servicios/             # Scripts de despliegue en API (FastAPI, Flask)
│   ├── pipelines/                 # ETL, automatizaciones
│   └── dashboards/                # Paneles web interactivos (Streamlit, Dash)
│
├── 📂 7_PowerBI_Excel/ (opcional para proyectos BI)
│   ├── reportes_powerbi.pbix
│   └── reportes_excel.xlsx
│
├── 📂 8_Tests/ (opcional avanzado)
│   ├── unit_tests.py              # Pruebas unitarias
│   └── integracion_tests.py       # Pruebas de integración
│
└── 📂 9_Entrega_Final/
    ├── informe_final.md           # Documento final del análisis
    ├── presentacion.pptx          # Presentación ejecutiva
    └── dashboard_link.txt         # Enlace a PowerBI/Streamlit/otro


## 1. Definir la fuente y permisos
- [ ] Identificar tipo de fuente: archivo (CSV/Excel/JSON/Parquet), API, BD, web scraping.  
- [ ] Verificar autorizaciones y credenciales (manejo seguro: `.env` o variables de entorno).  
- [ ] Establecer frecuencia de actualización y corte temporal.  

## 2. Trazabilidad mínima
- [ ] Guardar copia en `data/raw/` (datos tal cual llegan).  
- [ ] Calcular hash (sha256) del archivo fuente.  
- [ ] Registrar fecha/hora de ingesta.  
- [ ] Documentar responsable y parámetros de carga.  

## 3. Estándar de almacenamiento
- [ ] Convención de nombres: `dataset_YYYYMMDD_origen.ext`.  
- [ ] Subcarpetas: `raw/`, `staging/`, `logs/`.  

## 4. Detección inicial de formato
- [ ] Detectar encoding (UTF-8, latin-1).  
- [ ] Identificar delimitador (`,` `;` `\t`).  
- [ ] Confirmar formato de fechas, decimales y separadores de miles.  

## 5. Exploración preliminar (sin limpiar)
- [ ] Número de filas y columnas cargadas.  
- [ ] Nombres de columnas → estandarizar a `snake_case`.  
- [ ] Tipos inferidos (numérico, texto, fecha, categórico).  
- [ ] Conteo de nulos por columna.  
- [ ] Conteo de duplicados (fila completa y por PK).  
- [ ] Columnas constantes o con un solo valor.  

## 6. Diccionario preliminar de datos
- [ ] Tipo de dato por columna.  
- [ ] % de valores nulos.  
- [ ] Número de valores únicos.  
- [ ] Valores mínimos y máximos (numéricos/fechas).  
- [ ] Ejemplos de valores.  

## 7. Privacidad y compliance
- [ ] Identificar posibles datos sensibles (PII: cédula, correo, teléfono).  
- [ ] Enmascarar o excluir si no son necesarios para el análisis.  

## 8. Salidas de la fase
- [ ] Dataset en formato **Parquet** (colunar, comprimido).  
- [ ] Diccionario preliminar en CSV/Markdown.  
- [ ] Log JSON con metadatos de carga (origen, hash, encoding, filas, columnas, incidencias).  
# Ames Housing: predicción del precio de venta

Pipeline de Machine Learning supervisado para predecir `SalePrice` minimizando el
RMSE sobre el logaritmo del precio (RMSE_log). Incluye limpieza semántica,
ingeniería de características, validación anidada, regularización, árboles de
decisión, ensamble y explicabilidad con SHAP.

## Resumen de negocio

Una estimación de precio de venta sirve a compradores, vendedores y agentes para
fijar precios de referencia. El error se mide en escala logarítmica, así que una
diferencia porcentual pesa igual en una casa barata que en una cara. Con el
ensamble final, el RMSE_log medio en validación externa es de
**0.1107** (desviación estándar entre folds: 0.0155), lo que equivale aproximadamente
a un error típico del 11 % en el precio.

## Arquitectura del pipeline

```
src/
├── data_loader.py         Lectura de CSV y tratamiento del token "NA"
├── null_audit.py          Auditoría de nulos por columna
├── cleaning.py            Ausencia semántica y eliminación de outliers (solo train)
├── feature_engineering.py Mapeo ordinal y variables de negocio
├── preprocessing.py       ColumnTransformer: imputación, RobustScaler y One-Hot
├── evaluation.py          Métrica RMSE_log y esquemas K-Fold estratificados
├── models.py              Especificaciones y espacios de búsqueda de cada modelo
├── train.py               Validación anidada, pesos NNLS y predicción final
├── predict.py             Reversión exp(z)-1 y validación de la entrega
└── interpretation.py      Residuos, correlaciones, importancia y SHAP
```

Flujo: `raw → cleaning → feature engineering → Pipeline (dentro de cada fold) → modelos → ensamble → submission`.

## Decisiones de preprocesamiento

**1. Nulos semánticos vs. reales.** El diccionario usa `NA` tanto para "no existe el
atributo" como para dato faltante, así que la decisión se toma columna por columna.
Las categorías `None` (sin piscina, sin garaje, sin sótano) se conservan como
valor propio. Las columnas de garaje y sótano solo se rellenan cuando todo el
grupo está ausente, para no ocultar datos faltantes parciales.

**2. Outliers operacionales.** Se eliminaron 2 casas (Id 524 y 1299) con
`GrLivArea > 4000` pies² y `SalePrice < 300 000`. Son áreas muy grandes vendidas a
precios de casas pequeñas, un patrón que suele indicar error de registro o venta
no representativa. La casa grande del test se conserva, porque no se puede dejar
sin predecir una propiedad de la entrega.


**5. Transformación del objetivo.** Se modela `log(1 + SalePrice)`, porque el
precio tiene asimetría positiva (skewness 1.88) y la transformación la reduce a
0.12. Las predicciones se revierten con `exp(z) - 1`.

**6. Prevención de fuga de datos.** Las medianas, modas, escalas y categorías
One-Hot se ajustan dentro de cada fold del `Pipeline`. El test nunca participa en
ajuste ni en selección de hiperparámetros.

## Validación

- **Validación externa:** K-Fold estratificado por deciles del objetivo, K=5, semilla 42.
- **Búsqueda de hiperparámetros:** `RandomizedSearchCV` con K-Fold interno de 3 folds,
  dentro de cada fold externo (validación anidada). El error reportado no depende de
  los hiperparámetros elegidos con los mismos datos.
- **Pesos del ensamble:** se ajustan con NNLS sobre predicciones fuera de muestra, y
  su error se mide también con validación cruzada.

## Comparativa de modelos (RMSE_log, validación externa)

| Modelo | RMSE_log medio | Desv. estándar |
|---|---|---|
| **Ensamble NNLS** | **0.11065** | 0.01548 |
| Ensamble promedio | 0.11108 | 0.01545 |
| Lasso | 0.11238 | 0.01387 |
| Ridge | 0.11370 | 0.01313 |
| XGBoost | 0.11762 | 0.01766 |
| Regresión lineal | 0.12724 | 0.01053 |
| Random Forest | 0.13155 | 0.01720 |

Pesos del ensamble NNLS: Lasso 0.51, XGBoost 0.35, Ridge 0.14; los demás modelos
recibieron peso 0.

**Lectura:** la regularización L1 (Lasso) y L2 (Ridge) mejora sobre la regresión
lineal sin penalizar. XGBoost aporta información distinta de los modelos lineales,
y el ensamble la combina mejor que cualquiera por separado. Random Forest queda por
detrás de XGBoost y el NNLS le asigna peso cero. La diferencia entre el mejor
ensamble y Lasso es pequeña (0.0017) frente a la desviación entre folds, así que no
conviene sobreinterpretarla.

## Variables clave e impacto inmobiliario

Medido con SHAP medio absoluto sobre XGBoost ajustado en todo el train:

| Variable | SHAP medio |
|---|---|
| TotalSF | 0.1013 |
| OverallQual | 0.0892 |
| TotalBaths | 0.0295 |
| KitchenQual | 0.0222 |
| OverallCond | 0.0208 |

Las dos variables dominantes son el área total y la calidad global de la vivienda,
lo que coincide con la intuición de mercado: más metros construidos y mejor
material y acabado elevan el precio. La importancia nativa de XGBoost da más peso
a `ExterQual`, pero SHAP es más fiable porque mide el aporte real a la predicción y
no solo cuántas veces se usa una variable en los árboles.

Figuras en `reports/figures/`: `residuals_vs_predicted.png`,
`correlation_heatmap.png` y `top15_features.png`.

## Limitaciones

- **Casas extremas:** la predicción más alta del test es de 1.23 millones de dólares,
  por encima del máximo de train (755 000). Corresponde a una casa muy grande y de
  máxima calidad, pero el modelo tiene pocas observaciones en ese rango.
- **Outliers eliminados:** solo se quitaron 2 casas. Otros puntos atípicos se
  conservaron porque no hay evidencia operacional clara.
- **Interpretación:** SHAP explica el modelo, no establece causalidad.
- **Sesgo de retransformación:** `exp(z) - 1` estima aproximadamente la mediana
  condicional. Si el objetivo fuera el precio en dólares, se requeriría una corrección.

## Reproducción

```powershell
cd 'C:\Users\emanu\OneDrive\Documentos\Proyectos\ames-housing-prediction'
.\.venv\Scripts\Activate.ps1
python src\data_loader.py
python src\null_audit.py
python src\cleaning.py
python src\feature_engineering.py
python src\train.py
python src\predict.py
python src\interpretation.py
```

El entrenamiento completo (validación anidada con 5 modelos) tarda varios minutos.

**3. Codificación ordinal explícita.** Calidades (`Ex=5, Gd=4, TA=3, Fa=2, Po=1`),
exposición del sótano, acabado de sótano y de garaje se mapean a enteros con
`None=0`. Si aparece una categoría no contemplada, se lanza un error en lugar de
convertirla en nulo.

**4. Variables de negocio.** `TotalSF` (sótano + primer y segundo piso),
`Age` (año de venta menos año de construcción), `IsRemodeled` (1 si el año de
remodelación difiere del de construcción) y `TotalBaths` (baños completos + 0.5 ×
medios baños, sumando los del sótano y los de los pisos superiores).

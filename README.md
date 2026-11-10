# InduManufactura S.A.S. — Optimización de producción y distribución

Aplicación en Streamlit que resuelve, con PuLP (solucionador CBC), el modelo de programación lineal entera
que minimiza el costo total de producción y transporte de los productos P1, P2 y P3, fabricados en las
plantas Norte y Sur y enviados a los centros de Medellín, Rionegro y Bello.

**Aplicación publicada:** [ENLACE DE LA APLICACIÓN]

## Contenido

| Archivo | Descripción |
|---|---|
| `streamlit_app.py` | Modelo de optimización (datos, escenarios, variables, función objetivo, restricciones, solución y validación) e interfaz Streamlit |
| `requirements.txt` | Dependencias |

## Funcionalidades

- Edición de horas disponibles, tiempos de fabricación, costos de producción, demanda y costos de transporte.
- Optimización con estado del solucionador (óptimo, infactible, etc.); no se muestran soluciones inválidas.
- Producción y distribución óptimas, costo total desglosado y utilización de capacidad.
- Validación automática de cada restricción y de la función objetivo.
- Comparación del escenario base con los tres escenarios obligatorios y uno adicional.

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

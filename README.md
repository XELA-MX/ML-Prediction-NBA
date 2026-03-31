# NBA Game Predictor

Sistema de predicción de partidos NBA que combina un modelo de machine learning con un motor de scoring ponderado multifactor. Consume datos en tiempo real desde la NBA API, ESPN y opcionalmente Vegas odds.

---

## Características

- **Modelo ML**: HistGradientBoostingClassifier entrenado sobre 4 temporadas (2021-22 a 2024-25)
- **Scoring multifactor**: 13 factores ponderados como fallback o modo independiente
- **Datos en tiempo real**: forma reciente, stats avanzadas, lesiones (ESPN), fatiga y travel factor
- **Interfaz CLI** (`main.py`) e **interfaz gráfica** (`GUI.py` con Tkinter)
- **Exportación a PDF** con estadísticas por partido (requiere `reportlab`)
- **Integración con Vegas odds** vía The Odds API (opcional)

---

## Estructura

```
NBA-Predictor/
├── main.py                  # CLI interactivo
├── GUI.py                   # Interfaz gráfica con Tkinter
├── data/
│   ├── dataset.csv          # Dataset histórico (generado)
│   └── model.pkl            # Modelo entrenado (generado)
└── Servicios/
    ├── api_utils.py         # Decorator de retry con backoff exponencial
    ├── GetSchedule.py       # Próximo partido entre dos equipos (ScoreboardV2)
    ├── GetTier1.py          # Forma reciente, H2H, splits local/visitante
    ├── GetTier2.py          # Stats avanzadas, shooting, defensa, clutch
    ├── GetTier3.py          # Lesiones (ESPN), fatiga, minutos por jugador
    ├── Predict.py           # Inferencia con model.pkl + ajuste por lesiones
    ├── TrainModel.py        # Pipeline de entrenamiento y evaluación
    ├── BuildDataset.py      # Construcción del dataset histórico
    ├── team_locations.py    # Coordenadas y zonas horarias de los 30 equipos
    └── vegas_odds.py        # Integración con The Odds API
```

---

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Dependencias principales

| Paquete | Uso |
|---|---|
| `nba_api` | Estadísticas oficiales NBA |
| `pandas` | Manipulación de datos |
| `scikit-learn` | Modelo ML y evaluación |
| `joblib` | Serialización del modelo |
| `requests` | ESPN API y Vegas odds |
| `reportlab` | Exportación a PDF (opcional) |

---

## Uso rápido

### CLI

```bash
python main.py
```

Pedirá el nombre de los dos equipos y mostrará la predicción con el desglose de factores.

### GUI

```bash
python GUI.py
```

Carga los partidos del día y ejecuta predicciones en paralelo con threading.

---

## Entrenar el modelo

El modelo se entrena desde cero con datos históricos de la NBA API.

```bash
# 1. Construir el dataset (puede tardar varios minutos por los rate limits)
python Servicios/BuildDataset.py

# 2. Entrenar y guardar el modelo
python Servicios/TrainModel.py
```

El dataset incluye 4 temporadas con features rolling de 10 y 5 partidos, stats de temporada en modo expanding (sin data leakage), schedule strength, travel factor y head-to-head ratio.

---

## Variables de entorno

Crea un archivo `.env` en la raíz del proyecto:

```
ODDS_API_KEY=tu_api_key_aqui
```

La clave es opcional. Sin ella el sistema funciona normalmente, sin mostrar líneas de Vegas. Se puede obtener en [the-odds-api.com](https://the-odds-api.com) (tier gratuito: 500 peticiones/mes).

---

## Arquitectura del pipeline

```
nba_api / ESPN / Odds API
        │
        ├── GetTier1  →  forma reciente, H2H, splits, travel
        ├── GetTier2  →  stats avanzadas, shooting, defensa
        └── GetTier3  →  lesiones, fatiga, minutos por jugador
                │
                ├── Predict.py     → inferencia ML + ajuste lesiones
                └── calcular_prediccion()  → scoring ponderado (fallback)
```

### Features del modelo (11)

| Feature | Tipo |
|---|---|
| `diff_win_rate_10j` | Diferencial tasa de victorias (10 partidos) |
| `diff_pt_diff_10j` | Diferencial de puntos |
| `diff_pts_allowed_10j` | Diferencial de puntos permitidos |
| `diff_season_def_rtg` | Diferencial Defensive Rating de temporada |
| `diff_schedule_strength` | Diferencial de dificultad del calendario |
| `away_rest_days` | Días de descanso del visitante |
| `away_fatigue_7d` | Partidos del visitante en últimos 7 días |
| `home_b2b` | ¿El local juega back-to-back? |
| `home_rest_days` | Días de descanso del local |
| `travel_factor` | Factor de fatiga por viaje (distancia + zona horaria) |
| `h2h_ratio` | Ratio de victorias en enfrentamientos directos |

---

## Notas

- Los rate limits de la NBA API (stats.nba.com) requieren delays de ~0.6s entre llamadas. Una predicción completa tarda aproximadamente 60 segundos.
- El modelo se entrena con un split temporal: las temporadas anteriores a 2024-25 como train y 2024-25 como test.
- El ajuste por lesiones se aplica en espacio log-odds sobre la probabilidad del modelo ML para mantener el rango [0, 1].

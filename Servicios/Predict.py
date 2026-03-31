"""
NBA Predictor — Inferencia con Modelo ML
Carga model.pkl y predice usando datos en tiempo real de los tres tiers.
"""

import os
import logging
import joblib
import numpy as np
from vegas_odds import get_game_odds, odds_available

log = logging.getLogger(__name__)

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_PATH = os.path.join(DATA_DIR, "model.pkl")

_artifact = None  # cache en memoria


def _load():
    global _artifact
    if _artifact is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Modelo no encontrado en {MODEL_PATH}\n"
                "Ejecuta primero:\n"
                "  python Servicios/BuildDataset.py\n"
                "  python Servicios/TrainModel.py"
            )
        _artifact = joblib.load(MODEL_PATH)
    return _artifact


def _safe_get(d: dict, key: str, default=0.0, label: str = ""):
    """Safely extract a value from a dict, logging a warning on fallback."""
    val = d.get(key, default)
    if val is None:
        log.warning("Missing key '%s' in %s — using default %s", key, label, default)
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        log.warning("Non-numeric value for '%s' in %s: %r — using default %s",
                    key, label, val, default)
        return float(default)


def _build_vector(t1, t2, t3, equipo_a, equipo_b, equipo_a_es_local) -> list:
    """
    Convierte los datos de los tres tiers al vector de features
    en el mismo orden que feature_cols de TrainModel.build_features().

    Orden: diff_* (5) → context (5) → h2h_ratio
    """
    home = equipo_a if equipo_a_es_local else equipo_b
    away = equipo_b if equipo_a_es_local else equipo_a

    fr_h = t1.get("forma_reciente", {}).get(home, {})
    fr_a = t1.get("forma_reciente", {}).get(away, {})
    df_h = t2.get("defensa_reciente_10j", {}).get(home, {})
    df_a = t2.get("defensa_reciente_10j", {}).get(away, {})
    av_h = t2.get("advanced_stats_temporada", {}).get(home, {})
    av_a = t2.get("advanced_stats_temporada", {}).get(away, {})
    ca_a = t3.get("carga_reciente_7_dias", {}).get(away, {})
    
    # Schedule strength (from Tier 1)
    ss_h = t1.get("schedule_strength", {}).get(home, {})
    ss_a = t1.get("schedule_strength", {}).get(away, {})
    
    # Travel info (from Tier 1)
    travel = t1.get("travel_info", {})

    fat_a = _safe_get(ca_a, "partidos_ultimos_7_dias", 2, f"carga {away}")

    h2h = t1.get("head_to_head", {})
    if "wins_equipo_a" in h2h:
        total = h2h["wins_equipo_a"] + h2h["wins_equipo_b"]
        if total > 0:
            wins_home = h2h["wins_equipo_a"] if equipo_a_es_local else h2h["wins_equipo_b"]
            h2h_ratio = wins_home / total
        else:
            h2h_ratio = 0.5
    else:
        h2h_ratio = 0.5

    # Differential features (home - away) matching TrainModel.DIFF_COLS:
    # win_rate_10j, pt_diff_10j, pts_allowed_10j, season_def_rtg, schedule_strength
    diff = [
        _safe_get(fr_h, "win_rate", 0.5, home)       - _safe_get(fr_a, "win_rate", 0.5, away),
        _safe_get(fr_h, "point_differential", 0, home) - _safe_get(fr_a, "point_differential", 0, away),
        _safe_get(df_h, "pts_permitidos_prom", 110, home) - _safe_get(df_a, "pts_permitidos_prom", 110, away),
        _safe_get(av_h, "defensive_rating", 112, home) - _safe_get(av_a, "defensive_rating", 112, away),
        _safe_get(ss_h, "schedule_strength", 0.5, home) - _safe_get(ss_a, "schedule_strength", 0.5, away),
    ]

    # Context features (expanded)
    context = [
        _safe_get(fr_a, "dias_desde_ultimo_partido", 2, away),  # away_rest_days
        fat_a,                                                   # away_fatigue_7d
        _safe_get(fr_h, "back_to_back", 0, home),               # home_b2b
        _safe_get(fr_h, "dias_desde_ultimo_partido", 2, home),  # home_rest_days
        _safe_get(travel, "travel_factor", 0.0, "travel"),      # travel_factor
    ]

    vec = diff + context + [h2h_ratio]

    # Sanity check: no NaN/Inf
    for i, v in enumerate(vec):
        if not np.isfinite(v):
            log.warning("Non-finite value at feature index %d: %r — replacing with 0", i, v)
            vec[i] = 0.0

    return vec


def predecir(t1, t2, t3, equipo_a, equipo_b, equipo_a_es_local) -> dict:
    """
    Predice el ganador usando el modelo ML entrenado.

    Retorna el mismo formato que el scoring ponderado:
        {
          "ganador":      str,
          "probabilidad": {equipo_a: float, equipo_b: float},
          "factores":     dict,
          "modelo":       str,
          "modelo_auc":   float,
        }
    """
    art   = _load()
    model = art["model"]

    home = equipo_a if equipo_a_es_local else equipo_b
    away = equipo_b if equipo_a_es_local else equipo_a

    vec = _build_vector(t1, t2, t3, equipo_a, equipo_b, equipo_a_es_local)
    X   = np.array([vec])

    prob_home_raw = float(model.predict_proba(X)[0][1])   # P(local gana)

    # Ajuste post-hoc por lesiones en log-odds para mantener probabilidades en [0,1]
    INJURY_WEIGHT = 1.8

    inj_home = t3.get("impacto_lesiones", {}).get(home, 0.0)
    inj_away = t3.get("impacto_lesiones", {}).get(away, 0.0)
    inj_diff = inj_home - inj_away  # positivo = local más lesionado

    if abs(inj_diff) > 0.01:
        # Convertir probabilidad a log-odds, ajustar, reconvertir
        eps = 1e-6
        p = np.clip(prob_home_raw, eps, 1 - eps)
        log_odds = np.log(p / (1 - p))
        log_odds_adj = log_odds - INJURY_WEIGHT * inj_diff
        prob_home = float(1.0 / (1.0 + np.exp(-log_odds_adj)))
        log.info("Injury adjustment: home=%.1f%% away=%.1f%% diff=%.3f → "
                 "prob_home %.3f → %.3f",
                 inj_home * 100, inj_away * 100, inj_diff,
                 prob_home_raw, prob_home)
    else:
        prob_home = prob_home_raw

    prob_away = 1.0 - prob_home

    pct_a = round((prob_home if equipo_a_es_local else prob_away) * 100, 1)
    pct_b = round((prob_away if equipo_a_es_local else prob_home) * 100, 1)
    ganador = equipo_a if pct_a >= pct_b else equipo_b

    UMBRAL_MINIMO = 57.0

    confianza = max(pct_a, pct_b)
    if confianza >= 75:
        nivel_apuesta = "FUERTE"
        apostar = True
    elif confianza >= 65:
        nivel_apuesta = "MODERADA"
        apostar = True
    elif confianza >= UMBRAL_MINIMO:
        nivel_apuesta = "BAJA"
        apostar = True
    else:
        nivel_apuesta = "NO APOSTAR"
        apostar = False

    # Factores para mostrar en pantalla (mismo formato que antes)
    sp_a = t2["splits_local_visitante_avanzado"][equipo_a]
    sp_b = t2["splits_local_visitante_avanzado"][equipo_b]

    # H2H para mostrar en pantalla
    h2h_data = t1["head_to_head"]
    if "wins_equipo_a" in h2h_data:
        total_h2h = h2h_data["wins_equipo_a"] + h2h_data["wins_equipo_b"]
        if total_h2h > 0:
            h2h_display = {"A": h2h_data["wins_equipo_a"], "B": h2h_data["wins_equipo_b"]}
        else:
            h2h_display = "sin enfrentamientos aún"
    else:
        h2h_display = "sin datos"

    factores = {
        "local": home,
        "net_rating": {
            "A": t2["advanced_stats_temporada"][equipo_a]["net_rating"],
            "B": t2["advanced_stats_temporada"][equipo_b]["net_rating"],
        },
        "win_rate_10j": {
            "A": round(t1["forma_reciente"][equipo_a]["win_rate"] * 100, 1),
            "B": round(t1["forma_reciente"][equipo_b]["win_rate"] * 100, 1),
        },
        "win_rate_5j": {
            "A": round(t1["forma_reciente"][equipo_a]["win_rate_5j"] * 100, 1),
            "B": round(t1["forma_reciente"][equipo_b]["win_rate_5j"] * 100, 1),
        },
        "point_diff": {
            "A": t1["forma_reciente"][equipo_a]["point_differential"],
            "B": t1["forma_reciente"][equipo_b]["point_differential"],
        },
        "fg_pct_10j": {
            "A": round(t2["shooting_reciente_10j"][equipo_a]["fg_pct"] * 100, 1),
            "B": round(t2["shooting_reciente_10j"][equipo_b]["fg_pct"] * 100, 1),
        },
        "pts_permitidos": {
            "A": t2["defensa_reciente_10j"][equipo_a]["pts_permitidos_prom"],
            "B": t2["defensa_reciente_10j"][equipo_b]["pts_permitidos_prom"],
        },
        "h2h": h2h_display,
        "nr_contexto": {
            "A": sp_a.get("como_local" if equipo_a_es_local else "como_visitante", {}).get("net_rating", 0),
            "B": sp_b.get("como_visitante" if equipo_a_es_local else "como_local", {}).get("net_rating", 0),
            "rol_A": "local" if equipo_a_es_local else "visitante",
        },
        "net_tov": {
            "A": round(t2["defensa_reciente_10j"][equipo_a]["tov_forzados_aprox"]
                       - t2["shooting_reciente_10j"][equipo_a]["tov_prom"], 1),
            "B": round(t2["defensa_reciente_10j"][equipo_b]["tov_forzados_aprox"]
                       - t2["shooting_reciente_10j"][equipo_b]["tov_prom"], 1),
        },
        "pace": {
            "A": t2["advanced_stats_temporada"][equipo_a]["pace"],
            "B": t2["advanced_stats_temporada"][equipo_b]["pace"],
            "ventaja_local": round(abs(
                t2["advanced_stats_temporada"][equipo_a]["pace"]
                - t2["advanced_stats_temporada"][equipo_b]["pace"]
            ), 1),
        },
        "impacto_lesiones": {
            "A": round(t3["impacto_lesiones"][equipo_a] * 100, 1),
            "B": round(t3["impacto_lesiones"][equipo_b] * 100, 1),
            "ajuste_aplicado": abs(inj_diff) > 0.01,
        },
        "prob_sin_lesiones": {
            equipo_a: round((prob_home_raw if equipo_a_es_local else 1 - prob_home_raw) * 100, 1),
            equipo_b: round(((1 - prob_home_raw) if equipo_a_es_local else prob_home_raw) * 100, 1),
        },
        "dias_descanso": {
            "A": t1["forma_reciente"][equipo_a]["dias_desde_ultimo_partido"],
            "B": t1["forma_reciente"][equipo_b]["dias_desde_ultimo_partido"],
        },
        "fatiga_7d": {
            "A": t3["carga_reciente_7_dias"][equipo_a]["nivel_fatiga"],
            "B": t3["carga_reciente_7_dias"][equipo_b]["nivel_fatiga"],
        },
        "clutch_stats": {
            "A": t2.get("clutch_stats", {}).get(equipo_a, {}),
            "B": t2.get("clutch_stats", {}).get(equipo_b, {}),
        },
    }

    # ── Vegas odds (si están disponibles) ───────────────────────────
    if odds_available():
        vegas = get_game_odds(home, away)
        if vegas.get("available"):
            factores["vegas_odds"] = {
                "home_implied": round(vegas["home_implied_prob"] * 100, 1),
                "away_implied": round(vegas["away_implied_prob"] * 100, 1),
                "spread_home": vegas.get("spread_home", 0),
                "total": vegas.get("total", 0),
                "available": True,
            }
        else:
            factores["vegas_odds"] = {"available": False}
    else:
        factores["vegas_odds"] = {"available": False}

    return {
        "ganador":       ganador,
        "probabilidad":  {equipo_a: pct_a, equipo_b: pct_b},
        "confianza":     confianza,
        "apostar":       apostar,
        "nivel_apuesta": nivel_apuesta,
        "factores":      factores,
        "modelo":        art["model_name"],
        "modelo_auc":    art.get("test_auc", 0.0),
        "modelo_acc":    art.get("test_accuracy", 0.0),
    }

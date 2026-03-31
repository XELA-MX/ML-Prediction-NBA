"""
NBA Predictor — Datos Tier 2
Obtiene: Offensive Rating, Defensive Rating, Net Rating,
         Pace, FG%, 3P%, Opponent FG%, TS%
"""

from nba_api.stats.endpoints import (
    leaguedashteamstats,
    teamdashboardbygeneralsplits,
    teamgamelogs,
    leaguedashteamclutch,
)
from nba_api.stats.static import teams
import pandas as pd
import time
from api_utils import retry_api

# ─────────────────────────────────────────
# UTILIDAD
# ─────────────────────────────────────────
def get_team_id(nombre: str) -> int:
    all_teams = teams.get_teams()
    resultado = [t for t in all_teams if nombre.lower() in t["full_name"].lower()]
    if not resultado:
        raise ValueError(f"Equipo '{nombre}' no encontrado.")
    return resultado[0]["id"]


# ─────────────────────────────────────────
# TIER 2A: Estadísticas Avanzadas de Temporada
# Offensive Rating, Defensive Rating, Net Rating, Pace
# ─────────────────────────────────────────
@retry_api(max_retries=3, backoff=2.0)
def get_advanced_stats(team_id: int, season: str = "2024-25") -> dict:
    """
    Obtiene stats avanzadas del equipo en toda la temporada.
    Fuente: LeagueDashTeamStats con MeasureType='Advanced'
    """
    time.sleep(0.6)

    df = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]

    equipo = df[df["TEAM_ID"] == team_id]
    if equipo.empty:
        raise ValueError(f"No se encontraron datos avanzados para team_id={team_id}")

    row = equipo.iloc[0]

    return {
        "offensive_rating": round(float(row["OFF_RATING"]), 2),
        "defensive_rating": round(float(row["DEF_RATING"]), 2),
        "net_rating":       round(float(row["NET_RATING"]), 2),
        "pace":             round(float(row["PACE"]), 2),
        "ts_pct":           round(float(row["TS_PCT"]), 4),    # True Shooting %
        "ast_ratio":        round(float(row["AST_RATIO"]), 2), # Assist ratio
        "reb_pct":          round(float(row["REB_PCT"]), 4),   # Rebound %
    }


# ─────────────────────────────────────────
# TIER 2B: FG%, 3P% y stats ofensivas recientes
# Últimos N partidos (más relevante que toda la temporada)
# ─────────────────────────────────────────
@retry_api(max_retries=3, backoff=2.0)
def get_shooting_reciente(team_id: int, season: str = "2024-25", n: int = 10) -> dict:
    """
    Calcula FG%, 3P%, FT% y puntos promedio de los últimos N partidos.
    Más representativo del momento actual del equipo.
    """
    time.sleep(0.6)

    logs = teamgamelogs.TeamGameLogs(
        team_id_nullable=team_id,
        season_nullable=season,
        season_type_nullable="Regular Season",
    ).get_data_frames()[0].head(n).copy()

    return {
        "partidos_analizados": len(logs),
        "fg_pct":   round(logs["FG_PCT"].mean(), 4),   # Field Goal %
        "fg3_pct":  round(logs["FG3_PCT"].mean(), 4),  # 3-Point %
        "ft_pct":   round(logs["FT_PCT"].mean(), 4),   # Free Throw %
        "fg3a_prom": round(logs["FG3A"].mean(), 1),    # Intentos de 3 por partido
        "pts_prom":  round(logs["PTS"].mean(), 1),
        "ast_prom":  round(logs["AST"].mean(), 1),
        "tov_prom":  round(logs["TOV"].mean(), 1),     # Turnovers (menos = mejor)
        "reb_prom":  round(logs["REB"].mean(), 1),
    }


# ─────────────────────────────────────────
# TIER 2C: Rendimiento defensivo reciente
# Qué tan bien defienden (Opponent FG%)
# ─────────────────────────────────────────
@retry_api(max_retries=3, backoff=2.0)
def get_defensa_reciente(team_id: int, season: str = "2024-25", n: int = 10) -> dict:
    """
    Calcula el rendimiento defensivo usando PLUS_MINUS y PTS en últimos N partidos.
    También extrae el Defensive Rating de la temporada como referencia.
    """
    time.sleep(0.6)

    logs = teamgamelogs.TeamGameLogs(
        team_id_nullable=team_id,
        season_nullable=season,
        season_type_nullable="Regular Season",
    ).get_data_frames()[0].head(n).copy()

    logs["PTS_RIVAL"] = logs["PTS"] - logs["PLUS_MINUS"]

    # Opponent FG% no está directo en TeamGameLogs,
    # pero podemos estimar la solidez defensiva con:
    # - Puntos permitidos promedio
    # - STL (robos) y BLK (bloqueos) como indicadores defensivos
    return {
        "pts_permitidos_prom": round(logs["PTS_RIVAL"].mean(), 1),
        "stl_prom":            round(logs["STL"].mean(), 1),  # Steals
        "blk_prom":            round(logs["BLK"].mean(), 1),  # Blocks
        "dreb_prom":           round(logs["DREB"].mean(), 1), # Defensive rebounds
        "tov_forzados_aprox":  round(logs["STL"].mean() * 1.4, 1),  # Estimado
    }


# ─────────────────────────────────────────
# TIER 2D: Split local/visitante avanzado
# Stats avanzadas separadas por casa y fuera
# ─────────────────────────────────────────
@retry_api(max_retries=3, backoff=2.0)
def get_advanced_splits(team_id: int, season: str = "2024-25") -> dict:
    """
    Obtiene Offensive/Defensive Rating separados por local y visitante.
    Fuente: TeamDashboardByGeneralSplits
    """
    time.sleep(0.6)

    df = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
        team_id=team_id,
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
    ).get_data_frames()

    # El índice 1 corresponde al split de Location (Home/Road)
    location_df = df[1]

    def parse_split(label: str) -> dict:
        row = location_df[location_df["GROUP_VALUE"] == label]
        if row.empty:
            return {}
        r = row.iloc[0]
        return {
            "partidos":         int(r["GP"]),
            "win_pct":          round(float(r["W_PCT"]), 3),
            "offensive_rating": round(float(r["OFF_RATING"]), 2),
            "defensive_rating": round(float(r["DEF_RATING"]), 2),
            "net_rating":       round(float(r["NET_RATING"]), 2),
            "pace":             round(float(r["PACE"]), 2),
        }

    return {
        "como_local":     parse_split("Home"),
        "como_visitante": parse_split("Road"),
    }


# ─────────────────────────────────────────
# TIER 2F: Clutch Stats (últimos 5 min, margen ≤5 pts)
# ─────────────────────────────────────────
_clutch_cache: dict = {}

@retry_api(max_retries=3, backoff=2.0)
def get_clutch_stats(season: str = "2024-25") -> dict:
    """
    Obtiene estadísticas en momentos clutch para todos los equipos.
    Clutch = últimos 5 minutos con margen ≤ 5 puntos.
    
    Returns dict[team_name] = {clutch_win_pct, clutch_plus_minus, clutch_net_rtg}
    """
    cache_key = f"clutch_{season}"
    if cache_key in _clutch_cache:
        return _clutch_cache[cache_key]
    
    time.sleep(0.6)
    
    try:
        clutch = leaguedashteamclutch.LeagueDashTeamClutch(
            season=season,
            season_type_all_star="Regular Season",
            clutch_time="Last 5 Minutes",
            ahead_behind="Ahead or Behind",
            point_diff=5,
        )
        df = clutch.get_data_frames()[0]
    except Exception as e:
        print(f"    Clutch stats no disponible: {e}")
        return {}
    
    result = {}
    for _, row in df.iterrows():
        team_name = row["TEAM_NAME"]
        games = row["GP"]
        if games > 0:
            result[team_name] = {
                "clutch_win_pct": round(float(row["W_PCT"]), 3),
                "clutch_plus_minus": round(float(row["PLUS_MINUS"]), 1),
                "clutch_games": int(games),
            }
    
    _clutch_cache[cache_key] = result
    return result


# ─────────────────────────────────────────
# FUNCIÓN PRINCIPAL: Reúne todo el Tier 2
# ─────────────────────────────────────────
def get_tier2(equipo_a: str, equipo_b: str, season: str = "2025-26") -> dict:
    """
    Junta todos los datos Tier 2 para un partido entre dos equipos.

    Params:
        equipo_a : Nombre del primer equipo  (ej. "Golden State Warriors")
        equipo_b : Nombre del segundo equipo (ej. "Portland Trail Blazers")
        season   : Temporada (ej. "2024-25")
    """
    id_a = get_team_id(equipo_a)
    id_b = get_team_id(equipo_b)

    print(f"\nObteniendo datos Tier 2: {equipo_a} vs {equipo_b}\n")

    print("  Advanced stats equipo A...")
    adv_a = get_advanced_stats(id_a, season)

    print("  Advanced stats equipo B...")
    adv_b = get_advanced_stats(id_b, season)

    print("  Shooting reciente equipo A...")
    shoot_a = get_shooting_reciente(id_a, season)

    print("  Shooting reciente equipo B...")
    shoot_b = get_shooting_reciente(id_b, season)

    print("  Defensa reciente equipo A...")
    def_a = get_defensa_reciente(id_a, season)

    print("  Defensa reciente equipo B...")
    def_b = get_defensa_reciente(id_b, season)

    print("  Splits local/visitante avanzados equipo A...")
    splits_a = get_advanced_splits(id_a, season)

    print("  Splits local/visitante avanzados equipo B...")
    splits_b = get_advanced_splits(id_b, season)

    print("  Clutch stats...")
    clutch_stats = get_clutch_stats(season)

    return {
        "advanced_stats_temporada": {
            equipo_a: adv_a,
            equipo_b: adv_b,
        },
        "shooting_reciente_10j": {
            equipo_a: shoot_a,
            equipo_b: shoot_b,
        },
        "defensa_reciente_10j": {
            equipo_a: def_a,
            equipo_b: def_b,
        },
        "splits_local_visitante_avanzado": {
            equipo_a: splits_a,
            equipo_b: splits_b,
        },
        "clutch_stats": {
            equipo_a: clutch_stats.get(equipo_a, {}),
            equipo_b: clutch_stats.get(equipo_b, {}),
        },
    }



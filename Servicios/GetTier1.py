"""
NBA Predictor — Datos Tier 1
Obtiene: forma reciente, local/visitante y head-to-head
"""

from nba_api.stats.endpoints import teamgamelogs, leaguegamefinder, leaguestandings
from nba_api.stats.static import teams
import pandas as pd
import time
from datetime import date
from api_utils import retry_api
from team_locations import calc_travel_factor, get_team_data

# ─────────────────────────────────────────
# UTILIDAD: Buscar ID de equipo por nombre
# ─────────────────────────────────────────
def get_team_id(nombre: str) -> int:
    """Devuelve el team_id de la NBA dado un nombre parcial del equipo."""
    all_teams = teams.get_teams()
    resultado = [t for t in all_teams if nombre.lower() in t["full_name"].lower()]
    if not resultado:
        raise ValueError(f"Equipo '{nombre}' no encontrado.")
    return resultado[0]["id"]


# ─────────────────────────────────────────
# TIER 1A: Forma reciente (últimos N partidos)
# ─────────────────────────────────────────
@retry_api(max_retries=3, backoff=2.0)
def get_forma_reciente(team_id: int, season: str = "2024-25", n: int = 10) -> dict:
    """
    Retorna estadísticas de los últimos N partidos de un equipo:
    - Récord (wins/losses)
    - Puntos anotados promedio
    - Puntos recibidos promedio
    - Point differential promedio
    """
    time.sleep(0.6)  # Evitar rate limit de la API

    logs = teamgamelogs.TeamGameLogs(
        team_id_nullable=team_id,
        season_nullable=season,
        season_type_nullable="Regular Season"
    ).get_data_frames()[0]

    # Los más recientes primero
    ultimos = logs.head(n).copy()

    # Determinar si ganó o perdió
    ultimos["WIN"] = ultimos["WL"].apply(lambda x: 1 if x == "W" else 0)

    # Puntos recibidos = PTS del rival
    # La columna PLUS_MINUS = PTS_equipo - PTS_rival → despejamos rival
    ultimos["PTS_RIVAL"] = ultimos["PTS"] - ultimos["PLUS_MINUS"]

    wins        = int(ultimos["WIN"].sum())
    losses      = n - wins
    pts_favor   = round(ultimos["PTS"].mean(), 1)
    pts_contra  = round(ultimos["PTS_RIVAL"].mean(), 1)
    diff        = round(pts_favor - pts_contra, 1)

    # ── Win rate últimos 5 partidos ────────────────────────────────────
    ultimos5     = ultimos.head(5)
    wins_5       = int(ultimos5["WIN"].sum())
    win_rate_5j  = round(wins_5 / 5, 3)

    # ── Días de descanso y back-to-back ───────────────────────────────
    ultimos["GAME_DATE"] = pd.to_datetime(ultimos["GAME_DATE"])
    ultimo_partido  = ultimos["GAME_DATE"].iloc[0].date()
    dias_descanso   = (date.today() - ultimo_partido).days
    # diff(-1): diferencia con el partido siguiente (más antiguo), en días
    ultimos["DIAS_DESCANSO"] = ultimos["GAME_DATE"].diff(-1).dt.days.abs()
    back_to_back = int(ultimos.iloc[0]["DIAS_DESCANSO"] == 1)

    # Racha actual: cuántos wins consecutivos desde el partido más reciente
    racha = int(ultimos["WIN"].cumprod().sum())
    # ──────────────────────────────────────────────────────────────────

    return {
        "ultimos_n_partidos":       n,
        "wins":                     wins,
        "losses":                   losses,
        "win_rate":                 round(wins / n, 3),
        "win_rate_5j":              win_rate_5j,
        "pts_anotados_prom":        pts_favor,
        "pts_recibidos_prom":       pts_contra,
        "point_differential":       diff,
        "dias_desde_ultimo_partido": dias_descanso,
        "back_to_back":             back_to_back,
        "racha_victorias_actual":   racha,
    }


# ─────────────────────────────────────────
# TIER 1B: Head-to-head histórico
# ─────────────────────────────────────────
@retry_api(max_retries=3, backoff=2.0)
def get_head_to_head(team_id_a: int, team_id_b: int,
                     season: str = "2024-25", n_partidos: int = 5) -> dict:
    """
    Devuelve los últimos N enfrentamientos directos entre dos equipos.
    - Resultado de cada partido (ganador y marcador)
    - Récord general de A vs B
    """
    time.sleep(0.6)

    finder = leaguegamefinder.LeagueGameFinder(
        team_id_nullable=team_id_a,
        season_nullable=season,
        season_type_nullable="Regular Season"
    ).get_data_frames()[0]

    # Filtrar solo partidos donde el rival fue team_b
    # La columna MATCHUP tiene formato "GSW vs. POR" o "GSW @ POR"
    all_teams = teams.get_teams()
    team_b_info = next(t for t in all_teams if t["id"] == team_id_b)
    team_b_abbr = team_b_info["abbreviation"]
    h2h = finder[finder["MATCHUP"].str.contains(team_b_abbr)].head(n_partidos).copy()

    if h2h.empty:
        return {"mensaje": "Sin enfrentamientos en la temporada actual."}

    h2h["WIN"] = h2h["WL"].apply(lambda x: 1 if x == "W" else 0)
    h2h["PTS_RIVAL"] = h2h["PTS"] - h2h["PLUS_MINUS"]

    partidos = []
    for _, row in h2h.iterrows():
        partidos.append({
            "fecha": row["GAME_DATE"],
            "matchup": row["MATCHUP"],
            "resultado": row["WL"],
            "pts_equipo_a": int(row["PTS"]),
            "pts_equipo_b": int(row["PTS_RIVAL"]),
        })

    wins_a = int(h2h["WIN"].sum())

    return {
        "enfrentamientos_analizados": len(h2h),
        "wins_equipo_a": wins_a,
        "wins_equipo_b": len(h2h) - wins_a,
        "partidos": partidos,
    }


# ─────────────────────────────────────────
# TIER 1C: Local vs Visitante
# ─────────────────────────────────────────
@retry_api(max_retries=3, backoff=2.0)
def get_local_visitante(team_id: int, season: str = "2024-25", n: int = 10) -> dict:
    """
    Separa el rendimiento reciente en casa vs fuera.
    MATCHUP con 'vs.' = local | '@' = visitante
    """
    time.sleep(0.6)

    logs = teamgamelogs.TeamGameLogs(
        team_id_nullable=team_id,
        season_nullable=season,
        season_type_nullable="Regular Season"
    ).get_data_frames()[0].head(n).copy()

    logs["ES_LOCAL"] = logs["MATCHUP"].str.contains(r"vs\.")
    logs["WIN"] = logs["WL"].apply(lambda x: 1 if x == "W" else 0)
    logs["PTS_RIVAL"] = logs["PTS"] - logs["PLUS_MINUS"]

    local     = logs[logs["ES_LOCAL"]]
    visitante = logs[~logs["ES_LOCAL"]]

    def resumen(df):
        if df.empty:
            return {"partidos": 0}
        return {
            "partidos": len(df),
            "win_rate": round(df["WIN"].mean(), 3),
            "pts_anotados_prom": round(df["PTS"].mean(), 1),
            "pts_recibidos_prom": round(df["PTS_RIVAL"].mean(), 1),
            "point_differential": round((df["PTS"] - df["PTS_RIVAL"]).mean(), 1),
        }

    return {
        "como_local": resumen(local),
        "como_visitante": resumen(visitante),
    }


# ─────────────────────────────────────────
# TIER 1D: Schedule Strength (opponent win%)
# ─────────────────────────────────────────
_standings_cache: dict = {}

@retry_api(max_retries=3, backoff=2.0)
def _get_league_standings(season: str) -> pd.DataFrame:
    """Obtiene standings de la liga (cacheado por temporada)."""
    if season in _standings_cache:
        return _standings_cache[season]
    
    time.sleep(0.6)
    standings = leaguestandings.LeagueStandings(
        season=season,
        season_type="Regular Season"
    ).get_data_frames()[0]
    
    _standings_cache[season] = standings
    return standings


def get_schedule_strength(team_id: int, season: str = "2024-25", n: int = 10) -> dict:
    """
    Calcula la fortaleza del schedule reciente.
    
    Returns:
        - opponent_avg_win_pct: promedio de win% de los últimos N oponentes
        - opponents_above_500: cuántos oponentes tenían >50% win rate
    """
    time.sleep(0.6)
    
    # Get team's recent games
    logs = teamgamelogs.TeamGameLogs(
        team_id_nullable=team_id,
        season_nullable=season,
        season_type_nullable="Regular Season"
    ).get_data_frames()[0]
    
    ultimos = logs.head(n).copy()
    
    # Extract opponent abbreviations from MATCHUP
    # Format: "GSW vs. POR" or "GSW @ POR"
    ultimos["OPP_ABBR"] = ultimos["MATCHUP"].str.extract(r"(?:vs\.|@)\s*(\w+)")
    
    # Get standings for opponent win%
    try:
        standings = _get_league_standings(season)
        # Map team abbreviation to win%
        abbr_to_winpct = dict(zip(
            standings["TeamSlug"].str.upper(),
            standings["WinPCT"]
        ))
        # Also try TeamCity + TeamName
        for _, row in standings.iterrows():
            abbr = row.get("TeamAbbreviation", "")
            if abbr:
                abbr_to_winpct[abbr] = row["WinPCT"]
    except Exception:
        # Fallback: assume 0.5 for all opponents
        abbr_to_winpct = {}
    
    opp_win_pcts = []
    for abbr in ultimos["OPP_ABBR"]:
        if pd.notna(abbr):
            win_pct = abbr_to_winpct.get(abbr.upper(), 0.5)
            opp_win_pcts.append(win_pct)
    
    if not opp_win_pcts:
        return {
            "opponent_avg_win_pct": 0.5,
            "opponents_above_500": 0,
            "schedule_strength": 0.5,
        }
    
    avg_win_pct = sum(opp_win_pcts) / len(opp_win_pcts)
    above_500 = sum(1 for p in opp_win_pcts if p > 0.5)
    
    return {
        "opponent_avg_win_pct": round(avg_win_pct, 3),
        "opponents_above_500": above_500,
        "schedule_strength": round(avg_win_pct, 3),  # alias
    }


# ─────────────────────────────────────────
# TIER 1E: Travel Factor
# ─────────────────────────────────────────
def get_travel_info(from_team: str, to_team: str) -> dict:
    """
    Calcula información de viaje entre dos equipos.
    El equipo visitante viaja desde su ciudad a la del local.
    """
    travel = calc_travel_factor(from_team, to_team)
    
    # Denver altitude bonus
    to_data = get_team_data(to_team)
    denver_factor = 0.0
    if to_data and to_data.get("altitude", 0) > 1500:
        denver_factor = 0.15  # 15% adicional por altitud extrema
    
    return {
        **travel,
        "denver_altitude_bonus": denver_factor,
    }


# ─────────────────────────────────────────
# FUNCIÓN PRINCIPAL: Reúne todo el Tier 1
# ─────────────────────────────────────────
def get_tier1(equipo_a: str, equipo_b: str,
              equipo_a_es_local: bool, season: str = "2025-26") -> dict:
    """
    Junta todos los datos Tier 1 para un partido entre dos equipos.
    
    Params:
        equipo_a          : Nombre del primer equipo (ej. "Golden State Warriors")
        equipo_b          : Nombre del segundo equipo (ej. "Portland Trail Blazers")
        equipo_a_es_local : True si equipo_a juega en casa
        season            : Temporada (ej. "2025-26")
    """
    id_a = get_team_id(equipo_a)
    id_b = get_team_id(equipo_b)

    print(f"\nObteniendo datos Tier 1: {equipo_a} vs {equipo_b}\n")

    print("  Forma reciente equipo A...")
    forma_a = get_forma_reciente(id_a, season)

    print("  Forma reciente equipo B...")
    forma_b = get_forma_reciente(id_b, season)

    print("  Local/Visitante equipo A...")
    local_a = get_local_visitante(id_a, season)

    print("  Local/Visitante equipo B...")
    local_b = get_local_visitante(id_b, season)

    print("  Head-to-head...")
    h2h = get_head_to_head(id_a, id_b, season)

    print("  Schedule strength equipo A...")
    sched_a = get_schedule_strength(id_a, season)

    print("  Schedule strength equipo B...")
    sched_b = get_schedule_strength(id_b, season)

    # Travel factor: visitante viaja a casa del local
    home_team = equipo_a if equipo_a_es_local else equipo_b
    away_team = equipo_b if equipo_a_es_local else equipo_a
    print("  Travel factor...")
    travel = get_travel_info(away_team, home_team)

    return {
        "partido": {
            "equipo_a": equipo_a,
            "equipo_b": equipo_b,
            "equipo_local": equipo_a if equipo_a_es_local else equipo_b,
            "ventaja_local_para": "A" if equipo_a_es_local else "B",
        },
        "forma_reciente": {
            equipo_a: forma_a,
            equipo_b: forma_b,
        },
        "rendimiento_local_visitante": {
            equipo_a: local_a,
            equipo_b: local_b,
        },
        "head_to_head": h2h,
        "schedule_strength": {
            equipo_a: sched_a,
            equipo_b: sched_b,
        },
        "travel_info": travel,
    }



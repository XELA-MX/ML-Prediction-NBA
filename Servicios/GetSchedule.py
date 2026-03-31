"""
NBA Predictor — Servicio de Schedule
Encuentra el próximo partido programado entre dos equipos.
Fuente: nba_api ScoreboardV2 (stats.nba.com)
"""

import time
from datetime import date, timedelta

from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import teams as nba_teams


def _build_id_map() -> dict:
    return {t["id"]: t["full_name"] for t in nba_teams.get_teams()}


def _find_team(nombre: str) -> dict:
    resultados = [
        t for t in nba_teams.get_teams()
        if nombre.lower() in t["full_name"].lower()
    ]
    if not resultados:
        raise ValueError(f"Equipo '{nombre}' no encontrado.")
    return resultados[0]


def get_next_game(equipo_a: str, equipo_b: str, dias_max: int = 30) -> dict | None:
    """
    Busca el próximo partido programado entre equipo_a y equipo_b.

    Itera desde hoy hasta dias_max días adelante consultando el scoreboard
    de cada fecha. Se detiene en el primer partido que encuentre.

    Retorna:
        {
          "fecha":            "YYYY-MM-DD",
          "equipo_local":     nombre completo del equipo local,
          "equipo_visitante": nombre completo del equipo visitante,
          "equipo_a_es_local": bool,
        }
    o None si no hay partido en ese rango.
    """
    team_a = _find_team(equipo_a)
    team_b = _find_team(equipo_b)
    id_a = team_a["id"]
    id_b = team_b["id"]
    id_to_name = _build_id_map()

    hoy = date.today()

    for delta in range(dias_max + 1):
        fecha = hoy + timedelta(days=delta)
        fecha_str = fecha.strftime("%m/%d/%Y")

        time.sleep(0.4)
        try:
            sb = scoreboardv2.ScoreboardV2(game_date=fecha_str)
            games_df = sb.game_header.get_data_frame()
        except Exception:
            continue

        if games_df.empty:
            continue

        for _, row in games_df.iterrows():
            home_id  = int(row["HOME_TEAM_ID"])
            visit_id = int(row["VISITOR_TEAM_ID"])

            if {home_id, visit_id} == {id_a, id_b}:
                return {
                    "fecha":             fecha.strftime("%Y-%m-%d"),
                    "equipo_local":      id_to_name.get(home_id,  str(home_id)),
                    "equipo_visitante":  id_to_name.get(visit_id, str(visit_id)),
                    "equipo_a_es_local": home_id == id_a,
                }

    return None

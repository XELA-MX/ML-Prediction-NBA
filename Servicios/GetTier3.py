"""
NBA Predictor — Datos Tier 3
Obtiene: Injury report, jugadores OUT/Questionable, star player disponible
         y carga de partidos reciente (últimos 7 días)
Fuente:  ESPN API pública (sin key) + nba_api
"""

import requests
import time
import pandas as pd
from nba_api.stats.endpoints import teamgamelogs, leaguedashplayerstats
from nba_api.stats.static import teams as nba_teams
from api_utils import retry_api

ESPN_TEAMS_URL      = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
ESPN_INJURIES_URL   = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

# Cache del injury report global (se llena una vez por ejecución)
_injuries_cache: dict | None = None


# ─────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────
def get_team_id_nba(nombre: str) -> int:
    """Devuelve el team_id de nba_api dado un nombre parcial."""
    all_teams = nba_teams.get_teams()
    resultado = [t for t in all_teams if nombre.lower() in t["full_name"].lower()]
    if not resultado:
        raise ValueError(f"Equipo '{nombre}' no encontrado en nba_api.")
    return resultado[0]["id"]


def get_espn_team_id(nombre: str) -> tuple[int, str]:
    """
    Busca el team_id de ESPN y el displayName dado un nombre parcial.
    Llama a la lista de equipos de ESPN y busca coincidencia por nombre.
    Retorna (espn_id, display_name).
    """
    resp = requests.get(ESPN_TEAMS_URL, params={"limit": 40})
    resp.raise_for_status()
    data = resp.json()

    equipos = data["sports"][0]["leagues"][0]["teams"]
    nombre_lower = nombre.lower()

    for entrada in equipos:
        t = entrada["team"]
        display = t.get("displayName", "")
        short    = t.get("shortDisplayName", "")
        abbr     = t.get("abbreviation", "")
        location = t.get("location", "")
        if (
            nombre_lower in display.lower()
            or nombre_lower in short.lower()
            or nombre_lower in abbr.lower()
            or nombre_lower in location.lower()
        ):
            return int(t["id"]), display

    raise ValueError(f"Equipo '{nombre}' no encontrado en ESPN.")


# ─────────────────────────────────────────
# TIER 3A: Injury Report via ESPN
# Jugadores OUT / Questionable / Day-To-Day
# ─────────────────────────────────────────
def clear_injuries_cache():
    """Resetea el cache de lesiones para forzar una nueva descarga."""
    global _injuries_cache
    _injuries_cache = None


def _fetch_all_injuries() -> list[dict]:
    """
    Descarga el injury report global de ESPN (todos los equipos).
    Cachea el resultado para no repetir la llamada en la misma ejecución.
    """
    global _injuries_cache
    if _injuries_cache is not None:
        return _injuries_cache

    try:
        resp = requests.get(ESPN_INJURIES_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        _injuries_cache = data.get("injuries", [])
    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"    ESPN injuries endpoint no disponible: {e}. Se asume 0 bajas.")
        _injuries_cache = []

    return _injuries_cache


def get_injury_report(espn_team_id: int, star_players: list[str] = None) -> dict:
    """
    Obtiene el injury report actual del equipo desde la API pública de ESPN.

    Usa el endpoint global /injuries (el per-team endpoint ya no devuelve datos)
    y filtra por espn_team_id.

    Params:
        espn_team_id : ID del equipo en ESPN
        star_players : Lista de nombres a monitorear (ej. ["Steph Curry"])
    """
    all_teams = _fetch_all_injuries()

    # Buscar el equipo en la lista global
    lesionados_raw = []
    for team_entry in all_teams:
        if int(team_entry.get("id", -1)) == espn_team_id:
            lesionados_raw = team_entry.get("injuries", [])
            break

    out_count          = 0
    questionable_count = 0
    lista_lesionados   = []
    nombres_lesionados = []

    for entrada in lesionados_raw:
        atleta  = entrada.get("athlete", {})
        nombre  = atleta.get("displayName", "?")
        status  = entrada.get("status", "").strip()
        tipo    = entrada.get("shortComment", "")

        lista_lesionados.append({
            "jugador": nombre,
            "status":  status,
            "tipo":    tipo,
        })
        nombres_lesionados.append(nombre.lower())

        status_up = status.upper()
        if status_up == "OUT":
            out_count += 1
        elif status_up in ("QUESTIONABLE", "DOUBTFUL", "DAY-TO-DAY", "GTD"):
            questionable_count += 1

    # Star players
    stars_status = {}
    if star_players:
        for star in star_players:
            esta = any(star.lower() in n for n in nombres_lesionados)
            stars_status[star] = "OUT/DTD" if esta else "ACTIVO"

    return {
        "jugadores_out":          out_count,
        "jugadores_questionable": questionable_count,
        "total_en_injury_report": len(lista_lesionados),
        "star_players":           stars_status,
        "detalle":                lista_lesionados,
    }


# ─────────────────────────────────────────
# TIER 3B: Minutos por jugador (para ponderar lesiones)
# ─────────────────────────────────────────
@retry_api(max_retries=3, backoff=2.0)
def get_player_minutes(team_id: int, season: str = "2024-25") -> dict:
    """
    Devuelve {nombre_jugador: minutos_promedio} para todos los jugadores del equipo.
    Fuente: LeagueDashPlayerStats (nba_api).
    """
    time.sleep(0.6)

    df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]

    team_df = df[df["TEAM_ID"] == team_id]
    return {
        row["PLAYER_NAME"]: float(row["MIN"])
        for _, row in team_df.iterrows()
        if float(row["MIN"]) > 0
    }


def calcular_impacto_lesiones(out_players: list[str], player_minutes: dict) -> float:
    """
    Calcula el ratio de minutos perdidos por lesiones (0.0 a 1.0).
    0.0 = ningún minuto perdido, 1.0 = todo el equipo fuera.

    Hace match flexible por nombre: busca si el nombre del lesionado
    aparece (parcialmente) en algún jugador del roster.
    """
    total_min = sum(player_minutes.values())
    if total_min == 0 or not out_players:
        return 0.0

    min_out = 0.0
    for out_name in out_players:
        out_lower = out_name.lower()
        for player_name, minutes in player_minutes.items():
            player_lower = player_name.lower()
            if out_lower in player_lower or player_lower in out_lower:
                min_out += minutes
                break

    return round(min(min_out / total_min, 1.0), 4)


# ─────────────────────────────────────────
# TIER 3C: Carga de partidos últimos 7 días
# Detecta fatiga acumulada más allá del back-to-back
# ─────────────────────────────────────────
@retry_api(max_retries=3, backoff=2.0)
def get_carga_reciente(team_id_nba: int, season: str = "2024-25", dias: int = 7) -> dict:
    """
    Cuenta cuántos partidos jugó el equipo en los últimos N días.
    3+ partidos en 7 días = fatiga acumulada relevante.
    """
    time.sleep(0.6)

    logs = teamgamelogs.TeamGameLogs(
        team_id_nullable=team_id_nba,
        season_nullable=season,
        season_type_nullable="Regular Season",
    ).get_data_frames()[0].copy()

    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    fecha_corte = logs["GAME_DATE"].iloc[0] - pd.Timedelta(days=dias)
    recientes   = logs[logs["GAME_DATE"] >= fecha_corte]

    partidos_en_ventana = len(recientes)

    if partidos_en_ventana >= 4:
        fatiga = "ALTA"
    elif partidos_en_ventana == 3:
        fatiga = "MEDIA"
    else:
        fatiga = "BAJA"

    return {
        f"partidos_ultimos_{dias}_dias": partidos_en_ventana,
        "nivel_fatiga":                  fatiga,
        "fechas_recientes": recientes["GAME_DATE"].dt.strftime("%Y-%m-%d").tolist(),
    }


# ─────────────────────────────────────────
# FUNCIÓN PRINCIPAL: Reúne todo el Tier 3
# ─────────────────────────────────────────
def get_tier3(
    equipo_a:        str,
    equipo_b:        str,
    stars_a:         list[str] = None,
    stars_b:         list[str] = None,
    season:          str = "2024-25",
) -> dict:
    """
    Junta injury report (ESPN) + carga reciente (nba_api) para ambos equipos.

    Params:
        equipo_a : Nombre completo (ej. "Golden State Warriors")
        equipo_b : Nombre completo (ej. "Portland Trail Blazers")
        stars_a  : Jugadores estrella del equipo A a monitorear
        stars_b  : Jugadores estrella del equipo B a monitorear
        season   : Temporada (ej. "2024-25")
    """
    print(f"\nObteniendo datos Tier 3: {equipo_a} vs {equipo_b}\n")

    # Forzar descarga fresca de lesiones en cada predicción
    clear_injuries_cache()

    # IDs en NBA API
    id_a_nba = get_team_id_nba(equipo_a)
    id_b_nba = get_team_id_nba(equipo_b)

    # IDs en ESPN — buscar por último token del nombre (p.ej. "Warriors", "76ers")
    token_a = equipo_a.split()[-1]
    token_b = equipo_b.split()[-1]
    id_a_espn, _ = get_espn_team_id(token_a)
    id_b_espn, _ = get_espn_team_id(token_b)

    print("  Injury report equipo A (ESPN)...")
    injury_a = get_injury_report(id_a_espn, stars_a)

    print("  Injury report equipo B (ESPN)...")
    injury_b = get_injury_report(id_b_espn, stars_b)

    print("  Minutos por jugador equipo A...")
    minutos_a = get_player_minutes(id_a_nba, season)

    print("  Minutos por jugador equipo B...")
    minutos_b = get_player_minutes(id_b_nba, season)

    # Jugadores confirmados OUT en cada equipo
    out_a = [j["jugador"] for j in injury_a["detalle"] if j["status"].upper() == "OUT"]
    out_b = [j["jugador"] for j in injury_b["detalle"] if j["status"].upper() == "OUT"]

    impacto_a = calcular_impacto_lesiones(out_a, minutos_a)
    impacto_b = calcular_impacto_lesiones(out_b, minutos_b)

    print("  Carga reciente equipo A...")
    carga_a = get_carga_reciente(id_a_nba, season)

    print("  Carga reciente equipo B...")
    carga_b = get_carga_reciente(id_b_nba, season)

    return {
        "injury_report": {
            equipo_a: injury_a,
            equipo_b: injury_b,
        },
        "impacto_lesiones": {
            equipo_a: impacto_a,
            equipo_b: impacto_b,
        },
        "carga_reciente_7_dias": {
            equipo_a: carga_a,
            equipo_b: carga_b,
        },
    }


# ─────────────────────────────────────────
# EJEMPLO DE USO
# ─────────────────────────────────────────
if __name__ == "__main__":
    import json

    datos = get_tier3(
        equipo_a = "Chicago Bulls",
        equipo_b = "Philadelphia 76ers",
        season   = "2024-25",
    )

    print(json.dumps(datos, indent=2, ensure_ascii=False))

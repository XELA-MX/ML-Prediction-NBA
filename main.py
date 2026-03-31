"""
NBA Game Predictor — main.py
Predice el ganador de un partido usando datos en tiempo real (temporada 2025-26).

Uso:
    python main.py
"""

import sys
import os

# Agregar Servicios al path de importación
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Servicios"))

from nba_api.stats.static import teams as nba_teams
from GetTier1 import get_tier1
from GetTier2 import get_tier2
from GetTier3 import get_tier3
from GetSchedule import get_next_game

try:
    from Predict import predecir as _predecir_ml
    ML_DISPONIBLE = True
except Exception:
    ML_DISPONIBLE = False

SEASON = "2025-26"


# ──────────────────────────────────────────────────────────────────
# UTILIDAD: Validar y seleccionar equipo interactivamente
# ──────────────────────────────────────────────────────────────────
def buscar_equipo(query: str) -> list:
    """Busca equipos NBA por nombre parcial, apodo o abreviatura."""
    all_teams = nba_teams.get_teams()
    q = query.lower()
    return [
        t for t in all_teams
        if q in t["full_name"].lower()
        or q in t["nickname"].lower()
        or q in t["abbreviation"].lower()
        or q in t["city"].lower()
    ]


def seleccionar_equipo(prompt_texto: str) -> str:
    """Pide un nombre de equipo y lo valida contra la lista oficial de la NBA."""
    while True:
        nombre = input(prompt_texto).strip()
        if not nombre:
            continue

        matches = buscar_equipo(nombre)

        if len(matches) == 0:
            print(f"  No se encontró ningún equipo con '{nombre}'.")
            print("    Ejemplos válidos: Lakers, Warriors, Celtics, Bucks, Nuggets\n")

        elif len(matches) == 1:
            print(f"  Equipo: {matches[0]['full_name']}\n")
            return matches[0]["full_name"]

        else:
            print(f"  Varios equipos coinciden con '{nombre}':")
            for i, t in enumerate(matches, 1):
                print(f"    {i}. {t['full_name']}")
            sel = input("  Elige el número: ").strip()
            try:
                idx = int(sel) - 1
                if 0 <= idx < len(matches):
                    print(f"  Equipo: {matches[idx]['full_name']}\n")
                    return matches[idx]["full_name"]
                else:
                    print("  Número fuera de rango.\n")
            except ValueError:
                print("  Entrada inválida.\n")


# ──────────────────────────────────────────────────────────────────
# MOTOR DE PREDICCIÓN: Scoring ponderado multifactor
# ──────────────────────────────────────────────────────────────────
def calcular_prediccion(
    t1: dict, t2: dict, t3: dict,
    equipo_a: str, equipo_b: str,
    equipo_a_es_local: bool,
) -> dict:
    """
    Calcula un score ponderado para cada equipo y devuelve el ganador predicho.

    Distribución de pesos (115 puntos → normalizado a 100%):
      Net Rating temporada      20 pts
      Win rate últimos 10j      12 pts
      Win rate últimos 5j       10 pts
      Point differential        10 pts
      Rendimiento local/visit.  10 pts
      FG% reciente (10j)        10 pts
      Defensa reciente          10 pts
      Head-to-head               8 pts
      Impacto lesiones pond.    10 pts
      Días de descanso           5 pts
      Fatiga (7 días)            5 pts
    """
    sa = 0.0  # score equipo A
    sb = 0.0  # score equipo B
    factores = {}

    # ── 1. Net Rating de temporada (peso 20) ──────────────────────
    nr_a = t2["advanced_stats_temporada"][equipo_a]["net_rating"]
    nr_b = t2["advanced_stats_temporada"][equipo_b]["net_rating"]
    MAX_NR = 15.0
    sa += max(0.0, min(1.0, (nr_a + MAX_NR) / (2 * MAX_NR))) * 20
    sb += max(0.0, min(1.0, (nr_b + MAX_NR) / (2 * MAX_NR))) * 20
    factores["net_rating"] = {"A": nr_a, "B": nr_b}

    # ── 2. Win rate últimos 10 partidos (peso 12) ─────────────────
    wr_a = t1["forma_reciente"][equipo_a]["win_rate"]
    wr_b = t1["forma_reciente"][equipo_b]["win_rate"]
    sa += wr_a * 12
    sb += wr_b * 12
    factores["win_rate_10j"] = {
        "A": round(wr_a * 100, 1),
        "B": round(wr_b * 100, 1),
    }

    # ── 3. Win rate últimos 5 partidos — momentum (peso 10) ───────
    wr5_a = t1["forma_reciente"][equipo_a]["win_rate_5j"]
    wr5_b = t1["forma_reciente"][equipo_b]["win_rate_5j"]
    sa += wr5_a * 10
    sb += wr5_b * 10
    factores["win_rate_5j"] = {
        "A": round(wr5_a * 100, 1),
        "B": round(wr5_b * 100, 1),
    }

    # ── 4. Point differential (peso 10) ───────────────────────────
    pd_a = t1["forma_reciente"][equipo_a]["point_differential"]
    pd_b = t1["forma_reciente"][equipo_b]["point_differential"]
    MAX_PD = 20.0
    sa += max(0.0, min(1.0, (pd_a + MAX_PD) / (2 * MAX_PD))) * 10
    sb += max(0.0, min(1.0, (pd_b + MAX_PD) / (2 * MAX_PD))) * 10
    factores["point_diff"] = {"A": pd_a, "B": pd_b}

    # ── 5. Rendimiento en contexto local/visitante (peso 10) ──────
    # Usa el Net Rating real de cada equipo en su rol (casa o fuera)
    # en lugar de un bonus fijo — captura qué tan dominante es cada
    # equipo específicamente en ese contexto.
    splits_a = t2["splits_local_visitante_avanzado"][equipo_a]
    splits_b = t2["splits_local_visitante_avanzado"][equipo_b]

    if equipo_a_es_local:
        nr_ctx_a = splits_a.get("como_local",     {}).get("net_rating", 0.0)
        nr_ctx_b = splits_b.get("como_visitante", {}).get("net_rating", 0.0)
        factores["local"] = equipo_a
    else:
        nr_ctx_a = splits_a.get("como_visitante", {}).get("net_rating", 0.0)
        nr_ctx_b = splits_b.get("como_local",     {}).get("net_rating", 0.0)
        factores["local"] = equipo_b

    MAX_CTX = 15.0
    sc_ctx_a = max(0.0, min(1.0, (nr_ctx_a + MAX_CTX) / (2 * MAX_CTX)))
    sc_ctx_b = max(0.0, min(1.0, (nr_ctx_b + MAX_CTX) / (2 * MAX_CTX)))
    total_ctx = sc_ctx_a + sc_ctx_b or 1.0
    sa += (sc_ctx_a / total_ctx) * 10
    sb += (sc_ctx_b / total_ctx) * 10
    factores["nr_contexto"] = {
        "A": round(nr_ctx_a, 2),
        "B": round(nr_ctx_b, 2),
        "rol_A": "local" if equipo_a_es_local else "visitante",
    }

    # ── 6. FG% reciente (peso 10) ─────────────────────────────────
    fg_a = t2["shooting_reciente_10j"][equipo_a]["fg_pct"]
    fg_b = t2["shooting_reciente_10j"][equipo_b]["fg_pct"]
    sa += fg_a * 10
    sb += fg_b * 10
    factores["fg_pct_10j"] = {
        "A": round(fg_a * 100, 1),
        "B": round(fg_b * 100, 1),
    }

    # ── 7. Defensa: puntos permitidos (peso 10) ───────────────────
    pts_a = t2["defensa_reciente_10j"][equipo_a]["pts_permitidos_prom"]
    pts_b = t2["defensa_reciente_10j"][equipo_b]["pts_permitidos_prom"]
    MIN_PTS, MAX_PTS = 95.0, 135.0
    sa += max(0.0, min(1.0, 1 - (pts_a - MIN_PTS) / (MAX_PTS - MIN_PTS))) * 10
    sb += max(0.0, min(1.0, 1 - (pts_b - MIN_PTS) / (MAX_PTS - MIN_PTS))) * 10
    factores["pts_permitidos"] = {"A": pts_a, "B": pts_b}

    # ── 8. Head-to-head temporada actual (peso 8) ─────────────────
    h2h = t1["head_to_head"]
    if "wins_equipo_a" in h2h:
        total_h2h = h2h["wins_equipo_a"] + h2h["wins_equipo_b"]
        if total_h2h > 0:
            sa += (h2h["wins_equipo_a"] / total_h2h) * 8
            sb += (h2h["wins_equipo_b"] / total_h2h) * 8
            factores["h2h"] = {"A": h2h["wins_equipo_a"], "B": h2h["wins_equipo_b"]}
        else:
            sa += 4; sb += 4
            factores["h2h"] = "sin enfrentamientos aún"
    else:
        sa += 4; sb += 4
        factores["h2h"] = "sin datos"

    # ── 9. Impacto lesiones ponderado por minutos (peso 10) ───────
    imp_a = t3["impacto_lesiones"][equipo_a]
    imp_b = t3["impacto_lesiones"][equipo_b]
    sa += (1 - imp_a) * 10
    sb += (1 - imp_b) * 10
    factores["impacto_lesiones"] = {
        "A": round(imp_a * 100, 1),
        "B": round(imp_b * 100, 1),
    }

    # ── 10. Diferencial de turnovers (peso 5) ─────────────────────
    # net_tov = turnovers forzados − turnovers cometidos
    # Positivo = el equipo genera más pérdidas de las que comete
    tov_comm_a  = t2["shooting_reciente_10j"][equipo_a]["tov_prom"]
    tov_comm_b  = t2["shooting_reciente_10j"][equipo_b]["tov_prom"]
    tov_forc_a  = t2["defensa_reciente_10j"][equipo_a]["tov_forzados_aprox"]
    tov_forc_b  = t2["defensa_reciente_10j"][equipo_b]["tov_forzados_aprox"]
    net_tov_a   = tov_forc_a - tov_comm_a
    net_tov_b   = tov_forc_b - tov_comm_b
    MAX_TOV = 8.0
    sa += max(0.0, min(1.0, (net_tov_a + MAX_TOV) / (2 * MAX_TOV))) * 5
    sb += max(0.0, min(1.0, (net_tov_b + MAX_TOV) / (2 * MAX_TOV))) * 5
    factores["net_tov"] = {
        "A": round(net_tov_a, 1),
        "B": round(net_tov_b, 1),
    }

    # ── 11. Pace matchup (peso 3) ──────────────────────────────────
    # El equipo local tiende a imponer su ritmo preferido en casa.
    # Si el local juega más rápido que el visitante, se beneficia;
    # si juega más lento, también controla el ritmo (ventaja menor).
    pace_a = t2["advanced_stats_temporada"][equipo_a]["pace"]
    pace_b = t2["advanced_stats_temporada"][equipo_b]["pace"]
    pace_diff = pace_a - pace_b  # positivo = A juega más rápido
    pace_advantage = pace_diff if equipo_a_es_local else -pace_diff
    MAX_PACE = 6.0
    sa_pace = max(0.0, min(1.0, (pace_advantage + MAX_PACE) / (2 * MAX_PACE))) * 3
    sb_pace = 3.0 - sa_pace
    sa += sa_pace
    sb += sb_pace
    factores["pace"] = {
        "A": pace_a,
        "B": pace_b,
        "ventaja_local": round(abs(pace_a - pace_b), 1),
    }

    # ── 12. Días de descanso reales (peso 5) ──────────────────────
    def rest_pts(dias: int) -> float:
        if dias <= 0: return 0.0
        if dias == 1: return 2.0
        if dias == 2: return 3.5
        return 5.0   # 3+ días = descanso completo

    dias_a = t1["forma_reciente"][equipo_a]["dias_desde_ultimo_partido"]
    dias_b = t1["forma_reciente"][equipo_b]["dias_desde_ultimo_partido"]
    sa += rest_pts(dias_a)
    sb += rest_pts(dias_b)
    factores["dias_descanso"] = {"A": dias_a, "B": dias_b}

    # ── 13. Fatiga últimos 7 días (peso 5) ────────────────────────
    fatiga_pts = {"BAJA": 5, "MEDIA": 3, "ALTA": 0}
    fat_a = t3["carga_reciente_7_dias"][equipo_a]["nivel_fatiga"]
    fat_b = t3["carga_reciente_7_dias"][equipo_b]["nivel_fatiga"]
    sa += fatiga_pts.get(fat_a, 3)
    sb += fatiga_pts.get(fat_b, 3)
    factores["fatiga_7d"] = {"A": fat_a, "B": fat_b}

    # ── Resultado final ───────────────────────────────────────────
    total = sa + sb
    pct_a = round((sa / total) * 100, 1)
    pct_b = round((sb / total) * 100, 1)
    ganador = equipo_a if sa >= sb else equipo_b

    return {
        "ganador": ganador,
        "probabilidad": {equipo_a: pct_a, equipo_b: pct_b},
        "factores": factores,
    }


# ──────────────────────────────────────────────────────────────────
# PRESENTACIÓN DE RESULTADOS
# ──────────────────────────────────────────────────────────────────
def imprimir_resultado(pred: dict, equipo_a: str, equipo_b: str) -> None:
    sep = "─" * 54
    ganador  = pred["ganador"]
    perdedor = equipo_b if ganador == equipo_a else equipo_a
    pct_g    = pred["probabilidad"][ganador]
    pct_p    = pred["probabilidad"][perdedor]
    f        = pred["factores"]

    modelo_str = ""
    if "modelo" in pred:
        modelo_str = (
            f"  [{pred['modelo']}  "
            f"AUC {pred.get('modelo_auc', 0):.3f}  "
            f"Acc {pred.get('modelo_acc', 0):.3f}]"
        )

    print(f"\n{sep}")
    print("   PREDICCIÓN NBA  —  TEMPORADA 2025-26")
    if modelo_str:
        print(f"  {modelo_str}")
    print(sep)
    print(f"   {equipo_a}")
    print(f"       vs")
    print(f"   {equipo_b}")
    print(sep)
    print(f"\n   GANADOR PREDICHO:  {ganador.upper()}")
    print(f"   Confianza:  {ganador} {pct_g}%  /  {perdedor} {pct_p}%")
    print(f"\n   FACTORES CONSIDERADOS:")

    nr = f["net_rating"]
    print(f"   {'Net Rating (temporada)':<32} A={nr['A']:+.1f}   B={nr['B']:+.1f}")

    wr = f["win_rate_10j"]
    print(f"   {'Win rate últimos 10j':<32} A={wr['A']}%   B={wr['B']}%")

    wr5 = f["win_rate_5j"]
    print(f"   {'Win rate últimos 5j (momentum)':<32} A={wr5['A']}%   B={wr5['B']}%")

    pd = f["point_diff"]
    print(f"   {'Dif. puntos (10j)':<32} A={pd['A']:+.1f}   B={pd['B']:+.1f}")

    fg = f["fg_pct_10j"]
    print(f"   {'FG% reciente (10j)':<32} A={fg['A']}%   B={fg['B']}%")

    pts = f["pts_permitidos"]
    print(f"   {'Pts permitidos (10j)':<32} A={pts['A']}   B={pts['B']}")

    h2h = f["h2h"]
    if isinstance(h2h, dict):
        h2h_str = f"A={h2h['A']}W  B={h2h['B']}W"
    else:
        h2h_str = str(h2h)
    print(f"   {'H2H temporada actual':<32} {h2h_str}")

    ctx = f["nr_contexto"]
    rol = "LOCAL" if ctx["rol_A"] == "local" else "VISIT."
    print(f"   {'Net Rating casa/fuera':<32} A({rol})={ctx['A']:+.2f}   B={ctx['B']:+.2f}")

    tov = f["net_tov"]
    print(f"   {'Diferencial turnovers':<32} A={tov['A']:+.1f}   B={tov['B']:+.1f}")

    pac = f["pace"]
    print(f"   {'Pace (ritmo)':<32} A={pac['A']}   B={pac['B']}")

    imp = f["impacto_lesiones"]
    print(f"   {'Impacto lesiones (% min OUT)':<32} A={imp['A']}%   B={imp['B']}%")

    des = f["dias_descanso"]
    print(f"   {'Días de descanso':<32} A={des['A']}d   B={des['B']}d")

    fat = f["fatiga_7d"]
    print(f"   {'Fatiga últimos 7 días':<32} A={fat['A']}  B={fat['B']}")

    print(f"   {'Equipo local':<32} {f['local']}")
    print(f"\n{sep}\n")


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 54)
    print("   NBA GAME PREDICTOR  |  Temporada 2025-26")
    print("=" * 54)
    print("\nIngresa los dos equipos del partido.\n")

    # Selección de equipos
    equipo_a = seleccionar_equipo("Equipo 1 (ej. Lakers, Warriors, Celtics): ")

    while True:
        equipo_b = seleccionar_equipo("Equipo 2 (ej. Lakers, Warriors, Celtics): ")
        if equipo_b != equipo_a:
            break
        print("  Los dos equipos no pueden ser el mismo.\n")

    # ── Buscar partido en el schedule ──────────────────────────────
    print("Buscando el próximo partido en el calendario NBA...")
    juego = get_next_game(equipo_a, equipo_b)

    if juego:
        print(f"\n  Partido encontrado:")
        print(f"    Fecha:     {juego['fecha']}")
        print(f"    Local:     {juego['equipo_local']}")
        print(f"    Visitante: {juego['equipo_visitante']}")
        confirmacion = input("\n  ¿Usar estos datos? (s/n): ").strip().lower()
        if confirmacion == "s":
            equipo_a_es_local = juego["equipo_a_es_local"]
            local     = juego["equipo_local"]
            visitante = juego["equipo_visitante"]
            print(f"\n  Local: {local}  |  Visitante: {visitante}\n")
        else:
            juego = None

    if not juego:
        print(f"\n¿Cuál equipo juega en casa?\n  1. {equipo_a}\n  2. {equipo_b}")
        while True:
            resp = input("Elige (1/2): ").strip()
            if resp in ("1", "2"):
                break
            print("  Escribe 1 o 2.")
        equipo_a_es_local = (resp == "1")
        local     = equipo_a if equipo_a_es_local else equipo_b
        visitante = equipo_b if equipo_a_es_local else equipo_a
        print(f"\n  Local: {local}  |  Visitante: {visitante}\n")

    # Obtención de datos
    print(f"Obteniendo datos en tiempo real (temporada {SEASON})...")
    print("Esto puede tardar ~60 segundos por los límites de la API.\n")

    try:
        t1 = get_tier1(equipo_a, equipo_b, equipo_a_es_local, season=SEASON)
        t2 = get_tier2(equipo_a, equipo_b, season=SEASON)
        t3 = get_tier3(equipo_a, equipo_b, season=SEASON)
    except Exception as e:
        print(f"\nError al obtener datos de la API: {e}")
        sys.exit(1)

    # Predicción y resultado
    print("\nCalculando predicción...\n")
    if ML_DISPONIBLE:
        try:
            pred = _predecir_ml(t1, t2, t3, equipo_a, equipo_b, equipo_a_es_local)
        except FileNotFoundError:
            print("Modelo ML no encontrado — usando scoring ponderado.")
            print("Para entrenar: python Servicios/BuildDataset.py && python Servicios/TrainModel.py\n")
            pred = calcular_prediccion(t1, t2, t3, equipo_a, equipo_b, equipo_a_es_local)
    else:
        pred = calcular_prediccion(t1, t2, t3, equipo_a, equipo_b, equipo_a_es_local)
    imprimir_resultado(pred, equipo_a, equipo_b)


if __name__ == "__main__":
    main()

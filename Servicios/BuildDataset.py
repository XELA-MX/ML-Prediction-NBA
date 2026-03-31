"""
NBA Predictor — Construcción del Dataset Histórico
Genera un dataset con features pre-partido para 3 temporadas.
Fuente: nba_api LeagueGameFinder (una llamada por temporada)

Uso:
    python Servicios/BuildDataset.py
"""

import os
import time
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams as nba_teams
from team_locations import calc_travel_factor, ABBR_TO_FULL

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SEASONS  = ["2021-22", "2022-23", "2023-24", "2024-25"]

# Features que se computarán en rolling para cada equipo
ROLL_COLS = ["win_rate_10j", "win_rate_5j", "pt_diff_10j",
             "fg_pct_10j", "fg3_pct_10j", "pts_allowed_10j", "tov_net_10j",
             "pace_10j", "rest_days", "fatigue_7d", "b2b",
             "season_win_rate", "season_net_rtg", "season_off_rtg", "season_def_rtg",
             "schedule_strength"]

# Mapping de TEAM_ID a abreviatura
_TEAM_ID_TO_ABBR: dict = {}
def _get_team_abbr(team_id: int) -> str:
    if not _TEAM_ID_TO_ABBR:
        for t in nba_teams.get_teams():
            _TEAM_ID_TO_ABBR[t["id"]] = t["abbreviation"]
    return _TEAM_ID_TO_ABBR.get(team_id, "")


# ──────────────────────────────────────────────────────────────────
# 1. Descarga de partidos
# ──────────────────────────────────────────────────────────────────
def get_season_games(season: str) -> pd.DataFrame:
    """Descarga todos los partidos de temporada regular para una temporada."""
    print(f"  Descargando temporada {season}...")
    time.sleep(1.2)

    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable="Regular Season",
        league_id_nullable="00",
    )
    df = finder.get_data_frames()[0].copy()

    df["SEASON"]   = season
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["IS_HOME"]  = df["MATCHUP"].str.contains(r"vs\.")
    df["WIN"]      = (df["WL"] == "W").astype(int)
    df["PTS_OPP"]  = df["PTS"] - df["PLUS_MINUS"]

    # Posesiones aproximadas (proxy de pace, misma escala que NBA API ~95-105)
    df["POSS"] = df["FGA"] + 0.44 * df["FTA"] - df["OREB"] + df["TOV"]

    # Net turnovers: robos generados − pérdidas propias
    df["TOV_NET"] = df["STL"] * 1.4 - df["TOV"]

    return df


# ──────────────────────────────────────────────────────────────────
# 2. Features rolling por equipo
# ──────────────────────────────────────────────────────────────────
def compute_rolling_features(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features usando SOLO datos anteriores al partido actual
    (shift(1) antes del rolling para evitar data leakage).
    """
    df = team_df.sort_values("GAME_DATE").copy()

    def roll(col, n):
        return df[col].shift(1).rolling(n, min_periods=max(1, n // 2)).mean()

    df["win_rate_10j"]    = roll("WIN",      10)
    df["win_rate_5j"]     = roll("WIN",       5)
    df["pt_diff_10j"]     = roll("PLUS_MINUS", 10)
    df["fg_pct_10j"]      = roll("FG_PCT",   10)
    df["fg3_pct_10j"]     = roll("FG3_PCT",  10)
    df["pts_allowed_10j"] = roll("PTS_OPP",  10)
    df["tov_net_10j"]     = roll("TOV_NET",  10)
    df["pace_10j"]        = roll("POSS",     10)

    # Season-level expanding stats (all games before current, no leakage)
    def expanding(col):
        return df[col].shift(1).expanding(min_periods=5).mean()

    df["season_win_rate"] = expanding("WIN")
    df["season_net_rtg"]  = expanding("PLUS_MINUS")
    # Offensive/defensive rating proxies: points per 100 possessions
    df["_pts_per_poss"]    = df["PTS"] / df["POSS"].clip(lower=1) * 100
    df["_ptsopp_per_poss"] = df["PTS_OPP"] / df["POSS"].clip(lower=1) * 100
    df["season_off_rtg"]   = df["_pts_per_poss"].shift(1).expanding(min_periods=5).mean()
    df["season_def_rtg"]   = df["_ptsopp_per_poss"].shift(1).expanding(min_periods=5).mean()
    df.drop(columns=["_pts_per_poss", "_ptsopp_per_poss"], inplace=True)

    # Días de descanso entre partidos consecutivos
    df["rest_days"] = df["GAME_DATE"].diff().dt.days.clip(0, 10).fillna(3.0)

    # Partidos jugados en los últimos 7 días antes de este partido
    dates = df["GAME_DATE"].values
    fatigue = []
    for i, d in enumerate(dates):
        cutoff = d - np.timedelta64(7, "D")
        fatigue.append(int(np.sum((dates[:i] > cutoff) & (dates[:i] < d))))
    df["fatigue_7d"] = fatigue

    # Back-to-back
    df["b2b"] = (df["rest_days"] == 1).astype(int)

    # Schedule strength: average opponent win rate over last 10 games
    # This requires opponent win rates which we'll compute in the main pipeline
    # For now, initialize with 0.5 (will be updated later)
    df["schedule_strength"] = 0.5

    return df


# ──────────────────────────────────────────────────────────────────
# 3. H2H ratio (simétrico — usa TODOS los enfrentamientos)
# ──────────────────────────────────────────────────────────────────
def compute_h2h(home_df: pd.DataFrame, all_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada partido del equipo local, calcula el ratio de victorias H2H
    contra ese rival en la misma temporada, ANTES de este partido.

    Usa all_feat (home + away rows) para contar TODOS los enfrentamientos
    entre ambos equipos (home + away), no solo los que jugó como local.
    """
    df = home_df.copy()
    df["away_abbr"] = df["MATCHUP"].str.extract(r"vs\. (\w+)")

    # Build lookup: all games with opponent extracted
    af = all_feat[["TEAM_ID", "GAME_DATE", "SEASON", "WIN", "MATCHUP"]].copy()
    af["OPP_ABBR"] = af["MATCHUP"].str.extract(r"(?:vs\.|@)\s*(\w+)")
    af = af.sort_values("GAME_DATE")

    # Group key: team vs opponent in same season
    af["h2h_key"] = (
        af["TEAM_ID"].astype(str) + "_" + af["OPP_ABBR"] + "_" + af["SEASON"]
    )
    # Cumulative wins and games BEFORE current game (shift within group)
    af["_cum_wins"]  = af.groupby("h2h_key")["WIN"].transform("cumsum")
    af["_prior_wins"] = af.groupby("h2h_key")["_cum_wins"].shift(1).fillna(0)
    af["_prior_games"] = af.groupby("h2h_key").cumcount()  # 0-indexed = games before

    af["_h2h_ratio"] = (
        af["_prior_wins"] / af["_prior_games"].clip(lower=1)
    ).where(af["_prior_games"] > 0, 0.5)

    # Now map back to home_df: for each home game, find the matching row in af
    # (same TEAM_ID, same GAME_DATE, same SEASON)
    df = df.sort_values("GAME_DATE")
    h2h_key_home = (
        df["TEAM_ID"].astype(str) + "_" + df["away_abbr"] + "_" + df["SEASON"]
    )

    # Build index from af for fast lookup
    af_indexed = af.set_index(["h2h_key", "GAME_DATE"])["_h2h_ratio"]
    # Remove duplicates keeping first (same team-opp-date shouldn't happen, but safety)
    af_indexed = af_indexed[~af_indexed.index.duplicated(keep="first")]

    h2h_ratios = []
    for key, gdate in zip(h2h_key_home, df["GAME_DATE"]):
        if pd.isna(key):
            h2h_ratios.append(0.5)
        elif (key, gdate) in af_indexed.index:
            h2h_ratios.append(af_indexed.loc[(key, gdate)])
        else:
            h2h_ratios.append(0.5)

    df["h2h_ratio"] = h2h_ratios
    return df


# ──────────────────────────────────────────────────────────────────
# 4. Schedule Strength calculation
# ──────────────────────────────────────────────────────────────────
def compute_schedule_strength(all_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el schedule strength para cada fila basándose en el win rate
    del oponente al momento del partido (usando datos previos).
    """
    print("Calculando schedule strength...")
    
    # Extract opponent abbreviation
    all_feat = all_feat.copy()
    all_feat["OPP_ABBR"] = all_feat["MATCHUP"].str.extract(r"(?:vs\.|@)\s*(\w+)")
    
    # Build a lookup of team season win rate at each date
    # Group by team+season, compute expanding win rate
    all_feat = all_feat.sort_values(["TEAM_ID", "SEASON", "GAME_DATE"])
    
    # For each team-season, compute their win rate up to each game date
    team_win_rates = {}
    for (team_id, season), grp in all_feat.groupby(["TEAM_ID", "SEASON"]):
        abbr = _get_team_abbr(team_id)
        grp = grp.sort_values("GAME_DATE")
        # Expanding win rate BEFORE each game
        expanding_wr = grp["WIN"].shift(1).expanding(min_periods=1).mean().fillna(0.5)
        for gdate, wr in zip(grp["GAME_DATE"], expanding_wr):
            team_win_rates[(abbr, season, gdate)] = wr
    
    # Now for each game, look up opponent's win rate at that date
    def get_opp_wr(row):
        opp = row["OPP_ABBR"]
        season = row["SEASON"]
        gdate = row["GAME_DATE"]
        return team_win_rates.get((opp, season, gdate), 0.5)
    
    all_feat["_opp_wr"] = all_feat.apply(get_opp_wr, axis=1)
    
    # Rolling average of opponent win rates (last 10 games)
    all_feat = all_feat.sort_values(["TEAM_ID", "SEASON", "GAME_DATE"])
    all_feat["schedule_strength"] = (
        all_feat.groupby(["TEAM_ID", "SEASON"])["_opp_wr"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
        .fillna(0.5)
    )
    
    all_feat.drop(columns=["_opp_wr", "OPP_ABBR"], inplace=True, errors="ignore")
    return all_feat


# ──────────────────────────────────────────────────────────────────
# 5. Travel Factor calculation
# ──────────────────────────────────────────────────────────────────
def compute_travel_factor(home_df: pd.DataFrame, away_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el travel factor para el equipo visitante en cada partido.
    """
    print("Calculando travel factor...")
    
    home_df = home_df.copy()
    
    # Get team abbreviations
    home_df["home_abbr"] = home_df["MATCHUP"].str.extract(r"^(\w+)\s+vs\.")
    home_df["away_abbr"] = home_df["MATCHUP"].str.extract(r"vs\.\s+(\w+)")
    
    def get_travel(row):
        home_abbr = row.get("home_abbr", "")
        away_abbr = row.get("away_abbr", "")
        
        home_full = ABBR_TO_FULL.get(home_abbr, "")
        away_full = ABBR_TO_FULL.get(away_abbr, "")
        
        if home_full and away_full:
            travel = calc_travel_factor(away_full, home_full)
            return travel.get("travel_factor", 0.0)
        return 0.0
    
    home_df["travel_factor"] = home_df.apply(get_travel, axis=1)
    home_df.drop(columns=["home_abbr", "away_abbr"], inplace=True, errors="ignore")
    
    return home_df


# ──────────────────────────────────────────────────────────────────
# 6. Pipeline principal
# ──────────────────────────────────────────────────────────────────
def build_dataset() -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)

    # ── 6a. Descarga ─────────────────────────────────────────────
    print("Descargando datos de la NBA API...")
    frames = [get_season_games(s) for s in SEASONS]
    all_games = pd.concat(frames, ignore_index=True)
    print(f"  Total filas (equipo × partido): {len(all_games)}")

    # ── 6b. Rolling features por equipo POR TEMPORADA ────────────
    #    (evita data leakage cross-season)
    print("\nCalculando features rolling...")
    team_frames = []
    for (team_id, season), tdf in all_games.groupby(["TEAM_ID", "SEASON"]):
        team_frames.append(compute_rolling_features(tdf))

    all_feat = pd.concat(team_frames, ignore_index=True)

    # ── 6c. Schedule strength ────────────────────────────────────
    all_feat = compute_schedule_strength(all_feat)

    # ── 6d. H2H (solo para filas de equipo local) ────────────────
    print("Calculando H2H histórico...")
    home_feat = all_feat[all_feat["IS_HOME"]].copy()
    home_feat = compute_h2h(home_feat, all_feat)

    # ── 6e. Travel factor ────────────────────────────────────────
    away_feat = all_feat[~all_feat["IS_HOME"]].copy()
    home_feat = compute_travel_factor(home_feat, away_feat)

    # ── 6f. Merge local ↔ visitante ──────────────────────────────
    home_sel = home_feat[[
        "GAME_ID", "WIN", "h2h_ratio", "travel_factor", "SEASON",
    ] + ROLL_COLS].copy()
    home_sel.columns = (
        ["game_id", "home_win", "h2h_ratio", "travel_factor", "season"]
        + [f"home_{c}" for c in ROLL_COLS]
    )

    away_sel = away_feat[["GAME_ID"] + ROLL_COLS].copy()
    away_sel.columns = ["game_id"] + [f"away_{c}" for c in ROLL_COLS]

    dataset = home_sel.merge(away_sel, on="game_id").dropna()

    print(f"\nPartidos en el dataset (tras dropna): {len(dataset)}")
    print(f"Distribución por temporada:\n{dataset['season'].value_counts().to_string()}")
    print(f"Win rate local histórico: {dataset['home_win'].mean():.3f}")

    out = os.path.join(DATA_DIR, "dataset.csv")
    dataset.to_csv(out, index=False)
    print(f"\nDataset guardado en: {out}")

    return dataset


if __name__ == "__main__":
    print("=" * 56)
    print("  NBA PREDICTOR — Dataset histórico (3 temporadas)")
    print("=" * 56 + "\n")
    build_dataset()

"""
NBA Predictor — Vegas Odds
Obtiene líneas de apuestas desde The Odds API.

Requiere API key de https://the-odds-api.com (tier gratis: 500 req/mes)
Configurar en .env o variable de entorno: ODDS_API_KEY
"""

import os
import requests
from datetime import datetime, timedelta
from pathlib import Path

# Cargar .env si existe
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"

# Cache para no repetir llamadas
_odds_cache: dict = {}


def get_nba_odds() -> dict:
    """
    Obtiene las líneas de apuestas actuales para todos los partidos NBA.
    
    Returns:
        dict[game_key] = {
            "home_team": str,
            "away_team": str,
            "commence_time": str,
            "home_odds": float,  # decimal odds
            "away_odds": float,
            "home_implied_prob": float,  # probabilidad implícita
            "away_implied_prob": float,
            "spread_home": float,  # spread del local
            "total": float,  # over/under
        }
        
        game_key = "away_team @ home_team"
    """
    if not ODDS_API_KEY:
        return {}
    
    cache_key = datetime.now().strftime("%Y-%m-%d-%H")
    if cache_key in _odds_cache:
        return _odds_cache[cache_key]
    
    try:
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "decimal",
        }
        resp = requests.get(ODDS_API_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"    Vegas odds no disponible: {e}")
        return {}
    
    result = {}
    for game in data:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        game_key = f"{away} @ {home}"
        
        # Find best odds from bookmakers
        home_odds = None
        away_odds = None
        spread_home = None
        total = None
        
        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] == "h2h":
                    for outcome in market["outcomes"]:
                        if outcome["name"] == home:
                            home_odds = outcome["price"]
                        elif outcome["name"] == away:
                            away_odds = outcome["price"]
                elif market["key"] == "spreads":
                    for outcome in market["outcomes"]:
                        if outcome["name"] == home:
                            spread_home = outcome.get("point", 0)
                elif market["key"] == "totals":
                    for outcome in market["outcomes"]:
                        if outcome["name"] == "Over":
                            total = outcome.get("point", 0)
            
            if home_odds and away_odds:
                break
        
        if home_odds and away_odds:
            # Calcular probabilidad implícita (con vigorish)
            home_implied = 1 / home_odds
            away_implied = 1 / away_odds
            # Normalizar para quitar el vig
            total_implied = home_implied + away_implied
            home_prob = home_implied / total_implied
            away_prob = away_implied / total_implied
            
            result[game_key] = {
                "home_team": home,
                "away_team": away,
                "commence_time": game.get("commence_time", ""),
                "home_odds": home_odds,
                "away_odds": away_odds,
                "home_implied_prob": round(home_prob, 3),
                "away_implied_prob": round(away_prob, 3),
                "spread_home": spread_home,
                "total": total,
            }
    
    _odds_cache[cache_key] = result
    return result


def get_game_odds(home_team: str, away_team: str) -> dict:
    """
    Obtiene las odds para un partido específico.
    
    Args:
        home_team: Nombre del equipo local (ej. "Los Angeles Lakers")
        away_team: Nombre del equipo visitante (ej. "Golden State Warriors")
    
    Returns:
        {
            "home_implied_prob": float,
            "away_implied_prob": float,
            "spread_home": float,
            "total": float,
            "available": bool,
        }
    """
    all_odds = get_nba_odds()
    
    if not all_odds:
        return {
            "home_implied_prob": 0.5,
            "away_implied_prob": 0.5,
            "spread_home": 0.0,
            "total": 220.0,
            "available": False,
        }
    
    # Buscar el partido por nombres (fuzzy match)
    home_lower = home_team.lower()
    away_lower = away_team.lower()
    
    for game_key, odds in all_odds.items():
        game_home = odds["home_team"].lower()
        game_away = odds["away_team"].lower()
        
        # Check if teams match (partial match)
        home_match = any(w in game_home for w in home_lower.split()) or \
                     any(w in home_lower for w in game_home.split())
        away_match = any(w in game_away for w in away_lower.split()) or \
                     any(w in away_lower for w in game_away.split())
        
        if home_match and away_match:
            return {
                "home_implied_prob": odds["home_implied_prob"],
                "away_implied_prob": odds["away_implied_prob"],
                "spread_home": odds.get("spread_home", 0.0),
                "total": odds.get("total", 220.0),
                "available": True,
            }
    
    return {
        "home_implied_prob": 0.5,
        "away_implied_prob": 0.5,
        "spread_home": 0.0,
        "total": 220.0,
        "available": False,
    }


def odds_available() -> bool:
    """Check if odds API is configured."""
    return bool(ODDS_API_KEY)

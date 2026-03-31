"""
Datos de ubicación de equipos NBA para calcular travel_factor.
Incluye timezone, coordenadas y altitud.
"""

# Timezone offsets desde UTC (hora estándar, sin DST)
# Eastern = -5, Central = -6, Mountain = -7, Pacific = -8
TEAM_DATA = {
    # Eastern Conference
    "Atlanta Hawks":          {"tz": -5, "lat": 33.757, "lon": -84.396, "altitude": 320},
    "Boston Celtics":         {"tz": -5, "lat": 42.366, "lon": -71.062, "altitude": 6},
    "Brooklyn Nets":          {"tz": -5, "lat": 40.683, "lon": -73.976, "altitude": 10},
    "Charlotte Hornets":      {"tz": -5, "lat": 35.225, "lon": -80.839, "altitude": 229},
    "Chicago Bulls":          {"tz": -6, "lat": 41.881, "lon": -87.674, "altitude": 182},
    "Cleveland Cavaliers":    {"tz": -5, "lat": 41.496, "lon": -81.688, "altitude": 200},
    "Detroit Pistons":        {"tz": -5, "lat": 42.341, "lon": -83.055, "altitude": 183},
    "Indiana Pacers":         {"tz": -5, "lat": 39.764, "lon": -86.156, "altitude": 218},
    "Miami Heat":             {"tz": -5, "lat": 25.781, "lon": -80.187, "altitude": 2},
    "Milwaukee Bucks":        {"tz": -6, "lat": 43.045, "lon": -87.917, "altitude": 188},
    "New York Knicks":        {"tz": -5, "lat": 40.751, "lon": -73.994, "altitude": 10},
    "Orlando Magic":          {"tz": -5, "lat": 28.539, "lon": -81.384, "altitude": 25},
    "Philadelphia 76ers":     {"tz": -5, "lat": 39.901, "lon": -75.172, "altitude": 12},
    "Toronto Raptors":        {"tz": -5, "lat": 43.643, "lon": -79.379, "altitude": 76},
    "Washington Wizards":     {"tz": -5, "lat": 38.898, "lon": -77.021, "altitude": 22},
    
    # Western Conference
    "Dallas Mavericks":       {"tz": -6, "lat": 32.790, "lon": -96.810, "altitude": 131},
    "Denver Nuggets":         {"tz": -7, "lat": 39.749, "lon": -105.008, "altitude": 1609},
    "Golden State Warriors":  {"tz": -8, "lat": 37.768, "lon": -122.388, "altitude": 2},
    "Houston Rockets":        {"tz": -6, "lat": 29.751, "lon": -95.362, "altitude": 15},
    "Los Angeles Clippers":   {"tz": -8, "lat": 34.043, "lon": -118.267, "altitude": 71},
    "Los Angeles Lakers":     {"tz": -8, "lat": 34.043, "lon": -118.267, "altitude": 71},
    "Memphis Grizzlies":      {"tz": -6, "lat": 35.138, "lon": -90.051, "altitude": 102},
    "Minnesota Timberwolves": {"tz": -6, "lat": 44.980, "lon": -93.276, "altitude": 256},
    "New Orleans Pelicans":   {"tz": -6, "lat": 29.949, "lon": -90.082, "altitude": 2},
    "Oklahoma City Thunder":  {"tz": -6, "lat": 35.463, "lon": -97.515, "altitude": 390},
    "Phoenix Suns":           {"tz": -7, "lat": 33.446, "lon": -112.071, "altitude": 331},
    "Portland Trail Blazers": {"tz": -8, "lat": 45.532, "lon": -122.667, "altitude": 15},
    "Sacramento Kings":       {"tz": -8, "lat": 38.580, "lon": -121.500, "altitude": 9},
    "San Antonio Spurs":      {"tz": -6, "lat": 29.427, "lon": -98.438, "altitude": 198},
    "Utah Jazz":              {"tz": -7, "lat": 40.768, "lon": -111.901, "altitude": 1288},
}

# Abbreviation to full name mapping
ABBR_TO_FULL = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}

FULL_TO_ABBR = {v: k for k, v in ABBR_TO_FULL.items()}


def get_team_data(team_name: str) -> dict:
    """
    Obtiene datos de ubicación de un equipo.
    Acepta nombre completo o abreviatura.
    """
    # Try full name first
    if team_name in TEAM_DATA:
        return TEAM_DATA[team_name]
    
    # Try abbreviation
    if team_name in ABBR_TO_FULL:
        return TEAM_DATA[ABBR_TO_FULL[team_name]]
    
    # Try partial match
    name_lower = team_name.lower()
    for full_name, data in TEAM_DATA.items():
        if name_lower in full_name.lower():
            return data
    
    return None


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula distancia en millas entre dos puntos."""
    import math
    R = 3959  # Radio de la Tierra en millas
    
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


def calc_travel_factor(from_team: str, to_team: str) -> dict:
    """
    Calcula el factor de viaje entre dos equipos.
    
    Returns:
        {
            "distance_miles": float,
            "timezone_shift": int,  # horas (positivo = hacia oeste)
            "altitude_change": int,  # metros
            "travel_factor": float,  # 0-1, mayor = más cansancio
        }
    """
    from_data = get_team_data(from_team)
    to_data = get_team_data(to_team)
    
    if not from_data or not to_data:
        return {
            "distance_miles": 0,
            "timezone_shift": 0,
            "altitude_change": 0,
            "travel_factor": 0.0,
        }
    
    distance = haversine_miles(
        from_data["lat"], from_data["lon"],
        to_data["lat"], to_data["lon"]
    )
    
    tz_shift = from_data["tz"] - to_data["tz"]  # positivo = viajando al oeste
    altitude_change = to_data["altitude"] - from_data["altitude"]
    
    # Factor de viaje normalizado (0-1)
    # Componentes:
    #   - Distancia: >2000 millas es máximo impacto
    #   - Timezone: cada hora de shift = impacto
    #   - Altitud: Denver (1609m) es caso extremo
    
    dist_factor = min(distance / 2500, 1.0) * 0.4
    tz_factor = min(abs(tz_shift) / 3, 1.0) * 0.35
    alt_factor = min(max(altitude_change, 0) / 1600, 1.0) * 0.25
    
    travel_factor = dist_factor + tz_factor + alt_factor
    
    return {
        "distance_miles": round(distance),
        "timezone_shift": tz_shift,
        "altitude_change": altitude_change,
        "travel_factor": round(travel_factor, 3),
    }


def is_denver_home(team_name: str) -> bool:
    """Check if team is Denver (altitude advantage)."""
    return "nuggets" in team_name.lower() or "denver" in team_name.lower()

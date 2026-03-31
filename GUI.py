"""
NBA Predictor — Interfaz Gráfica
Muestra los partidos del día y calcula predicciones con un clic.

Uso: python GUI.py
"""

import sys
import os
import threading
import queue
import time
from datetime import date, datetime

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ── Path setup ────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "Servicios"))

from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import teams as nba_teams
from GetTier1 import get_tier1
from GetTier2 import get_tier2
from GetTier3 import get_tier3

try:
    from Predict import predecir
    ML_DISPONIBLE = True
except Exception:
    ML_DISPONIBLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.colors import HexColor, white, black
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table,
        TableStyle, HRFlowable, KeepTogether,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    PDF_DISPONIBLE = True
except ImportError:
    PDF_DISPONIBLE = False

MODEL_PATH = os.path.join(ROOT_DIR, "data", "model.pkl")
SEASON     = "2025-26"

# ── Algoritmo de distribución de presupuesto ─────────────────────
def calculate_budget_allocation(predictions: dict, budget: float) -> dict:
    """
    Distribuye el presupuesto entre los partidos según confianza.
    
    Reglas:
    - Solo apuesta en partidos donde apostar=True
    - Peso por nivel: FUERTE=3, MODERADA=2, BAJA=1
    - Distribución proporcional al peso * confianza
    
    Returns:
        {game_id: {"team": str, "amount": float, "nivel": str, "confianza": float}}
    """
    WEIGHTS = {"FUERTE": 3.0, "MODERADA": 2.0, "BAJA": 1.0}
    
    # Filtrar solo partidos apostables
    apostables = []
    for gid, pred in predictions.items():
        if pred.get("apostar", False):
            nivel = pred.get("nivel_apuesta", "BAJA")
            confianza = pred.get("confianza", 50) / 100.0
            peso = WEIGHTS.get(nivel, 1.0) * confianza
            apostables.append({
                "gid": gid,
                "team": pred["ganador"],
                "nivel": nivel,
                "confianza": pred.get("confianza", 50),
                "peso": peso,
                "prob": pred["probabilidad"],
            })
    
    if not apostables:
        return {}
    
    # Distribuir proporcionalmente
    total_peso = sum(a["peso"] for a in apostables)
    
    result = {}
    for ap in apostables:
        proporcion = ap["peso"] / total_peso
        monto = round(budget * proporcion, 2)
        result[ap["gid"]] = {
            "team": ap["team"],
            "amount": monto,
            "nivel": ap["nivel"],
            "confianza": ap["confianza"],
            "prob": ap["prob"],
        }
    
    return result

# ── Paleta UI ─────────────────────────────────────────────────────
BG      = "#0d1117"
CARD    = "#161b22"
BORDER  = "#30363d"
TEXT    = "#e6edf3"
DIM     = "#8b949e"
ACCENT  = "#58a6ff"
GREEN   = "#3fb950"
RED     = "#f85149"
YELLOW  = "#d29922"
BTN     = "#21262d"
BTN_HOV = "#30363d"
WARN_BG = "#2d2208"
WARN_FG = "#d29922"
ERR_BG  = "#2d0f0f"
ERR_FG  = "#f85149"

# ── Paleta PDF ────────────────────────────────────────────────────
PDF_DARK    = HexColor("#0d1117")
PDF_CARD    = HexColor("#f6f8fa")
PDF_BORDER  = HexColor("#d0d7de")
PDF_TEXT    = HexColor("#24292f")
PDF_DIM     = HexColor("#656d76")
PDF_ACCENT  = HexColor("#0969da")
PDF_GREEN   = HexColor("#1a7f37")
PDF_RED     = HexColor("#cf222e")
PDF_GREEN_L = HexColor("#dafbe1")
PDF_RED_L   = HexColor("#ffebe9")
PDF_BAR_G   = HexColor("#3fb950")
PDF_BAR_R   = HexColor("#f85149")


# ══════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════
def get_today_games() -> list[dict]:
    today = date.today().strftime("%m/%d/%Y")
    sb    = scoreboardv2.ScoreboardV2(game_date=today)
    df    = sb.game_header.get_data_frame()
    id2n  = {t["id"]: t["full_name"] for t in nba_teams.get_teams()}
    return [
        {
            "game_id": row["GAME_ID"],
            "home":    id2n.get(int(row["HOME_TEAM_ID"]),    "?"),
            "away":    id2n.get(int(row["VISITOR_TEAM_ID"]), "?"),
            "status":  str(row.get("GAME_STATUS_TEXT", "")).strip(),
        }
        for _, row in df.iterrows()
    ]


def run_prediction(home: str, away: str) -> dict:
    t1 = get_tier1(home, away, equipo_a_es_local=True, season=SEASON)
    t2 = get_tier2(home, away, season=SEASON)
    t3 = get_tier3(home, away, season=SEASON)
    if ML_DISPONIBLE:
        return predecir(t1, t2, t3, home, away, equipo_a_es_local=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "main", os.path.join(ROOT_DIR, "main.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.calcular_prediccion(t1, t2, t3, home, away, True)


def model_age_info() -> tuple[str, str]:
    """
    Retorna (mensaje, nivel) donde nivel es 'ok' | 'warn' | 'error'.
    """
    if not os.path.exists(MODEL_PATH):
        return ("Modelo no entrenado — usando scoring manual. "
                "Ejecuta BuildDataset.py y TrainModel.py.", "error")
    age_days = int((time.time() - os.path.getmtime(MODEL_PATH)) / 86400)
    trained  = datetime.fromtimestamp(
        os.path.getmtime(MODEL_PATH)).strftime("%d/%m/%Y")
    if age_days == 0:
        return (f"Modelo entrenado hoy ({trained})", "ok")
    if age_days <= 30:
        return (f"Modelo entrenado hace {age_days} días ({trained})", "ok")
    return (
        f"Modelo entrenado hace {age_days} días ({trained}) — reentrenamiento recomendado.",
        "warn"
    )


def nick(name: str) -> str:
    parts = name.split()
    return " ".join(parts[-2:]) if len(parts) > 2 else parts[-1]


# ══════════════════════════════════════════════════════════════════
# GENERADOR DE PDF
# ══════════════════════════════════════════════════════════════════
def generate_pdf(filepath: str, games: list[dict], predictions: dict[str, dict]):
    if not PDF_DISPONIBLE:
        raise ImportError("reportlab no instalado: pip install reportlab")

    doc = SimpleDocTemplate(
        filepath, pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=1.5*cm, bottomMargin=2*cm,
    )
    W = A4[0] - 3.6*cm   # ancho útil

    # ── Estilos ───────────────────────────────────────────────────
    styles = getSampleStyleSheet()

    def style(name, parent="Normal", **kw):
        s = ParagraphStyle(name, parent=styles[parent], **kw)
        return s

    s_title   = style("title",   fontSize=20, textColor=white,
                      fontName="Helvetica-Bold", spaceAfter=2)
    s_sub     = style("sub",     fontSize=10, textColor=HexColor("#8b949e"),
                      fontName="Helvetica")
    s_team    = style("team",    fontSize=14, textColor=PDF_TEXT,
                      fontName="Helvetica-Bold")
    s_winner  = style("winner",  fontSize=13, fontName="Helvetica-Bold")
    s_label   = style("label",   fontSize=8,  textColor=PDF_DIM,
                      fontName="Helvetica")
    s_cell    = style("cell",    fontSize=9,  textColor=PDF_TEXT,
                      fontName="Helvetica", alignment=TA_CENTER)
    s_footer  = style("footer",  fontSize=8,  textColor=PDF_DIM,
                      fontName="Helvetica", alignment=TA_CENTER)

    story = []

    # ── Cabecera del documento ────────────────────────────────────
    today_str = date.today().strftime("%A %d de %B de %Y").title()
    hdr_data  = [[
        Paragraph("NBA PREDICTOR", s_title),
        Paragraph(today_str, style("hdr_date", fontSize=9,
                                   textColor=HexColor("#8b949e"),
                                   fontName="Helvetica",
                                   alignment=TA_RIGHT)),
    ]]
    hdr_tbl = Table(hdr_data, colWidths=[W * 0.6, W * 0.4])
    hdr_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), PDF_DARK),
        ("TOPPADDING",  (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("LEFTPADDING",  (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 16),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("ROUNDEDCORNERS", [6]),
    ]))
    story.append(hdr_tbl)
    story.append(Spacer(1, 0.5*cm))

    # Subtítulo
    n_games = len(predictions)
    story.append(Paragraph(
        f"{n_games} predicción{'es' if n_games != 1 else ''}  ·  "
        f"Temporada {SEASON}  ·  "
        f"{'Modelo ML' if ML_DISPONIBLE else 'Scoring ponderado'}",
        style("intro", fontSize=9, textColor=PDF_DIM,
              fontName="Helvetica", spaceAfter=4)
    ))
    story.append(HRFlowable(width=W, thickness=1,
                             color=PDF_BORDER, spaceAfter=12))

    # ── Una tarjeta por partido ───────────────────────────────────
    for g in games:
        gid  = g["game_id"]
        pred = predictions.get(gid)
        if pred is None:
            continue

        home    = g["home"]
        away    = g["away"]
        ganador = pred["ganador"]
        ph      = pred["probabilidad"][home]
        pa      = pred["probabilidad"][away]
        f       = pred["factores"]

        es_local   = (ganador == home)
        win_color  = PDF_GREEN  if es_local else PDF_RED
        win_bg     = PDF_GREEN_L if es_local else PDF_RED_L

        # ── Encabezado del partido ────────────────────────────────
        game_hdr = Table(
            [[
                Paragraph(f"{away}", s_team),
                Paragraph("@", style("at", fontSize=12, textColor=PDF_DIM,
                                     fontName="Helvetica", alignment=TA_CENTER)),
                Paragraph(f"{home}", s_team),
                Paragraph(g["status"], style("st", fontSize=8,
                                             textColor=PDF_DIM,
                                             fontName="Helvetica",
                                             alignment=TA_RIGHT)),
            ]],
            colWidths=[W*0.38, W*0.08, W*0.38, W*0.16],
        )
        game_hdr.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), PDF_CARD),
            ("TOPPADDING",    (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING",   (0, 0), (-1, -1), 12),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("LINEBELOW",     (0, 0), (-1, -1), 0.5, PDF_BORDER),
        ]))

        # ── Ganador + probabilidad ────────────────────────────────
        arrow = "▲" if es_local else "▼"
        win_row = Table(
            [[
                Paragraph(
                    f'<font color="#{("1a7f37" if es_local else "cf222e")}">'
                    f'{arrow}  {ganador}</font>',
                    style("wr", fontSize=13, fontName="Helvetica-Bold",
                          textColor=win_color)
                ),
                Paragraph(
                    f"{nick(home)} {ph}%  /  {nick(away)} {pa}%",
                    style("pcts", fontSize=9, textColor=PDF_DIM,
                          fontName="Helvetica", alignment=TA_RIGHT)
                ),
            ]],
            colWidths=[W * 0.6, W * 0.4],
        )
        win_row.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), win_bg),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING",   (0, 0), (-1, -1), 12),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]))

        # ── Barra de probabilidad ─────────────────────────────────
        bar_w_home = max(0.5, W * ph / 100)
        bar_w_away = max(0.5, W * pa / 100)
        bar = Table(
            [["", ""]],
            colWidths=[bar_w_home, bar_w_away],
            rowHeights=[0.35*cm],
        )
        bar.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), PDF_BAR_G),
            ("BACKGROUND", (1, 0), (1, 0), PDF_BAR_R),
            ("LINEABOVE",  (0, 0), (-1, -1), 0, white),
        ]))

        # ── Recomendación de apuesta ─────────────────────────────
        apostar = pred.get("apostar", False)
        nivel = pred.get("nivel_apuesta", "")
        if apostar:
            if nivel == "FUERTE":
                bet_text = "APOSTAR — confianza alta"
                bet_color = HexColor("#22c55e")
            elif nivel == "MODERADA":
                bet_text = "APOSTAR — confianza moderada"
                bet_color = HexColor("#3b82f6")
            else:
                bet_text = "Apostar con cautela"
                bet_color = HexColor("#eab308")
        else:
            bet_text = "NO APOSTAR — muy parejo"
            bet_color = HexColor("#ef4444")

        bet_row = Paragraph(
            bet_text,
            style("bet", fontSize=10, fontName="Helvetica-Bold",
                  textColor=bet_color, alignment=TA_CENTER)
        )

        # ── Tabla de factores ─────────────────────────────────────
        def fval(key, suffix="", fmt=None):
            """Extrae A y B de un factor del dict."""
            d = f.get(key, {})
            if not isinstance(d, dict):
                return "—", "—"
            va, vb = d.get("A", "—"), d.get("B", "—")
            if fmt and va != "—":
                va = fmt.format(va)
            if fmt and vb != "—":
                vb = fmt.format(vb)
            return f"{va}{suffix}", f"{vb}{suffix}"

        nr_a,  nr_b  = fval("net_rating",      fmt="{:+.1f}")
        wr_a,  wr_b  = fval("win_rate_10j",    suffix="%")
        wr5_a, wr5_b = fval("win_rate_5j",     suffix="%")
        pd_a,  pd_b  = fval("point_diff",      fmt="{:+.1f}")
        fg_a,  fg_b  = fval("fg_pct_10j",      suffix="%")
        pt_a,  pt_b  = fval("pts_permitidos")
        il_a,  il_b  = fval("impacto_lesiones", suffix="%")
        ds_a,  ds_b  = fval("dias_descanso",    suffix="d")
        ft_a,  ft_b  = f.get("fatiga_7d", {}).get("A", "—"), \
                       f.get("fatiga_7d", {}).get("B", "—")

        rows = [
            ["ESTADÍSTICA",        nick(away), nick(home)],
            ["Net Rating (temp.)", nr_a, nr_b],
            ["Win rate 10j",       wr_a, wr_b],
            ["Win rate 5j",        wr5_a, wr5_b],
            ["Point diff. (10j)",  pd_a, pd_b],
            ["FG% (10j)",          fg_a, fg_b],
            ["Pts permitidos",     pt_a, pt_b],
            ["Impacto lesiones",   il_a, il_b],
            ["Días descanso",      ds_a, ds_b],
            ["Fatiga 7d",          ft_a, ft_b],
        ]

        col_w = [W * 0.46, W * 0.27, W * 0.27]
        stats_tbl = Table(rows, colWidths=col_w, repeatRows=1)
        ts = TableStyle([
            # Header row
            ("BACKGROUND",    (0, 0), (-1, 0), PDF_DARK),
            ("TEXTCOLOR",     (0, 0), (-1, 0), white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0), 8),
            ("ALIGN",         (0, 0), (-1, 0), "CENTER"),
            # Data rows
            ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 1), (-1, -1), 8),
            ("TEXTCOLOR",     (0, 1), (-1, -1), PDF_TEXT),
            ("ALIGN",         (1, 1), (-1, -1), "CENTER"),
            ("ALIGN",         (0, 1), (0, -1),  "LEFT"),
            # Alternating rows
            *[("BACKGROUND", (0, i), (-1, i), PDF_CARD)
              for i in range(2, len(rows), 2)],
            # Padding
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            # Grid
            ("GRID",          (0, 0), (-1, -1), 0.4, PDF_BORDER),
        ])
        stats_tbl.setStyle(ts)

        story.append(KeepTogether([
            game_hdr,
            win_row,
            bar,
            Spacer(1, 0.15*cm),
            bet_row,
            Spacer(1, 0.25*cm),
            stats_tbl,
            Spacer(1, 0.6*cm),
        ]))

    # ── Footer ────────────────────────────────────────────────────
    story.append(HRFlowable(width=W, thickness=0.5, color=PDF_BORDER,
                            spaceBefore=4, spaceAfter=6))
    modelo_info = ""
    if ML_DISPONIBLE and predictions:
        first = next(iter(predictions.values()), {})
        modelo_info = (f"Modelo: {first.get('modelo', 'ML')}  ·  "
                       f"AUC {first.get('modelo_auc', 0):.3f}  ·  ")
    generated = datetime.now().strftime("%d/%m/%Y %H:%M")
    story.append(Paragraph(
        f"{modelo_info}Generado el {generated}  ·  NBA Predictor",
        s_footer
    ))

    doc.build(story)


# ══════════════════════════════════════════════════════════════════
# WIDGET: TARJETA DE PARTIDO
# ══════════════════════════════════════════════════════════════════
class GameCard(tk.Frame):

    def __init__(self, parent, home: str, away: str, status: str):
        super().__init__(parent, bg=CARD,
                         highlightbackground=BORDER, highlightthickness=1)
        self.home = home
        self.away = away
        self._build(status)

    def _build(self, status: str):
        p = dict(padx=16)

        tk.Label(self, text=status, fg=DIM, bg=CARD,
                 font=("Helvetica", 9)).pack(anchor="e", **p, pady=(8, 2))

        row = tk.Frame(self, bg=CARD)
        row.pack(fill="x", **p, pady=2)
        tk.Label(row, text=nick(self.away), fg=TEXT, bg=CARD,
                 font=("Helvetica", 15, "bold")).pack(side="left")
        tk.Label(row, text="  @  ", fg=DIM, bg=CARD,
                 font=("Helvetica", 12)).pack(side="left")
        tk.Label(row, text=nick(self.home), fg=TEXT, bg=CARD,
                 font=("Helvetica", 15, "bold")).pack(side="left")

        tk.Label(self, text=f"{self.away}  vs  {self.home}",
                 fg=DIM, bg=CARD, font=("Helvetica", 9)).pack(
            anchor="w", **p, pady=(0, 6))

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", **p)

        res = tk.Frame(self, bg=CARD)
        res.pack(fill="x", **p, pady=8)

        self._res_var = tk.StringVar(value="—")
        self._res_lbl = tk.Label(res, textvariable=self._res_var,
                                 fg=DIM, bg=CARD,
                                 font=("Helvetica", 12, "bold"), anchor="w")
        self._res_lbl.pack(side="left", fill="x", expand=True)

        self._pct_lbl = tk.Label(res, text="", fg=DIM, bg=CARD,
                                 font=("Helvetica", 11))
        self._pct_lbl.pack(side="right")

        # Recomendación de apuesta
        self._bet_lbl = tk.Label(self, text="", fg=DIM, bg=CARD,
                                 font=("Helvetica", 10, "bold"))
        self._bet_lbl.pack(anchor="w", padx=16, pady=(0, 4))
        
        # Monto recomendado (se actualiza después)
        self._amount_lbl = tk.Label(self, text="", fg=ACCENT, bg=CARD,
                                    font=("Helvetica", 12, "bold"))
        self._amount_lbl.pack(anchor="w", padx=16, pady=(0, 4))

        self._bar = tk.Canvas(self, height=5, bg=CARD, highlightthickness=0)
        self._bar.pack(fill="x", padx=16, pady=(0, 12))

    def set_loading(self):
        self._res_var.set("Obteniendo datos...")
        self._res_lbl.config(fg=YELLOW)
        self._pct_lbl.config(text="")
        self._bet_lbl.config(text="")
        self._amount_lbl.config(text="")
        self._bar.delete("all")
    
    def set_bet_amount(self, amount: float, team: str):
        """Muestra el monto recomendado de apuesta."""
        if amount > 0:
            self._amount_lbl.config(
                text=f"Apostar ${amount:.2f} a {nick(team)}",
                fg="#22c55e"
            )
        else:
            self._amount_lbl.config(text="")

    def set_result(self, pred: dict):
        ganador  = pred["ganador"]
        ph = pred["probabilidad"][self.home]
        pa = pred["probabilidad"][self.away]
        color = GREEN if ganador == self.home else RED
        arrow = "▲" if ganador == self.home else "▼"
        self._res_var.set(f"{arrow}  {nick(ganador)}")
        self._res_lbl.config(fg=color)
        self._pct_lbl.config(
            text=f"{nick(self.home)} {ph}%  /  {nick(self.away)} {pa}%",
            fg=DIM)

        # Recomendación de apuesta
        apostar = pred.get("apostar", False)
        nivel = pred.get("nivel_apuesta", "")
        if apostar:
            if nivel == "FUERTE":
                self._bet_lbl.config(text="APOSTAR — confianza alta", fg="#22c55e")
            elif nivel == "MODERADA":
                self._bet_lbl.config(text="APOSTAR — confianza moderada", fg="#3b82f6")
            else:  # BAJA
                self._bet_lbl.config(text="Apostar con cautela", fg=YELLOW)
        else:
            self._bet_lbl.config(text="NO APOSTAR — muy parejo", fg="#ef4444")

        self._draw_bar(ph)

    def set_error(self):
        self._res_var.set("Error al obtener datos")
        self._res_lbl.config(fg=RED)

    def _draw_bar(self, pct_home: float):
        self._bar.update_idletasks()
        w = self._bar.winfo_width() or 400
        split = int(w * pct_home / 100)
        self._bar.delete("all")
        if split > 0:
            self._bar.create_rectangle(0, 0, split, 5, fill=GREEN, outline="")
        if split < w:
            self._bar.create_rectangle(split, 0, w, 5, fill=RED, outline="")


# ══════════════════════════════════════════════════════════════════
# APLICACIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════
class NBAApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("NBA Predictor")
        self.configure(bg=BG)
        self.geometry("820x660")
        self.minsize(640, 460)

        self._q:           queue.Queue           = queue.Queue()
        self._cards:       dict[str, GameCard]   = {}
        self._games:       list[dict]            = []
        self._predictions: dict[str, dict]       = {}
        self._budget_allocation: dict            = {}

        self._build_ui()
        self.after(100, self._check_queue)
        self._fetch_games()

    # ── Construcción de la UI ─────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=24, pady=(20, 0))

        tk.Label(hdr, text="NBA Predictor", fg=TEXT, bg=BG,
                 font=("Helvetica", 22, "bold")).pack(side="left")
        tk.Label(hdr, text=date.today().strftime("%A %d %b %Y").title(),
                 fg=DIM, bg=BG,
                 font=("Helvetica", 12)).pack(side="left", padx=14, pady=5)
        tk.Label(hdr,
                 text=("  ML activo  " if ML_DISPONIBLE else "  scoring manual  "),
                 fg=ACCENT if ML_DISPONIBLE else YELLOW,
                 bg=BG, font=("Helvetica", 9)).pack(side="left")

        # ── Input de presupuesto ────────────────────────────────────
        budget_frame = tk.Frame(hdr, bg=BG)
        budget_frame.pack(side="left", padx=20)
        
        tk.Label(budget_frame, text="💰 Presupuesto:", fg=DIM, bg=BG,
                 font=("Helvetica", 10)).pack(side="left")
        
        self._budget_var = tk.StringVar(value="100")
        self._budget_entry = tk.Entry(
            budget_frame, textvariable=self._budget_var,
            width=8, bg=CARD, fg=TEXT, insertbackground=TEXT,
            font=("Helvetica", 11), relief="flat",
            highlightbackground=BORDER, highlightthickness=1,
        )
        self._budget_entry.pack(side="left", padx=4, ipady=4)
        
        tk.Label(budget_frame, text="$", fg=DIM, bg=BG,
                 font=("Helvetica", 10)).pack(side="left")

        # Botón exportar PDF (oculto hasta tener predicciones)
        self._pdf_btn = tk.Button(
            hdr, text="  Exportar PDF  ",
            fg=TEXT, bg="#1c3a1c",
            activebackground="#2d5a2d", activeforeground=TEXT,
            relief="flat", cursor="hand2", font=("Helvetica", 11),
            command=self._export_pdf, state="disabled",
        )
        self._pdf_btn.pack(side="right", ipady=7, ipadx=6, padx=(8, 0))

        self._btn = tk.Button(
            hdr, text="  Calcular predicciones  ",
            fg=TEXT, bg=BTN,
            activebackground=BTN_HOV, activeforeground=TEXT,
            relief="flat", cursor="hand2", font=("Helvetica", 11),
            command=self._start_predictions, state="disabled",
        )
        self._btn.pack(side="right", ipady=7, ipadx=6)

        # Separador
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=24, pady=10)

        # ── Banner de estado del modelo ───────────────────────────
        msg, level = model_age_info()
        if level != "ok":
            banner_bg = WARN_BG if level == "warn" else ERR_BG
            banner_fg = WARN_FG if level == "warn" else ERR_FG
            banner = tk.Frame(self, bg=banner_bg,
                              highlightbackground=banner_fg,
                              highlightthickness=1)
            banner.pack(fill="x", padx=24, pady=(0, 6))
            tk.Label(banner, text=msg, fg=banner_fg, bg=banner_bg,
                     font=("Helvetica", 9), anchor="w").pack(
                anchor="w", padx=12, pady=5)

        # Barra de estado
        self._status_var = tk.StringVar(value="Cargando partidos del día...")
        tk.Label(self, textvariable=self._status_var,
                 fg=DIM, bg=BG, font=("Helvetica", 10)).pack(
            anchor="w", padx=24, pady=(4, 6))

        # Contenedor scrollable
        outer = tk.Frame(self, bg=BG)
        outer.pack(fill="both", expand=True, padx=24, pady=(0, 16))

        self._canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=self._canvas.yview)

        self._inner = tk.Frame(self._canvas, bg=BG)
        self._inner.bind(
            "<Configure>",
            lambda e: self._canvas.configure(
                scrollregion=self._canvas.bbox("all")))

        self._canvas.create_window((0, 0), window=self._inner, anchor="nw")
        self._canvas.configure(yscrollcommand=sb.set)
        self._canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        for seq, fn in (
            ("<MouseWheel>", lambda e: self._canvas.yview_scroll(
                -1*(e.delta//120), "units")),
            ("<Button-4>",   lambda e: self._canvas.yview_scroll(-1, "units")),
            ("<Button-5>",   lambda e: self._canvas.yview_scroll( 1, "units")),
        ):
            self._canvas.bind_all(seq, fn)

    # ── Carga de partidos ─────────────────────────────────────────
    def _fetch_games(self):
        def worker():
            try:
                self._q.put(("games", get_today_games()))
            except Exception as e:
                self._q.put(("load_err", str(e)))
        threading.Thread(target=worker, daemon=True).start()

    def _show_games(self, games: list[dict]):
        for w in self._inner.winfo_children():
            w.destroy()
        self._cards.clear()
        self._games = games
        self._predictions.clear()
        self._pdf_btn.config(state="disabled")

        if not games:
            tk.Label(self._inner,
                     text="No hay partidos programados para hoy.",
                     fg=DIM, bg=BG, font=("Helvetica", 13)).pack(pady=50)
            self._status_var.set("Sin partidos hoy.")
            return

        for g in games:
            card = GameCard(self._inner, g["home"], g["away"], g["status"])
            card.pack(fill="x", pady=5)
            self._cards[g["game_id"]] = card

        n = len(games)
        self._status_var.set(
            f"{n} partido{'s' if n > 1 else ''} hoy  —  "
            f"presiona «Calcular predicciones»")
        self._btn.config(state="normal")

    # ── Predicciones ──────────────────────────────────────────────
    def _start_predictions(self):
        self._btn.config(state="disabled", text="  Calculando...  ")
        self._pdf_btn.config(state="disabled")
        self._predictions.clear()
        self._status_var.set(
            "Obteniendo datos en tiempo real — puede tardar varios minutos...")
        for card in self._cards.values():
            card.set_loading()

        def worker():
            total = len(self._games)
            for i, g in enumerate(self._games, 1):
                gid = g["game_id"]
                self._q.put(("progress", gid, i, total))
                try:
                    pred = run_prediction(g["home"], g["away"])
                    self._q.put(("result", gid, pred, i, total))
                except Exception as e:
                    self._q.put(("game_err", gid, str(e), i, total))

        threading.Thread(target=worker, daemon=True).start()

    # ── Cálculo de presupuesto ─────────────────────────────────────
    def _calculate_and_show_budget(self):
        """Calcula la distribución del presupuesto y actualiza las tarjetas."""
        try:
            budget = float(self._budget_var.get().replace(",", "."))
            if budget <= 0:
                return
        except ValueError:
            return
        
        # Calcular distribución
        self._budget_allocation = calculate_budget_allocation(
            self._predictions, budget
        )
        
        # Actualizar cada tarjeta con su monto
        for gid, alloc in self._budget_allocation.items():
            if gid in self._cards:
                self._cards[gid].set_bet_amount(alloc["amount"], alloc["team"])
        
        # Mostrar resumen
        if self._budget_allocation:
            total_apostado = sum(a["amount"] for a in self._budget_allocation.values())
            n_apuestas = len(self._budget_allocation)
            self._show_budget_summary(total_apostado, n_apuestas)
    
    def _show_budget_summary(self, total: float, n_apuestas: int):
        """Muestra un resumen del presupuesto distribuido."""
        # Crear o actualizar el panel de resumen
        if hasattr(self, "_summary_frame"):
            self._summary_frame.destroy()
        
        self._summary_frame = tk.Frame(self, bg="#1a2332",
                                       highlightbackground=ACCENT,
                                       highlightthickness=1)
        self._summary_frame.pack(fill="x", padx=24, pady=(0, 10), before=self._canvas.master)
        
        # Título
        title_row = tk.Frame(self._summary_frame, bg="#1a2332")
        title_row.pack(fill="x", padx=12, pady=(8, 4))
        
        tk.Label(title_row, text="💰 DISTRIBUCIÓN DE PRESUPUESTO",
                 fg=ACCENT, bg="#1a2332",
                 font=("Helvetica", 11, "bold")).pack(side="left")
        
        tk.Label(title_row, text=f"Total: ${total:.2f} en {n_apuestas} apuesta(s)",
                 fg=TEXT, bg="#1a2332",
                 font=("Helvetica", 10)).pack(side="right")
        
        # Detalle por partido
        for gid, alloc in sorted(self._budget_allocation.items(), 
                                  key=lambda x: x[1]["amount"], reverse=True):
            row = tk.Frame(self._summary_frame, bg="#1a2332")
            row.pack(fill="x", padx=12, pady=2)
            
            nivel_colors = {"FUERTE": "#22c55e", "MODERADA": "#3b82f6", "BAJA": YELLOW}
            nivel_color = nivel_colors.get(alloc["nivel"], DIM)
            
            tk.Label(row, text=f"${alloc['amount']:.2f}",
                     fg="#22c55e", bg="#1a2332",
                     font=("Helvetica", 11, "bold"), width=8, anchor="w").pack(side="left")
            
            tk.Label(row, text=f"→ {nick(alloc['team'])}",
                     fg=TEXT, bg="#1a2332",
                     font=("Helvetica", 10)).pack(side="left", padx=(4, 8))
            
            tk.Label(row, text=f"({alloc['nivel']} · {alloc['confianza']:.0f}%)",
                     fg=nivel_color, bg="#1a2332",
                     font=("Helvetica", 9)).pack(side="left")
        
        # Mensaje si no hay apuestas
        if not self._budget_allocation:
            tk.Label(self._summary_frame, 
                     text="🚫 No hay partidos recomendados para apostar hoy",
                     fg=YELLOW, bg="#1a2332",
                     font=("Helvetica", 10)).pack(padx=12, pady=8)
        
        # Padding final
        tk.Frame(self._summary_frame, bg="#1a2332", height=8).pack()

    # ── Exportar PDF ──────────────────────────────────────────────
    def _export_pdf(self):
        if not PDF_DISPONIBLE:
            messagebox.showerror(
                "reportlab no instalado",
                "Instala reportlab para exportar PDF:\n\npip install reportlab")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
            initialfile=f"nba_predicciones_{date.today()}.pdf",
            title="Guardar predicciones como PDF",
        )
        if not filepath:
            return

        try:
            generate_pdf(filepath, self._games, self._predictions)
            self._status_var.set(f"✓  PDF guardado en {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Error al generar PDF", str(e))

    # ── Cola de mensajes ──────────────────────────────────────────
    def _check_queue(self):
        try:
            while True:
                msg = self._q.get_nowait()

                if msg[0] == "games":
                    self._show_games(msg[1])

                elif msg[0] == "load_err":
                    self._status_var.set(f"Error al cargar partidos: {msg[1]}")

                elif msg[0] == "progress":
                    _, gid, i, total = msg
                    self._status_var.set(f"Calculando partido {i}/{total}...")

                elif msg[0] == "result":
                    _, gid, pred, i, total = msg
                    self._predictions[gid] = pred
                    if gid in self._cards:
                        self._cards[gid].set_result(pred)
                    if i == total:
                        self._btn.config(state="normal",
                                         text="  Recalcular  ")
                        self._pdf_btn.config(state="normal")
                        # Calcular distribución de presupuesto
                        self._calculate_and_show_budget()
                        self._status_var.set(
                            f"✓  {total} predicciones completadas  —  "
                            f"puedes exportar a PDF")

                elif msg[0] == "game_err":
                    _, gid, err, i, total = msg
                    if gid in self._cards:
                        self._cards[gid].set_error()
                    if i == total:
                        self._btn.config(state="normal",
                                         text="  Recalcular  ")
                        if self._predictions:
                            self._pdf_btn.config(state="normal")
                        self._status_var.set(
                            "Predicciones completadas con errores.")

        except queue.Empty:
            pass

        self.after(100, self._check_queue)


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = NBAApp()
    app.mainloop()

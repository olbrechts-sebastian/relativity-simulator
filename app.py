from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
C = 299_792_458.0  # Speed of light in vacuum, m/s
MUON_REST_LIFETIME = 2.1969811e-6  # Mean muon lifetime at rest, seconds
EARTH_RADIUS_M = 6_371_000.0

PLOT_BG = "rgba(5, 10, 22, 0)"
PAPER_BG = "rgba(5, 10, 22, 0)"
GRID = "rgba(139, 233, 253, 0.13)"
CYAN = "#45f3ff"
BLUE = "#4c83ff"
VIOLET = "#b76cff"
MAGENTA = "#ff4fd8"
GREEN = "#53ffb8"
AMBER = "#ffc857"
RED = "#ff5c7a"


@dataclass(frozen=True)
class RelativityState:
    """Computed relativistic quantities for the current dashboard controls."""

    beta: float
    gamma: float
    velocity_m_s: float
    velocity_km_s: float
    proper_time_s: float
    observed_time_s: float
    rest_length_m: float
    contracted_length_m: float
    atmosphere_height_m: float
    contracted_atmosphere_m: float
    earth_frame_travel_time_s: float
    muon_frame_travel_time_s: float
    dilated_muon_lifetime_s: float
    classical_survival_ground: float
    relativistic_survival_ground: float
    detector_survival: float
    relativity_intensity: float


def clamp_beta(beta: float) -> float:
    """Keep beta inside a numerically stable open interval below light speed."""
    return float(np.clip(beta, 0.001, 0.999_999))


def lorentz_factor(beta: float) -> float:
    beta = clamp_beta(beta)
    return 1.0 / math.sqrt(1.0 - beta**2)


def survival_probability(distance_m: float, beta: float, gamma: float | None = None) -> float:
    """Muon survival probability after traveling a given distance in Earth's frame."""
    beta = clamp_beta(beta)
    gamma = lorentz_factor(beta) if gamma is None else gamma
    travel_time = distance_m / (beta * C)
    dilated_lifetime = gamma * MUON_REST_LIFETIME
    return float(math.exp(-travel_time / dilated_lifetime))


def classical_survival_probability(distance_m: float, beta: float) -> float:
    """Non-relativistic baseline for comparison: no time dilation."""
    beta = clamp_beta(beta)
    travel_time = distance_m / (beta * C)
    return float(math.exp(-travel_time / MUON_REST_LIFETIME))


def compute_state(
    beta: float,
    proper_time_us: float,
    rest_length_m: float,
    atmosphere_height_m: float,
    detector_depth_m: float,
) -> RelativityState:
    beta = clamp_beta(beta)
    gamma = lorentz_factor(beta)
    velocity_m_s = beta * C
    travel_distance_to_detector = max(atmosphere_height_m - detector_depth_m, 1.0)
    earth_frame_travel_time_s = atmosphere_height_m / velocity_m_s

    observed_time_s = gamma * proper_time_us * 1e-6
    contracted_length_m = rest_length_m / gamma
    contracted_atmosphere_m = atmosphere_height_m / gamma
    muon_frame_travel_time_s = earth_frame_travel_time_s / gamma
    dilated_lifetime_s = gamma * MUON_REST_LIFETIME
    classical_ground = classical_survival_probability(atmosphere_height_m, beta)
    relativistic_ground = survival_probability(atmosphere_height_m, beta, gamma)
    detector_survival = survival_probability(travel_distance_to_detector, beta, gamma)

    # A bounded 0..1 indicator that becomes intense once gamma leaves everyday physics.
    relativity_intensity = float(np.clip(math.log10(gamma) / math.log10(24), 0.0, 1.0))

    return RelativityState(
        beta=beta,
        gamma=gamma,
        velocity_m_s=velocity_m_s,
        velocity_km_s=velocity_m_s / 1_000,
        proper_time_s=proper_time_us * 1e-6,
        observed_time_s=observed_time_s,
        rest_length_m=rest_length_m,
        contracted_length_m=contracted_length_m,
        atmosphere_height_m=atmosphere_height_m,
        contracted_atmosphere_m=contracted_atmosphere_m,
        earth_frame_travel_time_s=earth_frame_travel_time_s,
        muon_frame_travel_time_s=muon_frame_travel_time_s,
        dilated_muon_lifetime_s=dilated_lifetime_s,
        classical_survival_ground=classical_ground,
        relativistic_survival_ground=relativistic_ground,
        detector_survival=detector_survival,
        relativity_intensity=relativity_intensity,
    )


@st.cache_data(show_spinner=False, max_entries=32)
def simulate_muon_decays(
    count: int,
    atmosphere_height_m: float,
    detector_depth_m: float,
    beta: float,
    seed: int,
) -> dict[str, np.ndarray | int | float]:
    """Monte Carlo atmospheric muon decay model.

    Muon lifetimes are sampled from an exponential distribution using the
    relativistically dilated mean lifetime in Earth's frame.
    """
    beta = clamp_beta(beta)
    gamma = lorentz_factor(beta)
    rng = np.random.default_rng(seed)

    lifetimes_s = rng.exponential(MUON_REST_LIFETIME * gamma, count)
    travel_distances_m = beta * C * lifetimes_s
    decay_altitudes_m = atmosphere_height_m - travel_distances_m
    detector_altitude_m = detector_depth_m
    arrivals = decay_altitudes_m <= detector_altitude_m

    return {
        "lifetimes_s": lifetimes_s,
        "travel_distances_m": travel_distances_m,
        "decay_altitudes_m": decay_altitudes_m,
        "arrivals": arrivals,
        "arrival_count": int(np.count_nonzero(arrivals)),
        "arrival_fraction": float(np.mean(arrivals)),
    }


def configure_page() -> None:
    st.set_page_config(
        page_title="Relativity & Muon Physics Simulator",
        page_icon="R",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #050a16;
            --panel: rgba(9, 18, 38, 0.78);
            --panel-strong: rgba(12, 25, 52, 0.92);
            --line: rgba(100, 210, 255, 0.22);
            --cyan: #45f3ff;
            --blue: #4c83ff;
            --violet: #b76cff;
            --magenta: #ff4fd8;
            --green: #53ffb8;
            --amber: #ffc857;
            --text: #e8f3ff;
            --muted: #91a6c4;
        }

        .stApp {
            color: var(--text);
            background:
                radial-gradient(circle at 18% 6%, rgba(69, 243, 255, 0.11), transparent 28%),
                radial-gradient(circle at 82% 9%, rgba(183, 108, 255, 0.13), transparent 26%),
                linear-gradient(135deg, #030711 0%, #07101f 43%, #090719 100%);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(6, 13, 28, 0.98), rgba(9, 12, 29, 0.98));
            border-right: 1px solid rgba(69, 243, 255, 0.16);
        }

        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span {
            color: #dbeafe;
        }

        div[data-testid="stVerticalBlock"] > div:has(.glass-panel) {
            gap: 0.8rem;
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 2.1rem 2rem 1.5rem;
            border: 1px solid rgba(69, 243, 255, 0.22);
            border-radius: 18px;
            background:
                linear-gradient(135deg, rgba(10, 22, 48, 0.88), rgba(9, 11, 31, 0.78)),
                repeating-linear-gradient(90deg, rgba(255,255,255,0.03) 0 1px, transparent 1px 88px);
            box-shadow: 0 0 42px rgba(69, 243, 255, 0.10), inset 0 0 60px rgba(76, 131, 255, 0.06);
        }

        .hero:before {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(90deg, transparent, rgba(69, 243, 255, 0.08), transparent);
            animation: sweep 7s linear infinite;
        }

        @keyframes sweep {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }

        .hero-content {
            position: relative;
            z-index: 1;
        }

        .eyebrow {
            color: var(--cyan);
            font-size: 0.78rem;
            letter-spacing: 0.18rem;
            text-transform: uppercase;
            font-weight: 800;
        }

        .hero-title {
            margin: 0.35rem 0 0.35rem;
            font-size: clamp(2.2rem, 4.2vw, 4.2rem);
            line-height: 1.02;
            font-weight: 900;
            letter-spacing: 0;
            color: #f5fbff;
            text-shadow: 0 0 28px rgba(69, 243, 255, 0.18);
        }

        .hero-subtitle {
            max-width: 980px;
            color: #b7c6df;
            font-size: 1.02rem;
            line-height: 1.65;
            margin: 0;
        }

        .status-row {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin-top: 1.1rem;
        }

        .status-pill {
            border: 1px solid rgba(69, 243, 255, 0.22);
            border-radius: 999px;
            padding: 0.42rem 0.74rem;
            background: rgba(9, 18, 38, 0.72);
            color: #dff8ff;
            font-size: 0.83rem;
            box-shadow: 0 0 18px rgba(69, 243, 255, 0.08);
        }

        .glass-panel {
            border: 1px solid rgba(100, 210, 255, 0.18);
            border-radius: 16px;
            background: rgba(8, 17, 36, 0.74);
            box-shadow: 0 0 34px rgba(0, 0, 0, 0.24), inset 0 0 30px rgba(76, 131, 255, 0.04);
            padding: 1.05rem;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 1rem 0 0.35rem;
        }

        @media (max-width: 1100px) {
            .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }

        @media (max-width: 720px) {
            .metric-grid { grid-template-columns: 1fr; }
            .hero { padding: 1.45rem; }
        }

        .metric-card {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(69, 243, 255, 0.18);
            border-radius: 14px;
            background: linear-gradient(145deg, rgba(12, 26, 55, 0.94), rgba(8, 12, 29, 0.88));
            padding: 1rem;
            min-height: 116px;
            box-shadow: 0 0 26px rgba(69, 243, 255, 0.08);
            transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
        }

        .metric-card:hover {
            transform: translateY(-3px);
            border-color: rgba(69, 243, 255, 0.42);
            box-shadow: 0 0 36px rgba(69, 243, 255, 0.14);
        }

        .metric-card:after {
            content: "";
            position: absolute;
            inset: auto -20% -45% -20%;
            height: 72px;
            background: radial-gradient(circle, rgba(69, 243, 255, 0.16), transparent 65%);
        }

        .metric-label {
            color: #9bb2d5;
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.08rem;
            font-weight: 800;
        }

        .metric-value {
            margin-top: 0.35rem;
            color: #f2fbff;
            font-size: 1.78rem;
            font-weight: 900;
            line-height: 1.1;
            font-variant-numeric: tabular-nums;
            animation: glowPulse 2.8s ease-in-out infinite;
        }

        @keyframes glowPulse {
            0%, 100% { text-shadow: 0 0 10px rgba(69, 243, 255, 0.10); }
            50% { text-shadow: 0 0 18px rgba(69, 243, 255, 0.28); }
        }

        .metric-note {
            color: #8fa4c2;
            font-size: 0.78rem;
            margin-top: 0.35rem;
            line-height: 1.35;
        }

        .section-title {
            margin: 1.2rem 0 0.55rem;
            color: #f5fbff;
            font-size: 1.25rem;
            font-weight: 850;
        }

        .ship-track {
            position: relative;
            height: 96px;
            overflow: hidden;
            margin: 0.6rem 0 0.9rem;
            border: 1px solid rgba(69, 243, 255, 0.18);
            border-radius: 14px;
            background:
                linear-gradient(90deg, rgba(69, 243, 255, 0.06), rgba(183, 108, 255, 0.06)),
                repeating-linear-gradient(90deg, transparent 0 52px, rgba(255, 255, 255, 0.055) 52px 53px);
        }

        .ship-track:before {
            content: "0c";
            position: absolute;
            left: 12px;
            bottom: 10px;
            color: #7590b4;
            font-size: 0.72rem;
        }

        .ship-track:after {
            content: "1c";
            position: absolute;
            right: 12px;
            bottom: 10px;
            color: var(--cyan);
            font-size: 0.72rem;
        }

        .ship {
            --x: 50%;
            position: absolute;
            left: calc(var(--x) - 34px);
            top: 27px;
            width: 68px;
            height: 30px;
            border-radius: 50% 8px 8px 50%;
            background: linear-gradient(90deg, #e7fbff, #45f3ff 42%, #4c83ff);
            box-shadow: 0 0 28px rgba(69, 243, 255, 0.48);
            animation: shipFloat 1.9s ease-in-out infinite;
        }

        .ship:before {
            content: "";
            position: absolute;
            left: -76px;
            top: 8px;
            width: 72px;
            height: 12px;
            border-radius: 999px;
            background: linear-gradient(90deg, transparent, rgba(69, 243, 255, 0.62));
            filter: blur(1px);
        }

        .ship:after {
            content: "";
            position: absolute;
            right: -7px;
            top: 9px;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: #f5fbff;
            box-shadow: 0 0 16px #f5fbff;
        }

        @keyframes shipFloat {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
        }

        .journey {
            display: grid;
            grid-template-columns: 1fr;
            gap: 0.55rem;
            margin-top: 0.6rem;
        }

        .journey-band {
            position: relative;
            height: 54px;
            overflow: hidden;
            border: 1px solid rgba(100, 210, 255, 0.16);
            border-radius: 14px;
            background: linear-gradient(90deg, rgba(14, 34, 70, 0.9), rgba(8, 13, 29, 0.9));
        }

        .journey-fill {
            height: 100%;
            border-radius: 14px;
            background: linear-gradient(90deg, rgba(69, 243, 255, 0.38), rgba(83, 255, 184, 0.30));
            box-shadow: 0 0 22px rgba(83, 255, 184, 0.16);
        }

        .journey-label {
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 0.85rem;
            color: #e9fbff;
            font-size: 0.86rem;
            font-weight: 750;
        }

        .warning-panel {
            border: 1px solid rgba(255, 200, 87, 0.28);
            background: rgba(255, 200, 87, 0.08);
            color: #ffe7a3;
            border-radius: 14px;
            padding: 0.85rem 1rem;
            line-height: 1.55;
        }

        .ok-panel {
            border: 1px solid rgba(83, 255, 184, 0.22);
            background: rgba(83, 255, 184, 0.08);
            color: #c7ffe8;
            border-radius: 14px;
            padding: 0.85rem 1rem;
            line-height: 1.55;
        }

        div[data-testid="stTabs"] button {
            color: #c9d8ee;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #ffffff;
            border-bottom-color: var(--cyan);
        }

        .stSlider [data-baseweb="slider"] div {
            color: var(--cyan);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_scientific(value: float, unit: str = "") -> str:
    if value == 0:
        return f"0 {unit}".strip()
    if abs(value) >= 10_000 or abs(value) < 0.001:
        return f"{value:.3e} {unit}".strip()
    return f"{value:,.3f} {unit}".strip()


def metric_card(label: str, value: str, note: str, accent: str = CYAN) -> str:
    # Keep this HTML unindented. Markdown treats four-space-indented HTML as
    # code, which would display the markup instead of rendering the cards.
    return (
        f'<div class="metric-card" style="border-top: 2px solid {accent};">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-note">{note}</div>'
        "</div>"
    )


def render_metric_grid(state: RelativityState) -> None:
    cards = [
        metric_card("Velocity", f"{state.beta:.5f} c", f"{state.velocity_km_s:,.0f} km/s", CYAN),
        metric_card("Lorentz factor", f"{state.gamma:,.3f}", "Multiplier for time dilation", VIOLET),
        metric_card("Observed time", f"{state.observed_time_s * 1e6:,.2f} us", "Lab-frame duration", BLUE),
        metric_card("Contracted length", f"{state.contracted_length_m:,.2f} m", "Measured along motion", GREEN),
    ]
    st.markdown(f"<div class='metric-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def render_hero(state: RelativityState) -> None:
    regime = classify_regime(state.beta, state.gamma)
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-content">
                <div class="eyebrow">CERN-inspired interactive physics console</div>
                <h1 class="hero-title">Relativity & Muon Physics Simulator</h1>
                <p class="hero-subtitle">
                    Explore time dilation, length contraction, proper time, cosmic-ray muon decay,
                    and detector arrival statistics in a single high-fidelity scientific dashboard.
                </p>
                <div class="status-row">
                    <span class="status-pill">Regime: {regime}</span>
                    <span class="status-pill">beta = {state.beta:.5f}</span>
                    <span class="status-pill">gamma = {state.gamma:,.3f}</span>
                    <span class="status-pill">Muon survival = {state.relativistic_survival_ground * 100:.2f}%</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def classify_regime(beta: float, gamma: float) -> str:
    if beta >= 0.995:
        return "extreme relativistic"
    if beta >= 0.95:
        return "deep relativistic"
    if beta >= 0.75:
        return "relativistic"
    if gamma >= 1.05:
        return "transitional"
    return "near-classical"


def render_spaceship(beta: float) -> None:
    x_position = 7 + beta * 86
    st.markdown(
        f"""
        <div class="glass-panel">
            <div class="section-title">Light-Speed Approach Vector</div>
            <div class="ship-track">
                <div class="ship" style="--x: {x_position:.2f}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_gauge(title: str, value: float, max_value: float, suffix: str, color: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": suffix, "font": {"size": 36, "color": "#f5fbff"}},
            title={"text": title, "font": {"size": 15, "color": "#c7d7ee"}},
            gauge={
                "axis": {"range": [0, max_value], "tickcolor": "#8aa4c7"},
                "bar": {"color": color, "thickness": 0.32},
                "bgcolor": "rgba(255,255,255,0.04)",
                "borderwidth": 1,
                "bordercolor": "rgba(100,210,255,0.22)",
                "steps": [
                    {"range": [0, max_value * 0.45], "color": "rgba(76,131,255,0.12)"},
                    {"range": [max_value * 0.45, max_value * 0.8], "color": "rgba(183,108,255,0.13)"},
                    {"range": [max_value * 0.8, max_value], "color": "rgba(255,79,216,0.14)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=260,
        margin={"l": 18, "r": 18, "t": 42, "b": 12},
        paper_bgcolor=PAPER_BG,
        font={"color": "#dbeafe"},
    )
    return fig


def apply_plotly_theme(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font={"color": "#dbeafe", "family": "Arial"},
        margin={"l": 48, "r": 28, "t": 54, "b": 46},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)",
        },
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor="rgba(255,255,255,0.12)", color="#bcd0eb")
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="rgba(255,255,255,0.12)", color="#bcd0eb")
    return fig


def make_lorentz_curve(beta: float) -> go.Figure:
    betas = np.linspace(0.01, 0.999, 600)
    gammas = 1 / np.sqrt(1 - betas**2)
    current_gamma = lorentz_factor(beta)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=betas,
            y=gammas,
            mode="lines",
            line={"color": CYAN, "width": 4},
            name="gamma(beta)",
            hovertemplate="beta=%{x:.4f}<br>gamma=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[beta],
            y=[current_gamma],
            mode="markers",
            marker={"size": 14, "color": MAGENTA, "line": {"color": "#ffffff", "width": 1}},
            name="current state",
            hovertemplate="current beta=%{x:.5f}<br>gamma=%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Lorentz Factor Divergence Near Light Speed",
        xaxis_title="Velocity fraction beta = v/c",
        yaxis_title="Lorentz factor gamma",
    )
    fig.update_yaxes(type="log")
    return apply_plotly_theme(fig, height=430)


def make_time_length_chart(state: RelativityState) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=["Time interval", "Object length", "Atmosphere thickness"],
            y=[
                state.observed_time_s / max(state.proper_time_s, 1e-30),
                state.contracted_length_m / state.rest_length_m,
                state.contracted_atmosphere_m / state.atmosphere_height_m,
            ],
            marker={
                "color": [VIOLET, GREEN, CYAN],
                "line": {"color": "rgba(255,255,255,0.28)", "width": 1},
            },
            text=[
                f"{state.gamma:.2f}x dilation",
                f"{state.contracted_length_m / state.rest_length_m:.3f}x length",
                f"{state.contracted_atmosphere_m / state.atmosphere_height_m:.3f}x thickness",
            ],
            textposition="outside",
            hovertemplate="%{x}<br>ratio=%{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Observed Relativistic Transformations",
        yaxis_title="Ratio relative to rest-frame value",
    )
    return apply_plotly_theme(fig, height=390)


def make_spacetime_chart(state: RelativityState) -> go.Figure:
    t_lab_us = np.linspace(0, max(state.observed_time_s * 1e6, 1), 160)
    t_proper_us = t_lab_us / state.gamma

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t_lab_us,
            y=t_lab_us,
            mode="lines",
            name="lab clock",
            line={"color": BLUE, "width": 3},
            hovertemplate="lab elapsed=%{y:.3f} us<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_lab_us,
            y=t_proper_us,
            mode="lines",
            name="moving clock",
            line={"color": CYAN, "width": 4},
            fill="tonexty",
            fillcolor="rgba(69,243,255,0.08)",
            hovertemplate="moving clock=%{y:.3f} us<extra></extra>",
        )
    )
    fig.update_layout(
        title="Proper Time vs Observed Time",
        xaxis_title="Earth/lab time (microseconds)",
        yaxis_title="Clock reading (microseconds)",
    )
    return apply_plotly_theme(fig, height=390)


def make_decay_curve(state: RelativityState) -> go.Figure:
    max_t = max(15e-6, state.earth_frame_travel_time_s * 1.35, state.dilated_muon_lifetime_s * 4.5)
    times = np.linspace(0, max_t, 800)
    classical = np.exp(-times / MUON_REST_LIFETIME)
    relativistic = np.exp(-times / state.dilated_muon_lifetime_s)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times * 1e6,
            y=classical,
            mode="lines",
            name="without relativity",
            line={"color": RED, "width": 3},
            hovertemplate="time=%{x:.3f} us<br>remaining=%{y:.5f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=times * 1e6,
            y=relativistic,
            mode="lines",
            name="with time dilation",
            line={"color": GREEN, "width": 4},
            hovertemplate="time=%{x:.3f} us<br>remaining=%{y:.5f}<extra></extra>",
        )
    )
    fig.add_vline(
        x=state.earth_frame_travel_time_s * 1e6,
        line_width=2,
        line_dash="dash",
        line_color=AMBER,
        annotation_text="arrival time",
        annotation_font_color=AMBER,
    )
    fig.update_layout(
        title="Muon Decay Law: Classical vs Relativistic Lifetime",
        xaxis_title="Earth-frame time (microseconds)",
        yaxis_title="Fraction of muons remaining",
    )
    fig.update_yaxes(range=[0, 1.03])
    return apply_plotly_theme(fig, height=440)


def make_altitude_histogram(decay_altitudes_m: np.ndarray, atmosphere_height_m: float, detector_depth_m: float) -> go.Figure:
    clipped = np.clip(decay_altitudes_m, -2_000, atmosphere_height_m)
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=clipped,
            nbinsx=70,
            marker={"color": "rgba(69, 243, 255, 0.70)", "line": {"color": "rgba(255,255,255,0.18)", "width": 0.5}},
            name="simulated decays",
            hovertemplate="altitude bin=%{x:.0f} m<br>muons=%{y}<extra></extra>",
        )
    )
    fig.add_vrect(
        x0=-2_000,
        x1=detector_depth_m,
        fillcolor="rgba(83,255,184,0.16)",
        line_width=0,
        annotation_text="detector arrival zone",
        annotation_font_color=GREEN,
    )
    fig.add_vline(x=0, line_color="#f5fbff", line_width=2, annotation_text="ground")
    fig.update_layout(
        title="Monte Carlo Decay Altitudes",
        xaxis_title="Decay altitude relative to ground (m)",
        yaxis_title="Simulated muons",
        bargap=0.04,
    )
    return apply_plotly_theme(fig, height=430)


def make_survival_heatmap(current_beta: float, current_altitude_m: float) -> go.Figure:
    betas = np.linspace(0.55, 0.999, 120)
    altitudes = np.linspace(1_000, 30_000, 100)
    survival = np.empty((len(altitudes), len(betas)))

    for i, altitude in enumerate(altitudes):
        gammas = 1 / np.sqrt(1 - betas**2)
        travel_times = altitude / (betas * C)
        survival[i, :] = np.exp(-travel_times / (gammas * MUON_REST_LIFETIME))

    fig = go.Figure(
        go.Heatmap(
            x=betas,
            y=altitudes / 1_000,
            z=survival * 100,
            colorscale=[
                [0.0, "#08111f"],
                [0.22, "#143c69"],
                [0.48, "#206ec9"],
                [0.72, "#45f3ff"],
                [1.0, "#53ffb8"],
            ],
            colorbar={"title": "survival %", "tickcolor": "#bcd0eb"},
            hovertemplate="beta=%{x:.4f}<br>altitude=%{y:.1f} km<br>survival=%{z:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[current_beta],
            y=[current_altitude_m / 1_000],
            mode="markers",
            marker={"size": 15, "color": MAGENTA, "line": {"color": "white", "width": 1}},
            name="current run",
            hovertemplate="current beta=%{x:.4f}<br>altitude=%{y:.1f} km<extra></extra>",
        )
    )
    fig.update_layout(
        title="Muon Survival Phase Map",
        xaxis_title="Velocity fraction beta",
        yaxis_title="Production altitude (km)",
    )
    return apply_plotly_theme(fig, height=450)


def make_matplotlib_detector_strip(state: RelativityState, arrival_fraction: float) -> plt.Figure:
    """Small Matplotlib engineering strip retained to demonstrate mixed plotting stack."""
    fig, ax = plt.subplots(figsize=(9, 1.8), facecolor="#050a16")
    ax.set_facecolor("#050a16")
    ax.set_xlim(0, state.atmosphere_height_m)
    ax.set_ylim(0, 1)
    ax.hlines(0.5, 0, state.atmosphere_height_m, color="#243a5a", linewidth=9)
    ax.hlines(0.5, 0, state.atmosphere_height_m * arrival_fraction, color=GREEN, linewidth=9)
    ax.scatter([0], [0.5], s=180, color=GREEN, edgecolor="white", linewidth=1.2, label="Earth detector")
    ax.scatter([state.atmosphere_height_m], [0.5], s=180, color=CYAN, edgecolor="white", linewidth=1.2, label="Muon generation")
    ax.set_title("Atmosphere-to-Detector Survival Corridor", color="#e8f3ff", pad=10)
    ax.set_xlabel("Distance from detector upward (m)", color="#bcd0eb")
    ax.set_yticks([])
    ax.tick_params(colors="#91a6c4")
    for spine in ax.spines.values():
        spine.set_color("#1d3355")
    ax.grid(axis="x", color="#1b345b", alpha=0.45)
    ax.legend(facecolor="#07101f", edgecolor="#24476e", labelcolor="#e8f3ff", loc="upper right")
    fig.tight_layout()
    return fig


def render_equation_panel(state: RelativityState) -> None:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("#### Relativistic Field Equations")
    st.latex(r"\gamma = \frac{1}{\sqrt{1 - \beta^2}}, \qquad \beta=\frac{v}{c}")
    st.latex(r"\Delta t = \gamma \Delta \tau, \qquad L = \frac{L_0}{\gamma}")
    st.latex(r"P_{\mathrm{survive}} = \exp\left(-\frac{d}{\beta c \gamma \tau_\mu}\right)")
    st.caption(
        f"Current state: gamma = {state.gamma:.3f}, observed/proper time ratio = {state.gamma:.3f}, "
        f"length contraction ratio = {1 / state.gamma:.4f}."
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_warning_panel(state: RelativityState) -> None:
    if state.beta >= 0.995:
        st.markdown(
            """
            <div class="warning-panel">
                Extreme relativistic regime: small slider changes now create very large changes in gamma,
                detector arrival probability, and contracted atmospheric distance.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif state.beta >= 0.9:
        st.markdown(
            """
            <div class="ok-panel">
                Deep relativistic behavior is active: time dilation and length contraction are both large
                enough to dominate the muon survival outcome.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="ok-panel">
                Transitional regime: relativistic effects are measurable, but cosmic muon survival is still
                strongly limited unless the speed approaches c.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_muon_journey(state: RelativityState, arrival_fraction: float, arrival_count: int, muon_count: int) -> None:
    analytic_pct = state.detector_survival * 100
    simulated_pct = arrival_fraction * 100
    classical_pct = state.classical_survival_ground * 100
    fill = min(100, max(0, simulated_pct))
    st.markdown(
        f"""
        <div class="glass-panel">
            <div class="section-title">Atmospheric Muon Journey</div>
            <div class="journey">
                <div class="journey-band">
                    <div class="journey-fill" style="width: {fill:.2f}%;"></div>
                    <div class="journey-label">
                        <span>Upper atmosphere: muon generation</span>
                        <span>Earth detector</span>
                    </div>
                </div>
            </div>
            <p style="color:#a9bbd6; margin: 0.7rem 0 0;">
                Analytical survival: <b style="color:{GREEN};">{analytic_pct:.2f}%</b> |
                Classical baseline: <b style="color:{RED};">{classical_pct:.6f}%</b> |
                Monte Carlo arrivals: <b style="color:{CYAN};">{arrival_count:,} / {muon_count:,}</b>
                ({simulated_pct:.2f}%)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_controls() -> tuple[float, float, float, int, int, float]:
    presets = {
        "Custom": None,
        "0.50c - training beam": 0.50,
        "0.90c - relativistic": 0.90,
        "0.99c - cosmic ray": 0.99,
        "0.995c - sea-level muons": 0.995,
        "0.999c - extreme lab": 0.999,
    }

    with st.sidebar:
        st.markdown("### Simulation Controls")
        preset_name = st.selectbox("Velocity preset", list(presets.keys()), index=4)
        preset_beta = presets[preset_name]

        default_beta = 0.995 if preset_beta is None else preset_beta
        beta = st.slider(
            "Velocity fraction beta = v/c",
            min_value=0.05,
            max_value=0.999,
            value=float(default_beta),
            step=0.001,
            format="%.3f c",
            help="As beta approaches 1, gamma grows nonlinearly and relativistic effects intensify.",
        )

        st.markdown("### Relativity Lab")
        proper_time_us = st.slider(
            "Proper time interval",
            min_value=0.1,
            max_value=20.0,
            value=2.2,
            step=0.1,
            format="%.1f microseconds",
            help="Time measured by the moving object or particle in its own rest frame.",
        )
        rest_length_m = st.slider(
            "Rest length L0",
            min_value=1.0,
            max_value=1_000.0,
            value=120.0,
            step=1.0,
            format="%.0f m",
            help="Length measured in the object's own rest frame.",
        )

        st.markdown("### Muon Observatory")
        atmosphere_height_m = st.slider(
            "Muon production altitude",
            min_value=1_000,
            max_value=30_000,
            value=10_000,
            step=500,
            format="%d m",
            help="Approximate altitude where secondary cosmic-ray muons are generated.",
        )
        muon_count = st.slider(
            "Monte Carlo muons",
            min_value=1_000,
            max_value=120_000,
            value=25_000,
            step=1_000,
            help="Higher counts reduce random noise but take slightly longer to render.",
        )
        detector_depth_m = st.slider(
            "Detector acceptance above ground",
            min_value=0,
            max_value=2_000,
            value=100,
            step=25,
            format="%d m",
            help="Muons decaying below this altitude are counted as detector arrivals.",
        )
        seed = st.number_input("Monte Carlo seed", min_value=1, max_value=999_999, value=42, step=1)

    return beta, proper_time_us, rest_length_m, atmosphere_height_m, muon_count, detector_depth_m, int(seed)


def render_relativity_tab(state: RelativityState) -> None:
    render_spaceship(state.beta)
    left, right = st.columns([1.25, 0.75], gap="large")
    with left:
        st.plotly_chart(make_lorentz_curve(state.beta), width="stretch")
    with right:
        st.plotly_chart(
            make_gauge(
                "Relativistic effect intensity",
                state.relativity_intensity * 100,
                100,
                "%",
                MAGENTA,
            ),
            width="stretch",
        )
        render_warning_panel(state)

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.plotly_chart(make_time_length_chart(state), width="stretch")
    with col_b:
        st.plotly_chart(make_spacetime_chart(state), width="stretch")

    with st.expander("Learn more: why gamma changes everything", expanded=False):
        st.write(
            "The Lorentz factor is the conversion key between inertial frames. "
            "At everyday speeds gamma is almost exactly 1. Near light speed it rises sharply, "
            "stretching observed lifetimes while compressing distances measured along the direction of motion."
        )
        st.latex(r"\gamma \rightarrow \infty \quad \mathrm{as} \quad v \rightarrow c")


def render_muon_tab(
    state: RelativityState,
    simulation: dict[str, np.ndarray | int | float],
    muon_count: int,
    detector_depth_m: float,
) -> None:
    arrival_fraction = float(simulation["arrival_fraction"])
    arrival_count = int(simulation["arrival_count"])
    decay_altitudes = simulation["decay_altitudes_m"]
    assert isinstance(decay_altitudes, np.ndarray)

    render_muon_journey(state, arrival_fraction, arrival_count, muon_count)

    top_cols = st.columns([1.15, 0.85], gap="large")
    with top_cols[0]:
        st.plotly_chart(make_decay_curve(state), width="stretch")
    with top_cols[1]:
        st.plotly_chart(
            make_gauge(
                "Detector arrival probability",
                state.detector_survival * 100,
                100,
                "%",
                GREEN,
            ),
            width="stretch",
        )
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("#### Frame Diagnostics")
        st.write(
            f"Earth-frame travel time: **{state.earth_frame_travel_time_s * 1e6:.3f} microseconds**"
        )
        st.write(
            f"Muon-frame travel time: **{state.muon_frame_travel_time_s * 1e6:.3f} microseconds**"
        )
        st.write(
            f"Dilated muon lifetime: **{state.dilated_muon_lifetime_s * 1e6:.3f} microseconds**"
        )
        st.write(
            f"Atmosphere in muon frame: **{state.contracted_atmosphere_m:,.0f} m**"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    hist_cols = st.columns([1.1, 0.9], gap="large")
    with hist_cols[0]:
        st.plotly_chart(
            make_altitude_histogram(decay_altitudes, state.atmosphere_height_m, detector_depth_m),
            width="stretch",
        )
    with hist_cols[1]:
        st.pyplot(
            make_matplotlib_detector_strip(state, arrival_fraction),
            clear_figure=True,
            width="stretch",
        )
        st.dataframe(
            {
                "Quantity": [
                    "Generated muons",
                    "Detector arrivals",
                    "Monte Carlo arrival rate",
                    "Analytical detector survival",
                    "Classical ground survival",
                    "Relativity boost over classical",
                ],
                "Value": [
                    f"{muon_count:,}",
                    f"{arrival_count:,}",
                    f"{arrival_fraction * 100:.3f}%",
                    f"{state.detector_survival * 100:.3f}%",
                    f"{state.classical_survival_ground * 100:.8f}%",
                    f"{state.relativistic_survival_ground / max(state.classical_survival_ground, 1e-300):,.0f}x",
                ],
            },
            hide_index=True,
            width="stretch",
        )

    st.plotly_chart(make_survival_heatmap(state.beta, state.atmosphere_height_m), width="stretch")

    with st.expander("Learn more: why cosmic muons reach Earth", expanded=False):
        st.write(
            "Muons are created when high-energy cosmic rays collide with nuclei in the upper atmosphere. "
            "Their rest lifetime is only about 2.2 microseconds, so a classical calculation predicts that "
            "very few should reach sea level. Relativity changes the outcome: in Earth's frame the muon clock "
            "runs slow, and in the muon's frame the atmosphere is length-contracted."
        )
        st.latex(r"\tau_{\mu,\,lab}=\gamma\tau_\mu")


def render_theory_tab(state: RelativityState) -> None:
    cols = st.columns([0.9, 1.1], gap="large")
    with cols[0]:
        render_equation_panel(state)
    with cols[1]:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("#### Scientific Readout")
        st.write(
            "This simulator models special relativity in flat spacetime, appropriate for high-speed "
            "particles over atmospheric distances where gravitational corrections are negligible."
        )
        st.write(
            "The muon simulation uses exponential decay with a gamma-dilated mean lifetime in the lab frame. "
            "The Monte Carlo output should converge toward the analytical survival probability as sample size grows."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    fact_cols = st.columns(3)
    fact_cols[0].info("At 0.99c, gamma is about 7.09, so a 2.2 us muon lifetime appears over 15 us in the lab.")
    fact_cols[1].info("Length contraction only affects distances parallel to motion; transverse dimensions do not contract.")
    fact_cols[2].info("No massive object reaches c; the energy required grows without bound as beta approaches 1.")


def main() -> None:
    configure_page()
    inject_css()

    (
        beta,
        proper_time_us,
        rest_length_m,
        atmosphere_height_m,
        muon_count,
        detector_depth_m,
        seed,
    ) = sidebar_controls()

    state = compute_state(
        beta=beta,
        proper_time_us=proper_time_us,
        rest_length_m=rest_length_m,
        atmosphere_height_m=float(atmosphere_height_m),
        detector_depth_m=float(detector_depth_m),
    )

    simulation = simulate_muon_decays(
        count=muon_count,
        atmosphere_height_m=float(atmosphere_height_m),
        detector_depth_m=float(detector_depth_m),
        beta=state.beta,
        seed=seed,
    )

    render_hero(state)
    render_metric_grid(state)

    tab_relativity, tab_muons, tab_theory = st.tabs(
        ["Relativity Console", "Muon Observatory", "Theory & Equations"]
    )

    with tab_relativity:
        render_relativity_tab(state)

    with tab_muons:
        render_muon_tab(state, simulation, muon_count, float(detector_depth_m))

    with tab_theory:
        render_theory_tab(state)


if __name__ == "__main__":
    main()

"""
Interactive figures (plotly HTML) of the MAP study on Hartmann 6D, built from the results of
run_map_benchmark.py and run_prior_sensitivity.py. Output: benchmarks/map_hartmann/figures/.

Colors: validated categorical / ordinal palette with light and dark steps; the dark steps are
swapped in by the browser when the system is in dark mode (prefers-color-scheme).

Run from the repository root (benchmark environment):
    .venv-benchmark\\Scripts\\python benchmarks\\map_hartmann\\make_figures.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
# pylint: disable=wrong-import-position,import-error
from priors import PRIORS, distribution, mode  # noqa: E402

RESULTS = HERE / "results"
FIGURES = HERE / "figures"
N_HF = [5, 10, 20, 40]
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"

# role -> (light, dark)
INK = {"surface": ("#fcfcfb", "#1a1a19"), "page": ("#f9f9f7", "#0d0d0d"),
       "primary": ("#0b0b0b", "#ffffff"), "secondary": ("#52514e", "#c3c2b7"),
       "muted": ("#898781", "#898781"), "grid": ("#e1e0d9", "#2c2c2a"),
       "axis": ("#c3c2b7", "#383835")}
CATEGORICAL = [("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70"),
               ("#eda100", "#c98500"), ("#e87ba4", "#d55181")]
ORDINAL = {3: [("#86b6ef", "#184f95"), ("#2a78d6", "#3987e5"), ("#0d366b", "#b7d3f6")],
           4: [("#86b6ef", "#184f95"), ("#3987e5", "#2a78d6"), ("#1c5cab", "#6da7ec"),
               ("#0d366b", "#b7d3f6")],
           5: [("#86b6ef", "#184f95"), ("#5598e7", "#256abf"), ("#2a78d6", "#3987e5"),
               ("#1c5cab", "#6da7ec"), ("#0d366b", "#b7d3f6")]}


def _dark_map() -> dict:
    """light hex -> dark hex for every color of the figures (one dark step per light step)."""
    pairs = list(INK.values()) + CATEGORICAL + [p for ramp in ORDINAL.values() for p in ramp]
    mapping = {}
    for light, dark in pairs:
        assert mapping.setdefault(light, dark) == dark, f"two dark steps for {light}"
    return mapping


DARK = _dark_map()
SURFACE, PRIMARY, SECONDARY = INK["surface"][0], INK["primary"][0], INK["secondary"][0]
MUTED, GRID, AXIS = INK["muted"][0], INK["grid"][0], INK["axis"][0]
COLOR = {"MAP": CATEGORICAL[0][0], "MLE": CATEGORICAL[1][0], "BoTorch": CATEGORICAL[2][0]}
LABEL = {"MAP": "mfego MAP (prior IG(3, 2))", "MLE": "mfego MLE (sans prior)",
         "BoTorch": "BoTorch (MAP, priors par défaut)"}

# swaps every light color of the rendered figure for its dark step when the system is dark
DARK_JS = """
(function () {
  if (!window.matchMedia || !window.matchMedia('(prefers-color-scheme: dark)').matches) return;
  var MAP = %s;
  function hex2(v) { return ('0' + parseInt(v, 10).toString(16)).slice(-2); }
  function swap(c) {
    if (typeof c !== 'string') return c;
    var s = c.trim().toLowerCase();
    if (s[0] === '#') return MAP[s] || c;
    var m = s.match(/^rgba?\\(([^)]+)\\)$/);
    if (!m) return c;
    var p = m[1].split(',').map(function (v) { return v.trim(); });
    var d = MAP['#' + hex2(p[0]) + hex2(p[1]) + hex2(p[2])];
    if (!d) return c;
    var rgb = [1, 3, 5].map(function (i) { return parseInt(d.substr(i, 2), 16); });
    return p.length > 3 ? 'rgba(' + rgb.join(',') + ',' + p[3] + ')' : 'rgb(' + rgb.join(',') + ')';
  }
  function deep(o) {
    if (o === null || o === undefined || ArrayBuffer.isView(o)) return o;
    if (Array.isArray(o)) return o.map(deep);
    if (typeof o === 'object') { var r = {}; for (var k in o) r[k] = deep(o[k]); return r; }
    return swap(o);
  }
  var gd = document.getElementById('{plot_id}');
  document.body.style.background = MAP['%s'];
  Plotly.react(gd, gd.data.map(deep), deep(gd.layout));
})();
""" % (json.dumps(DARK), INK["page"][0])


def rgba(hex_color: str, alpha: float) -> str:
    """'#rrggbb' -> 'rgba(r,g,b,alpha)' (the dark-mode swap keeps the alpha)."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def fr(value: float, digits: int = 2) -> str:
    """French decimal comma."""
    return f"{value:.{digits}f}".replace(".", ",")


def style(fig: go.Figure, title: str, subtitle: str, height: int) -> go.Figure:
    """Common chrome: surfaces, ink, hairline solid grid, legend above the plots."""
    fig.update_layout(
        template="none", height=height, separators=", ",
        title={"text": title, "x": 0.0, "xanchor": "left", "y": 1.0 - 14 / height,
               "yanchor": "top", "font": {"size": 18, "color": PRIMARY},
               "subtitle": {"text": subtitle, "font": {"size": 13, "color": SECONDARY}}},
        font={"family": FONT, "size": 13, "color": PRIMARY},
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, hovermode="closest",
        hoverlabel={"font": {"family": FONT}},
        legend={"orientation": "h", "x": 0.0, "xanchor": "left", "y": 1.0 + 46 / (height - 220),
                "yanchor": "bottom", "font": {"color": SECONDARY}, "bgcolor": "rgba(0,0,0,0)"},
        margin={"l": 70, "r": 30, "t": 150, "b": 70})
    fig.update_xaxes(showgrid=True, gridcolor=GRID, gridwidth=1, showline=True, linecolor=AXIS,
                     zeroline=False, ticks="outside", tickcolor=AXIS, tickfont={"color": MUTED},
                     title_font={"color": SECONDARY, "size": 12})
    fig.update_yaxes(showgrid=True, gridcolor=GRID, gridwidth=1, showline=True, linecolor=AXIS,
                     zeroline=False, ticks="outside", tickcolor=AXIS, tickfont={"color": MUTED},
                     title_font={"color": SECONDARY, "size": 12})
    # log axes: one tick per decade (the default minor labels "2, 5" read as values)
    for axis in list(fig.select_xaxes()) + list(fig.select_yaxes()):
        if axis.type == "log" and axis.tickvals is None:
            axis.update(dtick=1)
    # subplot titles (no font set yet): secondary ink; the other annotations keep their font
    fig.for_each_annotation(lambda a: a.update(font={"color": SECONDARY, "size": 13})
                            if a.font.size is None else None)
    return fig


def save(fig: go.Figure, name: str) -> Path:
    path = FIGURES / f"{name}.html"
    fig.write_html(path, include_plotlyjs="cdn", post_script=DARK_JS,
                   config={"displaylogo": False, "responsive": True})
    return path


def axis_refs(fig, row, col) -> tuple[str, str, bool]:
    """('x2', 'y2', y is log) of a subplot (annotations on a log axis take log10 values)."""
    subplot = fig.get_subplot(row, col)
    return (subplot.xaxis.plotly_name.replace("axis", ""),
            subplot.yaxis.plotly_name.replace("axis", ""), subplot.yaxis.type == "log")


def label(fig, y, text, row, col, position="top left"):
    """Text-ink label at height y of a subplot, on its left or right edge."""
    xref, yref, ylog = axis_refs(fig, row, col)
    vertical, horizontal = position.split()
    fig.add_annotation(x=0.0 if horizontal == "left" else 1.0, xref=f"{xref} domain",
                       xanchor=horizontal, y=np.log10(y) if ylog else y, yref=yref,
                       yanchor="bottom" if vertical == "top" else "top", text=text,
                       showarrow=False, font={"color": SECONDARY, "size": 11})


def hline(fig, y, text, row, col, color=MUTED, position="top left"):
    """Hairline reference (solid) with a text-ink label (set the axis type first)."""
    fig.add_hline(y=y, line={"color": color, "width": 1}, row=row, col=col)
    label(fig, y, text, row, col, position)


def ticks_125(fig, row, col, axis="y"):
    """1-2-5 ticks with French decimals on a log axis spanning about two decades."""
    values = [m * 10.0 ** e for e in range(-3, 4) for m in (1, 2, 5)]
    text = [f"{v:g}".replace(".", ",") for v in values]
    update = fig.update_yaxes if axis == "y" else fig.update_xaxes
    update(tickvals=values, ticktext=text, row=row, col=col)


def load_lists(frame: pd.DataFrame, columns: list) -> pd.DataFrame:
    """CSV cells holding JSON lists -> Python lists."""
    for column in columns:
        frame[column] = frame[column].apply(lambda s: json.loads(s) if isinstance(s, str)
                                            else None)
    return frame


# 1/7 ---------------------------------------------------------------------------------------------
def figure_accuracy(acc: pd.DataFrame) -> Path:
    """RMSE / NLPD / coverage vs n_HF, MF and SF rows, mfego MAP vs MLE vs BoTorch."""
    rows = {"MF": {"MAP": "mfego MF (MAP)", "MLE": "mfego MF (MLE)",
                   "BoTorch": "BoTorch MF-GP (MF)"},
            "SF": {"MAP": "mfego SF (MAP)", "MLE": "mfego SF (MLE)",
                   "BoTorch": "BoTorch SingleTaskGP (SF)"}}
    metrics = [("rmse_rel", "RMSE relative", False), ("nlpd", "NLPD (échelle log)", True),
               ("coverage_95", "Couverture de l'intervalle à 95 %", False)]
    titles = [f"{title} · {'multifidélité' if v == 'MF' else 'HF seule'}"
              for v in rows for _, title, _ in metrics]
    fig = make_subplots(rows=2, cols=3, subplot_titles=titles, horizontal_spacing=0.08,
                        vertical_spacing=0.16)
    offsets = {"MAP": 0.95, "MLE": 1.0, "BoTorch": 1.05}
    for r, (variant, models) in enumerate(rows.items(), start=1):
        for c, (metric, _, log) in enumerate(metrics, start=1):
            for est, model in models.items():
                grouped = acc[acc["model"] == model].groupby("n_hf")[metric]
                med, q1, q3 = grouped.median(), grouped.quantile(0.25), grouped.quantile(0.75)
                n = med.index.to_numpy()
                fig.add_trace(go.Scatter(
                    x=n * offsets[est], y=med, mode="lines+markers", name=LABEL[est],
                    legendgroup=est, showlegend=(r == 1 and c == 1),
                    line={"color": COLOR[est], "width": 2},
                    marker={"size": 8, "color": COLOR[est], "line": {"color": SURFACE,
                                                                     "width": 2}},
                    error_y={"type": "data", "symmetric": False, "array": q3 - med,
                             "arrayminus": med - q1, "color": COLOR[est], "thickness": 1,
                             "width": 0},
                    customdata=np.column_stack([n, q1, q3]),
                    hovertemplate=f"<b>{LABEL[est]}</b> · {variant}<br>n_HF = %{{customdata[0]}}"
                                  "<br>médiane %{y:.3f}<br>quartiles [%{customdata[1]:.3f} ; "
                                  "%{customdata[2]:.3f}]<extra></extra>"), row=r, col=c)
                if metric == "nlpd" and variant == "MF":
                    fig.add_annotation(x=np.log10(n[-1] * 1.12), y=np.log10(med.iloc[-1]),
                                       text=est, showarrow=False, xanchor="left",
                                       font={"color": SECONDARY, "size": 11}, row=r, col=c)
            if log:
                fig.update_yaxes(type="log", row=r, col=c)
            if metric == "coverage_95":
                hline(fig, 0.95, "nominal 95 %", r, c, position="bottom left")
    fig.update_xaxes(type="log", tickvals=N_HF, ticktext=[str(n) for n in N_HF],
                     range=[np.log10(4.2), np.log10(58)])
    fig.update_xaxes(title_text="n_HF (n_LF = 2 n_HF)", row=2)
    style(fig, "Précision du modèle de substitution sur Hartmann 6D",
          "Médiane sur 10 graines, barres = quartiles · 2000 points de test · même plan "
          "d'expériences pour tous les modèles d'une graine", 760)
    return save(fig, "precision_vs_n")


# 2/7 ---------------------------------------------------------------------------------------------
def figure_paired(acc: pd.DataFrame) -> Path:
    """MLE vs MAP per seed (same DOE): below the diagonal the MAP is better."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, subplot_titles=[
        "RMSE relative", "NLPD (échelle log)"])
    ramp = [light for light, _ in ORDINAL[4]]
    for c, (metric, log) in enumerate((("rmse_rel", False), ("nlpd", True)), start=1):
        values = []
        for variant, symbol in (("MF", "circle"), ("SF", "diamond")):
            mle, map_ = f"mfego {variant} (MLE)", f"mfego {variant} (MAP)"
            wide = acc[acc["model"].isin([mle, map_])].pivot_table(
                index=["n_hf", "seed"], columns="model", values=metric).reset_index()
            values += [wide[mle], wide[map_]]
            for i, n in enumerate(N_HF):
                sub = wide[wide["n_hf"] == n]
                fig.add_trace(go.Scatter(
                    x=sub[mle], y=sub[map_], mode="markers", name=f"n_HF = {n}",
                    legendgroup=f"n{n}", showlegend=(c == 1 and variant == "MF"),
                    marker={"symbol": symbol, "size": 10, "color": ramp[i],
                            "line": {"color": SURFACE, "width": 2}},
                    customdata=np.column_stack([sub["seed"], np.full(len(sub), n)]),
                    hovertemplate=f"{variant} · n_HF = %{{customdata[1]}} · graine "
                                  "%{customdata[0]}<br>MLE %{x:.3f} → MAP %{y:.3f}"
                                  "<extra></extra>"), row=1, col=c)
        all_values = pd.concat(values)
        low, high = all_values.min() * 0.9, all_values.max() * 1.1
        fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", hoverinfo="skip",
                                 line={"color": MUTED, "width": 1}, showlegend=False),
                      row=1, col=c)
        suffix = "" if c == 1 else str(c)
        fig.add_annotation(x=0.97, y=0.04, xref=f"x{suffix} domain", yref=f"y{suffix} domain",
                           text="MAP meilleur ↘ sous la diagonale", showarrow=False,
                           xanchor="right", font={"color": SECONDARY, "size": 11})
        axis = {"type": "log", "range": [np.log10(low), np.log10(high)]} if log \
            else {"range": [low, high]}
        fig.update_xaxes(title_text="MLE (sans prior)", row=1, col=c, **axis)
        fig.update_yaxes(title_text="MAP (prior IG(3, 2))", row=1, col=c, **axis)
    for symbol, text in (("circle", "multifidélité (MF)"), ("diamond", "HF seule (SF)")):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=text,
                                 marker={"symbol": symbol, "size": 10, "color": MUTED}))
    style(fig, "MAP contre MLE, graine par graine",
          "Chaque point = un plan d'expériences (10 graines × 4 tailles × MF/SF) ; "
          "sous la diagonale, le MAP fait mieux", 560)
    return save(fig, "map_vs_mle_par_graine")


# 3/7 ---------------------------------------------------------------------------------------------
def figure_lengthscales(acc: pd.DataFrame, reference: dict) -> Path:
    """Fitted lengthscales (all dimensions, all seeds): MLE runs to the bounds, MAP does not."""
    panels = (("SF", "HF", "HF seule : GP de Hartmann"),
              ("MF", "delta", "Multifidélité : GP du résidu δ (niveau 2)"))
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                        subplot_titles=[p[2] for p in panels])
    fig.update_yaxes(type="log", range=[np.log10(0.006), np.log10(15)])
    fig.update_xaxes(type="category", categoryorder="array",
                     categoryarray=[str(n) for n in N_HF])
    for c, (variant, ref_key, _) in enumerate(panels, start=1):
        for est in ("MLE", "MAP"):
            rows = acc[acc["model"] == f"mfego {variant} ({est})"]
            x = [str(n) for n, ls in zip(rows["n_hf"], rows["hf_lengthscales"]) for _ in ls]
            y = [v for ls in rows["hf_lengthscales"] for v in ls]
            fig.add_trace(go.Box(
                x=x, y=y, name=LABEL[est], legendgroup=est, showlegend=(c == 1),
                marker={"color": COLOR[est], "size": 5, "opacity": 0.55},
                line={"color": COLOR[est], "width": 1.5}, fillcolor=rgba(COLOR[est], 0.12),
                boxpoints="all", jitter=0.5, pointpos=0,
                hovertemplate=f"{LABEL[est]}<br>n_HF = %{{x}}<br>l = %{{y:.3f}}<extra></extra>"),
                row=1, col=c)
        ref = np.array(reference[ref_key])
        active = ref[ref < 1.0]
        fig.add_hrect(y0=active.min(), y1=active.max(), fillcolor=rgba(MUTED, 0.18),
                      line={"width": 0}, row=1, col=c)
        label(fig, active.min(), "référence (300 points)", 1, c, position="bottom left")
        for value in ref[ref >= 1.0]:
            hline(fig, value, f"référence, dimension quasi inactive ({fr(value, 1)})", 1, c)
        hline(fig, 0.01, "borne basse 0,01", 1, c, position="bottom right")
        hline(fig, 10.0, "borne haute 10", 1, c, position="top right")
        hline(fig, 0.5, "mode de la prior 0,5", 1, c, color=COLOR["MAP"],
              position="bottom right")
    fig.update_layout(boxmode="group", boxgap=0.25, boxgroupgap=0.1)
    fig.update_yaxes(title_text="longueur de corrélation l (entrées dans [0, 1])", row=1, col=1)
    fig.update_xaxes(title_text="n_HF (n_LF = 2 n_HF)")
    style(fig, "Longueurs de corrélation ajustées (niveau HF)",
          "6 dimensions × 10 graines par boîte · bande grise = longueurs de référence des "
          "dimensions actives (scikit-learn, 300 points)", 560)
    return save(fig, "longueurs_correlation")


# 4/7 ---------------------------------------------------------------------------------------------
METHODS = {"mfego NN-MF-EGO (MAP)": ("MAP", "MF", "NN-MF-EGO · MAP"),
           "mfego NN-MF-EGO (MLE)": ("MLE", "MF", "NN-MF-EGO · MLE"),
           "mfego SF-EGO (MAP)": ("MAP", "SF", "SF-EGO · MAP"),
           "mfego SF-EGO (MLE)": ("MLE", "SF", "SF-EGO · MLE"),
           "BoTorch qLogEI (SF)": ("BoTorch", "SF", "BoTorch qLogEI (SF)")}


def best_on_grid(history: dict, grid: np.ndarray) -> np.ndarray:
    """Best observed HF error at each cost of the grid (step function of the history)."""
    index = np.searchsorted(history["cost"], grid, side="right") - 1
    return np.asarray(history["best_observed"])[np.clip(index, 0, None)]


def figure_optimization(histories: list) -> Path:
    """Convergence (median, quartile band), final recommendation error and LF share."""
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.17, horizontal_spacing=0.08,
                        subplot_titles=["Convergence · multifidélité (NN-MF-EGO)",
                                        "Convergence · HF seule",
                                        "Erreur de la recommandation finale",
                                        "Part d'évaluations basse fidélité (NN-MF-EGO)"])
    grid = np.linspace(120, 250, 131)
    for method, (est, variant, label) in METHODS.items():
        runs = [h for h in histories if h["method"] == method and "error" not in h]
        color = COLOR[est]
        curves = np.maximum(np.array([best_on_grid(h, grid) for h in runs]), 1e-4)
        q1, med, q3 = np.percentile(curves, [25, 50, 75], axis=0)
        col = 1 if variant == "MF" else 2
        fig.add_trace(go.Scatter(x=np.concatenate([grid, grid[::-1]]),
                                 y=np.concatenate([q3, q1[::-1]]), fill="toself",
                                 fillcolor=rgba(color, 0.14), line={"width": 0},
                                 hoverinfo="skip", showlegend=False, legendgroup=method),
                      row=1, col=col)
        fig.add_trace(go.Scatter(x=grid, y=med, mode="lines", name=label, legendgroup=method,
                                 line={"color": color, "width": 2, "shape": "hv"},
                                 hovertemplate=f"<b>{label}</b><br>coût %{{x:.0f}}<br>"
                                               "f − f* médiane %{y:.3f}<extra></extra>"),
                      row=1, col=col)
        errors = [h["recommendation_error"] for h in runs]
        fig.add_trace(go.Box(x=[label] * len(runs), y=errors, name=label, legendgroup=method,
                             showlegend=False, marker={"color": color, "size": 8},
                             line={"color": color, "width": 1.5}, fillcolor=rgba(color, 0.12),
                             boxpoints="all", jitter=0.4, pointpos=0,
                             customdata=[h["seed"] for h in runs],
                             hovertemplate=f"{label}<br>graine %{{customdata}}<br>"
                                           "f(x_reco) − f* = %{y:.3f}<extra></extra>"),
                      row=2, col=1)
        if variant == "MF":
            shares = [float(np.mean(np.array(h["levels"][1:]) == 1)) for h in runs]
            fig.add_trace(go.Box(x=[label] * len(runs), y=shares, name=label,
                                 legendgroup=method, showlegend=False,
                                 marker={"color": color, "size": 8},
                                 line={"color": color, "width": 1.5},
                                 fillcolor=rgba(color, 0.12), boxpoints="all", jitter=0.4,
                                 pointpos=0, customdata=[h["seed"] for h in runs],
                                 hovertemplate=f"{label}<br>graine %{{customdata}}<br>"
                                               "part BF %{y:.0%}<extra></extra>"),
                          row=2, col=2)
    for col in (1, 2):
        fig.update_yaxes(type="log", title_text="f − f* (meilleure observation HF)", row=1,
                         col=col)
        fig.update_xaxes(title_text="coût cumulé (plan initial = 120)", row=1, col=col)
    fig.update_yaxes(type="log", title_text="f(x_reco) − f*", row=2, col=1)
    for row, col in ((1, 1), (1, 2), (2, 1)):
        ticks_125(fig, row, col)
    fig.update_yaxes(tickformat=".0%", range=[0, 1], title_text="évaluations BF / total",
                     row=2, col=2)
    style(fig, "Optimisation de Hartmann 6D à budget égal (250)",
          "10 graines · ligne = médiane, bande = quartiles · coût HF = 10, coût BF = 1", 860)
    return save(fig, "optimisation")


# 5/7 ---------------------------------------------------------------------------------------------
GROUP_TITLE = {"mode": "Position du mode (α = 3)", "strength": "Force (mode 0,5)",
               "family": "Autres lois (mode 0,5)", "library": "Défauts BoTorch"}


def prior_label(name: str) -> str:
    """'Gamma(3, 6) BoTorch MF' -> 'Gamma(3, 6)<br>BoTorch MF<br>mode 0,33'."""
    head, _, tail = name.partition(") ")
    head = head + ")" if tail else name
    lines = [head] + ([tail] if tail else []) + [f"mode {fr(mode(PRIORS[name][1]))}"]
    return "<br>".join(lines)


def figure_sensitivity(acc: pd.DataFrame, prior_acc: pd.DataFrame, prior_opt: pd.DataFrame,
                       histories: list) -> Path:
    """Accuracy and optimization error for every prior configuration (+ MLE baseline)."""
    baseline = "MLE<br>(sans prior)"
    order = [baseline] + [prior_label(name) for name in PRIORS]
    acc_mle = acc[acc["model"] == "mfego MF (MLE)"].assign(label=baseline)
    pacc = pd.concat([acc_mle, prior_acc.assign(label=prior_acc["prior"].map(prior_label))])
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    ramp = [light for light, _ in ORDINAL[4]]
    for r, metric in ((1, "nlpd"), (2, "rmse_rel")):
        for i, n in enumerate(N_HF):
            med = pacc[pacc["n_hf"] == n].groupby("label")[metric].median().reindex(order)
            fig.add_trace(go.Scatter(
                x=order, y=med, mode="markers", name=f"n_HF = {n}", legendgroup=f"n{n}",
                showlegend=(r == 1),
                marker={"size": 11, "color": ramp[i], "line": {"color": SURFACE, "width": 2}},
                hovertemplate=f"%{{x}}<br>n_HF = {n}<br>médiane %{{y:.3f}}<extra></extra>"),
                row=r, col=1)
    fig.update_yaxes(type="log", title_text="NLPD médiane (log)", row=1, col=1)
    fig.update_yaxes(title_text="RMSE relative médiane", row=2, col=1)
    opt_mle = [h["recommendation_error"] for h in histories
               if h["method"] == "mfego NN-MF-EGO (MLE)" and "error" not in h]
    boxes = [(baseline, opt_mle, COLOR["MLE"])] + [
        (prior_label(name), prior_opt[prior_opt["prior"] == name]["recommendation error"],
         COLOR["MAP"]) for name in PRIORS]
    for label, errors, color in boxes:
        fig.add_trace(go.Box(x=[label] * len(errors), y=list(errors), name=label,
                             showlegend=False, marker={"color": color, "size": 7},
                             line={"color": color, "width": 1.5}, fillcolor=rgba(color, 0.12),
                             boxpoints="all", jitter=0.4, pointpos=0,
                             hovertemplate="%{x}<br>f(x_reco) − f* = %{y:.3f}<extra></extra>"),
                      row=3, col=1)
    fig.update_yaxes(type="log", title_text="erreur de recommandation<br>NN-MF-EGO (log)",
                     row=3, col=1)
    ticks_125(fig, 1, 1)
    ticks_125(fig, 3, 1)
    fig.update_xaxes(tickangle=0, tickfont={"size": 11})
    # group separators and titles; light band behind the current default IG(3, 2)
    groups = [PRIORS[name][0] for name in PRIORS]
    position = 1
    for group in GROUP_TITLE:
        size = groups.count(group)
        for r in (1, 2, 3):
            fig.add_vline(x=position - 0.5, line={"color": AXIS, "width": 1}, row=r, col=1)
        fig.add_annotation(x=position + (size - 1) / 2, y=1.0, xref="x", yref="y domain",
                           text=GROUP_TITLE[group], showarrow=False, yanchor="bottom",
                           font={"color": SECONDARY, "size": 11})
        position += size
    default = order.index(prior_label("IG(3, 2)"))
    for r in (1, 2, 3):
        fig.add_vrect(x0=default - 0.5, x1=default + 0.5, fillcolor=rgba(MUTED, 0.12),
                      line={"width": 0}, row=r, col=1)
    fig.add_annotation(x=default, y=0.0, xref="x3", yref="y3 domain", text="défaut actuel",
                       showarrow=False, yanchor="bottom", font={"color": SECONDARY, "size": 11})
    style(fig, "Sensibilité du MAP au choix de la prior",
          "Hartmann 6D multifidélité · 10 graines · InvGamma = IG(α, β) ; Gamma(forme, taux) ; "
          "LogN(μ, σ) sur ln l", 1060)
    fig.update_layout(margin={"t": 190, "l": 90, "b": 90})
    return save(fig, "sensibilite_prior")


# 6/7 ---------------------------------------------------------------------------------------------
def figure_prior_shapes(reference: dict) -> Path:
    """Densities of ln l (l p(l)) of the tested priors, with the reference lengthscales."""
    panels = [("InvGamma, α = 3 : position du mode",
               ["IG(3, 0.4)", "IG(3, 1)", "IG(3, 2)", "IG(3, 4)", "IG(3, 8)"],
               [light for light, _ in ORDINAL[5]]),
              ("InvGamma, mode 0,5 : force de la prior",
               ["IG(1.5, 1.25)", "IG(3, 2)", "IG(10, 5.5)"], [light for light, _ in ORDINAL[3]]),
              ("Choix de la loi (mode 0,5) et défauts BoTorch",
               ["IG(3, 2)", "Gamma(3, 4)", "LogN(-0.13, 0.75)", "Gamma(3, 6) BoTorch MF",
                "LogN(2.31, 1.73) BoTorch"], [light for light, _ in CATEGORICAL])]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[p[0] for p in panels])
    lengthscales = np.logspace(np.log10(0.005), np.log10(300), 300)
    ref = np.concatenate([reference[key] for key in ("HF", "LF", "delta")])
    active = ref[ref < 1.0]
    for r, (_, names, colors) in enumerate(panels, start=1):
        for name, color in zip(names, colors):
            prior = PRIORS[name][1]
            density = lengthscales * distribution(prior).pdf(lengthscales)
            fig.add_trace(go.Scatter(
                x=lengthscales, y=density, mode="lines", name=name,
                legend="legend" if r == 1 else f"legend{r}", line={"color": color, "width": 2},
                hovertemplate=f"<b>{name}</b> (mode {fr(mode(prior))})<br>l = %{{x:.3f}}<br>"
                              "densité de ln l %{y:.3f}<extra></extra>"), row=r, col=1)
        fig.add_vrect(x0=active.min(), x1=active.max(), fillcolor=rgba(MUTED, 0.18),
                      line={"width": 0}, row=r, col=1)
        for bound in (0.01, 10.0):
            fig.add_vline(x=bound, line={"color": MUTED, "width": 1}, row=r, col=1)
    for bound, text in ((0.01, "borne basse"), (10.0, "borne haute")):
        fig.add_annotation(x=np.log10(bound), y=1.0, xref="x", yref="y domain", text=text,
                           showarrow=False, xanchor="left", yanchor="top", xshift=4,
                           font={"color": SECONDARY, "size": 11})
    fig.update_xaxes(type="log", range=[np.log10(0.005), np.log10(300)])
    fig.update_xaxes(title_text="longueur de corrélation l (entrées dans [0, 1])", row=3)
    fig.update_yaxes(title_text="densité de ln l", rangemode="tozero")
    style(fig, "Formes des priors testées",
          "Densité de ln l = l·p(l) (lisible en abscisse log) · bande grise = longueurs de "
          "référence des dimensions actives, HF, BF et résidu δ (scikit-learn, 300 points)", 900)
    fig.update_layout(margin={"r": 230, "t": 110}, **{
        "legend" if r == 1 else f"legend{r}": {
            "orientation": "v", "x": 1.02, "xanchor": "left", "yanchor": "top",
            "y": fig.layout["yaxis" if r == 1 else f"yaxis{r}"].domain[1],
            "font": {"color": SECONDARY}, "bgcolor": "rgba(0,0,0,0)"} for r in (1, 2, 3)})
    return save(fig, "formes_des_priors")


# 7/7 ---------------------------------------------------------------------------------------------
def figure_mle_diagnostic(diagnostic: pd.DataFrame, reference: dict) -> Path:
    """Where the mfego MLE fails: genuine overfitting (small n) vs optimizer failure (large n)."""
    diagnostic = diagnostic.assign(missed=diagnostic["nll_map"] < diagnostic["nll_mle"] - 1e-6)
    grouped = diagnostic.groupby("n")
    n = grouped.size().index.to_numpy()
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08, subplot_titles=[
        "Graines où l'optimiseur MLE<br>rate un meilleur optimum",
        "RMSE relative (GP seul, Hartmann)", "Longueur de corrélation médiane"])
    fig.update_xaxes(type="category", row=1, col=1)
    fig.update_yaxes(type="log", range=[np.log10(0.006), np.log10(15)], row=1, col=3)
    fig.add_trace(go.Bar(x=[str(v) for v in n], y=grouped["missed"].mean(), showlegend=False,
                         marker={"color": COLOR["MLE"], "line": {"width": 0}},
                         hovertemplate="n = %{x}<br>%{y:.0%} des graines<extra></extra>"),
                  row=1, col=1)
    for c, metric in ((2, "rmse_rel"), (3, "ls_median")):
        for est, label in (("mle", "MLE"), ("map", "MAP")):
            med = grouped[f"{metric}_{est}"].median()
            fig.add_trace(go.Scatter(
                x=n, y=med, mode="lines+markers", name=LABEL[label], legendgroup=label,
                showlegend=(c == 2), line={"color": COLOR[label], "width": 2},
                marker={"size": 8, "color": COLOR[label], "line": {"color": SURFACE,
                                                                   "width": 2}},
                hovertemplate=f"{LABEL[label]}<br>n = %{{x}}<br>médiane %{{y:.3f}}"
                              "<extra></extra>"), row=1, col=c)
        fig.update_xaxes(type="log", tickvals=list(n), ticktext=[str(v) for v in n], row=1,
                         col=c)
    ref = np.array(reference["HF"])
    fig.add_hrect(y0=ref[ref < 1].min(), y1=ref[ref < 1].max(), fillcolor=rgba(MUTED, 0.18),
                  line={"width": 0}, row=1, col=3)
    hline(fig, 0.01, "borne basse 0,01", 1, 3, position="top left")
    hline(fig, 10.0, "borne haute 10", 1, 3, position="bottom right")
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05], row=1, col=1)
    fig.update_layout(bargap=0.35)
    fig.update_xaxes(title_text="n (points HF)")
    style(fig, "Diagnostic de l'estimation MLE de mfego",
          "NLL pure au point MAP < NLL atteinte par le fit MLE ⇒ l'optimiseur MLE a raté un "
          "meilleur optimum · 10 graines par n", 520)
    return save(fig, "diagnostic_mle")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    acc = load_lists(pd.read_csv(RESULTS / "accuracy.csv"), ["hf_lengthscales"])
    acc = acc[acc["error"].isna()] if "error" in acc else acc
    histories = json.loads((RESULTS / "optimization_histories.json").read_text("utf-8"))
    prior_acc = pd.read_csv(RESULTS / "prior_accuracy.csv")
    prior_acc = prior_acc[prior_acc["error"].isna()] if "error" in prior_acc else prior_acc
    prior_opt = pd.read_csv(RESULTS / "prior_optimization.csv")
    reference = json.loads((RESULTS / "prior_reference_lengthscales.json").read_text("utf-8"))
    diagnostic = pd.read_csv(RESULTS / "mle_diagnostic.csv")
    paths = [figure_accuracy(acc), figure_paired(acc), figure_lengthscales(acc, reference),
             figure_optimization(histories),
             figure_sensitivity(acc, prior_acc, prior_opt, histories),
             figure_prior_shapes(reference), figure_mle_diagnostic(diagnostic, reference)]
    for path in paths:
        print(path.relative_to(HERE.parents[1]))


if __name__ == "__main__":
    main()

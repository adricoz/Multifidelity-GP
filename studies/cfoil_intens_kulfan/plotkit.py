"""
Common style of the interactive figures of the study (plotly, standalone HTML files).

Same chart system as benchmarks/map_hartmann/make_figures.py, in English:
* validated categorical palette, light and dark steps; the three fidelity levels always use the
  first three slots, which are colorblind-safe all-pairs in both modes (scatter plots included):
  L1 blue, L2 orange, L3 aqua; the baseline section is drawn in secondary ink, the optimum in
  primary ink, so that "reference vs result" never depends on a hue;
* text in ink tokens (never in a series color), hairline solid grid, legend above the plots;
* sequential blue ramp for magnitudes (GP mean / standard deviation maps);
* a small script swaps every light color for its dark step when the system is in dark mode.

Usage:
    fig = go.Figure(...); style(fig, "Title", "Subtitle", height=520); save(fig, "name")
"""
import json
from pathlib import Path

import plotly.graph_objects as go

FIGURES = Path(__file__).resolve().parent / "figures"
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"

# role -> (light, dark)
INK = {"surface": ("#fcfcfb", "#1a1a19"), "page": ("#f9f9f7", "#0d0d0d"),
       "primary": ("#0b0b0b", "#ffffff"), "secondary": ("#52514e", "#c3c2b7"),
       "muted": ("#898781", "#898781"), "grid": ("#e1e0d9", "#2c2c2a"),
       "axis": ("#c3c2b7", "#383835")}
CATEGORICAL = [("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70"),
               ("#eda100", "#c98500"), ("#e87ba4", "#d55181")]
# sequential blue ramp (light -> dark), the same in both modes (a magnitude scale)
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _dark_map() -> dict:
    """light hex -> dark hex of every color that has a dark step."""
    mapping = {}
    for light, dark in list(INK.values()) + CATEGORICAL:
        assert mapping.setdefault(light, dark) == dark, f"two dark steps for {light}"
    return mapping


DARK = _dark_map()
SURFACE, PRIMARY, SECONDARY = INK["surface"][0], INK["primary"][0], INK["secondary"][0]
MUTED, GRID, AXIS = INK["muted"][0], INK["grid"][0], INK["axis"][0]
LEVEL_COLOR = {1: CATEGORICAL[0][0], 2: CATEGORICAL[1][0], 3: CATEGORICAL[2][0]}
LEVEL_SYMBOL = {1: "circle", 2: "square", 3: "diamond"}
SEQUENTIAL_SCALE = [[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)]

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


def style(fig: go.Figure, title: str, subtitle: str, height: int, top: int = 150) -> go.Figure:
    """Common chrome: surfaces, ink, hairline solid grid, legend above the plots."""
    fig.update_layout(
        template="none", height=height,
        title={"text": title, "x": 0.0, "xanchor": "left", "y": 1.0 - 14 / height,
               "yanchor": "top", "font": {"size": 18, "color": PRIMARY},
               "subtitle": {"text": subtitle, "font": {"size": 13, "color": SECONDARY}}},
        font={"family": FONT, "size": 13, "color": PRIMARY},
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, hovermode="closest",
        hoverlabel={"font": {"family": FONT}},
        legend={"orientation": "h", "x": 0.0, "xanchor": "left", "y": 1.0 + 46 / (height - 220),
                "yanchor": "bottom", "font": {"color": SECONDARY}, "bgcolor": "rgba(0,0,0,0)"},
        margin={"l": 80, "r": 30, "t": top, "b": 70})
    axis_style = {"showgrid": True, "gridcolor": GRID, "gridwidth": 1, "showline": True,
                  "linecolor": AXIS, "zeroline": False, "ticks": "outside", "tickcolor": AXIS,
                  "tickfont": {"color": MUTED}, "title_font": {"color": SECONDARY, "size": 12}}
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    for axis in list(fig.select_xaxes()) + list(fig.select_yaxes()):
        if axis.type == "log" and axis.tickvals is None:
            axis.update(dtick=1)
    fig.for_each_annotation(lambda a: a.update(font={"color": SECONDARY, "size": 13})
                            if a.font.size is None else None)
    return fig


def save(fig: go.Figure, name: str, folder: Path = None) -> Path:
    """Standalone HTML (plotly from the CDN) with the dark-mode swap."""
    folder = folder or FIGURES
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.html"
    fig.write_html(path, include_plotlyjs="cdn", post_script=DARK_JS,
                   config={"displaylogo": False, "responsive": True})
    return path

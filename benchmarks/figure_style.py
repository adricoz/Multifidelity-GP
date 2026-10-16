"""
Shared chrome of the interactive benchmark figures (plotly HTML): validated categorical palette
with light and dark steps (the dark steps are swapped in by the browser when the system is in
dark mode), surfaces, ink, hairline grid, legend above the plots. Same palette and style as
benchmarks/map_hartmann/make_figures.py.
"""
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

FONT = "system-ui, -apple-system, Segoe UI, sans-serif"
# role -> (light, dark)
INK = {"surface": ("#fcfcfb", "#1a1a19"), "page": ("#f9f9f7", "#0d0d0d"),
       "primary": ("#0b0b0b", "#ffffff"), "secondary": ("#52514e", "#c3c2b7"),
       "muted": ("#898781", "#898781"), "grid": ("#e1e0d9", "#2c2c2a"),
       "axis": ("#c3c2b7", "#383835")}
# categorical slots in their fixed (validated) order
CATEGORICAL = [("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70"),
               ("#eda100", "#c98500"), ("#e87ba4", "#d55181"), ("#008300", "#008300"),
               ("#6250d6", "#9085e9"), ("#e34948", "#e66767")]
SERIES = [light for light, _ in CATEGORICAL]
SURFACE, PRIMARY, SECONDARY = INK["surface"][0], INK["primary"][0], INK["secondary"][0]
MUTED, GRID, AXIS = INK["muted"][0], INK["grid"][0], INK["axis"][0]
DARK = {light: dark for light, dark in list(INK.values()) + CATEGORICAL}

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


def style(fig: go.Figure, title: str, subtitle: str, height: int,
          legend_rows: int = 1) -> go.Figure:
    """Common chrome: surfaces, ink, hairline grid, legend above the plots (below the title and
    the subtitle; legend_rows = number of rows the legend needs)."""
    top, bottom = 118 + 24 * legend_rows, 70
    fig.update_layout(
        template="none", height=height,
        title={"text": title, "x": 0.0, "xanchor": "left", "y": 1.0 - 14 / height,
               "yanchor": "top", "font": {"size": 18, "color": PRIMARY},
               "subtitle": {"text": subtitle, "font": {"size": 13, "color": SECONDARY}}},
        font={"family": FONT, "size": 13, "color": PRIMARY},
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, hovermode="closest",
        hoverlabel={"font": {"family": FONT}},
        legend={"orientation": "h", "x": 0.0, "xanchor": "left",
                "y": 1.0 + (top - 80) / (height - top - bottom), "yanchor": "top",
                "font": {"color": SECONDARY}, "bgcolor": "rgba(0,0,0,0)"},
        margin={"l": 70, "r": 30, "t": top, "b": bottom})
    for update in (fig.update_xaxes, fig.update_yaxes):
        update(showgrid=True, gridcolor=GRID, gridwidth=1, showline=True, linecolor=AXIS,
               zeroline=False, ticks="outside", tickcolor=AXIS, tickfont={"color": MUTED},
               title_font={"color": SECONDARY, "size": 12})
    for axis in list(fig.select_xaxes()) + list(fig.select_yaxes()):
        if axis.type == "log" and axis.tickvals is None:
            axis.update(dtick=1)
    fig.for_each_annotation(lambda a: a.update(font={"color": SECONDARY, "size": 13})
                            if a.font.size is None else None)
    return fig


def hline(fig: go.Figure, y: float, text: str, row: int, col: int, color: str = MUTED,
          dash: str = "dot", position: str = "top left") -> None:
    """Hairline reference at height y of a subplot with a text-ink label. Works on log axes
    (set the axis type BEFORE calling it): annotations on a log axis take log10 values."""
    fig.add_hline(y=y, line={"color": color, "width": 1, "dash": dash}, row=row, col=col)
    subplot = fig.get_subplot(row, col)
    xref = subplot.xaxis.plotly_name.replace("axis", "")
    yref = subplot.yaxis.plotly_name.replace("axis", "")
    vertical, horizontal = position.split()
    fig.add_annotation(x=0.0 if horizontal == "left" else 1.0, xref=f"{xref} domain",
                       xanchor=horizontal, y=np.log10(y) if subplot.yaxis.type == "log" else y,
                       yref=yref, yanchor="bottom" if vertical == "top" else "top", text=text,
                       showarrow=False, font={"color": SECONDARY, "size": 11})


def save(fig: go.Figure, folder: Path, name: str) -> Path:
    """Writes folder/name.html (plotly.js from the CDN, dark-mode swap script)."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.html"
    fig.write_html(path, include_plotlyjs="cdn", post_script=DARK_JS,
                   config={"displaylogo": False, "responsive": True})
    return path

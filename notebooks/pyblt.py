"""
pyblt — a Python viewer for bio-logger HDF5 files produced by the R package rblt.

Reads the .h5 files written by rblt (axytrek2h5, cats2h5, lul2h5, wacu2h5 and
saveasloggerdata / LDATA), optionally overlays BORIS behaviour labels, and draws
synchronised, zoomable plots with plotly + plotly-resampler (works in Colab).

Inspired by rblt (Sebastien Geiger, CNRS IPHC, GPL >= 3): https://github.com/sg4r/rblt
"""
from __future__ import annotations

from dataclasses import dataclass, field

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

# Column layout per logger type, as defined in rblt's getdata()/initmetriclst()
# Each metric: (display name, column prefix, number of columns)
LOGGER_METRICS = {
    "AXYTREK": [("Accelerometer", "a", 3), ("Pressure", "p", 1), ("Temperature", "t", 1)],
    "CATS": [("Accelerometer", "a", 3), ("Gyroscope", "g", 3), ("Magnetometer", "m", 3),
             ("Temperature", "t", 1), ("Pressure", "p", 1), ("Light intensity", "l", 1)],
    "LUL": [("Temperature", "t", 1), ("Pressure", "p", 1), ("Light intensity", "l", 1)],
    "WACU": [("Temperature", "t", 1), ("Pressure", "p", 1), ("Light intensity", "l", 1),
             ("Accelerometer", "a", 3)],
}


def _colnames(metrics):
    names = []
    for _, prefix, n in metrics:
        names += [f"{prefix}{j}" for j in range(1, n + 1)] if n > 1 else [prefix]
    return names


def _attr(f, key, default=None):
    """Read an HDF5 attribute written by hdf5r and return a plain Python value."""
    if key not in f.attrs:
        return default
    v = f.attrs[key]
    if isinstance(v, np.ndarray):
        v = v.ravel()[0] if v.size == 1 else v
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, np.generic):
        v = v.item()
    return v


def _decode(arr):
    return np.vectorize(lambda x: x.decode() if isinstance(x, bytes) else str(x))(arr)


@dataclass
class Logger:
    """One bio-logger recording loaded from an rblt .h5 file."""
    path: str
    logger: str
    version: str
    datestart: pd.Timestamp
    accres: float
    rtctick: float
    filesrc: str
    metrics: list
    data: pd.DataFrame
    behaviors: pd.DataFrame | None = field(default=None)

    @property
    def dt_seconds(self):
        return self.rtctick / self.accres

    def summary(self):
        dur = pd.to_timedelta(len(self.data) * self.dt_seconds, unit="s")
        print(f"{self.path}\n  logger: {self.logger} (v{self.version}), source: {self.filesrc}\n"
              f"  start: {self.datestart} UTC, rate: {self.accres / self.rtctick:g} Hz, "
              f"rows: {len(self.data):,}, duration: {dur}\n"
              f"  metrics: {', '.join(m[0] for m in self.metrics)}")
        if self.behaviors is not None:
            counts = self.behaviors["Behavior"].value_counts()
            print("  behaviours:", ", ".join(f"{b} ({n})" for b, n in counts.items()))


def read_rblt_h5(path, dtype=np.float32) -> Logger:
    """Load an rblt .h5 file into a pandas DataFrame indexed by UTC time."""
    with h5py.File(path, "r") as f:
        logger = str(_attr(f, "logger", "")).upper()
        version = str(_attr(f, "version", ""))
        datestart = pd.Timestamp(str(_attr(f, "datestart")), tz="UTC")
        accres = float(_attr(f, "accres", 1))
        rtctick = float(_attr(f, "rtctick", 1))
        filesrc = str(_attr(f, "filesrc", ""))

        if logger == "LDATA":
            metrics = _read_ldata_metrics(f)
        elif logger in LOGGER_METRICS:
            metrics = LOGGER_METRICS[logger]
        else:
            raise ValueError(f"{path}: unknown logger type {logger!r}")
        cols = _colnames(metrics)

        raw = f["data"][()]
    # R (column-major) and Python (row-major) see the matrix transposed;
    # orient it so that rows = samples and columns = channels.
    if raw.shape[1] != len(cols) and raw.shape[0] == len(cols):
        raw = raw.T
    if raw.shape[1] != len(cols):
        raise ValueError(f"{path}: data has shape {raw.shape}, expected {len(cols)} columns {cols}")

    n = raw.shape[0]
    step = pd.to_timedelta(rtctick / accres, unit="s")
    index = pd.date_range(datestart, periods=n, freq=step, name="time")
    # one contiguous array per channel (the downsampler needs contiguous arrays)
    df = pd.DataFrame({c: np.ascontiguousarray(raw[:, i], dtype=dtype)
                       for i, c in enumerate(cols)}, index=index)
    return Logger(path, logger, version, datestart, accres, rtctick, filesrc, metrics, df)


def _read_ldata_metrics(f):
    """LDATA files (rblt saveasloggerdata) store their own metric table."""
    m = _decode(f["metriclst"][()])
    if m.ndim == 1:
        m = m.reshape(1, -1)
    # columns: name, colname, colid, colnb, enable, srcin, beobs, height
    if m.shape[1] != 8 or not np.isin(m[:, 4], ["TRUE", "FALSE"]).all():
        m = m.T
    return [(str(row[0]), str(row[1]), int(float(row[3]))) for row in m]


def read_boris(path, datestart, sep=","):
    """Read a BORIS aggregated-events CSV (Behavior, Start (s), Stop (s)).

    Times are seconds relative to the logger start, as in rblt.
    """
    b = pd.read_csv(path, sep=sep)
    b.columns = [c.strip() for c in b.columns]
    rename = {}
    for c in b.columns:
        key = c.lower().replace(".", " ").replace("(", " ").replace(")", " ").split()
        if key[:1] == ["start"]:
            rename[c] = "start_s"
        elif key[:1] == ["stop"]:
            rename[c] = "stop_s"
    b = b.rename(columns=rename)
    missing = {"Behavior", "start_s", "stop_s"} - set(b.columns)
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}; found {list(b.columns)}")
    b["start"] = datestart + pd.to_timedelta(b["start_s"], unit="s")
    b["stop"] = datestart + pd.to_timedelta(b["stop_s"], unit="s")
    return b


def load(h5_path, boris_path=None, sep=","):
    """Convenience: load a logger and (optionally) its behaviour file."""
    lg = read_rblt_h5(h5_path)
    if boris_path:
        lg.behaviors = read_boris(boris_path, lg.datestart, sep=sep)
    return lg


def plot(lg: Logger, metrics=None, behaviors=None, start=None, end=None,
         widget=True, n_shown_samples=2000, row_height=200, opacity=0.25):
    """Synchronised plots, one panel per metric, with behaviour shading.

    metrics   : list of metric names to show (default: all)
    behaviors : list of behaviour names to shade (default: all)
    start/end : optional time window, e.g. "2023-03-01 10:00"
    widget    : True -> FigureWidgetResampler (re-samples on zoom; Colab needs the
                custom widget manager). False -> plain downsampled plotly figure.
    """
    from plotly_resampler import FigureWidgetResampler, FigureResampler

    sel = [m for m in lg.metrics if metrics is None or m[0] in metrics]
    df = lg.data.loc[start:end]
    base = make_subplots(rows=len(sel), cols=1, shared_xaxes=True, vertical_spacing=0.03,
                         subplot_titles=[m[0] for m in sel])
    fig = (FigureWidgetResampler if widget else FigureResampler)(
        base, default_n_shown_samples=n_shown_samples)

    for r, (name, prefix, n) in enumerate(sel, start=1):
        cols = [f"{prefix}{j}" for j in range(1, n + 1)] if n > 1 else [prefix]
        for c in cols:
            fig.add_trace(go.Scattergl(name=c, line=dict(width=1), legendgroup=name),
                          hf_x=df.index, hf_y=np.ascontiguousarray(df[c].to_numpy()),
                          row=r, col=1)

    if lg.behaviors is not None:
        b = lg.behaviors
        names = sorted(b["Behavior"].unique())
        palette = qualitative.Plotly + qualitative.D3
        colour = {nm: palette[i % len(palette)] for i, nm in enumerate(names)}
        b = b[b["Behavior"].isin(behaviors if behaviors is not None else names)]
        if start is not None or end is not None:
            lo = pd.Timestamp(start, tz="UTC") if start is not None else df.index[0]
            hi = pd.Timestamp(end, tz="UTC") if end is not None else df.index[-1]
            b = b[(b["stop"] >= lo) & (b["start"] <= hi)]
        shapes = [dict(type="rect", xref="x", yref="paper", x0=s, x1=e, y0=0, y1=1,
                       fillcolor=colour[nm], opacity=opacity, line_width=0, layer="below")
                  for nm, s, e in zip(b["Behavior"], b["start"], b["stop"])]
        fig.update_layout(shapes=shapes)
        for nm in b["Behavior"].unique():  # legend entries for behaviours
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=nm,
                                     marker=dict(size=12, symbol="square", color=colour[nm]),
                                     legendgroup="behaviours",
                                     legendgrouptitle_text="Behaviours"))

    fig.update_layout(height=row_height * len(sel) + 120, title=lg.path.split("/")[-1],
                      margin=dict(l=60, r=20, t=80, b=40), hovermode="x unified")
    return fig


def show(lg: Logger, metrics=None, behaviors=None, start=None, end=None, **plot_kwargs):
    """Display plot() as a small inline Dash app that re-samples on zoom.

    Needs no custom widgets, so it works in Colab as well as VS Code / Jupyter.
    """
    fig = plot(lg, metrics=metrics, behaviors=behaviors, start=start, end=end,
               widget=False, **plot_kwargs)
    fig.show_dash(mode="inline")


def viewer(lg: Logger, widget=True, **plot_kwargs):
    """A small control panel (like rblt's Shiny sidebar) around plot().

    widget=True : the zoomable figure is placed directly in the panel (it is itself
                  a widget, so it must not be wrapped in an Output area in Colab).
    widget=False: a static plotly figure, drawn in an Output area (always displays,
                  but does not re-sample when you zoom).
    """
    import ipywidgets as w
    from IPython.display import display

    names = [m[0] for m in lg.metrics]
    m_sel = w.SelectMultiple(options=names, value=tuple(names), description="Metrics",
                             rows=min(len(names), 6))
    behs = sorted(lg.behaviors["Behavior"].unique()) if lg.behaviors is not None else []
    b_sel = w.SelectMultiple(options=behs, value=tuple(behs), description="Behaviours",
                             rows=min(max(len(behs), 1), 8), disabled=not behs)
    t0, t1 = lg.data.index[0], lg.data.index[-1]
    fmt = "%Y-%m-%d %H:%M:%S"
    start = w.Text(value=t0.strftime(fmt), description="Start (UTC)")
    end = w.Text(value=t1.strftime(fmt), description="End (UTC)")
    go_btn = w.Button(description="Plot", button_style="primary", icon="line-chart")
    status = w.HTML()
    plot_area = w.VBox()          # holds the figure widget (widget=True)
    out = w.Output()              # holds the static figure (widget=False)

    def _draw(_=None):
        status.value = "<i>Plotting…</i>"
        go_btn.disabled = True
        try:
            fig = plot(lg, metrics=list(m_sel.value) or None,
                       behaviors=list(b_sel.value) if behs else None,
                       start=start.value, end=end.value, widget=widget, **plot_kwargs)
            if widget:
                plot_area.children = [fig]
            else:
                out.clear_output(wait=True)
                with out:
                    fig.show()
            status.value = ""
        except Exception as e:  # show errors to students instead of failing silently
            status.value = f"<b style='color:#d62728'>Could not plot:</b> {e}"
        finally:
            go_btn.disabled = False

    go_btn.on_click(_draw)
    display(w.VBox([w.HBox([m_sel, b_sel]), w.HBox([start, end, go_btn]), status,
                    plot_area, out]))
    _draw()

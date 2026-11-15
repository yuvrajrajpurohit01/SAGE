"""Optional step-by-step terminal trace (enabled with --verbose). Never changes any result.
Uses the 'rich' package for colour when available; falls back to plain text otherwise."""
import os
import re
import sys

try:
    from rich.console import Console
    from rich.markup import escape as _escape
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    HAVE_RICH = True
except Exception:  # rich not installed: plain output
    Console = Panel = Table = box = None
    HAVE_RICH = False

    def _escape(s: str) -> str:
        return str(s).replace("[", "\\[")


def esc(s) -> str:
    return _escape(str(s))


def _f(v) -> float:
    if hasattr(v, "detach"):
        v = v.detach().reshape(-1).float().mean().item()
    elif isinstance(v, (list, tuple)):
        v = sum(v) / len(v)
    return float(v)


def fmt(v) -> str:
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return f"{v[0]:.2f}" if len(v) == 1 else "[" + ", ".join(f"{x:.2f}" for x in v) + "]"
    return f"{float(v):.2f}"


def _short(text: str, n: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


_UNI = dict(on="█", off="░", ok="✓", bad="✗", arrow="→", dot="·", tau="τ", delta="Δ", minus="−",
            steps=("①", "②", "③", "④"), le="≤", ge="≥", lt="<", gt=">")
_ASC = dict(on="#", off="-", ok="ok", bad="x", arrow="->", dot="-", tau="tau", delta="delta", minus="-",
            steps=("1", "2", "3", "4"), le="<=", ge=">=", lt="<", gt=">")
_ROLE_COLOR = {"context": "cyan", "macro": "magenta", "micro": "yellow", "synthesis": "green"}
_SRC = {"small": ("SHORT PATH", "green", "small model answered"),
        "heavy": ("HEAVY PATH", "blue", "agents' forecast accepted"),
        "fusion": ("FUSION", "yellow", "small + heavy averaged")}


class _Tracer:
    def __init__(self) -> None:
        self.enabled = False
        self.active = False
        self.modes = {m.strip() for m in os.getenv("SAGE_TRACE_MODES", "sage").split(",") if m.strip()}
        self.tally = {"small": 0, "heavy": 0, "fusion": 0}
        self._console = None
        self._status = None
        self._ready = False
        self.g = _ASC

    # ---------- plumbing ----------
    def _init(self) -> None:
        if self._ready:
            return
        self._ready = True
        if HAVE_RICH:
            self._console = Console(highlight=False)
            utf = (self._console.encoding or "").lower().startswith("utf")
        else:
            utf = (getattr(sys.stdout, "encoding", "") or "").lower().startswith("utf")
        self.g = _UNI if utf else _ASC
        if not HAVE_RICH:
            print("(tip: pip install rich  for a coloured trace)", flush=True)

    def wants(self, mode: str) -> bool:
        return self.enabled and mode in self.modes

    @staticmethod
    def _strip(m: str) -> str:
        return re.sub(r"(?<!\\)\[/?[a-z_ #0-9]+\]", "", m).replace("\\[", "[")

    def _out(self, markup: str = "") -> None:
        self._init()
        self._stop_status()
        if self._console:
            self._console.print(markup, highlight=False)
        else:
            print(self._strip(markup), flush=True)

    def _a(self, markup: str) -> None:  # only inside an active (traced) sample
        if self.enabled and self.active:
            self._out(markup)

    def _rule(self, title: str, style: str = "cyan") -> None:
        self._init()
        self._stop_status()
        if self._console:
            self._console.rule(title, style=style)
        else:
            print("\n" + "-" * 8 + " " + self._strip(title) + " " + "-" * 8, flush=True)

    def _bar(self, value: float, marker=None, width: int = 16) -> str:
        n = max(0, min(width, round(value * width)))
        chars = [self.g["on"]] * n + [self.g["off"]] * (width - n)
        if marker is not None:
            chars[max(0, min(width - 1, int(marker * width)))] = "|"
        return "".join(chars)

    def _stop_status(self) -> None:
        if self._status is not None:
            try:
                self._status.stop()
            finally:
                self._status = None

    # ---------- generic (also used by older call sites) ----------
    def banner(self, text: str) -> None:
        if self.enabled:
            self._rule(f"[bold]{esc(text)}[/]", "bold cyan")

    def info(self, text: str) -> None:
        if self.enabled:
            self._out(f"  [dim]{esc(text)}[/]")

    def line(self, text: str) -> None:
        self._a(f"       [dim]{esc(text)}[/]")

    def step(self, number, name: str, text: str) -> None:
        self._a(f"  [bold]{number}[/] [bold]{esc(name):<12}[/] {esc(text)}")

    # ---------- pipeline stages ----------
    def small(self, y_s) -> None:
        g = self.g
        self._a(f"  [bold cyan]{g['steps'][0]}[/] [bold]small model[/]     forecast  [bold white]{fmt(y_s)}[/]")

    def gate(self, mode: str, q_s: float, tau1: float, d1: bool) -> None:
        g = self.g
        head = f"  [bold cyan]{g['steps'][1]}[/] [bold]fuzzy gate[/]      "
        if mode == "small_only":
            self._a(head + "[dim]small-only mode: gate not used for routing[/]"); return
        if mode == "heavy_always":
            self._a(head + "[dim]heavy-always mode: gate bypassed[/]"); return
        colour = "green" if d1 else "yellow"
        badge = "[black on green] SHORT PATH [/]" if d1 else "[black on blue] HEAVY PATH [/]"
        self._a(f"{head}Qs [{colour}]{self._bar(q_s, tau1)}[/] [bold]{q_s:.3f}[/]  "
                f"[dim]{g['tau']}1 {tau1}[/]  {badge}")
        self._a("       [dim]" + ("confident: the small model answers, no agents called"
                                  if d1 else "not confident: invoking the 4-agent heavy path") + "[/]")

    def heavy_start(self) -> None:
        g = self.g
        self._a(f"  [bold cyan]{g['steps'][2]}[/] [bold]heavy path[/]      "
                f"[cyan]Context[/] {g['arrow']} [magenta]Macro[/] {g['arrow']} [yellow]Micro[/] {g['arrow']} [green]Synthesis[/]")

    def agent_start(self, role: str) -> None:
        if not (self.enabled and self.active):
            return
        self._init()
        self._stop_status()
        if self._console:
            colour = _ROLE_COLOR.get(role, "cyan")
            self._status = self._console.status(f"       [{colour}]{role}[/] agent: waiting for the model ...", spinner="dots")
            self._status.start()

    def agent_done(self, role: str, trace, in_tok: int, out_tok: int, secs: float, cached: bool = False) -> None:
        if not (self.enabled and self.active):
            return
        g, colour = self.g, _ROLE_COLOR.get(role, "cyan")
        tag = "cached" if cached else f"{secs:.1f}s"
        body = f"{len(trace.timeline)} row summaries" if role == "context" else f"forecast [bold]{fmt(trace.prediction)}[/]"
        self._out(f"       [{colour}]{g['ok']} {role:<9}[/] [dim]{tag:>7}  {in_tok}{g['arrow']}{out_tok} tok[/]   {body}")
        if role != "context" and trace.reasoning:
            self._out(f"         [dim italic]\"{esc(_short(trace.reasoning))}\"[/]")

    def agent_fail(self, role: str, exc) -> None:
        if not (self.enabled and self.active):
            self._stop_status(); return
        self._out(f"       [bold red]{self.g['bad']} {role} agent failed:[/] [red]{esc(type(exc).__name__)}: {esc(_short(exc, 200))}[/]")

    def verifier(self, q_s: float, q_l: float, delta: float, tau2: float, accept: bool, fusion: str) -> None:
        g = self.g
        colour = "blue" if accept else "yellow"
        badge = "[black on blue] ACCEPT HEAVY [/]" if accept else "[black on yellow] FUSE BOTH [/]"
        rel = g["le"] if accept else g["gt"]
        self._a(f"  [bold cyan]{g['steps'][3]}[/] [bold]verifier[/]        Ql [{colour}]{self._bar(q_l)}[/] [bold]{q_l:.3f}[/]  "
                f"[dim]{g['delta']} {delta:+.3f} {rel} {g['tau']}2 {tau2}[/] {badge}")
        self._a("       [dim]" + ("heavy not clearly less reliable: heavy forecast accepted"
                                  if accept else f"heavy clearly less reliable: forecasts combined ({fusion})") + "[/]")

    # ---------- evaluation ----------
    def begin(self, variant: str, mode: str, k: int, n: int, sid, truth) -> None:
        if not self.enabled:
            return
        if k == 1:
            self.tally = {"small": 0, "heavy": 0, "fusion": 0}
            hidden = "" if mode in self.modes else "   [dim](steps hidden, one line per sample)[/]"
            self._rule(f"[bold]VARIANT  {esc(variant)}[/]  [dim]{n} samples[/]{hidden}", "bold magenta")
        if self.wants(mode):
            self._out()
            self._rule(f"[bold]sample {k}/{n}[/] [dim]{self.g['dot']} id {esc(sid)} {self.g['dot']}[/] true value [bold]{fmt(truth)}[/]", "cyan")

    def end(self, variant: str, mode: str, k: int, n: int, sid, result, truth) -> None:
        if not self.enabled:
            return
        t, p = _f(truth), _f(result.prediction)
        err = p - t
        rel = abs(err) / max(abs(t), 1.0)
        ecol = "green" if rel <= 0.10 else ("yellow" if rel <= 0.25 else "red")
        if not self.wants(mode):
            self._out(f"  [dim]\\[{esc(variant)}] {k}/{n}  id {esc(sid)}  prediction {fmt(result.prediction)}  true {fmt(truth)}[/]  [{ecol}]error {err:+.2f}[/]")
            return
        self.tally[result.source] = self.tally.get(result.source, 0) + 1
        label, colour, note = _SRC.get(result.source, (result.source.upper(), "white", ""))
        tot = " ".join(f"[{_SRC[s][1]}]{s} {self.tally.get(s, 0)}[/]" for s in ("small", "heavy", "fusion"))
        body = (f"[black on {colour}] {label} [/]  [dim]{note}[/]\n"
                f"prediction [bold]{fmt(result.prediction)}[/]   true [bold]{fmt(truth)}[/]   "
                f"error [{ecol}]{err:+.2f}[/]   [dim]{result.elapsed_seconds:.1f}s[/]\n"
                f"[dim]running totals:[/] {tot}")
        self._init()
        self._stop_status()
        if self._console:
            self._console.print(Panel.fit(body, title="FINAL", title_align="left", border_style=colour, box=box.ROUNDED))
        else:
            print("  " + self._strip(body).replace("\n", "\n  "), flush=True)

    def summary(self, variant: str, m: dict) -> None:
        if self.enabled:
            self._out(f"\n  [bold]{esc(variant)}[/]  RMSE [bold]{m['rmse']:.3f}[/]  MAE [bold]{m['mae']:.3f}[/]  "
                      f"[dim]small {m['small_percent']:.0f}%  heavy {m['heavy_final_percent']:.0f}%  "
                      f"fusion {m['fusion_percent']:.0f}%  calls {m['heavy_invocation_percent']:.0f}%[/]")

    def final_table(self, variants: dict) -> None:
        if not self.enabled:
            return
        self._init()
        self._stop_status()
        best = min(v["rmse"] for v in variants.values())
        if self._console:
            t = Table(title="RESULTS", box=box.SIMPLE_HEAVY, title_style="bold magenta")
            for col in ("variant", "RMSE", "MAE", "short path", "heavy", "fusion", "heavy-path calls"):
                t.add_column(col, justify="left" if col == "variant" else "right")
            for name, v in variants.items():
                style = "bold green" if v["rmse"] == best else None
                t.add_row(name, f"{v['rmse']:.3f}", f"{v['mae']:.3f}", f"{v['small_percent']:.0f}%",
                          f"{v['heavy_final_percent']:.0f}%", f"{v['fusion_percent']:.0f}%",
                          f"{v['heavy_invocation_percent']:.0f}%", style=style)
            self._console.print()
            self._console.print(t)
        else:
            print("\nRESULTS (variant, RMSE, MAE, short%, heavy%, fusion%, heavy calls%)")
            for name, v in variants.items():
                print(f"  {name:<14} {v['rmse']:.3f} {v['mae']:.3f} {v['small_percent']:.0f} {v['heavy_final_percent']:.0f} "
                      f"{v['fusion_percent']:.0f} {v['heavy_invocation_percent']:.0f}", flush=True)

    # ---------- training ----------
    def epoch(self, label: str, k: int, n: int, loss: float) -> None:
        if self.enabled:
            self._out(f"  [cyan]{esc(label):<16}[/] epoch {k:>3}/{n}  [cyan]{self._bar(k / n, width=16)}[/]  loss [bold]{loss:.4f}[/]")

    def conf_begin(self, k: int, n: int, sid, truth, y_s) -> None:
        if not self.enabled:
            return
        self.active = True
        self._rule(f"[bold]confidence example {k}/{n}[/] [dim]{self.g['dot']} id {esc(sid)} {self.g['dot']}[/] true value [bold]{fmt(truth)}[/]", "dim cyan")
        self._out(f"  [bold cyan]{self.g['steps'][0]}[/] [bold]small model[/]     forecast  [bold white]{fmt(y_s)}[/]  [dim](every example goes through the heavy path)[/]")
        self._out(f"  [bold cyan]{self.g['steps'][2]}[/] [bold]heavy path[/]")

    def conf_end(self, y_l, truth, qs_label, ql_label) -> None:
        if self.enabled:
            self._out(f"       heavy forecast [bold]{fmt(y_l)}[/]   learning targets: Qs* [bold]{_f(qs_label):.3f}[/]  Ql* [bold]{_f(ql_label):.3f}[/]")


TRACE = _Tracer()

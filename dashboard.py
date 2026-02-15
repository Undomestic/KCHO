"""
Live dashboard using rich.
"""

import time
import numpy as np
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.table import Table
from rich import box
import config
from data_generation import generate_data_for_condition, smooth_all_variables
from topology import compute_persistence, betti_numbers_at_scale, persistent_entropy
from math_utils import lyapunov_spectrum, ricci_curvature
from health_score import compute_health_score, get_health_status

class LiveDashboard:
    def __init__(self, condition, update_interval=0.5):
        self.condition = condition
        self.update_interval = update_interval
        self.data_buffer = []  # list of dicts for each time point
        self.current_time = 0
        self.max_points = 100  # keep last 100 points for display

        self.full_data = generate_data_for_condition(condition, t_points=500, seed=config.RANDOM_SEED)
        self.times = self.full_data['time']
        self.n_points = len(self.times)

        self.data_buffer = [{var: self.full_data[var][i] for var in config.PHYSIO_VARS} for i in range(10)]

        self.current_metrics = {}

    def update_metrics(self):
        """Recompute metrics on the current data buffer."""
        if len(self.data_buffer) < 10:
            return
        t = np.arange(len(self.data_buffer))  # fake time
        data_dict = {'time': t}
        for var in config.PHYSIO_VARS:
            data_dict[var] = np.array([d[var] for d in self.data_buffer])

        smoothed = data_dict  # skip smoothing for dashboard speed

        point_cloud = np.column_stack([smoothed[var] for var in config.PHYSIO_VARS])

        diagrams = compute_persistence(point_cloud)
        betti = betti_numbers_at_scale(diagrams, scale=config.BETTI_SCALE_THRESHOLD)
        entropy_h1 = persistent_entropy(diagrams[1]) if len(diagrams) > 1 else 0.0
        lyap_spec = lyapunov_spectrum(smoothed['MAP'],
                                       emb_dim=config.LYAPUNOV_EMBEDDING_DIM,
                                       matrix_dim=config.LYAPUNOV_MATRIX_DIM)
        lyap_max = lyap_spec[0]
        curvatures = ricci_curvature(point_cloud)
        mean_curv = np.mean(curvatures)

        max_complex = 0.5  # placeholder

        from math_utils import approximate_entropy
        apen = approximate_entropy(smoothed['MAP'])

        self.current_metrics = {
            'betti': betti,
            'entropy_h1': entropy_h1,
            'lyap_max': lyap_max,
            'mean_curv': mean_curv,
            'max_complex': max_complex,
            'apen': apen
        }

    def get_health_score(self):
        if not self.current_metrics:
            return 50
        return compute_health_score(self.current_metrics)

    def generate_layout(self):
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )

        header_text = Text(f"KCTHO Live Dashboard - Condition: {self.condition.capitalize()}", style="bold cyan")
        layout["header"].update(Panel(header_text, style="white"))

        health = self.get_health_score()
        status = get_health_status(health)
        left_table = Table(show_header=False, box=box.SIMPLE)
        left_table.add_column("Metric", style="cyan")
        left_table.add_column("Value", style="magenta")
        left_table.add_row("Health Score", f"{health} / 100")
        left_table.add_row("Status", status)
        left_table.add_row("β0 (components)", str(self.current_metrics.get('betti', (0,0,0))[0]))
        left_table.add_row("β1 (loops)", str(self.current_metrics.get('betti', (0,0,0))[1]))
        left_table.add_row("β2 (voids)", str(self.current_metrics.get('betti', (0,0,0))[2]))
        left_table.add_row("Lyapunov λ", f"{self.current_metrics.get('lyap_max',0):.3f}")
        left_table.add_row("Mean curvature", f"{self.current_metrics.get('mean_curv',0):.3f}")
        left_table.add_row("Approx entropy", f"{self.current_metrics.get('apen',0):.3f}")
        layout["left"].update(Panel(left_table, title="Vitals & Topology", border_style="green"))

        right_table = Table(show_header=True, box=box.SIMPLE)
        right_table.add_column("Variable", style="cyan")
        right_table.add_column("Current", justify="right")
        right_table.add_column("Trend", justify="center")
        if self.data_buffer:
            latest = self.data_buffer[-1]
            for var in config.PHYSIO_VARS:
                val = latest[var]
                if len(self.data_buffer) >= 3:
                    prev2 = self.data_buffer[-3][var]
                    prev1 = self.data_buffer[-2][var]
                    if val > prev1 and prev1 > prev2:
                        trend = "↑↑"
                    elif val > prev1:
                        trend = "↑"
                    elif val < prev1 and prev1 < prev2:
                        trend = "↓↓"
                    elif val < prev1:
                        trend = "↓"
                    else:
                        trend = "→"
                else:
                    trend = " "
                right_table.add_row(var, f"{val:.1f}", trend)
        layout["right"].update(Panel(right_table, title="Current Values", border_style="blue"))

        footer_text = "Press Ctrl+C to exit dashboard. Updates every {:.1f}s".format(self.update_interval)
        layout["footer"].update(Panel(footer_text, style="white"))

        return layout

    def run(self):
        """Run the live dashboard, updating with new data points."""
        with Live(refresh_per_second=1/self.update_interval, screen=True) as live:
            try:
                for i in range(10, self.n_points):
                    new_point = {var: self.full_data[var][i] for var in config.PHYSIO_VARS}
                    self.data_buffer.append(new_point)
                    if len(self.data_buffer) > self.max_points:
                        self.data_buffer.pop(0)
                    self.update_metrics()
                    layout = self.generate_layout()
                    live.update(layout)
                    time.sleep(self.update_interval)
            except KeyboardInterrupt:
                pass
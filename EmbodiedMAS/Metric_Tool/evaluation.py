"""
Experiment metrics aggregation, optional live PyQt dashboard, and JSON/plot outputs for embodied benchmarks.

Privacy / repository hygiene: Saved plots, metrics JSON, and task identifiers may embed host-specific output
paths. Do not commit API keys or ``llm_profiles.json``; prefer environment variables. If sensitive paths or
secrets were ever committed, rewrite Git history with ``git filter-repo`` or BFG before pushing to a public
remote — old clones can still expose that material (see ``EmbodiedMAS/ExperimentRunning/Automation_runner.py``).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, TYPE_CHECKING, Tuple
import os
import time
import json
import math
from pathlib import Path
from datetime import datetime
from enum import Enum
from collections import deque
import asyncio

import tongsim as ts
from tongsim.core.world_context import WorldContext

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _safe_task_stem(task_id: str) -> str:
    """Sanitize a string for on-disk filenames; matches Automation_runner task directory naming."""
    s = (task_id or "").strip().replace(os.sep, "_").replace("/", "_")
    return s or "run"

try:
    from PyQt5 import QtWidgets, QtCore, QtGui
    from PyQt5.QtCore import Qt, QObject, pyqtSignal, pyqtSlot
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    HAS_PYQT5 = True
except ImportError:
    HAS_PYQT5 = False
    print("[WARN] PyQt5 not available")




   

class MetricType(Enum):
    """New metric types"""
    FIRE_SUPPRESSION_RATE = "FSR"
    RESCUE_RATE = "RR"
    NPC_SURVIVAL_RATIO = "NSR"
    REMAINING_VALUE_RATIO = "VDR"
    WATER_USE_EFFICIENCY = "WUE"
    GINI_FIRE_SUPPRESSION = "G-FS"
    GINI_RESCUE = "G-R"


def calculate_gini(data: List[float]) -> float:
    """Calculate Gini coefficient, range [0, 1], 0 means perfect equality"""
    if not data or len(data) == 0:
        return 0.0
    n = len(data)
    if n == 1:
        return 0.0
    sorted_data = sorted(data)
    cumsum = 0.0
    numerator = 0.0
    total = sum(sorted_data)
    if total == 0:
        return 0.0
    for i, value in enumerate(sorted_data):
        cumsum += value
        numerator += (i + 1) * value
    gini = (2 * numerator) / (n * total) - (n + 1) / n
    return round(max(0.0, gini), 4)


@dataclass
class AgentMetrics:
    id: str
    name: str
    health: float = 100.0
    water_used: float = 0.0
    water_limit: float = None
    cold_time: float = 0.0
    distance_traveled: float = 0.0
    npcs_rescued: int = 0
    extinguished_objects: int = 0  # NEW: count of extinguished objects
    saved_value: float = 0.0
    
    distance_history: deque = field(default_factory=lambda: deque(maxlen=10000))
    water_history: deque = field(default_factory=lambda: deque(maxlen=10000))
    extinguished_history: deque = field(default_factory=lambda: deque(maxlen=10000))  # NEW
    
    last_position: Optional[ts.Vector3] = field(default=None, repr=False)
    following_npcs: set = field(default_factory=set, repr=False)
    following_npcs_health: Dict[str, float] = field(default_factory=dict, repr=False)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "health": self.health,
            "water_used": self.water_used,
            "water_limit": self.water_limit,
            "cold_time": self.cold_time,
            "distance_traveled": self.distance_traveled,
            "npcs_rescued": self.npcs_rescued,
            "extinguished_objects": self.extinguished_objects,  # NEW
            "following_npcs": list(self.following_npcs),
            "following_npcs_health": self.following_npcs_health
        }


@dataclass
class NpcMetrics:
    id: str
    name: str
    health: float = 100.0
    initial_health: float = 100.0  # NEW: for calculating NSR
    status: str = "trapped"
    position: Optional[ts.Vector3] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "health": self.health,
            "initial_health": self.initial_health,  # NEW
            "status": self.status,
            "position": {
                "x": self.position.x if self.position else 0,
                "y": self.position.y if self.position else 0,
                "z": self.position.z if self.position else 0
            } if self.position else None
        }


# Fixed rescue safe region (axis-aligned box), per experiment spec — not configurable at runtime.
_SAFE_AABB_MIN = ts.Vector3(-500.0, -250.0, 900.0)
_SAFE_AABB_MAX = ts.Vector3(500.0, 250.0, 1100.0)


def _normalize_guid_key(s: Any) -> str:
    """Normalize a GUID-like string (case and hyphens ignored) for matching engine/RPC id keys."""
    if s is None:
        return ""
    t = str(s).strip().lower().replace("-", "")
    return t


@dataclass
class SafeZone:
    """Axis-aligned safe box (min/max corners, inclusive)."""

    min_c: ts.Vector3
    max_c: ts.Vector3

    def contains(self, pos: ts.Vector3) -> bool:
        return (
            self.min_c.x <= pos.x <= self.max_c.x
            and self.min_c.y <= pos.y <= self.max_c.y
            and self.min_c.z <= pos.z <= self.max_c.z
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "aabb",
            "min": {"x": self.min_c.x, "y": self.min_c.y, "z": self.min_c.z},
            "max": {"x": self.max_c.x, "y": self.max_c.y, "z": self.max_c.z},
        }


def save_json(data: Any, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# ========== PyQt5 Real-time Display Window ==========

if HAS_PYQT5:
    class LiveDisplayWindow(QtWidgets.QMainWindow):
        """Real-time metrics display window"""
        update_signal = QtCore.pyqtSignal()
        close_signal = QtCore.pyqtSignal()
        
        def __init__(self, experiment_result):
            super().__init__()
            self.exp = experiment_result
            self.setWindowTitle(
                f"Experiment: {experiment_result.task_id or 'unnamed'} - Live Metrics"
            )
            self.setGeometry(100, 100, 1600, 1000)
            self._closing = False
            self._closed = False
            self.central_widget = QtWidgets.QWidget()
            self.setCentralWidget(self.central_widget)
            layout = QtWidgets.QVBoxLayout(self.central_widget)
            self.fig = Figure(figsize=(16, 10), dpi=100)
            self.canvas = FigureCanvas(self.fig)
            self.toolbar = NavigationToolbar(self.canvas, self)
            layout.addWidget(self.toolbar)
            layout.addWidget(self.canvas)
            self._init_subplots()
            self.update_signal.connect(self._update_plots)
            self.close_signal.connect(self._safe_close)
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self._request_update)
            self.timer.start(500)
            self.show()
        
        def _init_subplots(self):
            """Initialize subplots - updated with new metrics"""
            self.axes = []
            
            # 1. New metrics display (0,0)
            ax1 = self.fig.add_subplot(3, 3, 1)
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.axis('off')
            ax1.set_title('Real-time Metrics', fontsize=12, fontweight='bold')
            self.metric_texts = {}
            # FSR, RR, NSR, VDR (remaining value ratio), WUE, G-FS, G-R
            metrics = ['FSR', 'RR', 'NSR', 'VDR', 'WUE', 'G-FS', 'G-R']
            colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
            for i, (metric, color) in enumerate(zip(metrics, colors)):
                y_pos = 0.90 - i * 0.13
                ax1.text(0.1, y_pos, f'{metric}:', fontsize=11, fontweight='bold', 
                        transform=ax1.transAxes, va='center')
                text_obj = ax1.text(0.45, y_pos, '0.0000', fontsize=12, fontweight='bold',
                                   color=color, transform=ax1.transAxes, va='center',
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                self.metric_texts[metric] = text_obj
            self.axes.append(ax1)
            
            # 2. Burning object count
            ax2 = self.fig.add_subplot(3, 3, 2)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Burning count')
            ax2.set_title('Burning Objects')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
            self.line_burning_num, = ax2.plot([], [], 'b-', linewidth=2, drawstyle='steps-post')
            self.axes.append(ax2)
                        
            # 3. Watered objects (burned_obj_num)
            ax3 = self.fig.add_subplot(3, 3, 3)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Watered count')
            ax3.set_title('Fire Watered (burned_obj_num)')
            ax3.grid(True, alpha=0.3)
            self.line_watered_num, = ax3.plot([], [], 'c-', linewidth=2, drawstyle='steps-post')
            self.axes.append(ax3)

            # 4. Burned area
            ax4 = self.fig.add_subplot(3, 3, 4)
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Total fired count')
            ax4.set_title('Fired objects number')
            ax4.grid(True, alpha=0.3)
            self.line_burned, = ax4.plot([], [], 'r-', linewidth=2, label='total_fired_num')
            self.axes.append(ax4)
            
            # 5. NPC health
            ax5 = self.fig.add_subplot(3, 3, 5)
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Total Health', color='g')
            ax5.set_title('NPC Health & Evacuation')
            ax5.grid(True, alpha=0.3)
            self.line_health, = ax5.plot([], [], 'g-', linewidth=2, label='Health')
            self.ax5_twin = ax5.twinx()
            self.ax5_twin.set_ylabel('Evacuation %', color='m')
            self.line_evacuation, = self.ax5_twin.plot([], [], 'm--', alpha=0.5, label='Evacuation %')
            self.axes.append(ax5)
            
            # 6. Remaining property
            ax6 = self.fig.add_subplot(3, 3, 6)
            ax6.set_xlabel('Time (s)')
            ax6.set_ylabel('Property Health Sum')
            ax6.set_title('Remaining Property')
            ax6.grid(True, alpha=0.3)
            self.line_property, = ax6.plot([], [], 'orange', linewidth=2)
            self.axes.append(ax6)
            
            # 7. Agent distance
            ax7 = self.fig.add_subplot(3, 3, 7)
            ax7.set_xlabel('Time (s)')
            ax7.set_ylabel('Distance (m)')
            ax7.set_title('Agent Distance Traveled')
            ax7.grid(True, alpha=0.3)
            self.agent_distance_lines = {}
            self.axes.append(ax7)
            
            # 8. Agent water usage
            ax8 = self.fig.add_subplot(3, 3, 8)
            ax8.set_xlabel('Time (s)')
            ax8.set_ylabel('Water Used')
            ax8.set_title('Agent Water Usage')
            ax8.grid(True, alpha=0.3)
            self.agent_water_lines = {}
            self.axes.append(ax8)
            
            # # 9. Agent extinguished count
            # ax9 = self.fig.add_subplot(3, 3, 9)
            # ax9.set_xlabel('Time (s)')
            # ax9.set_ylabel('Objects Extinguished')
            # ax9.set_title('Agent Fire Suppression')
            # ax9.grid(True, alpha=0.3)
            # self.agent_extinguished_lines = {}
            # self.axes.append(ax9)
            
            # self.fig.tight_layout()
        
        def _request_update(self):
            if not self._closing and not self._closed and self.exp.is_running():
                self.update_signal.emit()
        
        def _safe_slice_data(self, time_data, value_data):
            """Safely slice data to ensure consistent length"""
            min_len = min(len(time_data), len(value_data))
            return time_data[-min_len:], value_data[-min_len:]
        
        def _update_plots(self):
            if self._closing or self._closed or len(self.exp.history["time"]) < 2:
                return
            try:
                time_data = list(self.exp.history["time"])
                fire_reignited = self.exp._fire_reignited if hasattr(self.exp, '_fire_reignited') else False
                if fire_reignited and len(time_data) > 0:
                    current_metrics = {'WARNING': 'FIRE REIGNITED'}
                else:
                    if self.exp.history["FSR"]:
                        current_metrics = {
                            'FSR': self.exp.history["FSR"][-1],
                            'RR': self.exp.history["RR"][-1],
                            'NSR': self.exp.history["NSR"][-1],
                            'VDR': self.exp.history["VDR"][-1],
                            'WUE': self.exp.history["WUE"][-1],
                            'G-FS': self.exp.history["G-FS"][-1],
                            'G-R': self.exp.history["G-R"][-1],
                        }
                    else:
                        current_metrics = {}
                for metric, value in current_metrics.items():
                    if metric in self.metric_texts:
                        if metric == 'WARNING':
                            self.metric_texts[metric].set_text(value)
                            self.metric_texts[metric].set_color('red')
                        else:
                            self.metric_texts[metric].set_text(f'{value:.4f}')
                
                burning_series = list(self.exp.history["fire_burning_num"])
                t_data, b_data = self._safe_slice_data(time_data, burning_series)
                self.line_burning_num.set_data(t_data, b_data)
                self.axes[1].set_xlim(0, max(t_data[-1] if t_data else 1, 1))
                if b_data:
                    ymax = max(b_data)
                    self.axes[1].set_ylim(0, max(ymax * 1.1, 1))
                    self.axes[1].annotate(
                        f'{int(b_data[-1])}',
                        xy=(t_data[-1], b_data[-1]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        color='blue',
                    )
                
                burned_data = list(self.exp.history["total_fired_num"])
                t_data, b_data = self._safe_slice_data(time_data, burned_data)
                self.line_burned.set_data(t_data, b_data)
                self.axes[2].set_xlim(0, max(t_data[-1] if t_data else 1, 1))
                max_burned = max(b_data) if b_data else 1
                self.axes[2].set_ylim(0, max(max_burned, 1) * 1.1)
                if len(b_data) > 0:
                    self.axes[2].annotate(f'{b_data[-1]:.0f}', xy=(t_data[-1], b_data[-1]),
                                         xytext=(5, 5), textcoords='offset points',
                                         fontsize=8, color='red')
                
                watered_series = list(self.exp.history["watered_num"])
                t_w, wn_data = self._safe_slice_data(time_data, watered_series)
                self.line_watered_num.set_data(t_w, wn_data)
                self.axes[3].set_xlim(0, max(t_w[-1] if t_w else 1, 1))
                if wn_data:
                    ymax_w = max(wn_data)
                    self.axes[3].set_ylim(0, max(ymax_w * 1.1, 1))
                    self.axes[3].annotate(
                        f'{int(wn_data[-1])}',
                        xy=(t_w[-1], wn_data[-1]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        color='cyan',
                    )
                
                health_data = list(self.exp.history["total_npc_health"])
                t_data, h_data = self._safe_slice_data(time_data, health_data)
                self.line_health.set_data(t_data, h_data)
                evac_data = list(self.exp.history["evacuation_progress"])
                t_data2, e_data = self._safe_slice_data(time_data, evac_data)
                self.line_evacuation.set_data(t_data2, e_data)
                self.axes[4].set_xlim(0, max(t_data[-1] if t_data else 1, 1))
                if h_data:
                    max_health = max(h_data)
                    self.axes[4].set_ylim(0, max(max_health * 1.1, 100))
                    if len(h_data) > 0:
                        self.axes[4].annotate(f'{h_data[-1]:.1f}', xy=(t_data[-1], h_data[-1]),
                                             xytext=(5, 5), textcoords='offset points',
                                             fontsize=8, color='green')
                
                property_data = list(self.exp.history["remaining_property"])
                t_data, p_data = self._safe_slice_data(time_data, property_data)
                self.line_property.set_data(t_data, p_data)
                self.axes[5].set_xlim(0, max(t_data[-1] if t_data else 1, 1))
                if p_data:
                    max_prop = max(p_data)
                    self.axes[5].set_ylim(0, max(max_prop * 1.1, 100))
                    if len(p_data) > 0:
                        self.axes[5].annotate(f'{p_data[-1]:.0f}', xy=(t_data[-1], p_data[-1]),
                                             xytext=(5, 5), textcoords='offset points',
                                             fontsize=8, color='orange')
                
                colors = plt.cm.tab10(np.linspace(0, 1, max(len(self.exp._agents), 1)))
                for idx, (agent_id, agent) in enumerate(self.exp._agents.items()):
                    color = colors[idx % len(colors)]
                    if agent_id not in self.agent_distance_lines:
                        line, = self.axes[6].plot([], [], '-', linewidth=2, 
                                                 label=agent.name or agent_id[:8], color=color)
                        self.agent_distance_lines[agent_id] = line
                        self.axes[6].legend(loc='upper left', fontsize=8)
                    dist_history = list(agent.distance_history)
                    if dist_history:
                        t_plot, d_plot = self._safe_slice_data(time_data, dist_history)
                        self.agent_distance_lines[agent_id].set_data(t_plot, d_plot)
                        if len(d_plot) > 0 and idx == 0:
                            self.axes[6].annotate(f'{d_plot[-1]:.1f}m', 
                                                 xy=(t_plot[-1], d_plot[-1]),
                                                 xytext=(5, 5), textcoords='offset points',
                                                 fontsize=7, color=color)
                    
                    if agent_id not in self.agent_water_lines:
                        line, = self.axes[7].plot([], [], '-', linewidth=2, 
                                                 label=agent.name or agent_id[:8], color=color)
                        self.agent_water_lines[agent_id] = line
                        self.axes[7].legend(loc='upper left', fontsize=8)
                    water_history = list(agent.water_history)
                    if water_history:
                        t_plot, w_plot = self._safe_slice_data(time_data, water_history)
                        self.agent_water_lines[agent_id].set_data(t_plot, w_plot)
                        if len(w_plot) > 0 and idx == 0:
                            self.axes[7].annotate(f'{w_plot[-1]:.1f}', 
                                                 xy=(t_plot[-1], w_plot[-1]),
                                                 xytext=(5, 5), textcoords='offset points',
                                                 fontsize=7, color=color)
                    
                    if agent_id not in self.agent_extinguished_lines:
                        line, = self.axes[8].plot([], [], '-', linewidth=2, 
                                                 label=agent.name or agent_id[:8], color=color)
                        self.agent_extinguished_lines[agent_id] = line
                        self.axes[8].legend(loc='upper left', fontsize=8)
                    extinguished_history = list(agent.extinguished_history)
                    if extinguished_history:
                        t_plot, e_plot = self._safe_slice_data(time_data, extinguished_history)
                        self.agent_extinguished_lines[agent_id].set_data(t_plot, e_plot)
                        if len(e_plot) > 0 and idx == 0:
                            self.axes[8].annotate(f'{int(e_plot[-1])}', 
                                                 xy=(t_plot[-1], e_plot[-1]),
                                                 xytext=(5, 5), textcoords='offset points',
                                                 fontsize=7, color=color)
                
                for ax_idx in [6, 7, 8]:
                    self.axes[ax_idx].set_xlim(0, max(time_data[-1] if time_data else 1, 1))
                    all_values = []
                    lines_dict = [self.agent_distance_lines, self.agent_water_lines,
                                 self.agent_extinguished_lines][ax_idx - 6]
                    for line in lines_dict.values():
                        ydata = line.get_ydata()
                        if len(ydata) > 0:
                            all_values.extend(ydata)
                    if all_values:
                        self.axes[ax_idx].set_ylim(0, max(all_values) * 1.2)
                
                self.canvas.draw()
            except Exception as e:
                print(f"[WARN] Update plot failed: {e}")
        
        @pyqtSlot()
        def _safe_close(self):
            if self._closing or self._closed:
                return
            self._closing = True
            if self.timer:
                try:
                    self.timer.stop()
                except:
                    pass
            self.hide()
            self.close()
            self._closed = True
        
        def closeEvent(self, event):
            if not self._closing:
                self._closing = True
                if self.timer:
                    try:
                        self.timer.stop()
                    except:
                        pass
            self._closed = True
            event.accept()
        
        def request_close(self):
            if not self._closing and not self._closed:
                self.close_signal.emit()


@dataclass
class ExperimentResult:
    task_type: str
    task_id: str = ""
    agent_metrics: List[AgentMetrics] = field(default_factory=list)
    npc_metrics: List[NpcMetrics] = field(default_factory=list)
    scene_id: int = 0
    success: bool = False
    termination_reason: str = ""
    start_time: float = 0
    end_time: float = 0
    suppression_time: Optional[float] = None
    evacuation_time: Optional[float] = None
    total_fired_num: float = 0.0  # burning + extinguished + destroyed + watered (last term = API burned_obj_num)
    fire_unburned_num: int = 0
    fire_burning_num: int = 0
    fire_extinguished_num: int = 0
    fire_watered_num: int = 0  # get_burned_area burned_obj_num (watered / suppressed)
    fire_destroyed_count: int = 0
    npc_health: Dict[str, float] = field(default_factory=dict)
    initial_total_npc_health: float = 0.0  # NEW: for NSR calculation
    total_npc_health: float = 0.0
    initial_property_value: float = 0.0
    remaining_property_value: float = 0.0
    initial_property_num: int = 0
    remaining_property_num: int = 0
    # scene_total_object_count: int = 0  # len(query_info) at experiment start, for WUE undamaged count
    agent_extinguished_objects: Dict[str, int] = field(default_factory=dict)  # NEW
    safe_zones: List[SafeZone] = field(default_factory=list)
    npcs_status: Dict[str, str] = field(default_factory=dict)
    total_steps: int = 0
    dialogue_history: List[Dict] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    _fire_started: bool = False
    _fire_start_time: Optional[float] = None
    _fire_reignited: bool = False
    _consecutive_errors: int = 0
    MAX_CONSECUTIVE_ERRORS: int = 3
    _stop_requested: bool = False
    
    # Updated history with new metrics
    history: Dict[str, deque] = field(default_factory=lambda: {
        "time": deque(maxlen=10000),
        "fire_burning_num": deque(maxlen=10000),
        "suppression_time": deque(maxlen=10000),
        "total_fired_num": deque(maxlen=10000),
        "total_npc_health": deque(maxlen=10000),
        "remaining_property": deque(maxlen=10000),
        "evacuation_progress": deque(maxlen=10000),
        "watered_num": deque(maxlen=10000),
        # Scalar metrics (text panel only): keep latest value only
        "FSR": deque(maxlen=1),
        "RR": deque(maxlen=1),
        "NSR": deque(maxlen=1),
        "VDR": deque(maxlen=1),
        "WUE": deque(maxlen=1),
        "G-FS": deque(maxlen=1),
        "G-R": deque(maxlen=1),
    })
    
    _context: Optional[WorldContext] = field(default=None, repr=False)
    _agents: Dict[str, AgentMetrics] = field(default_factory=dict, repr=False)
    _npcs: Dict[str, NpcMetrics] = field(default_factory=dict, repr=False)
    _fire_extinguished_time: Optional[float] = None
    _fire_successfullly_supressed: bool = False
    _all_evacuated_time: Optional[float] = None
    _fire_successfullly_evacuated: bool = False
    _running: bool = False
    _update_task: Optional[asyncio.Task] = None
    _update_interval: float = 1.0
    _plot_output_dir: Path = field(default_factory=lambda: Path("./plots"))
    enable_live_display: bool = True
    _live_window: Any = None
    _qt_app: Any = None
    _qt_events_task: Optional[asyncio.Task] = field(default=None, repr=False)
    _agent_state_update_interval: float = 1.0
    _last_agent_state_update: float = 0
    _initial_property_num_snapshot_done: bool = field(default=False, repr=False)
    _prev_fire_burning_num: int = field(default=0, repr=False)
    
    def check_npc_in_safe_zone(self, npc_id: str) -> bool:
        _real_id, npc = self._resolve_npc_metrics(npc_id)
        if npc is None or npc.position is None:
            return False
        for zone in self.safe_zones:
            if zone.contains(npc.position):
                return True
        return False

    def _resolve_npc_metrics(self, npc_id: str) -> Tuple[Optional[str], Optional[NpcMetrics]]:
        """Resolve an entry in ``_npcs`` by raw key or normalized GUID."""
        if npc_id in self._npcs:
            return npc_id, self._npcs[npc_id]
        nk = _normalize_guid_key(npc_id)
        if not nk:
            return None, None
        for k, m in self._npcs.items():
            if _normalize_guid_key(k) == nk:
                return k, m
        return None, None
    
    def bind_context(self, context: WorldContext, update_interval: float = 1.0,
                    plot_output_dir: Optional[Path] = None,
                    enable_live_display: bool = True,
                    agent_state_update_interval: float = 1.0) -> 'ExperimentResult':
        self._context = context
        self._update_interval = update_interval
        self._enable_live_display = enable_live_display and HAS_PYQT5
        self._agent_state_update_interval = agent_state_update_interval
        if plot_output_dir:
            self._plot_output_dir = Path(plot_output_dir)
        self.safe_zones = [SafeZone(_SAFE_AABB_MIN, _SAFE_AABB_MAX)]
        return self
    
    def register_agents(self, agent_ids: List[str], agent_names: Optional[List[str]] = None) -> None:
        for i, aid in enumerate(agent_ids):
            name = agent_names[i] if agent_names and i < len(agent_names) else f"Agent_{aid}"
            if aid not in self._agents:
                metrics = AgentMetrics(id=aid, name=name)
                self._agents[aid] = metrics
                self.agent_metrics.append(metrics)
    
    def register_agent(self, agent_id: str, agent_name: Optional[str] = None) -> AgentMetrics:
        name = agent_name or f"Agent_{agent_id}"
        if agent_id not in self._agents:
            metrics = AgentMetrics(id=agent_id, name=name)
            self._agents[agent_id] = metrics
            self.agent_metrics.append(metrics)
        return self._agents[agent_id]
    
    async def start_async(self, enable_plotting: bool = True) -> 'ExperimentResult':
        if not self._context:
            raise RuntimeError("Please call bind_context() first")
        print(f"[INFO] Starting experiment: {self.task_id or 'unnamed'}")
        self.start_time = time.time()
        self._running = True
        self._stop_requested = False
        self._fire_reignited = False
        self._consecutive_errors = 0
        self._selfstate_none_warned: Set[str] = set()
        # qlist = await ts.UnaryAPI.query_info(self._context.conn)
        # self.scene_total_object_count = len(qlist) if qlist else 0
        self._update_task = asyncio.create_task(self._update_loop())
        if self._enable_live_display and self._init_live_display_main_thread():
            self._qt_events_task = asyncio.create_task(self._qt_pump_events())
        return self

    def _pyqt_plugin_root(self) -> Optional[str]:
        import os

        import PyQt5

        base = os.path.dirname(PyQt5.__file__)
        for sub in ("Qt5/plugins", "Qt/plugins"):
            root = os.path.join(base, sub)
            if os.path.isdir(os.path.join(root, "platforms")):
                return root
        return None

    def _init_live_display_main_thread(self) -> bool:
        """Create QApplication and window on the main thread; pump with processEvents from asyncio (no exec_ in a worker thread)."""
        import os

        if not HAS_PYQT5:
            return False
        if os.name == "posix" and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            print(
                "[WARN] Live display skipped: no DISPLAY / WAYLAND_DISPLAY (headless session)."
            )
            self._enable_live_display = False
            return False
        try:
            from PyQt5 import QtWidgets

            root = self._pyqt_plugin_root()
            if root:
                os.environ["QT_PLUGIN_PATH"] = root
            os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
            self._qt_app = QtWidgets.QApplication.instance()
            if self._qt_app is None:
                self._qt_app = QtWidgets.QApplication([])
            self._live_window = LiveDisplayWindow(self)
            print("[INFO] Live display window started (main thread + asyncio processEvents)")
            return True
        except Exception as e:
            print(f"[WARN] Live display failed to start: {e}")
            self._live_window = None
            self._qt_app = None
            self._enable_live_display = False
            return False

    async def _qt_pump_events(self) -> None:
        while self._running and self._live_window is not None and self._qt_app is not None:
            self._qt_app.processEvents()
            await asyncio.sleep(0.02)
    
    async def stop_async(self, success: bool = False, reason: str = "") -> None:
        if not self._running:
            return
        print(f"[INFO] Stopping experiment: {reason}")
        self._running = False
        self._stop_requested = True
        if self._update_task and not self._update_task.done():
            self._update_task.cancel()
            try:
                await asyncio.wait_for(self._update_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        if self._qt_events_task and not self._qt_events_task.done():
            self._qt_events_task.cancel()
            try:
                await self._qt_events_task
            except asyncio.CancelledError:
                pass
            self._qt_events_task = None
        if self._live_window is not None and HAS_PYQT5:
            try:
                self._live_window.request_close()
                if self._qt_app is not None:
                    for _ in range(40):
                        self._qt_app.processEvents()
                        await asyncio.sleep(0.01)
            except Exception as e:
                print(f"[WARN] Error closing Qt window: {e}")
            finally:
                self._live_window = None
        self.end_time = time.time()
        self.success = success
        self.termination_reason = reason
        try:
            await self.update_all_metrics()
        except Exception as e:
            print(f"[WARN] Final update failed: {e}")
        self.npc_metrics = list(self._npcs.values())
        try:
            self._save_final_plot()
        except Exception as e:
            print(f"[WARN] Save final plot failed: {e}")
        try:
            self.save()
        except Exception as e:
            print(f"[WARN] Save results failed: {e}")
    
    async def wait_async(self, seconds: float) -> 'ExperimentResult':
        await asyncio.sleep(seconds)
        return self
    
    def update_agent_distance(self, agent_id: str, new_position: ts.Vector3, 
                             last_position: Optional[ts.Vector3] = None) -> float:
        if agent_id not in self._agents:
            raise ValueError(f"Agent not registered: {agent_id}")
        agent = self._agents[agent_id]
        prev_pos = last_position if last_position is not None else agent.last_position
        if prev_pos is not None:
            dist = math.sqrt((new_position.x - prev_pos.x)**2 + 
                           (new_position.y - prev_pos.y)**2 +
                           (new_position.z - prev_pos.z)**2)
            agent.distance_traveled += dist
        agent.last_position = new_position
        agent.distance_history.append(agent.distance_traveled)
        return agent.distance_traveled
    
    def update_agent_water_used(self, agent_id: str, amount: float = 1.0) -> float:
        if agent_id not in self._agents:
            raise ValueError(f"Agent not registered: {agent_id}")
        self._agents[agent_id].water_used += amount
        self._agents[agent_id].water_history.append(self._agents[agent_id].water_used)
        return self._agents[agent_id].water_used
    
    def set_agent_water_setting(self, agent_id: str, water_limit: float, cold_time: float) -> None:
        if agent_id not in self._agents:
            raise ValueError(f"Agent not registered: {agent_id}")
        self._agents[agent_id].water_limit = water_limit
        self._agents[agent_id].cold_time = cold_time
    
    def update_agent_npcs_rescued(self, agent_id: str, count: int = 1) -> int:
        if agent_id not in self._agents:
            raise ValueError(f"Agent not registered: {agent_id}")
        self._agents[agent_id].npcs_rescued += count
        return self._agents[agent_id].npcs_rescued

    def _resolve_registered_agent_id(self, raw: Any) -> Optional[str]:
        """Map a spawn/UnaryAPI actor id to the key used in ``register_agents``."""
        if raw is None:
            return None
        candidates: List[str] = []
        if isinstance(raw, str):
            candidates.append(raw)
        elif isinstance(raw, bytes) and len(raw) == 16:
            try:
                from tongsim.connection.grpc.unary_api import _fguid_bytes_to_str

                candidates.append(_fguid_bytes_to_str(raw))
            except Exception:
                candidates.append(str(raw))
        else:
            candidates.append(str(raw))
        for c in candidates:
            if c in self._agents:
                return c
        nk0 = _normalize_guid_key(candidates[0]) if candidates else ""
        if nk0:
            for k in self._agents.keys():
                if _normalize_guid_key(k) == nk0:
                    return k
        return None

    async def record_rescue_after_stop_follow(self, rescuer_actor_id: Any) -> None:
        """
        Call after ``send_stop_follow`` / ``sendstopfollow`` RPC succeeds:
        increment ``npcs_rescued`` by 1 for the **agent that issued the command** only (no NPC id recorded).
        """
        if not self._context:
            return
        aid = self._resolve_registered_agent_id(rescuer_actor_id)
        if aid is None:
            print(
                f"[RESCUE] skip: rescuer actor id not registered in ExperimentResult: "
                f"{rescuer_actor_id!r}"
            )
            return
        self.update_agent_npcs_rescued(aid, 1)
        print(
            f"[RESCUE] Agent {self._agents[aid].name} rescue counted (send_stop_follow), "
            f"total: {self._agents[aid].npcs_rescued}"
        )
    
    def set_agent_npcs_rescued(self, agent_id: str, total_count: int) -> None:
        if agent_id not in self._agents:
            raise ValueError(f"Agent not registered: {agent_id}")
        self._agents[agent_id].npcs_rescued = total_count
    
    def update_agent_health(self, agent_id: str, health: float) -> None:
        if agent_id in self._agents:
            self._agents[agent_id].health = health

    async def _update_agent_states(self) -> None:
        if not self._context:
            return
        current_time = time.time()
        if current_time - self._last_agent_state_update < self._agent_state_update_interval:
            return
        self._last_agent_state_update = current_time
        for agent_id, agent in self._agents.items():
            try:
                # TongSim UnaryAPI.get_selfstate returns a dict on success (proto → SDK): location,
                # used_usage, following_NPC, etc. (see tongsim UnaryAPI.get_selfstate).
                # On RPC failure or disconnect, safe_async_rpc may return None — skip quietly; warn once per agent for id-format issues.
                state = await ts.UnaryAPI.get_selfstate(self._context.conn, agent_id)
                if not state or not isinstance(state, dict):
                    if agent_id not in self._selfstate_none_warned:
                        self._selfstate_none_warned.add(agent_id)
                        print(
                            f"[WARN] get_selfstate returned no data for registered id={agent_id!r} "
                            f"(agent {agent.name!r}); per-agent distance/water may stay zero. "
                            "Use the same actor id format as spawn_agent returns (canonical GUID string or 16-byte FGuid)."
                        )
                    continue
                location = state.get('location')
                if location and hasattr(location, 'x'):
                    new_pos = ts.Vector3(location.x, location.y, location.z)
                    if agent.last_position is not None:
                        dist = math.sqrt(
                            (new_pos.x - agent.last_position.x)**2 + 
                            (new_pos.y - agent.last_position.y)**2 +
                            (new_pos.z - agent.last_position.z)**2
                        )
                        if dist > 0.01:
                            agent.distance_traveled += dist
                    agent.last_position = new_pos
                used_usage = state.get('used_usage')
                if used_usage is not None and used_usage >= 0:
                    if used_usage > agent.water_used:
                        agent.water_used = float(used_usage)
                following_npc = state.get('following_NPC')
                if following_npc and isinstance(following_npc, dict):
                    current_following = set(following_npc.keys())
                else:
                    current_following = set()
                # Rescue counts are maintained by ActionAPI.sendstopfollow → record_rescue_after_stop_follow;
                # here we only sync the following set for health display, etc.
                agent.following_npcs_health = {}
                for npc_id in current_following:
                    health = self.npc_health.get(npc_id, 100.0)
                    agent.following_npcs_health[npc_id] = health
                agent.following_npcs = set(current_following)
                agent.distance_history.append(agent.distance_traveled)
                agent.water_history.append(agent.water_used)
                agent.extinguished_history.append(agent.extinguished_objects)
            except Exception as e:
                print(f"[WARN] Failed to update Agent {agent_id[:8]}: {e}")
    
    async def update_all_metrics(self) -> None:
        await self._update_agent_states()
        await self._update_npc_metrics()
        await self._update_fire_metrics()
        await self._update_property_metrics()
        await self._update_agent_extinguished_objects()  # NEW
        self._record_history()
        self.total_steps += 1
    
    async def _update_agent_extinguished_objects(self) -> None:
        """NEW: Get extinguished objects count for each agent"""
        if not self._context:
            return
        try:
            extinguished_data = await ts.UnaryAPI.get_agent_extinguished_objects(self._context.conn)
            if extinguished_data and isinstance(extinguished_data, dict):
                for agent_id, objects in extinguished_data.items():
                    if agent_id in self._agents:
                        agent = self._agents[agent_id]
                        new_count = len(objects) if isinstance(objects, list) else 0
                        agent.extinguished_objects = new_count
                        self.agent_extinguished_objects[agent_id] = new_count
        except Exception as e:
            pass  # API may not be available
    
    async def _auto_stop(self) -> None:
        if self._running and not self._stop_requested:
            await self.stop_async(success=False, reason="Connection failed, auto stop")
    
    def _record_history(self) -> None:
        current_time = time.time() - self.start_time if self.start_time > 0 else 0
        self.history["time"].append(current_time)
        self.history["fire_burning_num"].append(self.fire_burning_num)
        self.history["suppression_time"].append(self.suppression_time if self.suppression_time else 0)
        self.history["total_fired_num"].append(self.total_fired_num)
        self.history["total_npc_health"].append(self.total_npc_health)
        self.history["remaining_property"].append(self.remaining_property_value)
        total_npcs = len(self._npcs)
        if total_npcs > 0:
            rescued = sum(1 for n in self._npcs.values() if n.status == "rescued")
            progress = (rescued / total_npcs) * 100
        else:
            progress = 0
        self.history["evacuation_progress"].append(progress)
        self.history["watered_num"].append(float(self.fire_watered_num))
        # NEW: Record new metrics history
        metrics = self.calculate_metrics()
        self.history["FSR"].append(metrics['FSR'])
        self.history["RR"].append(metrics['RR'])
        self.history["NSR"].append(metrics['NSR'])
        self.history["VDR"].append(metrics["VDR"])
        self.history["WUE"].append(metrics['WUE'])
        self.history["G-FS"].append(metrics['G-FS'])
        self.history["G-R"].append(metrics['G-R'])
    
    async def _update_npc_metrics(self) -> None:
        if not self._context:
            return
        positions = await ts.UnaryAPI.get_npc_postions(self._context.conn)
        if positions and isinstance(positions, dict):
            for npc_id, pos in positions.items():
                if npc_id not in self._npcs:
                    self._npcs[npc_id] = NpcMetrics(id=npc_id, name=f"NPC_{npc_id[:8]}")
                if hasattr(pos, 'x') and hasattr(pos, 'y') and hasattr(pos, 'z'):
                    self._npcs[npc_id].position = ts.Vector3(pos.x, pos.y, pos.z)
                elif isinstance(pos, (list, tuple)) and len(pos) >= 3:
                    self._npcs[npc_id].position = ts.Vector3(pos[0], pos[1], pos[2])
        health_data = await ts.UnaryAPI.get_npc_health(self._context.conn)
        if health_data and isinstance(health_data, dict):
            self.npc_health = health_data
            for npc_id, health in health_data.items():
                if npc_id not in self._npcs:
                    self._npcs[npc_id] = NpcMetrics(id=npc_id, name=f"NPC_{npc_id[:8]}")
                npc = self._npcs[npc_id]
                if npc.initial_health == 100.0 and health < 100.0:
                    npc.initial_health = 100.0
                npc.health = float(health)
                in_safe_zone = self.check_npc_in_safe_zone(npc_id)
                if in_safe_zone and npc.status != "rescued":
                    npc.status = "rescued"
                    self.npcs_status[npc_id] = "rescued"
                    print(f"[RESCUE] NPC {npc_id[:8]} entered safe zone!")
                elif health <= 0 and npc.status != "dead":
                    npc.status = "dead"
                    self.npcs_status[npc_id] = "dead"
                elif not in_safe_zone and 0 < health < 90 and npc.status not in ["rescued", "dead"]:
                    npc.status = "trapped"
                    self.npcs_status[npc_id] = "trapped"
            if self.initial_total_npc_health == 0:
                self.initial_total_npc_health = sum(n.initial_health for n in self._npcs.values())
            alive_npcs = [n for n in self._npcs.values() if n.status != "dead"]
            self.total_npc_health = sum(n.health for n in alive_npcs)
            total = len(self._npcs)
            if total > 0 and self._all_evacuated_time is None:
                done = sum(1 for n in self._npcs.values() if n.status in ["rescued", "dead"])
                if done == total:
                    self._all_evacuated_time = time.time()
                    self._fire_successfullly_evacuated = True
                    if self._fire_started and self._fire_start_time:
                        self.evacuation_time = self._all_evacuated_time - self._fire_start_time
                    else:
                        self.evacuation_time = self._all_evacuated_time - self.start_time
    
    async def _update_fire_metrics(self) -> None:
        """Burned-area + destroyed once per tick; plots use burning_num; engine outfire = task completion only."""
        if not self._context:
            return

        burned_data = await ts.UnaryAPI.get_burned_area(self._context.conn)
        destroyed_raw = await ts.UnaryAPI.get_destroyed_objects(self._context.conn)
        self.fire_destroyed_count = len(destroyed_raw)

        self.fire_unburned_num = int(burned_data.get("unburned_num", 0))
        self.fire_burning_num = int(burned_data.get("burning_num", 0))
        self.fire_extinguished_num = int(burned_data.get("extinguished_num", 0))
        self.fire_watered_num = int(burned_data.get("burned_obj_num", 0))
        self.remaining_property_num = int(burned_data.get("total_num", 0))

        total_fired = (
            self.fire_burning_num
            + self.fire_extinguished_num
            + self.fire_destroyed_count
            + self.fire_watered_num
        )
        self.total_fired_num = float(total_fired)

        self.initial_property_num = self.remaining_property_num + self.fire_destroyed_count

        if not self._fire_started:
            self._fire_started = True
            self._fire_start_time = time.time()
            self._fire_reignited = False
            print(f"[INFO] Fire started at: {self._fire_start_time - self.start_time:.2f}s")

        if self._fire_started:
            if self._prev_fire_burning_num > 0 and self.fire_burning_num == 0:
                if self._fire_extinguished_time is None:
                    tnow = time.time()
                    self._fire_extinguished_time = tnow
                    base = self._fire_start_time if self._fire_start_time else self.start_time
                    self.suppression_time = tnow - base
                    print(f"[INFO] No burning objects (burning_num=0), elapsed since fire start: {self.suppression_time:.2f}s")
            if self._prev_fire_burning_num == 0 and self.fire_burning_num > 0 and self._fire_extinguished_time is not None:
                if not self._fire_reignited:
                    self._fire_reignited = True
                    print("[WARN] Fire reignited (burning_num > 0 again)!")
                self._fire_extinguished_time = None
                self.suppression_time = None

        self._prev_fire_burning_num = self.fire_burning_num

        fire_state = await ts.UnaryAPI.get_outfire_state(self._context.conn)
        is_engine_all_out = False
        if isinstance(fire_state, bool):
            is_engine_all_out = fire_state
        elif isinstance(fire_state, dict):
            is_engine_all_out = fire_state.get("extinguished", False) or fire_state.get("out", False)

        self._fire_successfullly_supressed = bool(is_engine_all_out)

        if not isinstance(burned_data, dict):
            if is_engine_all_out:
                if self._fire_extinguished_time is None:
                    self._fire_extinguished_time = time.time()
                    if self._fire_started and self._fire_start_time:
                        self.suppression_time = self._fire_extinguished_time - self._fire_start_time
                    else:
                        self.suppression_time = self._fire_extinguished_time - self.start_time
                    print(f"[INFO] Engine reports fire out (legacy path): {self.suppression_time:.2f}s")
                self._fire_reignited = False
            else:
                if self._fire_extinguished_time is not None:
                    if not self._fire_reignited:
                        self._fire_reignited = True
                        print("[WARN] Fire reignited!")
                    self._fire_extinguished_time = None
                    self.suppression_time = None
    
    async def _update_property_metrics(self) -> None:
        if not self._context:
            return
        self.remaining_property_value = await ts.UnaryAPI.get_obj_residual(self._context.conn)
    
    async def _update_loop(self) -> None:
        while self._running and not self._stop_requested:
            try:
                await self.update_all_metrics()
                self._consecutive_errors = 0
            except asyncio.CancelledError:
                print("[INFO] Update loop cancelled")
                break
            except Exception as e:
                error_msg = str(e)
                if "UNAVAILABLE" in error_msg or "failed to connect" in error_msg:
                    self._consecutive_errors += 1
                    print(f"[ERROR] Connection failed ({self._consecutive_errors}/{self.MAX_CONSECUTIVE_ERRORS})")
                    if self._consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                        print(f"[ERROR] Max consecutive errors reached, stopping")
                        self._running = False
                        break
                else:
                    print(f"[ERROR] Update failed: {type(e).__name__}: {e}")
            try:
                await asyncio.sleep(self._update_interval)
            except asyncio.CancelledError:
                print("[INFO] Update loop sleep cancelled")
                break
        print("[INFO] Update loop exited")
        if not self._stop_requested and self._consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
            await self._auto_stop()
    
    def _safe_slice_data(self, time_data, value_data):
        min_len = min(len(time_data), len(value_data))
        return time_data[-min_len:], value_data[-min_len:]
    
    def _add_value_annotation(self, ax, time_data, value_data, color='black', offset=(5, 5)):
        if len(value_data) > 0 and len(time_data) > 0:
            val = value_data[-1]
            if isinstance(val, (int, float)):
                text = f'{val:.1f}' if isinstance(val, float) else f'{val}'
                ax.annotate(text, xy=(time_data[-1], val),
                           xytext=offset, textcoords='offset points',
                           fontsize=9, color=color, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                    edgecolor=color, alpha=0.8))
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        - FSR: fire_watered_num / total_fired_num
        - VDR: remaining_property_value / initial_property_value (global remaining value ratio)
        - WUE: (remaining - max(0, scene_total@start - total_fired)*100) / total_water
        - RR, NSR, G-FS, G-R: as named
        """
        metrics = {}

        watered = float(self.fire_watered_num)
        fired = float(self.total_fired_num)

        # 1. FSR: Fire Suppression Rate (global API counts)
        if fired > 0:
            metrics["FSR"] = round(watered / fired, 4)
        else:
            metrics["FSR"] = 0.0
        
        # 2. RR: Rescue Rate
        total_npcs = len(self._npcs)
        if total_npcs > 0:
            rescued_npcs = sum(1 for n in self._npcs.values() if n.status == "rescued")
            metrics['RR'] = round(rescued_npcs / total_npcs, 4)
        else:
            metrics['RR'] = 0.0
        
        # 3. NSR: NPC Survival Ratio
        if self.initial_total_npc_health > 0:
            metrics['NSR'] = round(self.total_npc_health / self.initial_total_npc_health, 4)
        else:
            alive_npcs = [n for n in self._npcs.values() if n.status != "dead"]
            if alive_npcs:
                avg_health = sum(n.health for n in alive_npcs) / len(alive_npcs)
                metrics['NSR'] = round(avg_health / 100.0, 4)
            else:
                metrics['NSR'] = 0.0
        
        # 4. VDR: global remaining value ratio (same as historical remaining/initial property value)
        if self.initial_property_value > 0:
            metrics["VDR"] = round(self.remaining_property_value / self.initial_property_value, 4)
        else:
            metrics["VDR"] = 0.0
        
        # 5. WUE: saved_value / total_water; undamaged from query_info count at start minus total_fired_num
        total_water = sum(a.water_used for a in self._agents.values())
        # undamaged_objects = self.scene_total_object_count - int(round(self.total_fired_num))
        # undamaged_value = (self.remaining_property_num - self.fire_watered_num) * 100.0
        # TODO: Revisit this: fire_unburned_num is the unburned count; currently burning objects are also treated as suppressed here.
        undamaged_value = self.fire_unburned_num * 100.0
        self.saved_value = float(self.remaining_property_value) - undamaged_value
        if total_water > 0:
            metrics["WUE"] = round(max(0, self.saved_value) / total_water, 4)
        else:
            metrics["WUE"] = 0.0
        
        # 6. G-FS: Gini coefficient for Fire Suppression
        extinguished_counts = [agent.extinguished_objects for agent in self._agents.values()]
        if extinguished_counts:
            metrics['G-FS'] = calculate_gini(extinguished_counts)
        else:
            metrics['G-FS'] = 0.0
        
        # 7. G-R: Gini coefficient for Rescue
        rescue_counts = [agent.npcs_rescued for agent in self._agents.values()]
        if rescue_counts:
            metrics['G-R'] = calculate_gini(rescue_counts)
        else:
            metrics['G-R'] = 0.0
        
        return metrics
    
    def _save_final_plot(self) -> None:
        """Save final plot - updated with new metrics"""
        if len(self.history["time"]) < 2:
            return
        time_data = list(self.history["time"])
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        success_status = "SUCCESS" if (self._fire_successfullly_supressed and self._fire_successfullly_evacuated) else "INCOMPLETE" if self._fire_successfullly_supressed or self._fire_successfullly_evacuated else "FAILURE"
        fig.suptitle(f'Experiment: {self.task_id or "unnamed"} - Final Results\n'
                    f'Status: {success_status} | Duration: {self.end_time - self.start_time:.1f}s', 
                    fontsize=16, color='red' if self._fire_reignited else 'green' if success_status == "SUCCESS" else 'black')
        def safe_slice(t, v):
            min_len = min(len(t), len(v))
            return t[-min_len:], v[-min_len:]
        
        # 1. New metrics display (0,0)
        ax1 = axes[0, 0]
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        ax1.set_title('Final Metrics', fontsize=12, fontweight='bold')
        if self.history["FSR"]:
            final_metrics = {
                'FSR': self.history["FSR"][-1],
                'RR': self.history["RR"][-1],
                'NSR': self.history["NSR"][-1],
                'VDR': self.history["VDR"][-1],
                'WUE': self.history["WUE"][-1],
                'G-FS': self.history["G-FS"][-1],
                'G-R': self.history["G-R"][-1],
            }
            colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
            for i, ((metric, value), color) in enumerate(zip(final_metrics.items(), colors)):
                y_pos = 0.90 - i * 0.13
                ax1.text(0.1, y_pos, f'{metric}:', fontsize=11, fontweight='bold', 
                        transform=ax1.transAxes, va='center')
                ax1.text(0.45, y_pos, f'{value:.4f}', fontsize=12, fontweight='bold',
                        color=color, transform=ax1.transAxes, va='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Burning object count (0,1)
        ax2 = axes[0, 1]
        burning_data = list(self.history["fire_burning_num"])
        t_data, bn_data = safe_slice(time_data, burning_data)
        ax2.plot(t_data, bn_data, 'b-', linewidth=2.5, drawstyle='steps-post', label='burning_num')
        ax2.fill_between(t_data, bn_data, alpha=0.3, color='orange', step='post')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Burning count')
        ax2.set_title('Burning Objects')
        ax2.grid(True, alpha=0.3)
        if bn_data:
            ymax = max(bn_data)
            ax2.set_ylim(0, max(ymax * 1.1, 1))
            ax2.annotate(
                f'{int(bn_data[-1])}',
                xy=(t_data[-1], bn_data[-1]),
                xytext=(5, 10),
                textcoords='offset points',
                fontsize=10,
                color='blue',
                fontweight='bold',
            )
        if self._fire_extinguished_time and self.start_time > 0 and not self._fire_reignited:
            x_sup = self._fire_extinguished_time - self.start_time
            ax2.axvline(x=x_sup, color='g', linestyle='--', alpha=0.7,
                    label=f'Burning cleared at t={x_sup:.1f}s')
        ax2.legend(loc='upper left')

        # 3. Watered count (fire_watered_num / burned_obj_num) (0,2)
        ax3 = axes[0, 2]
        watered_data = list(self.history["watered_num"])
        t_w, wn_data = safe_slice(time_data, watered_data)
        ax3.plot(t_w, wn_data, 'c-', linewidth=2.5, drawstyle='steps-post', label='watered_num')
        ax3.fill_between(t_w, wn_data, alpha=0.3, color='cyan', step='post')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Watered count')
        ax3.set_title('Fire Watered (burned_obj_num)')
        ax3.grid(True, alpha=0.3)
        if wn_data:
            ymax_w = max(wn_data)
            ax3.set_ylim(0, max(ymax_w * 1.1, 1))
            ax3.annotate(
                f'{int(wn_data[-1])}',
                xy=(t_w[-1], wn_data[-1]),
                xytext=(5, 10),
                textcoords='offset points',
                fontsize=10,
                color='darkcyan',
                fontweight='bold',
            )
        ax3.legend(loc='upper left')

        # 4. Total fired count (1,0)
        ax4 = axes[1, 0]
        burned_data = list(self.history["total_fired_num"])
        t_data, b_data = safe_slice(time_data, burned_data)
        ax4.plot(t_data, b_data, 'r-', linewidth=2.5, label='total_fired_num')
        ax4.fill_between(t_data, b_data, alpha=0.3, color='red')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Total fired count')
        ax4.set_title('Fired objects number')
        ax4.grid(True, alpha=0.3)
        if len(b_data) > 0:
            ax4.annotate(f'{b_data[-1]:.0f}', xy=(t_data[-1], b_data[-1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, color='red', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        ax4.legend()
        
        # 5. NPC health (1,1)
        ax5 = axes[1, 1]
        health_data = list(self.history["total_npc_health"])
        t_data, h_data = safe_slice(time_data, health_data)
        ax5.plot(t_data, h_data, 'g-', linewidth=2.5, label='Health')
        ax5.fill_between(t_data, h_data, alpha=0.3, color='green')
        ax5_twin = ax5.twinx()
        evac_data = list(self.history["evacuation_progress"])
        t_data2, e_data = safe_slice(time_data, evac_data)
        ax5_twin.plot(t_data2, e_data, 'm--', alpha=0.6, linewidth=2, label='Evacuation %')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Total Health', color='g')
        ax5_twin.set_ylabel('Evacuation Progress (%)', color='m')
        ax5.set_title('NPC Health & Evacuation')
        ax5.grid(True, alpha=0.3)
        if len(h_data) > 0:
            ax5.annotate(f'{h_data[-1]:.1f}', xy=(t_data[-1], h_data[-1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, color='green', fontweight='bold')
        if len(e_data) > 0:
            ax5_twin.annotate(f'{e_data[-1]:.1f}%', xy=(t_data2[-1], e_data[-1]),
                             xytext=(5, -15), textcoords='offset points',
                             fontsize=9, color='magenta', fontweight='bold')
        
        # 6. Remaining property (1,2)
        ax6 = axes[1, 2]
        property_data = list(self.history["remaining_property"])
        t_data, p_data = safe_slice(time_data, property_data)
        ax6.plot(t_data, p_data, 'orange', linewidth=2.5)
        ax6.fill_between(t_data, p_data, alpha=0.3, color='orange')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Property Health Sum')
        ax6.set_title('Remaining Property')
        ax6.grid(True, alpha=0.3)
        if len(p_data) > 0:
            ax6.annotate(f'{p_data[-1]:.0f}', xy=(t_data[-1], p_data[-1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, color='darkorange', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(self._agents), 1)))
        
        # 7. Agent distance (2,0)
        ax7 = axes[2, 0]
        for idx, (agent_id, agent) in enumerate(self._agents.items()):
            color = colors[idx % len(colors)]
            dist_history = list(agent.distance_history)
            if dist_history:
                t_plot, d_plot = safe_slice(time_data, dist_history)
                ax7.plot(t_plot, d_plot, '-', linewidth=2,
                        label=agent.name or agent_id[:8], color=color)
                if len(d_plot) > 0:
                    ax7.annotate(f'{d_plot[-1]:.1f}cm',
                                xy=(t_plot[-1], d_plot[-1]),
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=9, color=color, fontweight='bold')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Distance (cm)')
        ax7.set_title('Agent Distance Traveled')
        ax7.grid(True, alpha=0.3)
        ax7.legend(loc='upper left', fontsize=8)
        
        # 8. Agent water usage (2,1)
        ax8 = axes[2, 1]
        for idx, (agent_id, agent) in enumerate(self._agents.items()):
            color = colors[idx % len(colors)]
            water_history = list(agent.water_history)
            if water_history:
                t_plot, w_plot = safe_slice(time_data, water_history)
                ax8.plot(t_plot, w_plot, '-', linewidth=2,
                        label=agent.name or agent_id[:8], color=color)
                if len(w_plot) > 0:
                    ax8.annotate(f'{w_plot[-1]:.1f}',
                                xy=(t_plot[-1], w_plot[-1]),
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=9, color=color, fontweight='bold')
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Water Used')
        ax8.set_title('Agent Water Usage')
        ax8.grid(True, alpha=0.3)
        ax8.legend(loc='upper left', fontsize=8)
        
        # 9. Agent statistics table (2,2) - with extinguished count
        ax9 = axes[2, 2]
        ax9.axis('off')
        ax9.set_title('Agent Final Statistics & Fire Suppression', fontsize=11, fontweight='bold')
        table_data = []
        colLabels = ['Agent', 'Distance(cm)', 'Water', 'Rescued', 'Extinguished']
        for agent_id, agent in self._agents.items():
            table_data.append([
                agent.name or agent_id[:8],
                f"{agent.distance_traveled:.1f}",
                f"{agent.water_used:.1f}",
                str(agent.npcs_rescued),
                str(agent.extinguished_objects)  # NEW
            ])
        if table_data:
            table = ax9.table(
                cellText=table_data,
                colLabels=colLabels,
                cellLoc='center',
                loc='center',
                bbox=[0.05, 0.1, 0.9, 0.8]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            for i in range(len(colLabels)):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
        plt.tight_layout()
        self._plot_output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{self.timestamp}"
        final_plot_file = self._plot_output_dir / f"{stem}_final.png"
        plt.savefig(final_plot_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[PLOT] Final plot saved: {final_plot_file}")
    
    def _history_for_json(self) -> Dict[str, List]:
        return {key: list(series) for key, series in self.history.items()}
    

    def to_dict(self) -> Dict:
        """Convert to dict with corrected metrics"""
        current_metrics = self.calculate_metrics()
        total_npcs = len(self._npcs)
        rescued_npcs = sum(1 for n in self._npcs.values() if n.status == "rescued")
        total_water = sum(a.water_used for a in self._agents.values())
        fired = float(self.total_fired_num)
        watered = float(self.fire_watered_num)

        return {
            "scene_id": self.scene_id,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "success": self.success,
            "termination_reason": self.termination_reason,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.end_time - self.start_time if self.end_time > 0 else 0,
            "suppression_time": self.suppression_time,
            "evacuation_time": self.evacuation_time,
            "fire_start_time": self._fire_start_time - self.start_time if self._fire_start_time else None,
            "total_fired_num": self.total_fired_num,
            "fire_state": {
                "unburned_num": self.fire_unburned_num,
                "burning_num": self.fire_burning_num,
                "extinguished_num": self.fire_extinguished_num,
                "burned_obj_num": self.fire_watered_num,
                "destroyed_count": self.fire_destroyed_count,
            },
            # "safe_zones": [z.to_dict() for z in self.safe_zones],
            "npc_health": self.npc_health,
            "total_npc_health": self.total_npc_health,
            "initial_total_npc_health": self.initial_total_npc_health,
            "initial_property_value": self.initial_property_value,
            "remaining_property_value": self.remaining_property_value,
            "initial_property_num": self.initial_property_num,
            "remaining_property_num": self.remaining_property_num,
            "scene_total_object_count": getattr(self, "scene_total_object_count", 0),
            "npcs_status": self.npcs_status,
            "total_steps": self.total_steps,
            "dialogue_history": self.dialogue_history,
            "timestamp": self.timestamp,
            "agent_metrics": [a.to_dict() for a in self.agent_metrics],
            "npc_metrics": [n.to_dict() for n in self.npc_metrics],
            "statistics": {
                "total_npcs": total_npcs,
                "rescued_npcs": rescued_npcs,
                "dead_npcs": sum(1 for n in self._npcs.values() if n.status == "dead"),
                "trapped_npcs": sum(1 for n in self._npcs.values() if n.status == "trapped"),
                "total_agents": len(self._agents),
                "total_water_used": total_water,
                "extinguished_fires": sum(a.extinguished_objects for a in self._agents.values()),
            },
            "calculated_metrics": {
                "FSR": {
                    "value": current_metrics["FSR"],
                    "definition": "Fire Suppression Rate: burned_obj_num (watered) / total_fired_num",
                    "norm": "100%",
                    "description": f"{int(watered)} / {int(fired)}",
                },
                "RR": {
                    "value": current_metrics['RR'],
                    "definition": "Rescue Rate: rescued NPCs / total NPCs",
                    "norm": "100%",
                    "description": f"{rescued_npcs} / {total_npcs}"
                },
                "NSR": {
                    "value": current_metrics['NSR'],
                    "definition": "NPC Survival Ratio: remaining health / initial health",
                    "norm": "100%",
                    "description": f"{self.total_npc_health:.1f} / {self.initial_total_npc_health:.1f}"
                },
                "VDR": {
                    "value": current_metrics.get("VDR", 0.0),
                    "definition": "Remaining value ratio (global): remaining_property_value / initial_property_value",
                    "norm": "100%",
                    "description": f"{self.remaining_property_value} / {self.initial_property_value}",
                },
                "WUE": {
                    "value": current_metrics["WUE"],
                    "definition": "Water Use Efficiency: max(0, remaining_property_value - undamaged_value) / total_water; undamaged_value = max(0, scene_total(query_info@start) - total_fired_num) * 100",
                    "norm": "[0, +inf]",
                    "description": f"{self.saved_value} / {total_water:.1f}",
                },
                "G-FS": {
                    "value": current_metrics['G-FS'],
                    "definition": "Gini - Fire Suppression: Gini(extinguished objects per agent)",
                    "norm": "[0, 1]",
                    "description": "Gini coefficient of extinguished objects distribution"
                },
                "G-R": {
                    "value": current_metrics['G-R'],
                    "definition": "Gini - Rescue: Gini(rescued NPCs per agent)",
                    "norm": "[0, 1]",
                    "description": "Gini coefficient of rescued NPCs distribution"
                }
            },
            "agent_distribution": {
                "extinguished_objects": {agent.name or agent.id[:8]: agent.extinguished_objects 
                                        for agent in self._agents.values()},
                "rescued_npcs": {agent.name or agent.id[:8]: agent.npcs_rescued 
                               for agent in self._agents.values()}
            },
            "history_series": self._history_for_json(),
        }
    
    def save(self, output_dir: Optional[Path] = None) -> Path:
        if output_dir is not None:
            out_dir = Path(output_dir)
        else:
            base = os.environ.get("EMBODIED_BENCHMARK_DATA_ROOT")
            if base:
                out_dir = Path(base) / "evaluation_results"
            else:
                out_dir = Path("./data_save/evaluation_results")
        stem = f"{self.timestamp}"
        out_file = out_dir / f"{stem}.json"
        save_json(self.to_dict(), out_file)
        print(f"[SAVE] {out_file}")
        return out_file
    
    def get_agent(self, agent_id: str) -> Optional[AgentMetrics]:
        return self._agents.get(agent_id)
    
    def get_npc(self, npc_id: str) -> Optional[NpcMetrics]:
        return self._npcs.get(npc_id)
    
    def is_running(self) -> bool:
        return self._running
    
    def get_rescued_count(self) -> int:
        return sum(1 for n in self._npcs.values() if n.status == "rescued")
    
    def get_total_water_used(self) -> float:
        return sum(a.water_used for a in self._agents.values())


def attach_experiment_result_to_base_agent(base_agent: Any, result: ExperimentResult) -> None:
    """
    Attach ``ExperimentResult`` to a ``BaseAgent*`` instance's ``ActionAPI`` so ``send_stop_follow`` can update ``npcs_rescued``.
    """
    actions = getattr(base_agent, "_actions", None)
    if actions is not None:
        setattr(actions, "experiment_result", result)


# ========== Usage Example ==========

async def experiment_main(ctx: WorldContext):
    """Main experiment function running in TongSim event loop"""
    
    result = ExperimentResult(
        task_type="fire_rescue",
        task_id="exp_new_metrics_001",
    ).bind_context(
        ctx,
        update_interval=1.0,
        plot_output_dir=Path("./data_save/experiment_plots"),
        enable_live_display=False,
        agent_state_update_interval=0.5
    )
    

    # Initialize: get initial property values (safe zone fixed in bind_context)
    print("\nRefresh_map_actors")
    context = ctx
    await ts.UnaryAPI.refresh_actors_map(context.conn)
    result.initial_property_value = await ts.UnaryAPI.get_obj_residual(context.conn)
    burned_state = await ts.UnaryAPI.get_burned_area(context.conn)
    result.initial_property_num = burned_state["total_num"]
    
    FireDog_BP = "/Game/Blueprint/BP_Firedog.BP_Firedog_C"
    Hall_Spawn = ts.Vector3(-200, 100, 1000)
    Hall_Target = ts.Vector3(200, 100, 1000)
    WashRoom_Spawn = ts.Vector3(1045, 0, 1000)
    WashRoom_Target = ts.Vector3(1045, 500, 1000)
    ACCEPT_RADIUS = 20.0
    ALLOW_PARTIAL = True
    SPEED_UU_PER_SEC = 300.0
    MOVE_TIMEOUT = 60.0
    
    print("\n[RL:1] Generate FireDog ...")
    actor = await ts.UnaryAPI.spawn_actor(
        context.conn,
        blueprint=FireDog_BP,
        transform=ts.Transform(
            location=Hall_Spawn,
            rotation=ts.math.euler_to_quaternion(ts.Vector3(0, 0, 180), is_degree=True),
        ),
        name="EmbodiedMAS_Spawned",
        tags=["FireDog"],
        timeout=5.0,
    )
    
    #start to burn the fire
    print("\n[EMAS:7] start_to_burn")
    await ts.UnaryAPI.start_to_burn(context.conn)
    time.sleep(3)
    result.register_agents([actor["id"]], ["EmbodiedMAS_Spawned"])
    result.set_agent_water_setting(actor["id"], 3, 5)
    await result.start_async(enable_plotting=True)
    
    print("[INFO] Live display window started")
    print("[INFO] New metrics: FSR, RR, NSR, VDR, WUE, G-FS, G-R")
    print("[INFO] Waiting for fire to start...")

    print(f"\n[EMAS:6] NavigateToLocation")
    resp = await ts.UnaryAPI.navigate_to_location(
        context.conn,
        actor_id=actor["id"],
        target_location=WashRoom_Target,
        accept_radius=ACCEPT_RADIUS,
        allow_partial=ALLOW_PARTIAL,
        speed_uu_per_sec=SPEED_UU_PER_SEC,
        timeout=MOVE_TIMEOUT,
    )
    print("navigate resp:", resp)
    time.sleep(3)

    print("\n[EMAS:3] ExtinguishFire")
    res = await ts.UnaryAPI.extinguish_fire(context.conn, actor["id"])
    print(res)

    print("\n[EMAS:1] GetPerceptionInfo")
    perception = await ts.UnaryAPI.get_embodied_perception(context.conn, actor["id"])
    print("actor count:", len(perception["actor_info"]))
    print("NPC count:", len(perception["npc_info"]))
    time.sleep(3)

    print("\n[EMAS:2-1] sendfollow")
    resfollow = await ts.UnaryAPI.sendfollow(context.conn, actor["id"])
    print(resfollow)
    time.sleep(5)

    print(f"\n[EMAS:6] NavigateToLocation")
    resp = await ts.UnaryAPI.navigate_to_location(
        context.conn,
        actor_id=actor["id"],
        target_location=Hall_Target,
        accept_radius=ACCEPT_RADIUS,
        allow_partial=ALLOW_PARTIAL,
        speed_uu_per_sec=SPEED_UU_PER_SEC,
        timeout=MOVE_TIMEOUT,
    )
    print("navigate resp:", resp)
    time.sleep(3)

    print("\n[EMAS:2-2] sendstopfollow")
    stop_follow_npc_ids = await ts.UnaryAPI.sendstopfollow(context.conn, actor["id"])
    print(stop_follow_npc_ids)
    await result.record_rescue_after_stop_follow(actor["id"])
    time.sleep(2)

    print("\n[EMAS:4] get_selfstate")
    resstopfollow = await ts.UnaryAPI.get_selfstate(context.conn, actor["id"])
    print("Current Agent state:")
    agent_obj = result.get_agent(actor['id'])
    print(f"  - Distance: {agent_obj.distance_traveled:.2f}m")
    print(f"  - Water: {agent_obj.water_used:.2f}")
    print(f"  - Rescued: {agent_obj.npcs_rescued}")
    print(f"  - Extinguished: {agent_obj.extinguished_objects}")  # NEW
    time.sleep(2)

    print("\n[EMAS:8] get_burned_area")
    burned_state = await ts.UnaryAPI.get_burned_area(context.conn)
    print(burned_state)
    time.sleep(3)

    print("\n[EMAS:9] get_obj_residual")
    obj_health = await ts.UnaryAPI.get_obj_residual(context.conn)
    print("objhealth:", obj_health)
    time.sleep(3)

    print("\n[EMAS:10] get_npc_health")
    npc_health = await ts.UnaryAPI.get_npc_health(context.conn)
    print("npc_health:", npc_health)
    time.sleep(3)

    print("\n[EMAS:11] GetFireState")
    OutFire = await ts.UnaryAPI.get_outfire_state(context.conn)
    print("OutFire:", OutFire)
    time.sleep(3)

    print("\n[EMAS:12] GetNPCPostions")
    postions = await ts.UnaryAPI.get_npc_postions(context.conn)
    print("NPCpostions:", postions)
    time.sleep(3)

    print("\n[RL:2] query_info")
    state_list = await ts.UnaryAPI.query_info(context.conn)
    print(f"  - actor count: {len(state_list)}")
    time.sleep(3)
    
    # NEW: Get extinguished objects for each agent
    print("\n[EMAS:14] get_agent_extinguished_objects")
    res = await ts.UnaryAPI.get_agent_extinguished_objects(context.conn)
    print(res)
    time.sleep(3)

    print(f"\n[EMAS:13] NavigateToLocation")
    resp = await ts.UnaryAPI.navigate_to_location(
        context.conn,
        actor_id=actor["id"],
        target_location=WashRoom_Spawn,
        accept_radius=ACCEPT_RADIUS,
        allow_partial=ALLOW_PARTIAL,
        speed_uu_per_sec=SPEED_UU_PER_SEC,
        timeout=MOVE_TIMEOUT,
    )
    print("navigate resp:", resp)
    time.sleep(3)

    print("\n[EMAS:14] ExtinguishFire")
    res = await ts.UnaryAPI.extinguish_fire(context.conn, actor["id"])
    print(res)
    time.sleep(13)

    print(f"\n[EMAS:15] NavigateToLocation")
    resp = await ts.UnaryAPI.navigate_to_location(
        context.conn,
        actor_id=actor["id"],
        target_location=Hall_Target,
        accept_radius=ACCEPT_RADIUS,
        allow_partial=ALLOW_PARTIAL,
        speed_uu_per_sec=SPEED_UU_PER_SEC,
        timeout=MOVE_TIMEOUT,
    )
    print("navigate resp:", resp)
    time.sleep(3)

    if result.is_running():
        await result.stop_async(success=True, reason="Experiment completed")
    
    print("[INFO] Experiment ended")
    print(f"[INFO] Final statistics:")
    for agent_id, agent in result._agents.items():
        print(f"  Agent {agent.name}:")
        print(f"    - Distance: {agent.distance_traveled:.2f}cm")
        print(f"    - Water: {agent.water_used:.2f}")
        print(f"    - Rescued: {agent.npcs_rescued}")
        print(f"    - Extinguished: {agent.extinguished_objects}")  # NEW
    
    final_metrics = result.calculate_metrics()
    print("\n[FINAL METRICS]")
    for metric, value in final_metrics.items():
        print(f"  {metric}: {value:.4f}")


def main():
    print("[INFO] Connecting to TongSim ...")
    if not HAS_PYQT5:
        print("[WARN] PyQt5 not installed")
    
    with ts.TongSim(grpc_endpoint="127.0.0.1:5726") as ue:
        ue.context.sync_run(experiment_main(ue.context))


if __name__ == "__main__":
    main()
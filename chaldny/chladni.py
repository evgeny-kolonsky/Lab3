"""
chladni.py - Chladni / mechanical-resonance experiment (single file).

    python chladni.py          # real Keysight DSO-X 2000 (via VISA)
    python chladni.py --sim    # simulator, no hardware needed

Requires: numpy, matplotlib, PyQt5, pyvisa (+ Keysight IO Libraries Suite),
optional scipy (peak search).

Hardware chain:
    DSO-X 2000 WaveGen  ->  power amplifier  ->  mechanical wave driver
    CH1 : voltage across low-value shunt (current), differential probe
    CH2 : voltage across the driver, differential probe (e.g. 1:10 / 1:20 / 1:50)

Measurement principle
    Both channels are digitised simultaneously (:DIGitize CHAN1,CHAN2) and a
    sine of the KNOWN generator frequency  y = a*cos(wt) + b*sin(wt) + c  is
    fitted to each record (digital lock-in).  This gives amplitude and phase of
    U and I with much better noise immunity than the scope's built-in :MEAS.
    From them:  |Z|, R = Re Z, X = Im Z, phase, active power P = Urms*Irms*cos(phi),
    and the "motional" power P_mech = P - Irms^2 * Re_dc  (power not dissipated
    in the voice-coil wire, i.e. going into mechanics).
"""

import math
import time
import threading
from dataclasses import dataclass, asdict, field

import numpy as np

# 1-2-5 sequence of V/div at the probe tip for a 1:1 probe
_BASE_SCALES = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
SCREEN_HALF_DIV = 4.0          # DSO-X 2000: 8 vertical divisions


def scale_list(probe_ratio):
    return [s * probe_ratio for s in _BASE_SCALES]


def pick_scale(peak_volts, probe_ratio, target_div=3.0):
    """Smallest 1-2-5 scale that puts |peak| at <= target_div divisions."""
    for s in scale_list(probe_ratio):
        if peak_volts / s <= target_div:
            return s
    return scale_list(probe_ratio)[-1]


# ----------------------------------------------------------------------------
# Configuration of one measurement
# ----------------------------------------------------------------------------
@dataclass
class MeasConfig:
    r_shunt: float = 1.0          # Ohm
    probe1: float = 10.0          # CH1 probe ratio (differential probe on shunt)
    probe2: float = 10.0          # CH2 probe ratio (differential probe)
    invert_current: bool = False  # flip sign of CH1 if the shunt is wired backwards
    re_dc: float = 0.0            # DC resistance of driver coil, Ohm (for P_mech); 0 = unknown
    wgen_vpp: float = 0.5         # generator amplitude, Vpp (into Hi-Z)
    wgen_max_vpp: float = 2.0     # never exceed this generator amplitude
    hold_vrms: float = 0.0        # >0: regulate generator so that Urms(driver) = hold_vrms
    hold_tol: float = 0.02        # relative tolerance of the regulation
    i_max_rms: float = 0.7        # current limit (driver has 1 A fuse)
    n_periods: float = 10.0       # periods in the acquisition window
    averages: int = 1             # scope averaging (1 = normal acquisition)
    points: int = 2000            # waveform points per channel
    settle_s: float = 0.4         # wait after frequency change (mechanical transient)


# ----------------------------------------------------------------------------
# Lock-in style sine fit
# ----------------------------------------------------------------------------
def fit_sine(t, y, f):
    """Fit y = A*cos(2*pi*f*t + phi) + c. Returns (A, phi_rad, c, resid_rms)."""
    w = 2 * math.pi * f
    M = np.column_stack([np.cos(w * t), np.sin(w * t), np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    a, b, c = coef
    A = math.hypot(a, b)
    phi = math.atan2(-b, a)
    resid = y - M @ coef
    return A, phi, c, float(np.sqrt(np.mean(resid ** 2)))


def wrap_deg(x):
    return (x + 180.0) % 360.0 - 180.0


def analyse(t, v_shunt, v_drv, f, cfg: MeasConfig):
    """Compute all electrical quantities from the two records."""
    sgn = -1.0 if cfg.invert_current else 1.0
    i = sgn * v_shunt / cfg.r_shunt
    Ia, Iph, Ic, Ires = fit_sine(t, i, f)
    Ua, Uph, Uc, Ures = fit_sine(t, v_drv, f)
    phi = wrap_deg(math.degrees(Uph - Iph))          # phase of Z (U leads I > 0)
    Urms, Irms = Ua / math.sqrt(2), Ia / math.sqrt(2)
    Z = Ua / Ia if Ia > 0 else float("nan")
    R = Z * math.cos(math.radians(phi))
    X = Z * math.sin(math.radians(phi))
    P = Urms * Irms * math.cos(math.radians(phi))
    # direct time-domain power (sanity check; exact only for integer periods)
    P_td = float(np.mean((v_drv - Uc) * (i - Ic)))
    P_mech = P - Irms ** 2 * cfg.re_dc if cfg.re_dc > 0 else float("nan")
    return dict(
        f=f, Urms=Urms, Irms=Irms, Z=Z, R=R, X=X, phase=phi,
        P=P, P_td=P_td, P_mech=P_mech, S=Urms * Irms,
        PF=math.cos(math.radians(phi)),
        U_snr=Ua / max(Ures, 1e-15), I_snr=Ia / max(Ires, 1e-15),
    )


# ----------------------------------------------------------------------------
# Keysight / Agilent DSO-X 2000 driver
# ----------------------------------------------------------------------------
class DSOX2000:
    def __init__(self, resource_name=None, rm=None):
        import pyvisa
        log = []
        if rm is None:
            try:
                rm = pyvisa.ResourceManager()            # Keysight / NI VISA
            except Exception as e:
                log.append(f"system VISA: {e}")
                rm = pyvisa.ResourceManager("@py")       # pyvisa-py fallback
        self.rm = rm
        self.inst = None
        try:
            log.append(f"VISA library: {self.rm.visalib}")
        except Exception:
            pass
        names = [resource_name] if resource_name else list(self.rm.list_resources())
        if not names:
            log.append("list_resources() returned nothing (USB cable / driver / scope off?)")
        # USB instruments first, serial ports last (they are slow to time out)
        names.sort(key=lambda r: (not r.upper().startswith("USB"), r.upper().startswith("ASRL")))
        for r in names:
            try:
                dev = self.rm.open_resource(r)
                dev.timeout = 3000
                idn = dev.query("*IDN?").strip()
                if any(k in idn.upper() for k in ("DSO-X", "DSOX", "KEYSIGHT", "AGILENT")):
                    self.inst, self.idn, self.resource = dev, idn, r
                    break
                log.append(f"{r}: not a DSO-X ({idn})")
                dev.close()
            except Exception as e:
                log.append(f"{r}: {type(e).__name__}: {e}")
                continue
        if self.inst is None:
            raise RuntimeError("Keysight DSO-X oscilloscope not found.\n" + "\n".join(log))
        self.inst.timeout = 30000
        self.lock = threading.RLock()
        self.scale = {1: None, 2: None}
        self.probe = {1: 1.0, 2: 1.0}
        self.trigger_source = None

    # -- low level -----------------------------------------------------------
    def w(self, cmd):
        with self.lock:
            self.inst.write(cmd)

    def q(self, cmd):
        with self.lock:
            return self.inst.query(cmd).strip()

    def errors(self):
        errs = []
        for _ in range(20):
            e = self.q(":SYST:ERR?")
            if e.startswith("+0") or e.startswith("0"):
                break
            errs.append(e)
        return errs

    # -- set-up --------------------------------------------------------------
    def setup(self, cfg: MeasConfig):
        with self.lock:
            self.w("*RST")
            time.sleep(2.0)
            self.w("*CLS")
            for ch, pr in ((1, cfg.probe1), (2, cfg.probe2)):
                self.probe[ch] = pr
                self.w(f":CHAN{ch}:DISP ON")
                self.w(f":CHAN{ch}:PROB {pr}")
                # DC coupling: AC coupling (~3.5 Hz HPF) would add phase error at low f
                self.w(f":CHAN{ch}:COUP DC")
                self.w(f":CHAN{ch}:BWL ON")
                self.w(f":CHAN{ch}:OFFS 0")
                self.set_scale(ch, scale_list(pr)[8])       # 0.5 V/div * probe
            self.w(":TIM:REF CENT")
            self.w(":TRIG:MODE EDGE")
            self.w(":TRIG:SWE AUTO")
            # Both measured channels are busy -> trigger on the generator itself
            self.errors()
            self.w(":TRIG:EDGE:SOUR WGEN")
            if self.errors():
                self.w(":TRIG:EDGE:SOUR CHAN2")
                self.w(":TRIG:EDGE:LEV 0")
                self.trigger_source = "CHAN2"
            else:
                self.trigger_source = "WGEN"
            self.set_averaging(cfg.averages)
            self.w(":WAV:POIN:MODE NORM")
            self.w(":WAV:FORM WORD")
            self.w(":WAV:BYT LSBF")
            self.w(":WAV:UNS 1")
            self.wgen_setup(cfg.wgen_vpp)
            self.errors()

    def set_averaging(self, n):
        if n and n > 1:
            self.w(":ACQ:TYPE AVER")
            self.w(f":ACQ:COUN {int(n)}")
        else:
            self.w(":ACQ:TYPE NORM")

    def set_scale(self, ch, volts_per_div):
        self.w(f":CHAN{ch}:SCAL {volts_per_div:.6g}")
        self.scale[ch] = volts_per_div

    # -- generator -----------------------------------------------------------
    def wgen_setup(self, vpp, freq=1000.0):
        self.w(":WGEN:OUTP OFF")
        self.w(":WGEN:FUNC SIN")
        self.w(":WGEN:OUTP:LOAD ONEM")        # amplifier input is high impedance
        self.w(f":WGEN:FREQ {freq:.6f}")
        self.w(f":WGEN:VOLT {vpp:.4f}")
        self.w(":WGEN:VOLT:OFFS 0")

    def wgen_on(self, on=True):
        self.w(f":WGEN:OUTP {'ON' if on else 'OFF'}")

    def set_amplitude(self, vpp):
        self.w(f":WGEN:VOLT {vpp:.4f}")

    def set_frequency(self, f, n_periods=10.0):
        self.w(f":WGEN:FREQ {f:.6f}")
        self.w(f":TIM:SCAL {n_periods / f / 10.0:.6g}")

    def get_frequency(self):
        return float(self.q(":WGEN:FREQ?"))

    # -- acquisition ---------------------------------------------------------
    def _read_channel(self, ch, points):
        self.inst.write(f":WAV:SOUR CHAN{ch}")
        self.inst.write(f":WAV:POIN {int(points)}")
        pre = self.inst.query(":WAV:PRE?").strip().split(",")
        xinc, xorg, xref = float(pre[4]), float(pre[5]), float(pre[6])
        yinc, yorg, yref = float(pre[7]), float(pre[8]), float(pre[9])
        raw = self.inst.query_binary_values(":WAV:DATA?", datatype="H",
                                            is_big_endian=False, container=np.array)
        v = (raw.astype(float) - yref) * yinc + yorg
        t = (np.arange(len(v)) - xref) * xinc + xorg
        return t, v

    def acquire(self, points=2000):
        """Single simultaneous acquisition of both channels. Returns t, v1, v2."""
        with self.lock:
            self.inst.write(":DIG CHAN1,CHAN2")
            self.inst.query("*OPC?")
            t, v1 = self._read_channel(1, points)
            _, v2 = self._read_channel(2, points)
        n = min(len(v1), len(v2))
        return t[:n], v1[:n], v2[:n]

    def run(self):
        self.w(":RUN")

    def close(self):
        try:
            self.wgen_on(False)
            self.run()
            self.inst.close()
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Simulator (same interface) - for testing the software without hardware
# ----------------------------------------------------------------------------
class SimulatedScope:
    """Moving-coil driver: Z = Re + jwL + sum_k (Bl^2)/(Rm_k + jw m_k + s_k/(jw)).
    The first term of the sum is the driver suspension, the others mimic plate modes."""

    def __init__(self, with_plate=True, seed=0):
        self.idn = "SIMULATED,DSO-X 2002A,SIM0001,0.0"
        self.resource = "SIM"
        self.trigger_source = "WGEN"
        self.lock = threading.RLock()
        self.rng = np.random.default_rng(seed)
        self.scale = {1: 0.05, 2: 1.0}
        self.probe = {1: 1.0, 2: 10.0}
        self.f = 1000.0
        self.vpp = 0.5
        self.on = False
        self.gain = 10.0              # amplifier voltage gain
        self.r_out = 0.05             # amplifier output resistance
        self.r_shunt = 0.5
        self.Re, self.L = 3.4, 0.6e-3
        self.with_plate = with_plate
        modes = [(28.0, 3.0)]         # (f0 Hz, Q) suspension resonance
        if with_plate:
            modes += [(f0, 60 + 0.02 * f0) for f0 in
                      (231, 347, 612, 788, 1043, 1260, 1590, 1888, 2240, 2675, 3150, 3604, 4210)]
        self.modes = modes

    # interface stubs
    def setup(self, cfg):
        self.probe = {1: cfg.probe1, 2: cfg.probe2}
        self.r_shunt = cfg.r_shunt
        self.vpp = cfg.wgen_vpp
        self.scale = {1: scale_list(cfg.probe1)[8], 2: scale_list(cfg.probe2)[8]}

    def errors(self): return []
    def set_averaging(self, n): self.avg = n
    def set_scale(self, ch, s): self.scale[ch] = s
    def wgen_setup(self, vpp, freq=1000.0): self.vpp, self.f = vpp, freq
    def wgen_on(self, on=True): self.on = on
    def set_amplitude(self, vpp): self.vpp = vpp
    def set_frequency(self, f, n_periods=10.0): self.f, self.np = f, n_periods
    def get_frequency(self): return self.f
    def run(self): pass
    def close(self): pass

    def impedance(self, f):
        w = 2 * math.pi * f
        Z = self.Re + 1j * w * self.L
        for k, (f0, Q) in enumerate(self.modes):
            # reflected motional impedance of a mechanical resonator =
            # parallel-RLC-like peak of height A at f0
            A = 15.0 if k == 0 else 2.0 + 4.0 * ((k * 7) % 5) / 4
            Z += A / (1 + 1j * Q * (f / f0 - f0 / f))
        return Z

    def acquire(self, points=2000):
        f = self.f
        n_per = getattr(self, "np", 10.0)
        t = np.linspace(-n_per / f / 2, n_per / f / 2, points)
        U0 = (self.vpp / 2) * self.gain if self.on else 0.0
        Z = self.impedance(f)
        Ic = U0 / (Z + self.r_shunt + self.r_out)
        w = 2 * math.pi * f
        i = abs(Ic) * np.cos(w * t + np.angle(Ic))
        u = abs(Ic * Z) * np.cos(w * t + np.angle(Ic * Z))
        v1 = i * self.r_shunt + self.rng.normal(0, 3e-4, points)
        v2 = u + self.rng.normal(0, 5e-3, points)
        lim1, lim2 = 4.2 * self.scale[1], 4.2 * self.scale[2]
        return t, np.clip(v1, -lim1, lim1), np.clip(v2, -lim2, lim2)


# ----------------------------------------------------------------------------
# One measurement point with auto-ranging, voltage hold and current limit
# ----------------------------------------------------------------------------
def autorange_acquire(scope, cfg: MeasConfig, max_iter=6):
    """Acquire, adjusting V/div of both channels until signals use 1..3.6 div."""
    for _ in range(max_iter):
        t, v1, v2 = scope.acquire(cfg.points)
        changed = False
        for ch, v, pr in ((1, v1, cfg.probe1), (2, v2, cfg.probe2)):
            s = scope.scale[ch] or scale_list(pr)[8]
            peak = float(np.max(np.abs(v))) if len(v) else 0.0
            div = peak / s
            sl = scale_list(pr)
            if div >= 3.9:                                   # clipped -> 2 steps up
                idx = min(sl.index(min(sl, key=lambda x: abs(x - s))) + 2, len(sl) - 1)
                new = sl[idx]
            elif div > 3.6 or div < 1.0:
                new = pick_scale(peak, pr, 3.0)
            else:
                new = s
            if not math.isclose(new, s, rel_tol=1e-6):
                scope.set_scale(ch, new)
                changed = True
        if not changed:
            return t, v1, v2, False
    return t, v1, v2, True          # True -> range not settled (warning)


def measure_point(scope, f, cfg: MeasConfig, settle=True, state=None, set_freq=True):
    """Set frequency, wait, measure. `state` dict carries current generator amplitude."""
    if state is None:
        state = {}
    vpp = state.get("vpp", cfg.wgen_vpp)
    if set_freq:
        scope.set_frequency(f, cfg.n_periods)
    if settle:
        time.sleep(cfg.settle_s)
    flags = []
    for it in range(6):
        t, v1, v2, unsettled = autorange_acquire(scope, cfg)
        if unsettled:
            flags.append("range")
        res = analyse(t, v1, v2, f, cfg)
        # --- current limit
        if res["Irms"] > cfg.i_max_rms:
            vpp = max(vpp * 0.7 * cfg.i_max_rms / res["Irms"], 0.01)
            scope.set_amplitude(vpp)
            flags.append("I-limit")
            time.sleep(min(cfg.settle_s, 0.3))
            continue
        # --- voltage hold (constant Urms on the driver)
        if cfg.hold_vrms > 0 and res["Urms"] > 0:
            ratio = cfg.hold_vrms / res["Urms"]
            if abs(ratio - 1) > cfg.hold_tol and it < 5:
                new = min(max(vpp * ratio, 0.01), cfg.wgen_max_vpp)
                if not math.isclose(new, vpp, rel_tol=1e-3):
                    vpp = new
                    scope.set_amplitude(vpp)
                    time.sleep(min(cfg.settle_s, 0.3))
                    continue
                flags.append("hold-sat")
        break
    state["vpp"] = vpp
    res.update(wgen_vpp=vpp, scale1=scope.scale[1], scale2=scope.scale[2],
               t_meas=time.time(), flags=";".join(sorted(set(flags))))
    return res


def frequency_list(f_start, f_stop, step=None, n_points=None, log=False):
    if log:
        n = int(n_points or 200)
        return list(np.geomspace(f_start, f_stop, n))
    step = float(step)
    n = int(math.floor((f_stop - f_start) / step + 1e-9)) + 1
    return [round(f_start + k * step, 6) for k in range(n)]


RESULT_FIELDS = ["f", "Urms", "Irms", "Z", "R", "X", "phase", "P", "P_td", "P_mech",
                 "S", "PF", "U_snr", "I_snr", "wgen_vpp", "scale1", "scale2",
                 "t_meas", "flags"]


# ============================================================================
#  GUI
# ============================================================================


import sys
import os
import csv
import json
import time
import math
from datetime import datetime
from collections import OrderedDict

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QTabWidget, QFileDialog,
    QMessageBox, QComboBox, QGroupBox, QCheckBox, QListWidget, QDialog,
    QDialogButtonBox, QFormLayout, QProgressBar, QScrollArea, QSizePolicy)
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure

core = sys.modules[__name__]   # single-file version: core functions live above

try:
    from scipy.signal import find_peaks
except Exception:            # scipy is optional
    find_peaks = None

SIM = "--sim" in sys.argv

QUANTITIES = OrderedDict([
    ("Z",      ("|Z|  (Ohm)",            1.0)),
    ("R",      ("R = Re Z  (Ohm)",       1.0)),
    ("X",      ("X = Im Z  (Ohm)",       1.0)),
    ("phase",  ("phase U-I  (deg)",      1.0)),
    ("P",      ("active power P  (mW)",  1e3)),
    ("P_mech", ("P - I²Re  (mW)",        1e3)),
    ("Irms",   ("I rms  (mA)",           1e3)),
    ("Urms",   ("U rms driver  (V)",     1.0)),
    ("PF",     ("power factor cos φ",    1.0)),
    ("wgen_vpp", ("generator ampl. (Vpp)", 1.0)),
])
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
          "#8c564b", "#e377c2", "#17becf", "#7f7f7f", "#bcbd22"]


def dspin(lo, hi, val, dec=2, step=None, suffix=""):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setValue(val)
    if step:
        s.setSingleStep(step)
    if suffix:
        s.setSuffix(suffix)
    s.setMinimumHeight(26)
    return s


# =============================================================================
# Connection dialog
# =============================================================================
class ConnectDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chladni experiment - connect oscilloscope")
        lay = QFormLayout(self)
        self.combo = QComboBox()
        self.combo.setMinimumWidth(380)
        self.combo.addItem("Auto-detect Keysight DSO-X", None)
        try:
            import pyvisa
            for r in pyvisa.ResourceManager().list_resources():
                self.combo.addItem(r, r)
        except Exception as e:
            self.combo.addItem(f"(VISA not available: {e})", None)
        self.combo.addItem("SIMULATOR (no hardware)", "SIM")
        if SIM:
            self.combo.setCurrentIndex(self.combo.count() - 1)
        lay.addRow("VISA resource:", self.combo)
        self.status = QLabel("")
        lay.addRow(self.status)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Connect")
        bb.accepted.connect(self.try_connect)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)
        self.scope = None

    def try_connect(self):
        res = self.combo.currentData()
        self.status.setText("Connecting...")
        QApplication.processEvents()
        try:
            self.scope = core.SimulatedScope() if res == "SIM" else core.DSOX2000(res)
            self.accept()
        except Exception as e:
            self.status.setText("<font color=red>Connection failed - details in the message box</font>")
            QMessageBox.critical(self, "Connection error", str(e))


# =============================================================================
# Worker threads
# =============================================================================
class SweepWorker(QObject):
    point = pyqtSignal(dict)
    message = pyqtSignal(str)
    finished = pyqtSignal(bool)          # True = completed, False = stopped / error

    def __init__(self, scope, freqs, cfg, setup_needed):
        super().__init__()
        self.scope, self.freqs, self.cfg = scope, freqs, cfg
        self.setup_needed = setup_needed
        self._run = True

    def stop(self):
        self._run = False

    def run(self):
        ok = True
        try:
            if self.setup_needed:
                self.message.emit("Configuring oscilloscope...")
                self.scope.setup(self.cfg)
            self.scope.set_averaging(self.cfg.averages)
            self.scope.set_amplitude(self.cfg.wgen_vpp)
            self.scope.set_frequency(self.freqs[0], self.cfg.n_periods)
            self.scope.wgen_on(True)
            time.sleep(max(1.0, self.cfg.settle_s))
            state = {"vpp": self.cfg.wgen_vpp}
            for k, f in enumerate(self.freqs):
                if not self._run:
                    ok = False
                    break
                r = core.measure_point(self.scope, f, self.cfg, state=state)
                r["index"], r["n_total"] = k, len(self.freqs)
                self.point.emit(r)
        except Exception as e:
            self.message.emit(f"ERROR: {e}")
            ok = False
        finally:
            try:
                self.scope.run()
            except Exception:
                pass
        self.finished.emit(ok)


class LiveWorker(QObject):
    """Continuous measurement at a fixed frequency (manual / Chladni-pattern mode)."""
    point = pyqtSignal(dict)
    message = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, scope, cfg):
        super().__init__()
        self.scope, self.cfg = scope, cfg
        self._run = True
        self.freq = None
        self.vpp = cfg.wgen_vpp
        self._pending = True

    def update(self, freq, vpp, cfg):
        self.freq, self.vpp, self.cfg = freq, vpp, cfg
        self._pending = True

    def stop(self):
        self._run = False

    def run(self):
        state = {}
        try:
            while self._run:
                new = self._pending
                if new:
                    self._pending = False
                    self.scope.set_averaging(self.cfg.averages)
                    self.scope.set_amplitude(self.vpp)
                    state["vpp"] = self.vpp
                r = core.measure_point(self.scope, self.freq, self.cfg,
                                       settle=new, set_freq=new, state=state)
                self.point.emit(r)
                time.sleep(0.05)
        except Exception as e:
            self.message.emit(f"ERROR: {e}")
        finally:
            try:
                self.scope.run()
            except Exception:
                pass
        self.finished.emit()


# =============================================================================
# Plot
# =============================================================================
class SweepCanvas(FigureCanvas):
    marker_set = pyqtSignal(float)

    def __init__(self):
        self.fig = Figure(figsize=(9, 7), tight_layout=True)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212, sharex=self.ax1)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.marker = None
        self.toolbar = None
        self.mpl_connect("button_press_event", self._click)

    def _click(self, ev):
        if self.toolbar is not None and self.toolbar.mode:
            return                          # zoom / pan active
        if ev.inaxes in (self.ax1, self.ax2) and ev.xdata is not None and ev.button == 1:
            self.marker = float(ev.xdata)
            self.marker_set.emit(self.marker)


# =============================================================================
# Main window
# =============================================================================
class MainWindow(QMainWindow):
    def __init__(self, scope):
        super().__init__()
        self.scope = scope
        self.scope_configured = False
        self.setWindowTitle(f"Chladni / mechanical resonance  —  {scope.idn.split(',')[1] if ',' in scope.idn else scope.idn}")
        self.resize(1450, 860)

        self.runs = OrderedDict()           # name -> list of result dicts
        self.run_meta = {}
        self.peaks = []
        self.sweep_thread = self.sweep_worker = None
        self.live_thread = self.live_worker = None
        self.current_run = None

        font = QFont()
        font.setPointSize(10)
        self.setFont(font)

        self._redraw_timer = QTimer(singleShot=True, interval=150, timeout=self.redraw)
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(self._build_sweep_tab(), "Frequency sweep")
        self.tabs.addTab(self._build_manual_tab(), "Manual / Chladni pattern")
        self.statusBar().showMessage(
            f"Connected: {scope.idn}   |   resource: {getattr(scope, 'resource', '?')}")

    # ------------------------------------------------------------------ UI
    def _build_settings_box(self):
        g = QGroupBox("Circuit && safety")
        f = QFormLayout(g)
        self.r_shunt = dspin(0.001, 100, 1.0, 4, 0.1, " Ω")
        self.probe1 = dspin(0.1, 1000, 10, 1, 1, " :1")
        self.probe2 = dspin(0.1, 1000, 10, 1, 1, " :1")
        self.invert_i = QCheckBox("invert current sign")
        self.re_dc = dspin(0, 100, 0, 3, 0.1, " Ω")
        self.re_dc.setToolTip("DC resistance of the driver coil (ohmmeter). Used for P_mech = P - I²·Re")
        self.i_max = dspin(0.01, 2, 0.7, 2, 0.05, " A rms")
        self.wgen_max = dspin(0.02, 5, 2.0, 3, 0.1, " Vpp")
        f.addRow("Shunt R:", self.r_shunt)
        f.addRow("CH1 diff. probe (shunt):", self.probe1)
        f.addRow("CH2 diff. probe (driver):", self.probe2)
        f.addRow(self.invert_i)
        f.addRow("Coil Re (DC):", self.re_dc)
        f.addRow("Current limit:", self.i_max)
        f.addRow("Max generator ampl.:", self.wgen_max)
        for w in (self.r_shunt, self.probe1, self.probe2):
            w.valueChanged.connect(lambda *_: setattr(self, "scope_configured", False))
        return g

    def _build_sweep_tab(self):
        w = QWidget()
        lay = QHBoxLayout(w)

        left = QVBoxLayout()
        left.addWidget(self._build_settings_box())

        # -- generator
        g = QGroupBox("Excitation")
        fl = QFormLayout(g)
        self.wgen_vpp = dspin(0.02, 5, 0.5, 3, 0.05, " Vpp")
        self.hold_chk = QCheckBox("hold U rms on driver at")
        self.hold_v = dspin(0.01, 50, 1.0, 3, 0.1, " V")
        fl.addRow("Generator amplitude:", self.wgen_vpp)
        fl.addRow(self.hold_chk, self.hold_v)
        left.addWidget(g)

        # -- sweep
        g = QGroupBox("Sweep")
        gl = QGridLayout(g)
        self.f_start = dspin(1, 20000, 100, 1, 10, " Hz")
        self.f_stop = dspin(1, 20000, 3000, 1, 10, " Hz")
        self.f_step = dspin(0.01, 1000, 2, 2, 0.5, " Hz")
        self.log_chk = QCheckBox("log spacing, points:")
        self.n_log = QSpinBox(); self.n_log.setRange(5, 20000); self.n_log.setValue(400)
        self.settle = dspin(0, 10, 0.4, 2, 0.1, " s")
        self.n_periods = dspin(2, 200, 10, 0, 1, "")
        self.averages = QComboBox(); self.averages.addItems(["1", "2", "4", "8", "16", "32", "64"])
        self.points = QComboBox(); self.points.addItems(["1000", "2000", "5000", "10000"]); self.points.setCurrentIndex(1)
        self.run_name = QComboBox(); self.run_name.setEditable(True)
        self.run_name.addItems(["no plate (reference)", "square plate", "circular plate",
                                "triangle plate", "violin plate"])
        self.npts_label = QLabel()
        rows = [("Start:", self.f_start), ("Stop:", self.f_stop), ("Step:", self.f_step)]
        for i, (t, wd) in enumerate(rows):
            gl.addWidget(QLabel(t), i, 0); gl.addWidget(wd, i, 1)
        gl.addWidget(self.log_chk, 3, 0); gl.addWidget(self.n_log, 3, 1)
        gl.addWidget(QLabel("Settle time:"), 4, 0); gl.addWidget(self.settle, 4, 1)
        gl.addWidget(QLabel("Periods in window:"), 5, 0); gl.addWidget(self.n_periods, 5, 1)
        gl.addWidget(QLabel("Scope averages:"), 6, 0); gl.addWidget(self.averages, 6, 1)
        gl.addWidget(QLabel("Waveform points:"), 7, 0); gl.addWidget(self.points, 7, 1)
        gl.addWidget(QLabel("Run name:"), 8, 0); gl.addWidget(self.run_name, 8, 1)
        gl.addWidget(self.npts_label, 9, 0, 1, 2)
        for wd in (self.f_start, self.f_stop, self.f_step, self.settle, self.n_periods):
            wd.valueChanged.connect(self.update_npts)
        self.n_log.valueChanged.connect(self.update_npts)
        self.averages.currentIndexChanged.connect(self.update_npts)
        self.log_chk.toggled.connect(self.update_npts)
        left.addWidget(g)

        # -- buttons
        g = QGroupBox("Control")
        bl = QGridLayout(g)
        self.btn_start = QPushButton("▶  START SWEEP")
        self.btn_start.setStyleSheet("QPushButton{background:#4CAF50;color:white;font-weight:bold;padding:6px}")
        self.btn_stop = QPushButton("■  STOP"); self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("QPushButton:enabled{background:#f44336;color:white;font-weight:bold;padding:6px}")
        self.btn_fine = QPushButton("Fine scan around marker")
        self.fine_span = dspin(0.5, 500, 10, 1, 1, " ±Hz")
        self.fine_step = dspin(0.01, 10, 0.1, 2, 0.05, " Hz")
        self.progress = QProgressBar()
        bl.addWidget(self.btn_start, 0, 0); bl.addWidget(self.btn_stop, 0, 1)
        bl.addWidget(self.btn_fine, 1, 0, 1, 2)
        bl.addWidget(QLabel("span / step:"), 2, 0)
        h = QHBoxLayout(); h.addWidget(self.fine_span); h.addWidget(self.fine_step)
        bl.addLayout(h, 2, 1)
        bl.addWidget(self.progress, 3, 0, 1, 2)
        self.btn_start.clicked.connect(lambda: self.start_sweep(fine=False))
        self.btn_fine.clicked.connect(lambda: self.start_sweep(fine=True))
        self.btn_stop.clicked.connect(self.stop_sweep)
        left.addWidget(g)

        self.live_label = QLabel("—")
        self.live_label.setStyleSheet("QLabel{font-family:Consolas,monospace;background:#f4f4f4;padding:4px}")
        left.addWidget(self.live_label)
        left.addStretch()

        lw = QWidget(); lw.setLayout(left)
        sc = QScrollArea(); sc.setWidget(lw); sc.setWidgetResizable(True); sc.setMinimumWidth(340)
        lay.addWidget(sc)

        # -- centre: plot
        mid = QVBoxLayout()
        top = QHBoxLayout()
        self.q1 = QComboBox(); self.q2 = QComboBox()
        for k, (lbl, _) in QUANTITIES.items():
            self.q1.addItem(lbl, k); self.q2.addItem(lbl, k)
        self.q1.setCurrentIndex(0); self.q2.setCurrentIndex(3)
        self.sub_ref = QCheckBox("subtract reference (net)")
        self.ref_combo = QComboBox()
        top.addWidget(QLabel("Top:")); top.addWidget(self.q1)
        top.addWidget(QLabel("Bottom:")); top.addWidget(self.q2)
        top.addWidget(self.sub_ref); top.addWidget(self.ref_combo)
        self.lock_zoom = QCheckBox("lock zoom")
        self.lock_zoom.setToolTip("keep axis limits while new points arrive (set automatically when you zoom)")
        self.lock_zoom.toggled.connect(self.schedule_redraw)
        top.addWidget(self.lock_zoom)
        top.addStretch()
        mid.addLayout(top)
        self.canvas = SweepCanvas()
        self.toolbar = NavToolbar(self.canvas, self)
        self.canvas.toolbar = self.toolbar
        self.canvas.marker_set.connect(self.on_marker)
        self.canvas.mpl_connect("button_release_event",
                                lambda ev: self.toolbar.mode and self.lock_zoom.setChecked(True))
        mid.addWidget(self.toolbar)
        mid.addWidget(self.canvas, 1)
        for wd in (self.q1, self.q2, self.ref_combo):
            wd.currentIndexChanged.connect(self.schedule_redraw)
        self.sub_ref.toggled.connect(self.schedule_redraw)
        lay.addLayout(mid, 1)

        # -- right: runs and peaks
        right = QVBoxLayout()
        g = QGroupBox("Runs")
        v = QVBoxLayout(g)
        self.run_list = QListWidget()
        self.run_list.itemChanged.connect(self.schedule_redraw)
        v.addWidget(self.run_list)
        b1 = QPushButton("Delete selected"); b1.clicked.connect(self.delete_run)
        b2 = QPushButton("Save CSV..."); b2.clicked.connect(self.save_csv)
        b3 = QPushButton("Load CSV..."); b3.clicked.connect(self.load_csv)
        for b in (b1, b2, b3):
            v.addWidget(b)
        right.addWidget(g)

        g = QGroupBox("Resonance peaks (top curve)")
        v = QVBoxLayout(g)
        h = QHBoxLayout()
        self.peak_chk = QCheckBox("find peaks"); self.peak_chk.setChecked(find_peaks is not None)
        self.peak_chk.setEnabled(find_peaks is not None)
        self.peak_prom = dspin(0.001, 1, 0.03, 3, 0.01)
        self.peak_prom.setToolTip("minimum prominence as a fraction of the curve's total range")
        self.peak_dips = QCheckBox("dips")
        h.addWidget(self.peak_chk); h.addWidget(QLabel("prom.:")); h.addWidget(self.peak_prom); h.addWidget(self.peak_dips)
        v.addLayout(h)
        self.peak_list = QListWidget()
        self.peak_list.itemDoubleClicked.connect(self.peak_to_manual)
        v.addWidget(self.peak_list)
        v.addWidget(QLabel("<small>double-click → set this frequency in Manual tab</small>"))
        bp = QPushButton("Save peak list..."); bp.clicked.connect(self.save_peaks)
        v.addWidget(bp)
        self.peak_chk.toggled.connect(self.schedule_redraw)
        self.peak_prom.valueChanged.connect(self.schedule_redraw)
        self.peak_dips.toggled.connect(self.schedule_redraw)
        right.addWidget(g)
        rw = QWidget(); rw.setLayout(right); rw.setFixedWidth(260)
        lay.addWidget(rw)

        self.update_npts()
        return w

    def _build_manual_tab(self):
        w = QWidget()
        lay = QHBoxLayout(w)
        left = QVBoxLayout()

        g = QGroupBox("Generator")
        gl = QGridLayout(g)
        self.m_freq = dspin(1, 20000, 500, 2, 1, " Hz")
        self.m_freq.setMinimumHeight(34)
        bf = QFont(); bf.setPointSize(14); bf.setBold(True)
        self.m_freq.setFont(bf)
        gl.addWidget(QLabel("Frequency:"), 0, 0); gl.addWidget(self.m_freq, 0, 1, 1, 4)
        for i, d in enumerate((-10, -1, -0.1, +0.1, +1, +10)):
            b = QPushButton(f"{d:+g}")
            b.clicked.connect(lambda _, d=d: self.m_freq.setValue(self.m_freq.value() + d))
            gl.addWidget(b, 1 + i // 3, 1 + i % 3)
        self.m_vpp = dspin(0.02, 5, 0.5, 3, 0.05, " Vpp")
        gl.addWidget(QLabel("Amplitude:"), 3, 0); gl.addWidget(self.m_vpp, 3, 1, 1, 3)
        self.m_btn = QPushButton("OUTPUT ON  +  live measurement")
        self.m_btn.setCheckable(True)
        self.m_btn.setMinimumHeight(44)
        self.m_btn.setStyleSheet("QPushButton{background:#4CAF50;color:white;font-weight:bold}"
                                 "QPushButton:checked{background:#f44336}")
        self.m_btn.toggled.connect(self.toggle_live)
        gl.addWidget(self.m_btn, 4, 0, 1, 4)
        left.addWidget(g)

        g = QGroupBox("Live values")
        v = QVBoxLayout(g)
        self.m_display = QLabel("—")
        mf = QFont("Consolas"); mf.setPointSize(15)
        self.m_display.setFont(mf)
        self.m_display.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self.m_display)
        self.m_note = QComboBox(); self.m_note.setEditable(True)
        self.m_note.addItems(["pattern observed", "pattern unclear", "no pattern"])
        b = QPushButton("Log this point (with note)"); b.clicked.connect(self.log_manual_point)
        v.addWidget(self.m_note); v.addWidget(b)
        left.addWidget(g)
        left.addWidget(QLabel("<small>Generator settings are applied ~0.3 s after you stop "
                              "changing them. Circuit/safety settings are taken from the sweep tab.</small>"))
        left.addStretch()
        lay.addLayout(left)

        right = QVBoxLayout()
        self.m_fig = Figure(figsize=(7, 5), tight_layout=True)
        self.m_ax = self.m_fig.add_subplot(111)
        self.m_ax2 = self.m_ax.twinx()
        self.m_canvas = FigureCanvas(self.m_fig)
        right.addWidget(QLabel("Last acquisition: U (CH2) and I (CH1/R)"))
        right.addWidget(self.m_canvas, 1)
        self.m_log = QListWidget()
        self.m_log.setMaximumHeight(170)
        right.addWidget(QLabel("Logged points:"))
        right.addWidget(self.m_log)
        bs = QPushButton("Save logged points..."); bs.clicked.connect(self.save_manual_log)
        right.addWidget(bs)
        lay.addLayout(right, 1)

        self.manual_log = []
        self.last_live = None
        self._live_timer = QTimer(singleShot=True, interval=300, timeout=self.push_live_settings)
        self.m_freq.valueChanged.connect(lambda *_: self._live_timer.start())
        self.m_vpp.valueChanged.connect(lambda *_: self._live_timer.start())
        return w

    # ------------------------------------------------------------- config
    def config(self):
        return core.MeasConfig(
            r_shunt=self.r_shunt.value(), probe1=self.probe1.value(), probe2=self.probe2.value(),
            invert_current=self.invert_i.isChecked(), re_dc=self.re_dc.value(),
            wgen_vpp=min(self.wgen_vpp.value(), self.wgen_max.value()),
            wgen_max_vpp=self.wgen_max.value(),
            hold_vrms=self.hold_v.value() if self.hold_chk.isChecked() else 0.0,
            i_max_rms=self.i_max.value(), n_periods=self.n_periods.value(),
            averages=int(self.averages.currentText()), points=int(self.points.currentText()),
            settle_s=self.settle.value())

    def freq_list(self, fine=False):
        if fine:
            c = self.canvas.marker
            if c is None:
                return None
            span, st = self.fine_span.value(), self.fine_step.value()
            return core.frequency_list(max(1.0, c - span), c + span, step=st)
        a, b = self.f_start.value(), self.f_stop.value()
        if b <= a:
            return []
        if self.log_chk.isChecked():
            return core.frequency_list(a, b, n_points=self.n_log.value(), log=True)
        return core.frequency_list(a, b, step=self.f_step.value())

    def update_npts(self):
        fl = self.freq_list()
        n = len(fl) if fl else 0
        if n:
            av = int(self.averages.currentText())
            t = sum(self.settle.value() + 0.35 + av * self.n_periods.value() / f for f in fl)
            self.npts_label.setText(f"<b>{n}</b> points, ≈ {t / 60:.1f} min")
        else:
            self.npts_label.setText("<font color=red>invalid range</font>")

    # -------------------------------------------------------------- sweep
    def start_sweep(self, fine=False):
        freqs = self.freq_list(fine)
        if fine and freqs is None:
            QMessageBox.information(self, "Marker", "Click on the plot first to set a marker frequency.")
            return
        if not freqs:
            QMessageBox.warning(self, "Sweep", "Invalid frequency range.")
            return
        if self.live_thread is not None:
            return
        cfg = self.config()
        base = self.run_name.currentText().strip() or "run"
        if fine:
            base = f"{base} fine {self.canvas.marker:.1f}Hz"
        name, k = base, 2
        while name in self.runs:
            name = f"{base} #{k}"; k += 1
        self.current_run = name
        self.runs[name] = []
        meta = {k: v for k, v in core.asdict(cfg).items()}
        meta.update(started=datetime.now().isoformat(timespec="seconds"),
                    scope=self.scope.idn, f_first=freqs[0], f_last=freqs[-1], n=len(freqs))
        self.run_meta[name] = meta
        self._add_run_item(name)
        self.progress.setRange(0, len(freqs)); self.progress.setValue(0)

        self.sweep_thread = QThread()
        self.sweep_worker = SweepWorker(self.scope, freqs, cfg, not self.scope_configured)
        self.sweep_worker.moveToThread(self.sweep_thread)
        self.sweep_thread.started.connect(self.sweep_worker.run)
        self.sweep_worker.point.connect(self.on_point)
        self.sweep_worker.message.connect(self.on_message)
        self.sweep_worker.finished.connect(self.sweep_thread.quit, Qt.DirectConnection)
        self.sweep_worker.finished.connect(self.on_sweep_done)
        self.scope_configured = True
        self.set_running(True)
        self.sweep_thread.start()

    def stop_sweep(self):
        if self.sweep_worker:
            self.sweep_worker.stop()
            self.btn_stop.setEnabled(False)

    def set_running(self, running):
        self.btn_start.setEnabled(not running)
        self.btn_fine.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.tabs.setTabEnabled(1, not running)
        for wd in (self.r_shunt, self.probe1, self.probe2, self.invert_i):
            wd.setEnabled(not running)

    def on_point(self, r):
        self.runs[self.current_run].append(r)
        self.progress.setValue(r["index"] + 1)
        self.live_label.setText(
            f"f    = {r['f']:9.2f} Hz\n|Z|  = {r['Z']:9.3f} Ω   φ = {r['phase']:+6.1f}°\n"
            f"Urms = {r['Urms']:9.4f} V   Irms = {1e3 * r['Irms']:7.2f} mA\n"
            f"P    = {1e3 * r['P']:9.3f} mW  gen = {r['wgen_vpp']:.3f} Vpp\n"
            f"{('flags: ' + r['flags']) if r['flags'] else ''}")
        self.schedule_redraw()

    def on_message(self, m):
        self.statusBar().showMessage(m, 10000)
        if m.startswith("ERROR"):
            self.scope_configured = False
            QMessageBox.critical(self, "Error", m)

    def on_sweep_done(self, ok):
        if self.sweep_thread is None:
            return
        self.sweep_thread.quit(); self.sweep_thread.wait()
        try:
            self.scope.wgen_on(False)
        except Exception:
            pass
        self.sweep_thread = self.sweep_worker = None
        self.set_running(False)
        self.statusBar().showMessage(f"Sweep '{self.current_run}' "
                                     f"{'completed' if ok else 'stopped'} — generator OFF", 15000)
        self._autosave(self.current_run)
        self.redraw()

    def _autosave(self, name):
        """Each run is also written to ./data automatically (safety against crashes)."""
        if not self.runs.get(name):
            return
        os.makedirs("data", exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        fn = os.path.join("data", f"{datetime.now():%Y%m%d_%H%M%S}_{safe}.csv")
        self._write_csv(fn, [name])

    # ------------------------------------------------------------ runs UI
    def _add_run_item(self, name):
        from PyQt5.QtWidgets import QListWidgetItem
        it = QListWidgetItem(name)
        it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
        it.setCheckState(Qt.Checked)
        self.run_list.addItem(it)
        self._refresh_ref_combo()

    def _refresh_ref_combo(self):
        cur = self.ref_combo.currentText()
        self.ref_combo.blockSignals(True)
        self.ref_combo.clear()
        self.ref_combo.addItems(list(self.runs.keys()))
        idx = self.ref_combo.findText(cur)
        if idx < 0:
            idx = next((i for i, n in enumerate(self.runs) if "ref" in n.lower() or "no plate" in n.lower()), 0)
        self.ref_combo.setCurrentIndex(idx)
        self.ref_combo.blockSignals(False)

    def visible_runs(self):
        out = []
        for i in range(self.run_list.count()):
            it = self.run_list.item(i)
            if it.checkState() == Qt.Checked and it.text() in self.runs:
                out.append(it.text())
        return out

    def delete_run(self):
        for it in self.run_list.selectedItems():
            self.runs.pop(it.text(), None)
            self.run_list.takeItem(self.run_list.row(it))
        self._refresh_ref_combo()
        self.redraw()

    # --------------------------------------------------------------- plot
    def schedule_redraw(self, *_):
        self._redraw_timer.start()

    def series(self, name, key):
        rows = self.runs[name]
        f = np.array([r["f"] for r in rows], float)
        y = np.array([r.get(key, np.nan) for r in rows], float) * QUANTITIES[key][1]
        o = np.argsort(f)
        f, y = f[o], y[o]
        if self.sub_ref.isChecked() and key != "wgen_vpp":
            ref = self.ref_combo.currentText()
            if ref and ref != name and self.runs.get(ref):
                fr = np.array([r["f"] for r in self.runs[ref]], float)
                yr = np.array([r.get(key, np.nan) for r in self.runs[ref]], float) * QUANTITIES[key][1]
                o = np.argsort(fr)
                y = y - np.interp(f, fr[o], yr[o], left=np.nan, right=np.nan)
        return f, y

    def redraw(self):
        c = self.canvas
        k1, k2 = self.q1.currentData(), self.q2.currentData()
        keep = self.lock_zoom.isChecked() and bool(c.ax1.lines)
        lims = (c.ax1.get_xlim(), c.ax1.get_ylim(), c.ax2.get_ylim())
        c.ax1.clear(); c.ax2.clear()
        names = self.visible_runs()
        ref = self.ref_combo.currentText() if self.sub_ref.isChecked() else None
        self.peaks = []
        for i, n in enumerate(self.runs):
            if n not in names or not self.runs[n]:
                continue
            if ref and n == ref:
                continue
            col = COLORS[i % len(COLORS)]
            f, y1 = self.series(n, k1)
            _, y2 = self.series(n, k2)
            ms = 2 if len(f) > 300 else 3
            c.ax1.plot(f, y1, "-", color=col, lw=0.9, marker=".", ms=ms, label=n)
            c.ax2.plot(f, y2, "-", color=col, lw=0.9, marker=".", ms=ms)
            if self.peak_chk.isChecked() and find_peaks is not None and len(f) > 5:
                good = np.isfinite(y1)
                ff, yy = f[good], y1[good]
                if len(yy) > 5:
                    sig = -yy if self.peak_dips.isChecked() else yy
                    rng = np.nanmax(yy) - np.nanmin(yy)
                    pk, _ = find_peaks(sig, prominence=self.peak_prom.value() * rng if rng > 0 else None)
                    c.ax1.plot(ff[pk], yy[pk], "v" if not self.peak_dips.isChecked() else "^",
                               color=col, ms=7, mfc="none")
                    self.peaks += [(n, ff[p], yy[p]) for p in pk]
        if c.marker is not None:
            for ax in (c.ax1, c.ax2):
                ax.axvline(c.marker, color="k", ls="--", lw=0.8)
        suffix = "  [net = run − reference]" if ref else ""
        c.ax1.set_ylabel(QUANTITIES[k1][0] + suffix)
        c.ax2.set_ylabel(QUANTITIES[k2][0] + suffix)
        c.ax2.set_xlabel("frequency (Hz)")
        for ax in (c.ax1, c.ax2):
            ax.grid(True, alpha=0.3)
        if c.ax1.lines:
            c.ax1.legend(loc="best", fontsize=8)
        if keep:
            c.ax1.set_xlim(lims[0]); c.ax1.set_ylim(lims[1]); c.ax2.set_ylim(lims[2])
        c.draw_idle()
        # peak list
        self.peak_list.clear()
        unit = QUANTITIES[k1][0].split("(")[-1].rstrip(")") if "(" in QUANTITIES[k1][0] else ""
        for n, f, y in sorted(self.peaks, key=lambda p: (p[0], p[1])):
            self.peak_list.addItem(f"{f:9.2f} Hz   {y:9.3g} {unit}   [{n}]")

    def on_marker(self, f):
        self.statusBar().showMessage(f"Marker: {f:.2f} Hz  (Fine scan uses it; manual tab frequency set)", 8000)
        self.m_freq.setValue(round(f, 2))
        self.redraw()

    def peak_to_manual(self, item):
        f = float(item.text().split("Hz")[0])
        self.canvas.marker = f
        self.m_freq.setValue(f)
        self.tabs.setCurrentIndex(1)
        self.redraw()

    # ------------------------------------------------------------- manual
    def toggle_live(self, on):
        if on:
            if self.sweep_thread is not None:
                self.m_btn.setChecked(False)
                return
            cfg = self.config()
            try:
                if not self.scope_configured:
                    self.statusBar().showMessage("Configuring oscilloscope...")
                    QApplication.processEvents()
                    self.scope.setup(cfg)
                    self.scope_configured = True
                self.scope.set_frequency(self.m_freq.value(), cfg.n_periods)
                self.scope.set_amplitude(min(self.m_vpp.value(), cfg.wgen_max_vpp))
                self.scope.wgen_on(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
                self.m_btn.setChecked(False)
                return
            self.m_btn.setText("OUTPUT OFF")
            self.live_thread = QThread()
            self.live_worker = LiveWorker(self.scope, cfg)
            self.live_worker.update(self.m_freq.value(), min(self.m_vpp.value(), cfg.wgen_max_vpp), cfg)
            self.live_worker.moveToThread(self.live_thread)
            self.live_thread.started.connect(self.live_worker.run)
            self.live_worker.point.connect(self.on_live_point)
            self.live_worker.message.connect(self.on_message)
            self.live_worker.finished.connect(self.live_thread.quit, Qt.DirectConnection)
            self.live_worker.finished.connect(self.on_live_done)
            self.tabs.setTabEnabled(0, False)
            self.live_thread.start()
        else:
            if self.live_worker:
                self.live_worker.stop()
            self.m_btn.setText("OUTPUT ON  +  live measurement")

    def push_live_settings(self):
        if self.live_worker:
            cfg = self.config()
            self.live_worker.update(self.m_freq.value(), min(self.m_vpp.value(), cfg.wgen_max_vpp), cfg)

    def on_live_done(self):
        if self.live_thread is None:
            return
        self.live_thread.quit(); self.live_thread.wait()
        try:
            self.scope.wgen_on(False)
        except Exception:
            pass
        self.live_thread = self.live_worker = None
        self.tabs.setTabEnabled(0, True)
        if self.m_btn.isChecked():
            self.m_btn.setChecked(False)
        self.statusBar().showMessage("Generator OFF", 5000)

    def on_live_point(self, r):
        self.last_live = r
        self.m_display.setText(
            f"f     {r['f']:10.2f} Hz\n"
            f"Urms  {r['Urms']:10.4f} V\n"
            f"Irms  {1e3 * r['Irms']:10.3f} mA\n"
            f"|Z|   {r['Z']:10.3f} Ω\n"
            f"φ     {r['phase']:+10.2f} °\n"
            f"P     {1e3 * r['P']:10.3f} mW\n"
            + (f"P-I²R {1e3 * r['P_mech']:10.3f} mW\n" if np.isfinite(r['P_mech']) else "")
            + f"gen   {r['wgen_vpp']:10.3f} Vpp\n"
            + (f"<{r['flags']}>" if r['flags'] else ""))
        # draw the waveforms re-synthesised from the fit (cheap and noise-free)
        f = r["f"]
        t = np.linspace(0, 3 / f, 400)
        ax, ax2 = self.m_ax, self.m_ax2
        ax.clear(); ax2.clear()
        ph = math.radians(r["phase"])
        ax.plot(t * 1e3, math.sqrt(2) * r["Urms"] * np.cos(2 * math.pi * f * t + ph), color="#d62728", label="U driver")
        ax2.plot(t * 1e3, 1e3 * math.sqrt(2) * r["Irms"] * np.cos(2 * math.pi * f * t), color="#1f77b4", label="I")
        ax.set_xlabel("t (ms)"); ax.set_ylabel("U (V)", color="#d62728"); ax2.set_ylabel("I (mA)", color="#1f77b4")
        ax2.yaxis.set_label_position("right"); ax2.yaxis.tick_right()
        ax.grid(True, alpha=0.3)
        ax.set_title(f"φ = {r['phase']:+.1f}°   (fitted sinusoids, 3 periods)")
        self.m_canvas.draw_idle()

    def log_manual_point(self):
        if not self.last_live:
            return
        r = dict(self.last_live)
        r["note"] = self.m_note.currentText()
        self.manual_log.append(r)
        self.m_log.addItem(f"{r['f']:9.2f} Hz  |Z|={r['Z']:.3f} Ω  φ={r['phase']:+.1f}°  "
                           f"P={1e3 * r['P']:.2f} mW   — {r['note']}")

    def save_manual_log(self):
        if not self.manual_log:
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Save logged points", "chladni_manual.csv", "CSV (*.csv)")
        if not fn:
            return
        fields = core.RESULT_FIELDS + ["note"]
        with open(fn, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in self.manual_log:
                w.writerow(r)

    # ----------------------------------------------------------------- IO
    def _write_csv(self, fn, names):
        with open(fn, "w", newline="") as fh:
            for n in names:
                fh.write(f"# run: {n} | " + json.dumps(self.run_meta.get(n, {})) + "\n")
            w = csv.DictWriter(fh, fieldnames=["run"] + core.RESULT_FIELDS, extrasaction="ignore")
            w.writeheader()
            for n in names:
                for r in self.runs[n]:
                    w.writerow(dict(r, run=n))

    def save_csv(self):
        names = self.visible_runs() or list(self.runs)
        if not names:
            QMessageBox.warning(self, "Save", "No data.")
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Save runs", f"chladni_{datetime.now():%Y%m%d_%H%M}.csv",
                                            "CSV (*.csv)")
        if fn:
            self._write_csv(fn, names)
            self.statusBar().showMessage(f"Saved {len(names)} run(s) to {fn}", 8000)

    def load_csv(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Load runs", "", "CSV (*.csv)")
        if not fn:
            return
        with open(fn, newline="") as fh:
            lines = fh.readlines()
        for ln in lines:
            if ln.startswith("# run:"):
                try:
                    n, meta = ln[6:].split(" | ", 1)
                    self.run_meta[n.strip()] = json.loads(meta)
                except Exception:
                    pass
        rd = csv.DictReader([ln for ln in lines if not ln.startswith("#")])
        loaded = OrderedDict()
        for row in rd:
            r = {}
            for k, v in row.items():
                if k in ("run", "flags"):
                    r[k] = v
                else:
                    try:
                        r[k] = float(v)
                    except (TypeError, ValueError):
                        r[k] = float("nan")
            loaded.setdefault(row.get("run") or os.path.basename(fn), []).append(r)
        for n, rows in loaded.items():
            name, k = n, 2
            while name in self.runs:
                name = f"{n} (loaded {k})"; k += 1
            self.runs[name] = rows
            self._add_run_item(name)
        self.redraw()

    def save_peaks(self):
        if not self.peaks:
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Save peaks", "chladni_peaks.csv", "CSV (*.csv)")
        if fn:
            with open(fn, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["run", "f_Hz", self.q1.currentData() + (" (net)" if self.sub_ref.isChecked() else "")])
                for n, f, y in sorted(self.peaks, key=lambda p: (p[0], p[1])):
                    w.writerow([n, f"{f:.3f}", f"{y:.6g}"])

    # -------------------------------------------------------------- close
    def closeEvent(self, ev):
        for wk, th in ((self.sweep_worker, self.sweep_thread), (self.live_worker, self.live_thread)):
            if wk is not None:
                wk.stop()
                th.wait(15000)
        self.sweep_worker = self.sweep_thread = self.live_worker = self.live_thread = None
        if any(self.runs.values()) and QMessageBox.question(
                self, "Exit", "Save all runs before exit?") == QMessageBox.Yes:
            self.save_csv()
        self.scope.close()
        ev.accept()


def main():
    app = QApplication(sys.argv)
    dlg = ConnectDialog()
    if dlg.exec_() != QDialog.Accepted or dlg.scope is None:
        sys.exit(0)
    win = MainWindow(dlg.scope)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

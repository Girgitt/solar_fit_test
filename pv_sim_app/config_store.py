from __future__ import annotations

import copy
import json
import os
import threading
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Any] = {
    "mqtt": {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 1883,
        "username": "",
        "password": "",
        "keepalive_sec": 30,
        "measurement_topic": "pv/sim/measurement",
        "measurement_channel_key": "channel_1",
        "duty_topic": "pv/sim/set_duty",
        "publish_simulated_measurements": True,
        "subscribe_measurement_topic": False,
        "publish_interval_sec": 0.5,
        "qos": 0,
        "retain_measurements": False,
        "client_id": "pv-mppt-simulator",
    },
    "irradiance": {
        "min": 100.0,
        "max": 1200.0,
        "initial": 800.0,
        "reference": 1000.0,
        "slider_step": 10.0,
        "voc_log_coeff": 1.25,
    },
    "pv": {
        "isc_ref": 5.20,
        "voc_ref": 22.00,
        "imp_ref": 4.80,
        "vmp_ref": 18.00,
        "curve_samples": 180,
    },
    "load": {
        "lamp_rated_voltage": 12.0,
        "lamp_rated_power": 20.0,
        "hot_to_cold_ratio": 10.0,
        "thermal_tau_sec": 2.5,
        "thermal_power_exponent": 0.65,
    },
    "pwm": {
        "duty_min": 0.0,
        "duty_max": 1.0,
        "duty_initial": 0.25,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class ConfigStore:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.RLock()
        self._cfg = copy.deepcopy(DEFAULT_CONFIG)
        self._load_or_init()

    def _load_or_init(self) -> None:
        with self._lock:
            if os.path.exists(self.path):
                try:
                    with open(self.path, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                    self._cfg = _deep_merge(DEFAULT_CONFIG, raw)
                    return
                except Exception:
                    # fall back to defaults and rewrite a clean file
                    self._cfg = copy.deepcopy(DEFAULT_CONFIG)
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            self._write_unlocked(self._cfg)

    def _write_unlocked(self, cfg: Dict[str, Any]) -> None:
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)

    def get(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._cfg)

    def save(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            merged = _deep_merge(DEFAULT_CONFIG, cfg)
            self._cfg = merged
            self._write_unlocked(self._cfg)
            return copy.deepcopy(self._cfg)

    def update(self, partial_cfg: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self._cfg = _deep_merge(self._cfg, partial_cfg)
            self._write_unlocked(self._cfg)
            return copy.deepcopy(self._cfg)

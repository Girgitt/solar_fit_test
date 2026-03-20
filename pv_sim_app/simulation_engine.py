from __future__ import annotations

import copy
import threading
import time
from typing import Any, Dict, Optional

from models import clamp, lamp_resistance, normalize_duty, solve_operating_point, step_lamp_temperature


class SimulationEngine:
    def __init__(self, config_store):
        self.config_store = config_store
        cfg = self.config_store.get()

        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._state: Dict[str, Any] = {
            "irradiance": float(cfg["irradiance"]["initial"]),
            "mqtt_duty": float(cfg["pwm"]["duty_initial"]),
            "manual_override": False,
            "manual_duty": float(cfg["pwm"]["duty_initial"]),
            "lamp_temp_state": 0.0,
            "effective_duty": float(cfg["pwm"]["duty_initial"]),
            "effective_duty_norm": normalize_duty(cfg, float(cfg["pwm"]["duty_initial"])),
            "duty_source": "mqtt",
            "voltage": 0.0,
            "current": 0.0,
            "power": 0.0,
            "resistance_ohm": 0.0,
            "voc": 0.0,
            "isc": 0.0,
            "shape_a": 0.0,
            "shape_b": 0.0,
            "last_step_monotonic": time.monotonic(),
            "mqtt_connected": False,
            "mqtt_status": "starting",
            "last_mqtt_message_monotonic": None,
            "last_external_measurement": None,
            "loop_hz": 10.0,
        }

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, name="pv-sim-engine", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            thread = self._thread
        if thread:
            thread.join(timeout=2.0)

    def _run_loop(self) -> None:
        prev = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            dt = max(now - prev, 1e-3)
            prev = now
            cfg = self.config_store.get()

            with self._lock:
                irr = clamp(
                    float(self._state["irradiance"]),
                    float(cfg["irradiance"]["min"]),
                    float(cfg["irradiance"]["max"]),
                )
                self._state["irradiance"] = irr

                manual_override = bool(self._state["manual_override"])
                if manual_override:
                    duty = float(self._state["manual_duty"])
                    duty_source = "manual"
                else:
                    duty = float(self._state["mqtt_duty"])
                    duty_source = "mqtt"

                duty = clamp(duty, float(cfg["pwm"]["duty_min"]), float(cfg["pwm"]["duty_max"]))

                operating = solve_operating_point(
                    cfg=cfg,
                    irradiance=irr,
                    duty=duty,
                    lamp_temp_state=float(self._state["lamp_temp_state"]),
                )
                next_lamp_state = step_lamp_temperature(
                    cfg["load"],
                    lamp_temp_state=float(self._state["lamp_temp_state"]),
                    power_w=float(operating["power"]),
                    dt_s=dt,
                )
                next_resistance = lamp_resistance(cfg["load"], next_lamp_state)

                self._state.update(
                    {
                        "effective_duty": duty,
                        "effective_duty_norm": float(operating["duty_norm"]),
                        "duty_source": duty_source,
                        "voltage": float(operating["voltage"]),
                        "current": float(operating["current"]),
                        "power": float(operating["power"]),
                        "resistance_ohm": float(next_resistance),
                        "voc": float(operating["voc"]),
                        "isc": float(operating["isc"]),
                        "shape_a": float(operating["shape_a"]),
                        "shape_b": float(operating["shape_b"]),
                        "overload_ratio": float(operating["overload_ratio"]),
                        "lamp_temp_state": float(next_lamp_state),
                        "last_step_monotonic": now,
                    }
                )

            target_dt = 0.1
            sleep_time = max(0.02, target_dt - (time.monotonic() - now))
            time.sleep(sleep_time)

    def set_irradiance(self, irradiance: float) -> None:
        with self._lock:
            self._state["irradiance"] = float(irradiance)

    def set_mqtt_duty(self, duty: float) -> None:
        with self._lock:
            self._state["mqtt_duty"] = float(duty)
            self._state["last_mqtt_message_monotonic"] = time.monotonic()

    def set_manual_override(self, enabled: bool) -> None:
        with self._lock:
            self._state["manual_override"] = bool(enabled)

    def set_manual_duty(self, duty: float) -> None:
        with self._lock:
            self._state["manual_duty"] = float(duty)

    def reset_lamp(self) -> None:
        with self._lock:
            self._state["lamp_temp_state"] = 0.0

    def update_external_measurement(self, v: float, i: float, age_ms: Optional[float], raw_payload: str) -> None:
        with self._lock:
            self._state["last_external_measurement"] = {
                "v": float(v),
                "i": float(i),
                "p": float(v) * float(i),
                "age_ms": None if age_ms is None else float(age_ms),
                "raw_payload": raw_payload,
                "received_monotonic": time.monotonic(),
            }

    def set_mqtt_status(self, connected: bool, status: str) -> None:
        with self._lock:
            self._state["mqtt_connected"] = bool(connected)
            self._state["mqtt_status"] = str(status)

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            snap = copy.deepcopy(self._state)
        snap["state_age_ms"] = int((time.monotonic() - snap["last_step_monotonic"]) * 1000.0)
        #snap["state_age_ms"] = int((time.monotonic()) * 1000.0)
        if snap.get("last_mqtt_message_monotonic") is not None:
            snap["duty_age_ms"] = int((time.monotonic() - snap["last_mqtt_message_monotonic"]) * 1000.0)
        else:
            snap["duty_age_ms"] = None
        if snap.get("last_external_measurement") and snap["last_external_measurement"].get("received_monotonic") is not None:
            snap["external_age_local_ms"] = int(
                (time.monotonic() - snap["last_external_measurement"]["received_monotonic"]) * 1000.0
            )
        else:
            snap["external_age_local_ms"] = None
        return snap

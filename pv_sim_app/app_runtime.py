from __future__ import annotations

import os
from pathlib import Path

from config_store import ConfigStore
from mqtt_service import MQTTService
from simulation_engine import SimulationEngine


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = os.getenv("PV_SIM_CONFIG_PATH", str(BASE_DIR / "pv_sim_config.json"))

config_store = ConfigStore(CONFIG_PATH)
engine = SimulationEngine(config_store)
mqtt_service = MQTTService(config_store, engine)

_started = False


def ensure_started() -> None:
    global _started
    if _started:
        return
    engine.start()
    mqtt_service.start()
    _started = True

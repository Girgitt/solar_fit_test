from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, Optional

try:
    import paho.mqtt.client as mqtt
except Exception:  # pragma: no cover - handled at runtime for environments without dependency
    mqtt = None


class MQTTService:
    def __init__(self, config_store, engine):
        self.config_store = config_store
        self.engine = engine

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._cfg_sig: Optional[str] = None
        self._client = None
        self._current_cfg: Optional[Dict[str, Any]] = None
        self._connected = False
        self._last_publish_monotonic = 0.0
        self._lock = threading.RLock()

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, name="pv-sim-mqtt", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._shutdown_client()
        with self._lock:
            thread = self._thread
        if thread:
            thread.join(timeout=2.0)

    def _run_loop(self) -> None:
        if mqtt is None:
            self.engine.set_mqtt_status(False, "paho-mqtt not installed")
            return

        while not self._stop_event.is_set():
            cfg = self.config_store.get().get("mqtt", {})
            cfg_sig = json.dumps(cfg, sort_keys=True, ensure_ascii=False)

            if cfg_sig != self._cfg_sig:
                self._apply_config(cfg, cfg_sig)

            self._publish_if_due()
            time.sleep(0.2)

    def _apply_config(self, cfg: Dict[str, Any], cfg_sig: str) -> None:
        self._shutdown_client()
        self._cfg_sig = cfg_sig
        self._current_cfg = cfg
        self._last_publish_monotonic = 0.0

        if not cfg.get("enabled", True):
            self.engine.set_mqtt_status(False, "MQTT disabled in configuration")
            return

        client_id = cfg.get("client_id") or f"pv-mppt-simulator-{int(time.time())}"
        client = mqtt.Client(client_id=client_id, clean_session=True)
        if cfg.get("username"):
            client.username_pw_set(cfg.get("username"), cfg.get("password") or None)

        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message

        try:
            client.connect_async(str(cfg.get("host", "127.0.0.1")), int(cfg.get("port", 1883)), int(cfg.get("keepalive_sec", 30)))
            client.loop_start()
            self._client = client
            self.engine.set_mqtt_status(False, f"connecting to {cfg.get('host')}:{cfg.get('port')}")
        except Exception as exc:
            self.engine.set_mqtt_status(False, f"MQTT connect error: {exc}")
            self._client = None

    def _shutdown_client(self) -> None:
        client = self._client
        self._client = None
        self._connected = False
        if client is not None:
            try:
                client.loop_stop()
            except Exception:
                pass
            try:
                client.disconnect()
            except Exception:
                pass

    def _on_connect(self, client, userdata, flags, rc):  # pragma: no cover - callback driven
        cfg = self._current_cfg or {}
        if rc == 0:
            self._connected = True
            duty_topic = (cfg.get("duty_topic") or "").strip()
            meas_topic = (cfg.get("measurement_topic") or "").strip()
            qos = int(cfg.get("qos", 0))

            if duty_topic:
                client.subscribe(duty_topic, qos=qos)
            if cfg.get("subscribe_measurement_topic") and meas_topic:
                client.subscribe(meas_topic, qos=qos)

            self.engine.set_mqtt_status(True, f"connected; subscribed to duty={duty_topic or '-'}")
        else:
            self._connected = False
            self.engine.set_mqtt_status(False, f"MQTT connect failed rc={rc}")

    def _on_disconnect(self, client, userdata, rc):  # pragma: no cover - callback driven
        self._connected = False
        if self._stop_event.is_set():
            self.engine.set_mqtt_status(False, "stopped")
        else:
            self.engine.set_mqtt_status(False, f"disconnected rc={rc}")

    def _on_message(self, client, userdata, msg):  # pragma: no cover - callback driven
        cfg = self._current_cfg or {}
        topic = msg.topic or ""
        payload = msg.payload.decode("utf-8", errors="replace").strip()

        try:
            duty_topic = (cfg.get("duty_topic") or "").strip()
            meas_topic = (cfg.get("measurement_topic") or "").strip()

            if topic == duty_topic:
                duty = self._parse_duty_payload(payload)
                if duty is not None:
                    self.engine.set_mqtt_duty(duty)
                return

            if topic == meas_topic:
                parsed = self._parse_measurement_payload(payload, str(cfg.get("measurement_channel_key") or "channel_1"))
                if parsed is not None:
                    self.engine.update_external_measurement(
                        v=parsed["v"],
                        i=parsed["i"],
                        age_ms=parsed.get("age_ms"),
                        raw_payload=payload,
                    )
        except Exception as exc:
            self.engine.set_mqtt_status(self._connected, f"MQTT message parse error: {exc}")

    def _parse_duty_payload(self, payload: str) -> Optional[float]:
        try:
            return float(payload)
        except Exception:
            pass

        try:
            obj = json.loads(payload)
        except Exception:
            return None

        for key in ("set_duty", "duty", "value"):
            if key in obj:
                try:
                    return float(obj[key])
                except Exception:
                    return None
        return None

    def _parse_measurement_payload(self, payload: str, channel_key: str) -> Optional[Dict[str, Any]]:
        obj = json.loads(payload)
        channel = obj.get(channel_key)
        if not isinstance(channel, dict):
            return None
        v = channel.get("v")
        i = channel.get("i")
        if v is None or i is None:
            return None
        return {
            "v": float(v),
            "i": float(i),
            "age_ms": obj.get("age_ms"),
        }

    def _publish_if_due(self) -> None:
        client = self._client
        cfg = self._current_cfg or {}
        if client is None or not self._connected:
            return
        if not cfg.get("publish_simulated_measurements", True):
            return

        topic = (cfg.get("measurement_topic") or "").strip()
        if not topic:
            return

        interval_sec = max(float(cfg.get("publish_interval_sec", 0.5)), 0.05)
        now = time.monotonic()
        if now - self._last_publish_monotonic < interval_sec:
            return

        channel_key = str(cfg.get("measurement_channel_key") or "channel_1")
        snap = self.engine.get_snapshot()
        payload = {
            "age_ms": int(snap.get("state_age_ms") or 0),
            channel_key: {
                "v": round(float(snap.get("voltage", 0.0)), 5),
                "i": round(float(snap.get("current", 0.0)), 5),
            },
            "meta": {
                "p": round(float(snap.get("power", 0.0)), 5),
                "irradiance": round(float(snap.get("irradiance", 0.0)), 3),
                "duty": round(float(snap.get("effective_duty", 0.0)), 6),
                "source": "pv_mqtt_sim",
            },
        }

        try:
            client.publish(
                topic,
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                qos=int(cfg.get("qos", 0)),
                retain=bool(cfg.get("retain_measurements", False)),
            )
            self._last_publish_monotonic = now
        except Exception as exc:
            self.engine.set_mqtt_status(self._connected, f"MQTT publish error: {exc}")

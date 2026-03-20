from __future__ import annotations

import math
from typing import Dict, List, Tuple


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def solve_shape_params(isc_ref: float, voc_ref: float, imp_ref: float, vmp_ref: float) -> Tuple[float, float]:
    """
    Finds parameters a, b for the family:
        I(V) = Isc * (1 - (V/Voc)^a)^b

    such that:
      * the curve passes through the configured MPP point (Vmp, Imp)
      * the configured MPP is indeed the power maximum of the curve

    This gives a smooth one-diode-like curve without requiring scipy.
    """
    if isc_ref <= 0 or voc_ref <= 0:
        raise ValueError("Isc and Voc must be > 0")
    if not (0 < imp_ref < isc_ref):
        raise ValueError("Imp must be between 0 and Isc")
    if not (0 < vmp_ref < voc_ref):
        raise ValueError("Vmp must be between 0 and Voc")

    x_m = vmp_ref / voc_ref
    y_m = imp_ref / isc_ref

    def residual(a: float) -> float:
        x_a = x_m ** a
        one_minus = 1.0 - x_a
        if x_a <= 0.0 or one_minus <= 0.0:
            return float("nan")
        b = one_minus / (a * x_a)
        return b * math.log(one_minus) - math.log(y_m)

    samples: List[Tuple[float, float]] = []
    for n in range(-80, 81):
        a = 10.0 ** (n / 20.0)  # 1e-4 .. 1e4
        r = residual(a)
        if math.isfinite(r):
            samples.append((a, r))

    lo = hi = None
    r_lo = r_hi = None
    prev = None
    for item in samples:
        if prev is not None:
            a0, r0 = prev
            a1, r1 = item
            if r0 == 0.0:
                lo = hi = a0
                r_lo = r_hi = r0
                break
            if r0 * r1 < 0.0:
                lo, hi = a0, a1
                r_lo, r_hi = r0, r1
                break
        prev = item

    if lo is None or hi is None:
        raise ValueError("Could not derive IV shape parameters from reference point")

    if lo == hi:
        a = lo
    else:
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            r_mid = residual(mid)
            if not math.isfinite(r_mid):
                break
            if abs(r_mid) < 1e-12:
                lo = hi = mid
                break
            if r_lo * r_mid <= 0.0:
                hi = mid
                r_hi = r_mid
            else:
                lo = mid
                r_lo = r_mid
        a = 0.5 * (lo + hi)

    x_a = x_m ** a
    b = (1.0 - x_a) / (a * x_a)
    return a, b


def pv_params_for_irradiance(cfg: Dict, irradiance: float) -> Dict[str, float]:
    irr_cfg = cfg["irradiance"]
    pv_cfg = cfg["pv"]

    g_ref = max(irr_cfg["reference"], 1e-9)
    g_eff = max(irradiance, 1e-6)
    g_ratio = g_eff / g_ref

    a, b = solve_shape_params(
        isc_ref=float(pv_cfg["isc_ref"]),
        voc_ref=float(pv_cfg["voc_ref"]),
        imp_ref=float(pv_cfg["imp_ref"]),
        vmp_ref=float(pv_cfg["vmp_ref"]),
    )

    isc = float(pv_cfg["isc_ref"]) * g_ratio
    voc = max(0.25, float(pv_cfg["voc_ref"]) + float(irr_cfg["voc_log_coeff"]) * math.log(g_ratio))

    return {
        "g": g_eff,
        "g_ratio": g_ratio,
        "isc": isc,
        "voc": voc,
        "shape_a": a,
        "shape_b": b,
    }


def pv_current_from_voltage(voltage: float, pv_params: Dict[str, float]) -> float:
    voc = max(pv_params["voc"], 1e-9)
    v = clamp(voltage, 0.0, voc)
    x = clamp(v / voc, 0.0, 1.0)
    return pv_params["isc"] * ((1.0 - (x ** pv_params["shape_a"])) ** pv_params["shape_b"])


def resistance_hot_and_cold(load_cfg: Dict) -> Tuple[float, float]:
    rated_voltage = max(float(load_cfg["lamp_rated_voltage"]), 1e-9)
    rated_power = max(float(load_cfg["lamp_rated_power"]), 1e-9)
    hot_to_cold_ratio = max(float(load_cfg["hot_to_cold_ratio"]), 1.01)

    r_hot = (rated_voltage ** 2) / rated_power
    r_cold = r_hot / hot_to_cold_ratio
    return r_hot, r_cold


def lamp_resistance(load_cfg: Dict, lamp_temp_state: float) -> float:
    r_hot, r_cold = resistance_hot_and_cold(load_cfg)
    theta = max(lamp_temp_state, 0.0)
    return r_cold + (r_hot - r_cold) * theta


def lamp_power_target_state(load_cfg: Dict, power_w: float) -> float:
    rated_power = max(float(load_cfg["lamp_rated_power"]), 1e-9)
    exponent = max(float(load_cfg["thermal_power_exponent"]), 0.05)
    normalized = max(power_w, 0.0) / rated_power
    return normalized ** exponent


def normalize_duty(cfg: Dict, duty: float) -> float:
    pwm_cfg = cfg["pwm"]
    duty_min = float(pwm_cfg["duty_min"])
    duty_max = float(pwm_cfg["duty_max"])
    duty_clamped = clamp(float(duty), duty_min, duty_max)
    span = duty_max - duty_min
    if span <= 1e-12:
        return 0.0
    return clamp((duty_clamped - duty_min) / span, 0.0, 1.0)


def step_lamp_temperature(load_cfg: Dict, lamp_temp_state: float, power_w: float, dt_s: float) -> float:
    tau = max(float(load_cfg["thermal_tau_sec"]), 1e-3)
    target = lamp_power_target_state(load_cfg, power_w)
    alpha = clamp(dt_s / tau, 0.0, 1.0)
    next_state = lamp_temp_state + alpha * (target - lamp_temp_state)
    return max(next_state, 0.0)


def load_current_for_voltage(voltage: float, duty_norm: float, resistance_ohm: float, load_cfg: Dict) -> float:
    voltage = max(float(voltage), 0.0)
    duty_norm = clamp(duty_norm, 0.0, 1.0)
    if duty_norm <= 0.0 or voltage <= 0.0:
        return 0.0

    return duty_norm * voltage / max(resistance_ohm, 1e-9)


def solve_operating_point(cfg: Dict, irradiance: float, duty: float, lamp_temp_state: float) -> Dict[str, float]:
    load_cfg = cfg["load"]
    resistance_ohm = lamp_resistance(load_cfg, lamp_temp_state)
    pv_params = pv_params_for_irradiance(cfg, irradiance)
    duty = clamp(duty, float(cfg["pwm"]["duty_min"]), float(cfg["pwm"]["duty_max"]))
    duty_norm = normalize_duty(cfg, duty)

    voc = pv_params["voc"]
    isc = pv_params["isc"]

    if duty_norm <= 0.0:
        voltage = voc
        current = 0.0
    else:
        def balance(v: float) -> float:
            return pv_current_from_voltage(v, pv_params) - load_current_for_voltage(v, duty_norm, resistance_ohm, load_cfg)

        lo = 0.0
        hi = voc
        f_lo = balance(lo)
        f_hi = balance(hi)

        if f_lo <= 0.0:
            voltage = 0.0
        elif f_hi >= 0.0:
            voltage = hi
        else:
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                f_mid = balance(mid)
                if abs(f_mid) < 1e-9:
                    lo = hi = mid
                    break
                if f_lo * f_mid > 0.0:
                    lo = mid
                    f_lo = f_mid
                else:
                    hi = mid
                    f_hi = f_mid
            voltage = 0.5 * (lo + hi)

        current = load_current_for_voltage(voltage, duty_norm, resistance_ohm, load_cfg)

    power = voltage * current
    rated_power = max(float(load_cfg["lamp_rated_power"]), 1e-9)
    overload_ratio = power / rated_power

    return {
        "voltage": voltage,
        "current": current,
        "power": power,
        "resistance_ohm": resistance_ohm,
        "voc": voc,
        "isc": isc,
        "shape_a": pv_params["shape_a"],
        "shape_b": pv_params["shape_b"],
        "duty_norm": duty_norm,
        "overload_ratio": overload_ratio,
    }


def build_iv_curve(cfg: Dict, irradiance: float, samples: int | None = None) -> Dict[str, List[float]]:
    pv_params = pv_params_for_irradiance(cfg, irradiance)
    count = int(samples or cfg["pv"].get("curve_samples", 180))
    count = max(count, 40)
    voc = pv_params["voc"]

    points = []
    for idx in range(count + 1):
        voltage = voc * idx / count
        current = pv_current_from_voltage(voltage, pv_params)
        points.append((current, voltage, current * voltage))

    points.sort(key=lambda item: item[0])
    currents = [p[0] for p in points]
    voltages = [p[1] for p in points]
    powers = [p[2] for p in points]
    return {
        "currents": currents,
        "voltages": voltages,
        "powers": powers,
        "voc": pv_params["voc"],
        "isc": pv_params["isc"],
    }

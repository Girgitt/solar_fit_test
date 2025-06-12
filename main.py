# ========================================================================== #
#  Polynomial regression demo: align "power_hi" sensor to "power_reference"
#  Adds sun-elevation & time-of-day features and produces 4 diagnostic plots
# ========================================================================== #

import math
import datetime as dt
from pathlib import Path

import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ----------------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------------
CSV_PATH = "./eds_trend__power_hi.csv"  # ↔ adjust if your file lives elsewhere
OUT_DIR = Path("./plots")  # all PNGs go here
OUT_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------------
# 2. HELPER: very-lightweight solar-elevation approximation
# ----------------------------------------------------------------------------
def solar_elevation(lat, lon, tz_offset, dt_local):
    """Return solar elevation angle (degrees) at *dt_local*."""
    n = dt_local.timetuple().tm_yday
    lt = dt_local.hour + dt_local.minute / 60 + dt_local.second / 3600  # local clock time
    B = math.radians((360 / 365) * (n - 81))
    eot = 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)  # Eq. of Time [min]
    lstm = 15 * tz_offset
    tc = 4 * (lon - lstm) + eot  # Time-corr [min]
    lst = lt + tc / 60  # Local solar time
    omega = math.radians(15 * (lst - 12))  # Hour angle
    delta = math.radians(23.45 * math.sin(math.radians(360 * (284 + n) / 365)))
    phi = math.radians(lat)
    cos_z = (math.sin(phi) * math.sin(delta) +
             math.cos(phi) * math.cos(delta) * math.cos(omega))
    z = math.acos(max(-1, min(1, cos_z)))  # clamp
    return math.degrees(math.pi / 2 - z)


# ----------------------------------------------------------------------------
# 3. DATA LOAD & FEATURE ENGINEERING
# ----------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH, parse_dates=["time"])

# 3a. solar elevation for Warsaw (52.2297 N, 21.0122 E, UTC+2 in summer)
df["sun_elev"] = df["time"].apply(lambda t: solar_elevation(52.2297, 21.0122, 2, t))

# 3b. smooth time-of-day encodings
df["hour_decimal"] = (df["time"].dt.hour
                      + df["time"].dt.minute / 60
                      + df["time"].dt.second / 3600)
hour_angle = 2 * np.pi * df["hour_decimal"] / 24
df["hour_sin"] = np.sin(hour_angle)
df["hour_cos"] = np.cos(hour_angle)

# 3c. drop rows with missing values
df = df.dropna(subset=["power_reference.common@sensor_1:VALUE",
                       "power_hi.common@sensor_1:VALUE"])

# ----------------------------------------------------------------------------
# 4. BUILD FEATURE MATRIX & TARGET
# ----------------------------------------------------------------------------
X = df[["power_hi.common@sensor_1:VALUE",
        "sun_elev",
        "hour_sin",
        "hour_cos"]]
y = df["power_reference.common@sensor_1:VALUE"]

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.20, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------------------------------------------------------
# 5. EVALUATE
# ----------------------------------------------------------------------------
y_pred_test = model.predict(X_test)
r2 = r2_score(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
print(f"R² = {r2:.3f}   MAE = {mae:.1f} W")

# predict full set (used in several plots)
df["pred_reference"] = model.predict(X_poly)

timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

# ----------------------------------------------------------------------------
# 6. PLOTS (all saved as PNG + shown on screen)
# ----------------------------------------------------------------------------

# -- Plot 1: raw series vs time ----------------------------------------------
plt.figure(figsize=(9, 4))
plt.plot(df["time"], df["power_hi.common@sensor_1:VALUE"],
         label="Power HI sensor", linewidth=0.9)
plt.plot(df["time"], df["power_reference.common@sensor_1:VALUE"],
         label="Power Reference (actual)", linewidth=0.9)
plt.title("Raw input series over time")
plt.xlabel("Time")
plt.ylabel("Power [W]")
plt.legend()
plt.grid(True)
plt.tight_layout()
fname = OUT_DIR / f"series_vs_time_{timestamp}.png"
plt.savefig(fname, dpi=300)
print("Saved:", fname)
plt.show()

# -- Plot 2: model fit curve (reference vs HI) -------------------------------
df_sorted = df.sort_values("power_hi.common@sensor_1:VALUE")
plt.figure(figsize=(6, 5))
plt.scatter(df["power_hi.common@sensor_1:VALUE"],
            df["power_reference.common@sensor_1:VALUE"],
            s=8, alpha=0.3, label="Actual reference")
plt.plot(df_sorted["power_hi.common@sensor_1:VALUE"],
         df_sorted["pred_reference"],
         linewidth=2, label="Model prediction")
plt.title("Model fit: output vs input")
plt.xlabel("Power HI sensor [W]")
plt.ylabel("Reference power [W]")
plt.legend()
plt.grid(True)
plt.tight_layout()
fname = OUT_DIR / f"fit_curve_{timestamp}.png"
plt.savefig(fname, dpi=300)
print("Saved:", fname)
plt.show()

# -- Plot 3: predicted vs actual (test scatter) ------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_test, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], linewidth=2)
plt.title("Model alignment: predicted vs actual (test set)")
plt.xlabel("Actual reference power [W]")
plt.ylabel("Predicted reference power [W]")
plt.grid(True)
plt.tight_layout()
fname = OUT_DIR / f"pred_vs_actual_{timestamp}.png"
plt.savefig(fname, dpi=300)
print("Saved:", fname)
plt.show()

# -- Plot 4: reference vs fitted over time -----------------------------------
plt.figure(figsize=(9, 4))
plt.plot(df["time"], df["power_reference.common@sensor_1:VALUE"],
         label="Power Reference (actual)", linewidth=0.9)
plt.plot(df["time"], df["pred_reference"],
         label="Model output (fitted)", linewidth=0.9)
plt.title("Reference sensor vs fitted model over time")
plt.xlabel("Time")
plt.ylabel("Reference power [W]")
plt.legend()
plt.grid(True)
plt.tight_layout()
fname = OUT_DIR / f"reference_vs_fit_{timestamp}.png"
plt.savefig(fname, dpi=300)
print("Saved:", fname)
plt.show()

names = ["x1", "x2", "x3", "x4",  # first-order
         "x1²", "x1x2", "x1x3", "x1x4",
         "x2²", "x2x3", "x2x4",
         "x3²", "x3x4",
         "x4²"]  # total = 14

coeff = np.r_[model.coef_]  # 1-D vector
template = textwrap.dedent("""\
    // Auto-generated coefficients (float32)
    constexpr float INTERCEPT = {inter:.8f}f;
    constexpr float COEF[14] = {{
    {coef_body}
    }};
    """)
body = ",\n".join([f"    /*{n:>4}*/ {c:.8f}f" for n, c in zip(names, coeff)])
print(template.format(inter=model.intercept_, coef_body=body))

from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import pvlib
from pvlib import clearsky, atmosphere, solarposition, irradiance
from pvlib.location import Location
from pvlib.iotools import read_tmy3
from pvlib.clearsky import detect_clearsky
from pvanalytics.features.clearsky import reno

from plot_functions import plot_clear_sky, plot_poa_components
from model_params import ClearSkyParameters, SolarDataForLocationAndTime
from calibrate_methods import sanitize_filename
from save_functions import save_dataframe_to_csv

def get_solar_data_for_location_and_time(clear_sky_parameters: ClearSkyParameters) -> SolarDataForLocationAndTime:
    tus = Location(
        latitude=clear_sky_parameters.warsaw_lat,
        longitude=clear_sky_parameters.warsaw_lon,
        tz=clear_sky_parameters.tz,
        altitude=clear_sky_parameters.altitude,
        name=clear_sky_parameters.name
    )

    times = pd.date_range(
        start=clear_sky_parameters.start_time,
        end=clear_sky_parameters.end_time,
        freq=clear_sky_parameters.frequency
    )

    sol = pvlib.solarposition.get_solarposition(times, clear_sky_parameters.warsaw_lat, clear_sky_parameters.warsaw_lon)
    cs = tus.get_clearsky(times)

    return tus, times, sol, cs

def clear_sky(
        clear_sky_parameters: ClearSkyParameters,
        show: bool = False,
        save_dir_plot: Path = None,
        save_dir_data: Path = None,
) -> pd.DataFrame:
    tus, times, sol, cs = get_solar_data_for_location_and_time(clear_sky_parameters)

    dni = cs['dni']
    dhi = cs['dhi']
    ghi = cs['ghi']

    dni_extra = irradiance.get_extra_radiation(times)
    solarpos = solarposition.get_solarposition(times, clear_sky_parameters.warsaw_lat, clear_sky_parameters.warsaw_lon)

    # panel orientation
    surface_tilt = clear_sky_parameters.surface_tilt
    surface_azimuth = clear_sky_parameters.surface_azimuth

    # get POA
    poa = irradiance.get_total_irradiance(
        surface_tilt,
        surface_azimuth,
        solarpos['zenith'],
        solarpos['azimuth'],
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        dni_extra=dni_extra,
        albedo=clear_sky_parameters.albedo,  # ground reflectance for ground‐reflected component
        model='perez'  # you can choose 'isotropic', 'haydavies', 'dirint', etc.
    )

    plot_clear_sky(cs, save_dir=save_dir_plot, show=show)
    plot_poa_components(poa, save_dir=save_dir_plot, show=show)

    if save_dir_data is not None:
        save_dir_data = Path(save_dir_data)
        output_path = save_dir_data.parent / "calculated_data" / save_dir_data.stem / ("poa_values" + save_dir_data.suffix)
        save_dataframe_to_csv(poa, output_path, index=True, index_label="time")

    return poa

def calculate_adaptive_best_mask(pair: pd.DataFrame) -> pd.DataFrame:
    poa_global_ref = pair['poa_global'].quantile(0.95)
    mean_percentage_grid = [0.08, 0.09, 0.10] #[0.06, 0.07, 0.08]
    max_percentage_grid = [0.12, 0.15] #[0.10, 0.12]

    step = pair.index.to_series().diff().median()
    window_minutes = int(max(3, round(pd.Timedelta('10min') / step))) * int(step / pd.Timedelta('1min'))
    window_length = max(6, min(20, window_minutes))

    best_mask, best_score = None, -np.inf
    for mean_pct, max_pct in product(mean_percentage_grid, max_percentage_grid):
        mean_diff = mean_pct * poa_global_ref
        max_diff = max_pct * poa_global_ref

        m = detect_clearsky(
            pair['measured'], pair['poa_global'],
            window_length=window_length, #10
            mean_diff=mean_diff, #100
            max_diff=max_diff, #100
        )

        mask = m.astype(bool)
        if mask.any():
            corr = pair.loc[mask, ['measured', 'poa_global']].corr().iloc[0, 1]
            nmid = (pair['poa_global'] > 0.4 * poa_global_ref).sum()
            nsel = (mask & (pair['poa_global'] > 0.4 * poa_global_ref)).sum()
            cover = nsel / max(1, nmid)
            score = (max(corr, 0) if pd.notna(corr) else 0) + 0.6 * cover
            if score > best_score:
                best_score, best_mask = score, mask

    sunny_subset = best_mask if best_mask is not None else pd.Series(False, index=pair.index)

    return sunny_subset

def calculate_my_own_mask(
        pair: pd.DataFrame,
        ratio: float = 0.90, # percentage
        time_period: int = 10 # minutes
) -> pd.Series:
    diff = (pair["measured"] - pair["poa_global"]).abs()
    tolerance = (1.0 - ratio) * pair["poa_global"]
    base = diff.le(tolerance) & diff.notna() & tolerance.gt(0)
    groups = base.ne(base.shift(fill_value=False)).cumsum()
    run_len = base.groupby(groups).transform("size")
    mask = base & run_len.ge(time_period)

    return mask.astype(bool)

def detect_clearsky_periods(
        df: pd.DataFrame,
        poa: pd.DataFrame,
        sensor_names: list[str] = None,
        sensor_name_ref: str = None,
        save_dir: Optional[Path] = None,
) -> pd.Series:
    df = df.copy()
    poa = poa.copy()

    df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
    df = df.set_index('time').tz_convert('Europe/Warsaw').sort_index()

    measured = df[sensor_name_ref].astype(float)
    poa_global = poa['poa_global'].astype(float)
    if poa_global.index.tz is None:
        poa_global.index = poa_global.index.tz_localize('Europe/Warsaw')
    else:
        poa_global = poa_global.tz_convert('Europe/Warsaw')
    poa_global = poa_global.sort_index()

    step = df.index.to_series().diff().median()
    if pd.isna(step):
        step = pd.Timedelta('1min')
    tol = step / 2

    poa_on_meas = poa_global.reindex(measured.index, method='nearest', tolerance=tol)

    pair = pd.concat(
        [measured.astype(float), poa_global.astype(float)],
        axis=1,
        keys=["measured", "poa_global"],
        join="inner",
    ).sort_index()

    #sunny_subset = calculate_adaptive_best_mask(pair)
    sunny_subset = detect_clearsky(
        pair['measured'], pair['poa_global'],
        window_length=4,
        mean_diff=100,
        max_diff=125
    )
    #sunny_subset = calculate_my_own_mask(pair, ratio=0.90, time_period=10)

    sunny_subset.index.name = 'time'
    df_sunny = sunny_subset.to_frame(name='if_sunny').reset_index()

    if save_dir is not None:
        save_dir = Path(save_dir)
        s_name = sanitize_filename(sensor_name_ref)
        output_path = save_dir.parent / "calculated_data" / save_dir.stem / (s_name + "_sunny_periods" + save_dir.suffix)
        save_dataframe_to_csv(df_sunny, output_path, index=False)

    return sunny_subset

import numpy as np
import pandas as pd
import pvlib

from itertools import product
from pathlib import Path
from typing import Optional
from pvlib import clearsky, atmosphere, solarposition, irradiance
from pvlib.location import Location
from pvlib.iotools import read_tmy3
from pvlib.clearsky import detect_clearsky
from pvanalytics.features.clearsky import reno
from datetime import time

from pvtools.visualisation.plotter import plot_clear_sky, plot_poa_components
from pvtools.config.params import ClearSkyParameters, SolarDataForLocationAndTime, ClearSkyCalculatedValues
from pvtools.preprocess.preprocess_data import sanitize_filename
from pvtools.io_file.writer import save_dataframe_to_csv
from pvtools.preprocess.preprocess_data import delete_night_period

def clear_sky(
        clearsky_parameters: ClearSkyParameters,
        show: bool = False,
        save_dir_plot: Path = None,
        save_dir: Path = None,
        filename: str = None
) -> pd.DataFrame:
    tus, times, sol, cs = get_solar_data_for_location_and_time(clearsky_parameters)

    dni = cs['dni']
    dhi = cs['dhi']
    ghi = cs['ghi']

    dni_extra = irradiance.get_extra_radiation(times)
    solarpos = solarposition.get_solarposition(times, clearsky_parameters.warsaw_lat, clearsky_parameters.warsaw_lon)

    # panel orientation
    surface_tilt = clearsky_parameters.surface_tilt
    surface_azimuth = clearsky_parameters.surface_azimuth

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
        albedo=clearsky_parameters.albedo,  # ground reflectance for ground‐reflected component
        model='perez'  # you can choose 'isotropic', 'haydavies', 'dirint', etc.
    )

    poa = poa.rename_axis('time').reset_index()
    poa_filtered = delete_night_period(
        df=poa,
        start=time(3, 0),  # 3:00 GMT -> 5:00 UTC+2
        end=time(18, 0)  # 18:00 GMT -> 20:00 UTC+2
    )

    plot_clear_sky(cs, save_dir=save_dir_plot, show=show)
    plot_poa_components(poa_filtered, save_dir=save_dir_plot, show=show)

    if save_dir is not None:
        save_dir = Path(save_dir)
        output_path = save_dir / "calculated_data" / filename / ("poa_values" + ".csv")
        save_dataframe_to_csv(poa_filtered, output_path, index=False)

    return poa_filtered

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

'''
def detect_clearsky_periods(
        df: pd.DataFrame,
        poa: pd.DataFrame,
        sensor_names: list[str] = None,
        sensor_name_ref: str = None,
        save_dir: Optional[Path] = None,
        filename: str = None,
) -> pd.Series:
    df = df.copy()
    poa = poa.copy()

    df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
    df = df.set_index('time').tz_convert('Europe/Warsaw').sort_index()

    if poa['time'].dt.tz is None:
        poa['time'] = poa['time'].dt.tz_localize('Europe/Warsaw')
    else:
        poa['time'] = poa['time'].dt.tz_convert('Europe/Warsaw')
    poa = poa.sort_values('time')

    poa['time'] = pd.to_datetime(poa['time'])
    poa = poa.set_index('time')
    poa_global = poa['poa_global'].astype(float)

    measured = df[sensor_name_ref].astype(float)
    measured = measured.rename('measured')

    pair = pd.concat(
        [measured, poa_global],
        axis=1,
        join='inner',
    ).sort_index()

    ------------------------------------------------------------------------------------------------
    #sunny_subset = calculate_adaptive_best_mask(pair)
    sunny_subset = detect_clearsky(
        pair['measured'], pair['poa_global'],
        window_length=4,
        mean_diff=100,
        max_diff=125
    )
    #sunny_subset = calculate_my_own_mask(pair, ratio=0.90, time_period=10)
    ----------------------------------------------------------------------------------------------------

    masks = []
    for day, grp in pair.groupby(pair.index.normalize()):
        grp = grp.asfreq('1min')

        sub = grp[['measured', 'poa_global']].dropna()
        if sub.empty:
            continue

        mask = detect_clearsky(
            sub['measured'],
            sub['poa_global'],
            window_length=4,
            mean_diff=100,
            max_diff=125,
        )
        masks.append(mask)

    sunny_subset = pd.concat(masks).sort_index()

    sunny_subset.index.name = 'time'
    df_sunny = sunny_subset.to_frame(name='if_sunny').reset_index()

    if save_dir is not None:
        save_dir = Path(save_dir)
        s_name = sanitize_filename(sensor_name_ref)
        output_path = save_dir / "calculated_data" / filename / (s_name + "_sunny_periods" + ".csv")
        save_dataframe_to_csv(df_sunny, output_path, index=False)

    return sunny_subset
'''
''' ---------------------------------------->>>>>>>>>>>>>>>>>>>> CHAT GPT
def detect_clearsky_periods(
        df: pd.DataFrame,
        poa: pd.DataFrame,
        sensor_names: list[str] = None,
        sensor_name_ref: str = None,
        save_dir: Optional[Path] = None,
        filename: str = None,
) -> pd.Series:
    df = df.copy()
    poa = poa.copy()

    df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
    df = df.set_index('time').tz_convert('Europe/Warsaw').sort_index()

    if poa['time'].dt.tz is None:
        poa['time'] = poa['time'].dt.tz_localize('Europe/Warsaw')
    else:
        poa['time'] = poa['time'].dt.tz_convert('Europe/Warsaw')
    poa = poa.sort_values('time')

    poa['time'] = pd.to_datetime(poa['time'])
    poa = poa.set_index('time')
    poa_global = poa['poa_global'].astype(float)

    measured = df[sensor_name_ref].astype(float)
    measured = measured.rename('measured')

    pair = pd.concat(
        [measured, poa_global],
        axis=1,
        join='inner',
    ).sort_index()

    masks = []
    for _, grp in pair.groupby(pair.index.normalize()):
        grp = grp.asfreq('1min')
        if grp.isna().all(axis=None):
            continue

        cs = grp['poa_global']
        daymask = cs > 200.0  # try 150–250 depending on site
        if not daymask.any():
            continue

        s = grp.loc[daymask, 'measured'].dropna()
        cs = cs.loc[daymask].reindex(s.index)

        if len(s) < 10:
            continue

        sub = grp[['measured', 'poa_global']].dropna()
        if sub.empty:
            continue

        mask, comps, alpha = detect_clearsky(
            measured=s,
            clearsky=cs,
            window_length='10min',
            mean_diff=40,  # tighten vs your 100/125
            max_diff=60,
            return_components=True,
            infer_limits=True,
        )

        strict = mask & alpha.between(0.9, 1.1)

        sunny_day = pd.Series(False, index=grp.index, name='if_sunny')
        sunny_day.loc[daymask] = strict.reindex(s.index, fill_value=False)
        masks.append(sunny_day)

    sunny_subset = pd.concat(masks).sort_index() if masks else pd.Series(False, index=pair.index, name='if_sunny')
    sunny_subset.index.name = 'time'

    if save_dir is not None:
        save_dir = Path(save_dir)
        s_name = sanitize_filename(sensor_name_ref)
        output_path = save_dir / "calculated_data" / filename / (s_name + "_sunny_periods" + ".csv")
        save_dataframe_to_csv(sunny_subset, output_path, index=False)

    return sunny_subset
'''

def detect_clearsky_periods(
        df: pd.DataFrame,
        poa: pd.DataFrame,
        sensor_names: list[str] = None,
        sensor_name_ref: str = None,
        save_dir: Optional[Path] = None,
        filename: str = None,
) -> pd.Series:
    df = df.copy()
    poa = poa.copy()

    df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
    df = df.set_index('time').tz_convert('Europe/Warsaw').sort_index()

    if poa['time'].dt.tz is None:
        poa['time'] = poa['time'].dt.tz_localize('Europe/Warsaw')
    else:
        poa['time'] = poa['time'].dt.tz_convert('Europe/Warsaw')
    poa = poa.sort_values('time')

    poa['time'] = pd.to_datetime(poa['time'])
    poa = poa.set_index('time')
    poa_global = poa['poa_global'].astype(float)

    measured = df[sensor_name_ref].astype(float)
    measured = measured.rename('measured')

    pair = pd.concat(
        [measured, poa_global],
        axis=1,
        join='inner',
    ).sort_index()

    tolerance = 0.3 # 30%
    series_mask = pair['measured'].between(pair['poa_global'] * (1-tolerance), pair['poa_global'] * (1+tolerance))

    masks = []
    for day, grp in pair.groupby(pair.index.normalize()):
        grp = grp.asfreq('1min')

        sub = grp[['measured', 'poa_global']].dropna()
        if sub.empty:
            continue


        mask = detect_clearsky(
            sub['measured'],
            sub['poa_global'],
            window_length=4,
            mean_diff=100,
            max_diff=125,
        )
        masks.append(mask)

    sunny_subset = pd.concat(masks).sort_index()

    sunny_subset.index.name = 'time'
    df_sunny = sunny_subset.to_frame(name='if_sunny').reset_index()

    series_sunny = df_sunny.set_index('time')['if_sunny']
    combined_masks = series_sunny & series_mask
    combined_masks.name = 'if_sunny'

    if save_dir is not None:
        save_dir = Path(save_dir)
        s_name = sanitize_filename(sensor_name_ref)
        output_path = save_dir / "calculated_data" / filename / (s_name + "_sunny_periods" + ".csv")
        save_dataframe_to_csv(combined_masks, output_path, index=True)

    return combined_masks

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



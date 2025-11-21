import numpy as np
import pandas as pd
import pvlib

from itertools import product
from pathlib import Path
from typing import Optional

from pandas import DatetimeIndex
from pvlib import irradiance
from pvlib.location import Location
from pvlib.clearsky import detect_clearsky, simplified_solis
from pvlib_mod.clearsky_mod import detect_clearsky_mod

from pvtools.config.params import ClearSkyParameters, ModelTimes, ModelDirectories
from pvtools.preprocess.preprocess_data import sanitize_filename
from pvtools.io_file.writer import save_dataframe_to_csv


def clear_sky(
        clearsky_params: ClearSkyParameters,
        model_dirs: ModelDirectories,
        model_times: ModelTimes,
) -> list[pd.DataFrame]:

    filename = model_dirs.filename
    save_dir = model_dirs.data_dir

    (tus, times, solarpos, cs, airmass, dni_extra) = get_solar_data_for_location_and_time(clearsky_params, model_times)

    dni = cs['dni']
    dhi = cs['dhi']
    ghi = cs['ghi']

    # panel orientation
    surface_tilt = clearsky_params.surface_tilt
    surface_azimuth = clearsky_params.surface_azimuth

    # get POA
    poa = irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=solarpos['zenith'],
        solar_azimuth=solarpos['azimuth'],
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        dni_extra=dni_extra,
        airmass=airmass,
        albedo=clearsky_params.albedo,
        model='perez-driesse'
    )

    poa = poa.rename_axis('time').reset_index()
    poa = pd.DataFrame(poa)

    if save_dir is not None:
        save_dir = Path(save_dir)
        output_path = save_dir / "calculated_data" / filename / ("poa_values" + ".csv")
        save_dataframe_to_csv(poa, output_path, index=False)

    return [poa, cs]


def get_solar_data_for_location_and_time(
        clearsky_params: ClearSkyParameters,
        model_times: ModelTimes
) -> tuple[Location, DatetimeIndex, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    tus = Location(
        latitude=clearsky_params.warsaw_lat,
        longitude=clearsky_params.warsaw_lon,
        tz=clearsky_params.tz,
        altitude=clearsky_params.altitude,
        name=clearsky_params.name
    )

    times = pd.date_range(
        start=model_times.start_time,
        end=model_times.end_time,
        freq=model_times.frequency
    )

    solpos = pvlib.solarposition.get_solarposition(
        time=times,
        latitude=clearsky_params.warsaw_lat,
        longitude=clearsky_params.warsaw_lon,
        altitude=clearsky_params.altitude,
        method='pyephem', #'nrel_numba'
    )

    '''
    apparent_zenith = solpos['apparent_zenith']
    airmass = pvlib.atmosphere.get_relative_airmass(apparent_zenith)
    pressure = pvlib.atmosphere.alt2pres(clearsky_params.altitude)
    airmass = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)
    linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(times, clearsky_params.warsaw_lat, clearsky_params.warsaw_lon)

    cs = tus.get_clearsky(
        times=times,
        model='ineichen',
        solar_position=solpos,
        linke_turbidity=linke_turbidity - 3, #linke_turbidity, #0.75
        #airmass=airmass
    )
    
    cs = tus.get_clearsky(
        times=times,
        model='simplified_solis',
        solar_position=solpos,
        # linke_turbidity=linke_turbidity - 3, #linke_turbidity, #0.75
        # airmass=airmass
    )
    '''

    apparent_zenith = solpos['apparent_zenith']
    airmass = pvlib.atmosphere.get_relative_airmass(apparent_zenith)

    apparent_elevation = solpos['apparent_elevation']
    aod700 = 0.1
    precipitable_water = 1.5 #FIXME This parameters shouldn't be hard-coded. Wait for response to download data!
    pressure = pvlib.atmosphere.alt2pres(clearsky_params.altitude)

    #FIXME - variables should be input variables. Hard coded for testing
    dni_extra = pvlib.irradiance.get_extra_radiation(
        datetime_or_doy=times,
        solar_constant=1366.1,
        method='pyephem', #'nrel'
        )

    cs = simplified_solis(
        apparent_elevation=apparent_elevation,
        aod700=aod700,
        precipitable_water=precipitable_water,
        pressure=pressure,
        dni_extra=dni_extra
    )

    return tus, times, solpos, cs, airmass, dni_extra


def detect_clearsky_periods(
        df: pd.DataFrame,
        poa: pd.DataFrame,
        sensor_name_ref: str = None,
        save_dir: Optional[Path] = None,
        filename: str = None,
) -> [pd.Series, pd.Series]:

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
            window_length=10, #before 4
            mean_diff=100,
            max_diff=125,
        )
        masks.append(mask)

    sunny_subset = pd.concat(masks).sort_index()

    sunny_subset.index.name = 'time'
    df_sunny = sunny_subset.to_frame(name='if_sunny').reset_index()

    series_sunny = df_sunny.set_index('time')['if_sunny']
    sunny_mask = series_sunny & series_mask
    sunny_mask.name = 'if_sunny'

    cloudy_mask = ~sunny_mask

    sunny_intervals = detect_sunny_cloudy_intervals(sunny_mask)
    cloudy_intervals = detect_sunny_cloudy_intervals(cloudy_mask)

    sunny_periods_cutted_short, cloudy_periods_cutted_short = delete_short_periods(
        sunny_mask=sunny_mask,
        cloudy_mask=cloudy_mask,
        sunny_intervals=sunny_intervals,
        cloudy_intervals=cloudy_intervals,
        min_length=30
    )

    if save_dir is not None:
        save_dir = Path(save_dir)
        s_name = sanitize_filename(sensor_name_ref)
        output_path_sunny = save_dir / "calculated_data" / filename / (s_name + "_sunny_periods_all" + ".csv")
        save_dataframe_to_csv(sunny_mask, output_path_sunny, index=True)

        output_path_cloudy = save_dir / "calculated_data" / filename / (s_name + "_cloudy_periods_all" + ".csv")
        save_dataframe_to_csv(cloudy_mask, output_path_cloudy, index=True)

        output_path_sunny_intervals = save_dir / "calculated_data" / filename / "sunny_intervals.csv"
        save_dataframe_to_csv(sunny_intervals, output_path_sunny_intervals, index=False)

        output_path_cloudy_intervals = save_dir / "calculated_data" / filename / "cloudy_intervals.csv"
        save_dataframe_to_csv(cloudy_intervals, output_path_cloudy_intervals, index=False)

        output_path_sunny = save_dir / "calculated_data" / filename / (s_name + "_sunny_periods_cutted_short" + ".csv")
        save_dataframe_to_csv(sunny_periods_cutted_short, output_path_sunny, index=True)

        output_path_cloudy = save_dir / "calculated_data" / filename / (s_name + "_cloudy_periods_cutted_short" + ".csv")
        save_dataframe_to_csv(cloudy_periods_cutted_short, output_path_cloudy, index=True)

    return sunny_mask, cloudy_mask

def detect_clearsky_periods_v2(
        measured: pd.Series,
        clearsky: pd.Series,
        times: pd.Series,
        sensor_name_ref: str = None,
        save_dir: Optional[Path] = None,
        filename: str = None,
) -> [pd.Series, pd.Series]:

    measured.index = times
    clearsky.index = times

    clear, comp, alpha = detect_clearsky_mod(
        measured=measured,
        clearsky=clearsky,
        window_length=7,
        mean_diff=200,
        max_diff=300,
        lower_line_length=3,
        upper_line_length=20,
        var_diff=0.01,
        slope_dev=30,
        max_iterations=20,
        return_components=True
    )

    clear.name = 'if_sunny'
    cloudy = ~clear

    if save_dir is not None:
        save_dir = Path(save_dir)
        s_name = sanitize_filename(sensor_name_ref)
        output_path_sunny = save_dir / "calculated_data" / filename / (s_name + "_sunny_periods_all" + ".csv")
        save_dataframe_to_csv(clear, output_path_sunny, index=True)

        output_path_cloudy = save_dir / "calculated_data" / filename / (s_name + "_cloudy_periods_all" + ".csv")
        save_dataframe_to_csv(cloudy, output_path_cloudy, index=True)

    return clear, cloudy


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

def detect_sunny_cloudy_intervals(s: pd.Series) -> pd.DataFrame:

    groups = (s != s.shift()).cumsum()
    true_groups = s[s].groupby(groups)

    data = [
        {
            "start": group.index[0],
            "end": group.index[-1],
            "length": len(group)
        }
        for _, group in true_groups
    ]

    intervals = pd.DataFrame(data)

    return intervals

def delete_short_periods(
        sunny_mask: pd.Series,
        cloudy_mask: pd.Series,
        sunny_intervals: pd.DataFrame,
        cloudy_intervals: pd.DataFrame,
        min_length: int
) -> [pd.DataFrame, pd.DataFrame]:

    sunny_mask_filtered = []
    cloudy_mask_filtered = []

    for _, row in sunny_intervals.iterrows():
        if row["length"] >= min_length:
            start_time = row["start"]
            end_time = row["end"]
            sunny_mask_filtered.append(sunny_mask.loc[start_time:end_time])

    for _, row in cloudy_intervals.iterrows():
        if row["length"] >= min_length:
            start_time = row["start"]
            end_time = row["end"]
            cloudy_mask_filtered.append(cloudy_mask.loc[start_time:end_time])

    sunny_mask_combined = pd.DataFrame()
    cloudy_mask_combined = pd.DataFrame()

    if sunny_mask_filtered:
        sunny_mask_combined = pd.concat(sunny_mask_filtered)

    if cloudy_mask_filtered:
        cloudy_mask_combined = pd.concat(cloudy_mask_filtered)

    return sunny_mask_combined, cloudy_mask_combined

import pandas as pd
import numpy as np
import pvlib

from typing import List
from pvanalytics import system
from pvanalytics.system import _peak_times # infer_orientation_daily_peak

from clear_sky_model import get_solar_data_for_location_and_time
from model_params import ClearSkyParameters

def determine_system_azimuth_and_tilt(
        clear_sky_parameters: ClearSkyParameters,
        df: pd.DataFrame,
        sunny_mask: pd.Series,
        sensor_names: List[str] = None,
        sensor_name_ref: str = None,
        tilts: np.ndarray = None,
        azimuths: np.ndarray = None
) -> List[float]:
    measured = pd.Series(df[sensor_name_ref].values, index=df['time'])

    # candidate grid to search
    if tilts is None:
        tilts = np.arange(0, 30, 1)
    if azimuths is None:
        azimuths = np.arange(170, 190, 1) # 180 is south

    tus, times, sol, cs = get_solar_data_for_location_and_time(clear_sky_parameters)

    freq = pd.Timedelta(clear_sky_parameters.frequency)
    measured = measured.reindex(times, method="nearest", tolerance=freq)
    sunny_mask = sunny_mask.reindex(times, method="nearest", tolerance=freq).fillna(False)

    tilt_deg, azimuth_deg = infer_orientation_daily_peak(
        power_or_poa=measured,
        sunny=sunny_mask,
        tilts=tilts,
        azimuths=azimuths,
        solar_azimuth=sol['azimuth'],
        solar_zenith=sol['apparent_zenith'],
        ghi=cs['ghi'],
        dhi=cs['dhi'],
        dni=cs['dni'],
    )

    print(f"Estimated tilt: {tilt_deg:.1f}°, azimuth: {azimuth_deg:.1f}°")

    return float(tilt_deg), float(azimuth_deg)

def infer_orientation_daily_peak(
        power_or_poa,
        sunny,
        tilts,
        azimuths,
        solar_azimuth,
        solar_zenith,
        ghi,
        dhi,
        dni
) -> List[float]:
    peak_times = _peak_times(power_or_poa[sunny])
    azimuth_by_minute = solar_azimuth.resample('1min').interpolate(method='linear')
    modeled_azimuth = azimuth_by_minute[peak_times]
    best_azimuth = None
    best_tilt = None
    smallest_sse = None

    for azimuth in azimuths:
        for tilt in tilts:
            poa = pvlib.irradiance.get_total_irradiance(
                tilt,
                azimuth,
                solar_zenith,
                solar_azimuth,
                ghi=ghi,
                dhi=dhi,
                dni=dni
            ).poa_global
            idx_daily_max = by_day(poa).idxmax()
            poa_azimuths = azimuth_by_minute.reindex(idx_daily_max, method="nearest", tolerance="30s")
            #poa_azimuths = azimuth_by_minute[by_day(poa).idxmax()] - this was originally in library. But it does not work!
            filtered_azimuths = poa_azimuths[np.isin(
                poa_azimuths.index.date,
                modeled_azimuth.index.date
            )]
            sum_of_squares = sum((filtered_azimuths.values - modeled_azimuth.values)**2)

            if (smallest_sse is None) or (smallest_sse > sum_of_squares):
                smallest_sse = sum_of_squares
                best_azimuth = azimuth
                best_tilt = tilt

    return best_azimuth, best_tilt

def by_day(data):
    return data.groupby(pd.to_datetime(data.index.date).tz_localize(data.index.tz)) # original code
    #return data.groupby(pd.Grouper(freq="D"))
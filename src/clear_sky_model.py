from pathlib import Path

import pandas as pd
import pvlib
from pvlib import clearsky, atmosphere, solarposition, irradiance
from pvlib.location import Location
from pvlib.iotools import read_tmy3

from plot_functions import plot_clear_sky, plot_poa_components
from model_params import ClearSkyParameters

def clear_sky(
        clear_sky_parameters: ClearSkyParameters,
        show: bool = False,
        save_dir: Path = None,
) -> pd.DataFrame:
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
        freq=clear_sky_parameters.frequency,
        tz=tus.tz)

    cs = tus.get_clearsky(times)  # ineichen with climatology table by default

    dni = cs['dni']
    dhi = cs['dhi']
    ghi = cs['ghi']

    dni_extra = irradiance.get_extra_radiation(times)
    solarpos = solarposition.get_solarposition(times, clear_sky_parameters.warsaw_lat, clear_sky_parameters.warsaw_lon)

    # panel orientation
    surface_tilt = 30  # degrees from horizontal
    surface_azimuth = 180  # south-facing

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

    plot_clear_sky(cs, save_dir=save_dir, show=show)
    plot_poa_components(poa, save_dir=save_dir, show=show)

    return poa

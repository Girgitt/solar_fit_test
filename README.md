## Table of contents

* [About](#-about)
* [How to build](#-how-to-build)
* [Usage](#-usage)
* [Configuration options](#-configuration-options)
* [Documentation](#-documentation)
* [Examples](#-examples)
* [License](#-license)
* [Contacts](#-contacts)

## 🚀 About
The solar fitting library is designed to provide comprehensive support for solar irradiance data analysis. It adheres
to standards of flexibility, reusability, and reliability. Utilizing well-known software design patterns, 
which offer benefits like:

* __Modularity__: The library consists of many smaller parts, different parts can function independently. 
* __Testability__: Smaller parts of code are testable. 
* __Maintainability__: Management of the codebase due to clean structure and separation is understandable.

Typical use cases include:

* Calibrating low-cost irradiance sensors against a reference sensor
* Performing clear-sky detection and filtering
* Fitting regression models (linear, polynomial, GP, splines, etc.)
* Generating plots and diagnostics for solar data analysis
* Exporting calibration coefficients for use on microcontrollers or other systems

The source code is organized as a Python package under `solar_fit_test/src`, with separate modules for loading data,
calibration, modeling, plotting, and utilities.

## 📝 How to build
```
# Open a terminal (Command Prompt or PowerShell for Windows, Terminal for macOS or Linux)

# Ensure Git is installed
# Visit https://git-scm.com to download and install console Git if not already installed

# Clone the repository
git clone https://github.com/Girgitt/solar_fit_test.git

# Navigate to the project directory
cd solar_fit_test

# Create virtual environment

#Linux/macOS:
   python -m venv .venv
   source .venv/bin/activate
   
#Windows:
   python -m venv .venv
   .venv\Scripts\activate
   
# Install dependencies
pip install -e .

# Verify installation
python -c "import solar_fit_test; print('OK')"
```

--------------------------------------------------------------------

To generate SPHINX documentation of the project open command line and move to ``solar_fit_test/docs``
then execute following commands:
* ``make clean``
* ``make html``

Then navigate to ``solar_fit_test/docs/build/html/index.html`` and open the documentation.

--------------------------------------------------------------------

To generate dependency graph (by using pydeps library) open command line and move to the project folder
``solar_fit_test`` then execute the command:

general:

``pydeps <filename> -o <output_name> -T <output_type>``

example:

``pydeps src/pvtools --only main pvtools --max-module-depth 2 --max-bacon 2 --cluster --rmprefix pvtools. --rankdir LR -o graph.svg``

Check ``--help`` for other options.


## ⚙️ Configuration options

There are several input parameters to properly set all calibration pipeline. Below is list of configuration parameters:
* __--action__: Specify whether to 'update' (train/save) or 'execute' (load/apply) the model.
* __--model_id__: Model identifier used for saving/loading coefficients.
* __--csv__: Path to CSV file with input data.
* __--calibration__: Defines which calibration method use to calibrate sensors.
* __--sensors__: List of sensors to calibrate. Number of specified column, counting from 0, skipping time column. 
Accept multiple numbers separated by space.
* __--reference__: Number of specified column, counting from 0, skipping time column. Accept single number.
* __--project_dir__: Force specific data directory to store logs, plots etc. Default: current working directory.
* __--calibration_metrics_dir__: Directory which contains all metrics needed for calibration.
* __--start_time_hour__: Start time (hour) for filter only day time period (GMT).
* __--start_time_minute__: Start time (minute) for filter only day time period.
* __--end_time_hour__: Start time (hour) for filter only day time period (GMT).
* __--end_time_minute__: Start time (minute) for filter only day time period.
* __--latitude__: Decimal latitude coordinates of measurement station (default Warsaw).
* __--longtitude__: Decimal longtitude coordinates of measurement station (default Warsaw).
* __--timezone__: Time zone of measurement station (default Europe/Warsaw). Check 'pytz.all_timezones' for all
available options.
* __--altitude__: Altitude of measurement station in meters.
* __--name__: Name for measurement station.
* __--frequency__: Target timestamps for filterenig dataset. Available formats: 'xs' 'xmin' 'xh' 'xms' 
where x is a number.
* __--albedo__: Ratio of reflected solar irradiance to global horizontal irradiance (unitless).
* __--surface_tilt__: Surface tilt of the sensor in degrees (default 0, horizontal).
* __--surface_azimuth__: Surface azimuth of the sensor in degrees (default 180, south).
* __--divided_linear_regression_intervals__: Time interval for one block in divided linear regression.
For all possibilities refer to: https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases.

## 📚 Documentation

## 🖥️ Usage

## 🌇 Examples

## 📃 License
All rights reserved.  
You may not use, copy, modify, or distribute this project without explicit permission.

## 🗨️ Contacts
For questions or collaboration inquiries, feel free to reach out:

**Mateusz Kania**

📧Email: mateusz.kania@ttas.pl

🐙GitHub: https://github.com/kaniamateusz
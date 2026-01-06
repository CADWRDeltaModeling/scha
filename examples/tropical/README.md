# Tropical example

This directory contains a small tropical example to exercise the SCHA Kalman-smoother pipeline.

Files:
- `tropical_sample.csv`: sample hourly time series with columns `time` and `y`. Note: the script also accepts CSVs with a numeric first column used as the time index (e.g., hours 0,1,2,...), but in that case the YAML *must* include `time_unit` (e.g., `hour` or `1h`) to indicate the units of the numeric index.
- `tropical_params.yml`: human-friendly parameters (frequencies in cycles-per-day, `dt` in days per sample, or `time_unit` when CSV index is numeric).
- `run_tropical_example.py`: script that reads the CSV and YAML, converts frequencies to radians per time-step, builds `SCHAParams`, runs the Kalman smoother, and writes `tropical_smoothed.csv`.
- `tropical_test.py`: basic test invoking the run script.

Notes:
- Frequencies in the YAML are in cycles per day (cpd). The script converts to radians/time-step using the `dt` value or by inferring sampling interval from the CSV index.
- Default variances are set to `sigma2_zeta = 1e-10` and `sigma2_kappa = 1.35e-13` to match typical values you suggested. Adjust these as needed for your R-computed parameters.
- The script depends on `PyYAML` to read the YAML file. Install: `pip install pyyaml`.

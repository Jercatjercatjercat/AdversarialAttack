"""Small, explicit settings for the second iteration; paths follow this file."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'local_data' / 'era5_five_variables.npz'
OUTPUT = ROOT / 'outputs' / 'era5'
# Channel order is fixed everywhere: input, output, statistics and result tables.
VARIABLES = ('t2m', 'u10', 'v10', 'mslp', 'q850')
SOURCE_NAMES = ('2m_temperature', '10m_u_component_of_wind',
                '10m_v_component_of_wind', 'mean_sea_level_pressure',
                'specific_humidity')
UNITS = ('K', 'm/s', 'm/s', 'hPa', 'g/kg')
SOURCE = ('https://storage.googleapis.com/weatherbench2/datasets/era5/'
          '1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr')

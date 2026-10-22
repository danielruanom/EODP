# eodp
EODP main repository

Earth Observation Data Processing simulator of a multi-band VNIR imager (Master in Space Engineering, UC3M).

## Modules
- `ism`: Instrument Module. Optical stage (ISRF, radiances to irradiances, system MTF), detection stage (photons, electrons, PRNU, dark signal, bad/dead pixels) and video chain (voltage, DN).
- `l1b`: Level-1B. Equalization and restoration (DN to radiances).
- `l1c`: Level-1C. Orthorectification onto the MGRS grid.
- `config`: configuration of each module. `ismConfig.py` and `l1bConfig.py` have the unit test values commented, the active values are the E2E ones.
- `auxiliary`: ISRF, equalization factors and logging configuration.

## Execution
Each module is launched with its main (input and output folders are set in the main):
```
python -m ism.mainIsm
python -m l1b.mainL1b
python -m l1c.mainL1c
```
Each module writes its log (`ISM.log`, `L1B.log`, `L1C.log`) in the output folder.

## Tests
The test scripts compare the outputs with the reference outputs (tolerance 1e-3 %) and generate the plots of the test report:
```
python -m ism.test.testIsm
python -m l1b.test.testL1b
python -m l1c.test.testL1c
python -m e2e.test.testE2e
```
- ISM: cross validation of all the stages, conversion factors and system MTF with its contributors.
- L1B: equalized vs non equalized TOA, ARA and SNR. The L1B is run twice, outputs in `output/equalized` and `output/non_equalized`.
- L1C: cross validation by MGRS point, GSD and footprint.
- E2E: cross validation of the ISM and L1B, L1B TOA maps and histograms, altitude map.

Test data in `data/EODP-TS-ISM`, `data/EODP-TS-L1B`, `data/EODP-TS-L1C` and `data/EODP-TS-E2E` (not versioned).

## Requirements
numpy, scipy (< 1.14, uses interp2d), netCDF4, mgrs, matplotlib

from common.io.writeToa import readToa
from common.io.readGeodetic import readGeodetic
from common.io.l1cProduct import readL1c
import numpy as np
import matplotlib.pyplot as plt
import mgrs
import os

# Directorio raiz del EODP (sirve en Linux y en Windows)
eodp_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
gm_dir = eodp_dir + '/data/EODP-TS-L1C/input/gm_alt100_act_150/'
l1b_dir = eodp_dir + '/data/EODP-TS-L1C/input/l1b_output/'
output_dir = eodp_dir + '/data/EODP-TS-L1C/output/'
validation_dir = eodp_dir + '/data/EODP-TS-L1C/output_validation/'

bands = ['VNIR-0', 'VNIR-1', 'VNIR-2', 'VNIR-3']
threshold = 1e-3 # [%] maxima diferencia relativa permitida

def distance(lat1, lon1, lat2, lon2):
    # Distancia sobre la Tierra entre dos puntos (haversine) [m]
    lat1, lon1, lat2, lon2 = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    a = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1)/2)**2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(a))

lat, lon = readGeodetic(gm_dir, 'geolocation.nc')
lon_l1b_boundary = np.concatenate([lon[0, :], lon[:, -1], lon[-1, ::-1], lon[::-1, 0]])
lat_l1b_boundary = np.concatenate([lat[0, :], lat[:, -1], lat[-1, ::-1], lat[::-1, 0]])

plt.figure(figsize=(16, 12))
for i, band in enumerate(bands):
    print('Processing band ' + band)
    toa_l1c_val, lat_l1c_val, lon_l1c_val = readL1c(validation_dir, 'l1c_toa_' + band + '.nc')
    toa_l1c, lat_l1c, lon_l1c = readL1c(output_dir, 'l1c_toa_' + band + '.nc')
    toa_l1b = readToa(l1b_dir, 'l1b_toa_' + band + '.nc')

    # Cross validation
    # Los puntos MGRS no tienen un orden fijo, asi que los ordeno por lat/lon antes de comparar
    print('Cross validation for band ' + band)
    idx = np.lexsort((lon_l1c, lat_l1c))
    idx_val = np.lexsort((lon_l1c_val, lat_l1c_val))
    same_grid = np.allclose(lat_l1c[idx], lat_l1c_val[idx_val]) and np.allclose(lon_l1c[idx], lon_l1c_val[idx_val])
    print('Same MGRS grid as the reference: ' + str(same_grid) + ' (' + str(len(toa_l1c)) + ' points)')
    # La referencia tiene valores fuera del rango del L1B (extrapolacion del spline) que nosotros recortamos con clip
    inside = (toa_l1c_val[idx_val] >= np.min(toa_l1b)) & (toa_l1c_val[idx_val] <= np.max(toa_l1b))
    print('Reference points outside the L1B range (clipped in our L1C): ' + str(np.sum(~inside)))
    diff = np.max(np.abs(toa_l1c[idx][inside] - toa_l1c_val[idx_val][inside])/np.abs(toa_l1c_val[idx_val][inside]))*100
    if diff < threshold:
        print('L1C TOA matches the reference TOA within the threshold of ' + str(threshold) + ' %')
    else:
        print('L1C TOA does NOT match the reference TOA within the threshold of ' + str(threshold) + ' %')
    print('Difference: ' + str(diff) + ' %')

    # Footprint: contorno del L1B con los puntos MGRS del L1C encima
    plt.subplot(2, 2, i+1)
    plt.scatter(lon_l1c, lat_l1c, c=toa_l1c, cmap='jet', s=5, label='L1C MGRS')
    plt.colorbar(label='TOA [mW/m2/sr]')
    plt.plot(lon_l1b_boundary, lat_l1b_boundary, color='black', linewidth=1.5, label='L1B')
    plt.title('Projection on ground ' + band)
    plt.xlabel('Longitude [deg]')
    plt.ylabel('Latitude [deg]')
    plt.grid(True)
    plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig(output_dir + 'footprint_all_bands.png')
plt.close()

# Ground sampling distance
# L1B: distancia entre pixeles vecinos de la geolocalizacion
gsd_act_l1b = distance(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
gsd_alt_l1b = distance(lat[:-1, :], lon[:-1, :], lat[1:, :], lon[1:, :])
# L1C: paso entre puntos MGRS vecinos en easting (~ACT) y northing (~ALT)
m = mgrs.MGRS()
easting = np.zeros(len(lat_l1c))
northing = np.zeros(len(lat_l1c))
for k in range(len(lat_l1c)):
    zone, hemisphere, easting[k], northing[k] = m.MGRSToUTM(m.toMGRS(float(lat_l1c[k]), float(lon_l1c[k]), MGRSPrecision=5))
#lat/lon estan guardados en float32, asi que redondeo a la malla de 100 m
easting = np.round(easting/100)*100
northing = np.round(northing/100)*100
gsd_act_l1c = np.concatenate([np.diff(np.sort(easting[northing == n])) for n in np.unique(northing)])
gsd_alt_l1c = np.concatenate([np.diff(np.sort(northing[easting == e])) for e in np.unique(easting)])

print('GSD ACT L1B: ' + str(np.mean(gsd_act_l1b)) + ' m, L1C: ' + str(np.mean(gsd_act_l1c)) + ' m')
print('GSD ALT L1B: ' + str(np.mean(gsd_alt_l1b)) + ' m, L1C: ' + str(np.mean(gsd_alt_l1c)) + ' m')

print('GSD ACT L1B variation: ' + str(np.max(gsd_act_l1b) - np.min(gsd_act_l1b)) + ' m, ALT: ' + str(np.max(gsd_alt_l1b) - np.min(gsd_alt_l1b)) + ' m')

# Para los plots: linea central del L1B y la fila/columna MGRS con mas puntos del L1C
idalt = int(lat.shape[0]/2)
idact = int(lat.shape[1]/2)
n_row = np.unique(northing)[np.argmax([np.sum(northing == n) for n in np.unique(northing)])]
e_col = np.unique(easting)[np.argmax([np.sum(easting == e) for e in np.unique(easting)])]
row_l1c = np.sort(easting[northing == n_row])
col_l1c = np.sort(northing[easting == e_col])
#distancia sobre el suelo desde el primer punto [km]
dist_act_l1b = np.cumsum(gsd_act_l1b[idalt, :])/1000
dist_alt_l1b = np.cumsum(gsd_alt_l1b[:, idact])/1000
dist_act_l1c = (row_l1c[1:] - row_l1c[0])/1000
dist_alt_l1c = (col_l1c[1:] - col_l1c[0])/1000

# Zoom en el centro de la imagen: centros de pixel L1B y puntos MGRS L1C en metros respecto al centro
lat0 = lat[idalt, idact]
lon0 = lon[idalt, idact]
x_l1b = np.radians(lon - lon0) * np.cos(np.radians(lat0)) * 6371008.8
y_l1b = np.radians(lat - lat0) * 6371008.8
x_l1c = np.radians(lon_l1c - lon0) * np.cos(np.radians(lat0)) * 6371008.8
y_l1c = np.radians(lat_l1c - lat0) * 6371008.8

plt.figure(figsize=(20, 6.5))
plt.subplot(1, 3, 1)
plt.scatter(x_l1b, y_l1b, color='blue', s=12, label='L1B pixel centres (' + f'{np.mean(gsd_act_l1b):.1f}' + ' m)')
plt.scatter(x_l1c, y_l1c, color='red', marker='s', s=30, label='L1C MGRS points (100 m)')
#linea y columna centrales del L1B para ver la orientacion de la malla del instrumento
plt.plot(x_l1b[idalt, :], y_l1b[idalt, :], color='blue', linewidth=1, label='L1B ACT line')
plt.plot(x_l1b[:, idact], y_l1b[:, idact], color='blue', linewidth=1, linestyle='--', label='L1B ALT column')
plt.xlim([-300, 300])
plt.ylim([-300, 300])
plt.gca().set_aspect('equal')
plt.title('Zoom at the image centre')
plt.xlabel('East [m]')
plt.ylabel('North [m]')
plt.grid()
plt.legend(loc='upper right')

plt.subplot(1, 3, 2)
plt.plot(dist_act_l1b, gsd_act_l1b[idalt, :], 'o-', color='blue', markersize=3, label='L1B: ' + f'{np.mean(gsd_act_l1b):.2f}' + ' m')
plt.plot(dist_act_l1c, np.diff(row_l1c), 's-', color='red', markersize=4, label='L1C: ' + f'{np.mean(gsd_act_l1c):.2f}' + ' m')
plt.ylim([0, 120])
plt.title('GSD across track')
plt.xlabel('Distance on ground [km]')
plt.ylabel('GSD [m]')
plt.grid()
plt.legend(loc='lower right')

plt.subplot(1, 3, 3)
plt.plot(dist_alt_l1b, gsd_alt_l1b[:, idact], 'o-', color='blue', markersize=3, label='L1B: ' + f'{np.mean(gsd_alt_l1b):.2f}' + ' m')
plt.plot(dist_alt_l1c, np.diff(col_l1c), 's-', color='red', markersize=4, label='L1C: ' + f'{np.mean(gsd_alt_l1c):.2f}' + ' m')
plt.ylim([0, 120])
plt.title('GSD along track')
plt.xlabel('Distance on ground [km]')
plt.ylabel('GSD [m]')
plt.grid()
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(output_dir + 'gsd.png')
plt.close()

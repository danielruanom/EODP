from common.io.writeToa import readToa
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import os

# Directorio raiz del EODP (sirve en Linux y en Windows)
eodp_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
e2e_dir = eodp_dir + '/data/EODP-TS-E2E/'

bands = ['VNIR-0','VNIR-1','VNIR-2','VNIR-3']
phases = ['_isrf', '_optical', '_e', '_detection', '']
threshold = 1e-3 # [%] maxima diferencia relativa permitida

#Cross validation del ISM y del L1B con las salidas de referencia (ism_out y l1b_out)
for band in bands:
    print('Processing band ' + band)
    for phase in phases:
        toa = readToa(e2e_dir + 'output_ism/', 'ism_toa' + phase + '_' + band + '.nc')
        toa_validation = readToa(e2e_dir + 'ism_out/', 'ism_toa' + phase + '_' + band + '.nc')
        mask = toa_validation != 0
        diff = np.max(np.abs(toa[mask] - toa_validation[mask])/np.abs(toa_validation[mask]))*100
        print('ISM' + phase + ' difference: ' + str(diff) + ' %')
        if diff > threshold:
            print('Difference exceeds threshold of ' + str(threshold) + ' %')
    #en DN la diferencia es de cuantificacion: compruebo que como mucho es 1 DN
    diff_dn = np.abs(toa - toa_validation)
    print('Final DN: max difference ' + str(np.max(diff_dn)) + ' DN in ' + str(np.sum(diff_dn > 0)) + ' of ' + str(toa.size) + ' pixels')

    toa_l1b = readToa(e2e_dir + 'output_l1b/', 'l1b_toa_' + band + '.nc')
    toa_l1b_validation = readToa(e2e_dir + 'l1b_out/', 'l1b_toa_' + band + '.nc')
    diff = np.max(np.abs(toa_l1b - toa_l1b_validation)/np.abs(toa_l1b_validation))*100
    print('L1B difference: ' + str(diff) + ' %, pixels above threshold: ' +
          str(np.sum(np.abs(toa_l1b - toa_l1b_validation)/np.abs(toa_l1b_validation)*100 > threshold)))

# Mapas de la TOA del L1B (como en Panoply) y sus histogramas (como hist_E2E)
plt.figure(figsize=(16, 14))
for i, band in enumerate(bands):
    toa_l1b = readToa(e2e_dir + 'output_l1b/', 'l1b_toa_' + band + '.nc')
    plt.subplot(2, 2, i+1)
    plt.imshow(toa_l1b, cmap='jet', origin='lower', aspect='auto')
    plt.colorbar(label='TOA [mW/m2/sr]')
    plt.title('L1B TOA ' + band)
    plt.xlabel('ACT pixel [-]')
    plt.ylabel('ALT line [-]')
plt.tight_layout()
plt.savefig(e2e_dir + 'output_l1b/l1b_toa_maps.png')
plt.close()

plt.figure(figsize=(16, 10))
for i, band in enumerate(bands):
    toa_l1b = readToa(e2e_dir + 'output_l1b/', 'l1b_toa_' + band + '.nc')
    print(band + ' L1B TOA min ' + str(np.min(toa_l1b)) + ', max ' + str(np.max(toa_l1b)) + ', mean ' + str(np.mean(toa_l1b)))
    plt.subplot(2, 2, i+1)
    plt.hist(toa_l1b.flatten(), bins=256, color='grey', edgecolor='black', linewidth=0.3)
    plt.title('Raster Histogram ' + band)
    plt.xlabel('Pixel Value [mW/m2/sr]')
    plt.ylabel('Frequency')
    plt.grid()
plt.tight_layout()
plt.savefig(e2e_dir + 'output_l1b/l1b_toa_histograms.png')
plt.close()

# Altitud sobre el mapa, del fichero de geometria del GM
dset = Dataset(e2e_dir + 'gm_out/geolocation.nc')
lat = np.array(dset.groups['projection'].variables['latitude'][:])
lon = np.array(dset.groups['projection'].variables['longitude'][:])
alt = np.array(dset.groups['projection'].variables['altitude'][:])
dset.close()
print('Altitude min ' + str(np.min(alt)) + ' m, max ' + str(np.max(alt)) + ' m')

plt.figure(figsize=(10, 12))
plt.scatter(lon[::2, ::2], lat[::2, ::2], c=alt[::2, ::2], cmap='terrain', s=1)
plt.colorbar(label='Altitude [m]')
plt.title('Altitude over the map')
plt.xlabel('Longitude [deg]')
plt.ylabel('Latitude [deg]')
plt.grid()
plt.savefig(e2e_dir + 'output_l1c/altitude_map.png')
plt.close()

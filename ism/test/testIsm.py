from common.io.writeToa import readToa
import numpy as np
import matplotlib.pyplot as plt

bands = ['VNIR-0','VNIR-1','VNIR-2','VNIR-3']
phases = ['_isrf', '_optical', '_e', '_prnu', '_ds', '_detection', '']
band_error = np.zeros((len(bands),len(phases)))
threshold = 1e-6
for band in bands:
    for phase in phases:
        print('Processing band ' + band)
        toa = readToa('/home/daniel/EODP/data/EODP-TS-ISM/output_validation/', 'ism_toa' + phase + '_' + band + '.nc')
        toa_central = toa[int(toa.shape[0]/2)][int(toa.shape[1]/2)]
        toa_validation = readToa('/home/daniel/EODP/data/EODP-TS-ISM/output/', 'ism_toa' + phase + '_' + band + '.nc')
        toa_validation_central = toa_validation[int(toa_validation.shape[0]/2)][int(toa_validation.shape[1]/2)]
        diff = np.max(np.abs(toa - toa_validation)/np.abs(toa_validation))*1e6


        print('TOA central pixel: ' + str(toa_central))
        print('TOA validation central pixel: ' + str(toa_validation_central))
        print('Difference: ' + str(diff))
        if abs(diff) > threshold:
            print('Difference exceeds threshold of ' + str(threshold))
            band_error[bands.index(band), phases.index(phase)] = abs(diff)

print('\n===Finished processing all bands and phases.===')
band_width = 10
col_width = 14

phases_labels = ['isrf', 'optical', 'e', 'prnu', 'ds', 'detection', 'final']

print("Band error matrix:")
print(f'{"":<{band_width}}', end='') # Espacio vacío para la esquina superior izquierda
for phase in phases_labels:
    print(f'{phase:>{col_width}}', end='')
print()

for i, band in enumerate(bands):
    print(f'{band:<{band_width}}', end='')
    for j in range(len(phases)):
        # Formateamos el número junto con "ppm" y alineamos el string completo
        val_str = f"{band_error[i, j]:.2f} ppm"
        print(f'{val_str:>{col_width}}', end='')
    print()

plt.imshow(band_error, cmap='gray')
#seteo techo de colorbar entre 0 y 100%
plt.clim(0, 1)
plt.colorbar(label='Difference [ppm]')
#muestro numeros encima de cada celda
for i in range(len(bands)):
    for j in range(len(phases)):
        plt.text(j, i, str(np.round(band_error[i,j], 2))+'ppm', ha='center', va='center', color='red')
phases_labels = ['isrf', 'optical', 'e', 'prnu', 'ds', 'detection', 'final']
plt.xticks(np.arange(len(phases)), phases_labels)
plt.yticks(np.arange(len(bands)), bands)
plt.show()
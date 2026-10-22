from common.io.writeToa import readToa
import numpy as np
import matplotlib.pyplot as plt
import os

# Directorio raiz del EODP (sirve en Linux y en Windows)
eodp_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
input_dir = eodp_dir + '/data/EODP-TS-L1B/input/'
output_dir = eodp_dir + '/data/EODP-TS-L1B/output/'
validation_dir = eodp_dir + '/data/EODP-TS-L1B/output_validation/'

bands = ['VNIR-0','VNIR-1','VNIR-2','VNIR-3']
threshold = 1e-3 # [%] maxima diferencia relativa permitida

plt.figure(figsize=(16, 10))
for i, band in enumerate(bands):
    print('Processing band ' + band)
    toa_equalized_val = readToa(validation_dir, 'l1b_toa_' + band + '.nc')
    toa_equalized = readToa(output_dir + 'equalized/', 'l1b_toa_' + band + '.nc') #black
    toa_final = readToa(input_dir, 'ism_toa_isrf_' + band + '.nc') #blue
    toa_non_equalized = readToa(output_dir + 'non_equalized/', 'l1b_toa_' + band + '.nc') #red

    #Cross validation
    print('Cross validation for band ' + band)
    diff = np.max(np.abs(toa_equalized_val - toa_equalized)/np.abs(toa_equalized_val))*100
    if diff < threshold:
        print('Equalized TOA matches the reference TOA within the threshold of ' + str(threshold) + ' %')
    else:
        print('Equalized TOA does NOT match the reference TOA within the threshold of ' + str(threshold) + ' %')
    print('Difference: ' + str(diff) + ' %')

    #ARA (azul-negro) y SNR (ruido = rojo-negro)
    #quito el pixel malo (col 5), los bordes (efecto de la FFT de la MTF) y los pixeles a 0 o saturados (4095 DN)
    toa_dn = readToa(input_dir, 'ism_toa_' + band + '.nc')
    valid = (toa_final > 0) & (toa_dn < 4095)
    valid[:, [0, 1, 5, -2, -1]] = False
    ara = np.abs(toa_equalized[valid] - toa_final[valid])/toa_final[valid]*100
    noise = toa_non_equalized[valid] - toa_equalized[valid]
    snr = np.mean(toa_equalized[valid])/np.std(noise)
    print('ARA mean: ' + str(np.mean(ara)) + ' %, ARA median: ' + str(np.median(ara)) + ' %')
    print('Noise std: ' + str(np.std(noise)) + ' mW/m2/sr, SNR: ' + str(snr))

    #Plotting (figura 8-3 del ATBD)
    idalt = int(toa_final.shape[0]/2)
    plt.subplot(2, 2, i+1)
    plt.title('Band ' + band)
    plt.plot(toa_final[idalt,:], label='Final TOA (ISM input)', color='blue')
    plt.plot(toa_equalized[idalt,:], label='Equalized TOA', color='black')
    plt.plot(toa_non_equalized[idalt,:], label='Non-Equalized TOA', color='red')
    plt.legend()
    plt.xlabel('ACT pixel [-]')
    plt.ylabel('TOA [mW/m2/sr]')
    plt.grid()

plt.tight_layout()
plt.savefig(output_dir + 'l1b_toa_comparison.png')
plt.close()

from common.io.writeToa import readToa
from common.io.readMat import readMat
from config.ismConfig import ismConfig
from ism.src.mtf import mtf
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

# Directorio raiz del EODP (sirve en Linux y en Windows)
eodp_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
output_dir = eodp_dir + '/data/EODP-TS-ISM/output/'
validation_dir = eodp_dir + '/data/EODP-TS-ISM/output_validation/'

bands = ['VNIR-0','VNIR-1','VNIR-2','VNIR-3']
phases = ['_isrf', '_optical', '_e', '_prnu', '_ds', '_detection', '']
band_error = np.zeros((len(bands),len(phases)))
threshold = 1e-3 # [%] maxima diferencia relativa permitida

#Cross validation
for band in bands:
    for phase in phases:
        print('Processing band ' + band + ' phase ' + phase)
        toa = readToa(output_dir, 'ism_toa' + phase + '_' + band + '.nc')
        toa_central = toa[int(toa.shape[0]/2)][int(toa.shape[1]/2)]
        toa_validation = readToa(validation_dir, 'ism_toa' + phase + '_' + band + '.nc')
        toa_validation_central = toa_validation[int(toa_validation.shape[0]/2)][int(toa_validation.shape[1]/2)]
        #diferencia relativa en %, sin los pixeles a 0 para no dividir entre 0
        mask = toa_validation != 0
        diff = np.max(np.abs(toa[mask] - toa_validation[mask])/np.abs(toa_validation[mask]))*100
        band_error[bands.index(band), phases.index(phase)] = diff

        print('TOA central pixel: ' + str(toa_central))
        print('TOA validation central pixel: ' + str(toa_validation_central))
        print('Difference: ' + str(diff) + ' %')
        if diff > threshold:
            print('Difference exceeds threshold of ' + str(threshold) + ' %')

print('\n===Finished processing all bands and phases.===')
band_width = 10
col_width = 14

phases_labels = ['isrf', 'optical', 'e', 'prnu', 'ds', 'detection', 'final']

print("Band error matrix [%]:")
print(f'{"":<{band_width}}', end='') # Espacio vacío para la esquina superior izquierda
for phase in phases_labels:
    print(f'{phase:>{col_width}}', end='')
print()

for i, band in enumerate(bands):
    print(f'{band:<{band_width}}', end='')
    for j in range(len(phases)):
        val_str = f"{band_error[i, j]:.2e}"
        print(f'{val_str:>{col_width}}', end='')
    print()
if np.all(band_error < threshold):
    print('ALL ISM OUTPUTS MATCH THE REFERENCE WITHIN ' + str(threshold) + ' %')

plt.figure(figsize=(12, 5))
plt.imshow(band_error, cmap='gray')
#seteo techo de colorbar en el threshold
plt.clim(0, threshold)
plt.colorbar(label='Max relative difference [%]')
#muestro numeros encima de cada celda
for i in range(len(bands)):
    for j in range(len(phases)):
        plt.text(j, i, f'{band_error[i,j]:.1e}%', ha='center', va='center', color='red')
plt.xticks(np.arange(len(phases)), phases_labels)
plt.yticks(np.arange(len(bands)), bands)
plt.title('ISM cross validation')
plt.savefig(output_dir + 'ism_crossvalidation.png')
plt.close()

# Factores de conversion analiticos de cada paso: radiancia -> irradiancia -> fotones -> electrones -> voltaje -> DN
config = ismConfig()
h = 6.62606896e-34  # Planck constant [J*s]
c = 2.99792458e8  # Speed of light [m/s]
print('\n===Conversion factors===')
print(f'{"":<10}{"rad2irrad":>12}{"irrad2ph":>12}{"ph2e":>8}{"e2v":>12}{"v2dn":>10}{"total":>10}{"1/total":>12}')
for i, band in enumerate(bands):
    k_rad2irrad = config.Tr * np.pi/4 * (config.D/config.f)**2 # [mW/m2] por cada [mW/m2/sr]
    k_irrad2ph = config.pix_size**2 * config.t_int * 1e-3 / (h*c/config.wv[i]) # [ph] por cada [mW/m2]
    k_ph2e = config.QE # [e-/ph]
    k_e2v = config.OCF * config.ADC_gain # [V/e-]
    k_v2dn = (2**config.bit_depth - 1)/(config.max_voltage - config.min_voltage) # [DN/V]
    k_total = k_rad2irrad * k_irrad2ph * k_ph2e * k_e2v * k_v2dn # [DN] por cada [mW/m2/sr]
    print(f'{band:<10}{k_rad2irrad:>12.4e}{k_irrad2ph:>12.4e}{k_ph2e:>8.2f}{k_e2v:>12.4e}{k_v2dn:>10.2f}{k_total:>10.4f}{1/k_total:>12.4e}')

    #comprobacion con las salidas, solo en pixeles no saturados
    toa_optical = readToa(output_dir, 'ism_toa_optical_' + band + '.nc')
    toa_e = readToa(output_dir, 'ism_toa_e_' + band + '.nc')
    no_sat = (toa_e < config.FWC) & (toa_optical > 0)
    print('Check e-/irradiance from the outputs: ' + str(np.median(toa_e[no_sat]/toa_optical[no_sat])) +
          ' (analytical ' + str(k_irrad2ph * k_ph2e) + '), saturated pixels: ' + str(np.sum(~no_sat)))

# MTF del sistema con sus 6 contribuciones (como la figura 7-51 del ATBD)
myMtf = mtf(logging.getLogger('ISM'), output_dir)
nlines, ncolumns = toa.shape
mtf_names = ['Hdiff', 'Hdefoc', 'Hwfe', 'Hdet', 'Hsmear', 'Hmotion', 'Hsys']
mtf_labels = ['Diffraction MTF', 'Defocus MTF', 'WFE Aberrations MTF', 'Detector MTF', 'Smearing MTF', 'Motion Blur MTF']
print('\n===MTF===')
fig, axs = plt.subplots(2, 4, figsize=(22, 10))
for i, band in enumerate(bands):
    fn2D, fr2D, fnAct, fnAlt = myMtf.freq2d(nlines, ncolumns, config.D, config.wv[i], config.f, config.pix_size)
    Hdiff = myMtf.mtfDiffract(fr2D)
    Hdefoc = myMtf.mtfDefocus(fr2D, config.defocus, config.f, config.D)
    Hwfe = myMtf.mtfWfeAberrations(fr2D, config.wv[i], config.kLF, config.wLF, config.kHF, config.wHF)
    Hdet = myMtf.mtfDetector(fn2D)
    Hsmear = myMtf.mtfSmearing(fnAlt, ncolumns, config.ksmear)
    Hmotion = myMtf.mtfMotion(fn2D, config.kmotion)
    Hsys = Hdiff * Hdefoc * Hwfe * Hdet * Hsmear * Hmotion
    mtfs = [Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion, Hsys]

    #Cross validation de las MTFs
    for name, H in zip(mtf_names, mtfs):
        H_validation = readMat(validation_dir, name + '_' + band + '.nc')
        print(name + ' ' + band + ' max difference: ' + str(np.max(np.abs(H - H_validation))))

    #cortes ACT (fila central, fnAlt=0) y ALT (columna central, fnAct=0)
    for H, label in zip(mtfs[:-1], mtf_labels):
        axs[0, i].plot(fnAct, H[nlines//2, :], label=label)
        axs[1, i].plot(fnAlt, H[:, ncolumns//2], label=label)
    axs[0, i].plot(fnAct, Hsys[nlines//2, :], label='System MTF', linewidth=2, color='black')
    axs[1, i].plot(fnAlt, Hsys[:, ncolumns//2], label='System MTF', linewidth=2, color='black')

    #valor en Nyquist (la primera frecuencia es -0.5) y MTF que limita el sistema
    nyquist_act = [H[nlines//2, 0] for H in mtfs]
    nyquist_alt = [H[0, ncolumns//2] for H in mtfs]
    limit_act = mtf_labels[np.argmin(nyquist_act[:-1])]
    limit_alt = mtf_labels[np.argmin(nyquist_alt[:-1])]
    print(band + ' MTF at Nyquist ACT: ' + ', '.join(f'{v:.4f}' for v in nyquist_act) + ' -> limited by ' + limit_act)
    print(band + ' MTF at Nyquist ALT: ' + ', '.join(f'{v:.4f}' for v in nyquist_alt) + ' -> limited by ' + limit_alt)

    axs[0, i].set_title(band + ' - slice ALT (ACT frequencies)')
    axs[1, i].set_title(band + ' - slice ACT (ALT frequencies)')
    for ax in axs[:, i]:
        ax.axvline(x=0.5, color='black', linestyle='--')
        ax.set_xlim([0, 0.5])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel('Spatial Frequency (f/(1/w)) [-]')
        ax.set_ylabel('MTF')
        ax.grid()
axs[0, 0].legend()
plt.tight_layout()
plt.savefig(output_dir + 'system_mtf_contributors_all_bands.png')
plt.close()

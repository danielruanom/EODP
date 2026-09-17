from common.io.writeToa import readToa
import numpy as np
import matplotlib.pyplot as plt

bands = ['VNIR-0','VNIR-1','VNIR-2','VNIR-3']
threshold = 1e-6
for band in bands:
    print('Processing band ' + band)
    toa_equalized_val = readToa('/home/daniel/EODP/data/EODP-TS-L1B/output_validation/', 'l1b_toa_' + band + '.nc')
    toa_equalized = readToa('/home/daniel/EODP/data/EODP-TS-L1B/output/equalized/', 'l1b_toa_' + band + '.nc') #black
    toa_final = readToa('/home/daniel/EODP/data/EODP-TS-L1B/input/', 'ism_toa_isrf_' + band + '.nc') #blue
    toa_non_equalized = readToa('/home/daniel/EODP/data/EODP-TS-L1B/output/non_equalized/', 'l1b_toa_' + band + '.nc') #red

    #Cross validation
    print('Cross validation for band ' + band)
    diff = np.sum(np.abs(toa_equalized_val-toa_equalized))
    if diff < threshold:
        print('Equalized TOA matches the reference TOA within the threshold of ' + str(threshold))
    else:
        print('Equalized TOA does NOT match the reference TOA within the threshold of ' + str(threshold))
        print('Difference: ' + str(diff))

    #Plotting
    plt.figure(figsize=(10, 6))
    plt.title('Band ' + band)
    plt.plot(toa_final[0,:], label='Final TOA', color='blue')
    plt.plot(toa_equalized[0,:], label='Equalized TOA', color='black')
    plt.plot(toa_non_equalized[0,:], label='Non-Equalized TOA', color='red')
    plt.legend()
    plt.xlabel('ACT pixel [-]')
    plt.ylabel('TOA [W/m2/sr]')
    plt.grid()
    plt.savefig('/home/daniel/EODP/data/EODP-TS-L1B/output/' + band + '_toa_comparison.png')

    input('Press Enter to continue to the next band...')
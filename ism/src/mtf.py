from math import pi
from config.ismConfig import ismConfig
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import j1
from numpy.matlib import repmat
from common.io.readMat import writeMat
from common.plot.plotMat2D import plotMat2D
from scipy.interpolate import interp2d
from numpy.fft import fftshift, ifft2
import os

class mtf:
    """
    Class MTF. Collects the analytical modelling of the different contributions
    for the system MTF
    """
    def __init__(self, logger, outdir):
        self.ismConfig = ismConfig()
        self.logger = logger
        self.outdir = outdir

    def system_mtf(self, nlines, ncolumns, D, lambd, focal, pix_size,
                   kLF, wLF, kHF, wHF, defocus, ksmear, kmotion, directory, band):
        """
        System MTF
        :param nlines: Lines of the TOA
        :param ncolumns: Columns of the TOA
        :param D: Telescope diameter [m]
        :param lambd: central wavelength of the band [m]
        :param focal: focal length [m]
        :param pix_size: pixel size in meters [m]
        :param kLF: Empirical coefficient for the aberrations MTF for low-frequency wavefront errors [-]
        :param wLF: RMS of low-frequency wavefront errors [m]
        :param kHF: Empirical coefficient for the aberrations MTF for high-frequency wavefront errors [-]
        :param wHF: RMS of high-frequency wavefront errors [m]
        :param defocus: Defocus coefficient (defocus/(f/N)). 0-2 low defocusing
        :param ksmear: Amplitude of low-frequency component for the motion smear MTF in ALT [pixels]
        :param kmotion: Amplitude of high-frequency component for the motion smear MTF in ALT and ACT
        :param directory: output directory
        :return: mtf
        """

        self.logger.info("Calculation of the System MTF")

        # Calculate the 2D relative frequencies
        self.logger.debug("Calculation of 2D relative frequencies")
        fn2D, fr2D, fnAct, fnAlt = self.freq2d(nlines, ncolumns, D, lambd, focal, pix_size)

        # Diffraction MTF
        self.logger.debug("Calculation of the diffraction MTF")
        Hdiff = self.mtfDiffract(fr2D)

        # Defocus
        Hdefoc = self.mtfDefocus(fr2D, defocus, focal, D)

        # WFE Aberrations
        Hwfe = self.mtfWfeAberrations(fr2D, lambd, kLF, wLF, kHF, wHF)

        # Detector
        Hdet  = self. mtfDetector(fn2D)

        # Smearing MTF
        Hsmear = self.mtfSmearing(fnAlt, ncolumns, ksmear)

        # Motion blur MTF
        Hmotion = self.mtfMotion(fn2D, kmotion)

        # Calculate the System MTF
        self.logger.debug("Calculation of the Sysmtem MTF by multiplying the different contributors")
        Hsys = Hdiff * Hdefoc * Hwfe * Hdet * Hsmear * Hmotion

        # Plot cuts ACT/ALT of the MTF
        self.plotMtf(Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion, Hsys, nlines, ncolumns, fnAct, fnAlt, directory, band)


        return Hsys

    def freq2d(self,nlines, ncolumns, D, lambd, focal, w):
        """
        Calculate the relative frequencies 2D (for the diffraction MTF)
        :param nlines: Lines of the TOA
        :param ncolumns: Columns of the TOA
        :param D: Telescope diameter [m]
        :param lambd: central wavelength of the band [m]
        :param focal: focal length [m]
        :param w: pixel size in meters [m]
        :return fn2D: normalised frequencies 2D (f/(1/w))
        :return fr2D: relative frequencies 2D (f/(1/fc))
        :return fnAct: 1D normalised frequencies 2D ACT (f/(1/w))
        :return fnAlt: 1D normalised frequencies 2D ALT (f/(1/w))
        """
        fs = 1.0 / w
        fc = D / (lambd * focal)
        fstepAlt = 1/nlines/w 
        fstepAct = 1/ncolumns/w 
        eps = 1e-9
        fAlt = np.arange(-1/(2*w),1/(2*w)-eps,fstepAlt)  
        fAct = np.arange(-1/(2*w),1/(2*w)-eps,fstepAct)  
        fnAlt = fAlt/fs
        fnAct = fAct/fs
        fnAlt2D, fnAct2D = np.meshgrid(fnAlt, fnAct, indexing='ij')
        fn2D = np.sqrt(fnAct2D**2 + fnAlt2D**2)
        fr2D = fn2D * (fs / fc)
        return fn2D, fr2D, fnAct, fnAlt

    def mtfDiffract(self,fr2D):
        """
        Optics Diffraction MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :return: diffraction MTF
        """
        Hdiff = 2/np.pi * (np.arccos(fr2D) - fr2D * np.sqrt(1 - fr2D**2))
        Hdiff[fr2D*fr2D>1] = 0
        return Hdiff


    def mtfDefocus(self, fr2D, defocus, focal, D):
        """
        Defocus MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :param defocus: Defocus coefficient (defocus/(f/N)). 0-2 low defocusing
        :param focal: focal length [m]
        :param D: Telescope diameter [m]
        :return: Defocus MTF
        """
        x = np.pi * defocus * fr2D * (1 - fr2D)
        Hdefoc = np.zeros(fr2D.shape)
        Hdefoc[fr2D*fr2D>0] =2/x[fr2D*fr2D>0] * j1(x[fr2D*fr2D>0]) # 2/x[fr2D*fr2D>0] * (x[fr2D*fr2D>0]/2 - x[fr2D*fr2D>0]**3/16 + x[fr2D*fr2D>0]**5/384 - x[fr2D*fr2D>0]**7/18432) #Approx version
        Hdefoc[fr2D*fr2D>1] = 0
        Hdefoc[fr2D==0] = 1
        return Hdefoc

    def mtfWfeAberrations(self, fr2D, lambd, kLF, wLF, kHF, wHF):
        """
        Wavefront Error Aberrations MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :param lambd: central wavelength of the band [m]
        :param kLF: Empirical coefficient for the aberrations MTF for low-frequency wavefront errors [-]
        :param wLF: RMS of low-frequency wavefront errors [m]
        :param kHF: Empirical coefficient for the aberrations MTF for high-frequency wavefront errors [-]
        :param wHF: RMS of high-frequency wavefront errors [m]
        :return: WFE Aberrations MTF
        """
        Hwfe = np.exp(-fr2D*(1-fr2D)*(kLF*(wLF/lambd)**2 + kHF*(wHF/lambd)**2))
        Hwfe[fr2D*fr2D>1] = 0
        return Hwfe

    def mtfDetector(self,fn2D):
        """
        Detector MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :return: detector MTF
        """
        Hdet = np.abs(np.sinc(fn2D))
        return Hdet

    def mtfSmearing(self, fnAlt, ncolumns, ksmear):
        """
        Smearing MTF
        :param ncolumns: Size of the image ACT
        :param fnAlt: 1D normalised frequencies 2D ALT (f/(1/w))
        :param ksmear: Amplitude of low-frequency component for the motion smear MTF in ALT [pixels]
        :return: Smearing MTF
        """
        Hsmear = np.zeros((len(fnAlt), ncolumns))
        for i in range(ncolumns):
            Hsmear[:, i] = np.sinc(ksmear * fnAlt)
        return Hsmear

    def mtfMotion(self, fn2D, kmotion):
        """
        Motion blur MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :param kmotion: Amplitude of high-frequency component for the motion smear MTF in ALT and ACT
        :return: detector MTF
        """
        Hmotion = np.sinc(kmotion * fn2D)
        return Hmotion

    def plotMtf(self,Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion, Hsys, nlines, ncolumns, fnAct, fnAlt, directory, band):
        """
        Plotting the system MTF and all of its contributors
        :param Hdiff: Diffraction MTF
        :param Hdefoc: Defocusing MTF
        :param Hwfe: Wavefront electronics MTF
        :param Hdet: Detector MTF
        :param Hsmear: Smearing MTF
        :param Hmotion: Motion blur MTF
        :param Hsys: System MTF
        :param nlines: Number of lines in the TOA
        :param ncolumns: Number of columns in the TOA
        :param fnAct: normalised frequencies in the ACT direction (f/(1/w))
        :param fnAlt: normalised frequencies in the ALT direction (f/(1/w))
        :param directory: output directory
        :param band: band
        :return: N/A
        """
        plt.imshow(Hsys, origin='lower', cmap='jet')
        plt.colorbar(label='MTF')
        plt.title(f'System MTF for band {band}')
        plt.xlabel('ACT')
        plt.ylabel('ALT')
        plt.savefig(os.path.join(directory, f'system_mtf_2d_{band}.png'))

        plt.figure(figsize=(10, 6))
        plt.plot(fnAct, Hdiff[nlines//2, :], label='Diffraction MTF')
        plt.plot(fnAct, Hdefoc[nlines//2, :], label='Defocus MTF')
        plt.plot(fnAct, Hwfe[nlines//2, :], label='WFE Aberrations MTF')
        plt.plot(fnAct, Hdet[nlines//2, :], label='Detector MTF')
        plt.plot(fnAct, Hsmear[nlines//2, :], label='Smearing MTF')
        plt.plot(fnAct, Hmotion[nlines//2, :], label='Motion Blur MTF')
        plt.plot(fnAct, Hsys[nlines//2, :], label='System MTF', linewidth=2, color='black')
        plt.axvline(x=0.5, color='black', linestyle='--', label='Nyquist Frequency')
        plt.title(f'System MTF - slice ALT for {band}')
        plt.xlim([0, 0.5])
        plt.xlabel('Spatial Frequency (f/(1/w)) [-]')
        plt.ylabel('MTF')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(directory, f'system_mtf_cutAct_{band}.png'))

        plt.figure(figsize=(10, 6))
        plt.plot(fnAlt, Hdiff[:, ncolumns//2], label='Diffraction MTF')
        plt.plot(fnAlt, Hdefoc[:, ncolumns//2], label='Defocus MTF')
        plt.plot(fnAlt, Hwfe[:, ncolumns//2], label='WFE Aberrations MTF')
        plt.plot(fnAlt, Hdet[:, ncolumns//2], label='Detector MTF')
        plt.plot(fnAlt, Hsmear[:, ncolumns//2], label='Smearing MTF')
        plt.plot(fnAlt, Hmotion[:, ncolumns//2], label='Motion Blur MTF')
        plt.plot(fnAlt, Hsys[:, ncolumns//2], label='System MTF', linewidth=2, color='black')
        plt.axvline(x=0.5, color='black', linestyle='--', label='Nyquist Frequency')
        plt.title(f'System MTF - slice ACT for {band}')
        plt.xlim([0, 0.5])
        plt.xlabel('Spatial Frequency (f/(1/w)) [-]')
        plt.ylabel('MTF')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(directory, f'system_mtf_cutAlt_{band}.png'))



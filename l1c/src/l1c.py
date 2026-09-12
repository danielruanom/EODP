
# LEVEL-1C

from l1c.src.initL1c import initL1c
from common.io.writeToa import writeToa, readToa
from common.io.readGeodetic import readGeodetic, getCorners
import mgrs
import numpy as np
from scipy.interpolate import bisplrep, bisplev
import matplotlib.pyplot as plt
from common.io.l1cProduct import writeL1c
from matplotlib import cm

class l1c(initL1c):

    def __init__(self, auxdir, indir, outdir):
        super().__init__(auxdir, indir, outdir)

    def processModule(self):

        self.logger.info("Start of the L1C Processing Module")

        for band in self.globalConfig.bands:

            self.logger.info("Start of BAND " + band)

            # Read TOA - output of the L1B in Radiances
            # -------------------------------------------------------------------------------
            toa = readToa(self.l1bdir, self.globalConfig.l1b_toa + band + '.nc')
            lat,lon = readGeodetic(self.gmdir, self.globalConfig.gm_geoloc)
            self.checkSize(lat,toa)

            # L1C reprojection onto the MGRS grid
            # -------------------------------------------------------------------------------
            lat_l1c, lon_l1c, toa_l1c = self.l1cProjtoa(lat, lon, toa, band)

            # Write output TOA
            # -------------------------------------------------------------------------------
            writeL1c(self.outdir, self.globalConfig.l1c_toa + band, lat_l1c, lon_l1c, toa_l1c)

            # Plot the L1B and L1C grids
            # -------------------------------------------------------------------------------
            self.plotProjGrid(lat, lon, lat_l1c, lon_l1c, band)

            # Plot the L1C TOA
            # -------------------------------------------------------------------------------
            self.plotProjtoa(lat_l1c, lon_l1c, toa_l1c, band)

            self.logger.info("End of BAND " + band)

        self.logger.info("End of the L1C Module!")


    def l1cProjtoa(self, lat, lon, toa, band):
        '''
        This function reprojects the L1B radiances into the MGRS grid.

        The MGRS reference system
        https://www.bluemarblegeo.com/knowledgebase/calculator-2020/Military_Grid_Reference_System_(MGRS).htm
        MGRS: '31REQ4367374067'
        31 is the UTM zone, R is the UTM latitude band; EQ are the MGRS column and row band letters
        43673 is the MGRS Easting (5 dig); 74067 is the MGRS Northing (5dig)

        Python mgrs library documentation
        https://pypi.org/project/mgrs/

        :param lat: L1B latitudes [deg]
        :param lon: L1B longitudes [deg]
        :param toa: L1B radiances
        :param band: band
        :return: L1C radiances, L1C latitude and longitude in degrees
        '''
        tck = bisplrep(lat.flatten(), lon.flatten(), toa.flatten())
        self.logger.info("Create interpolant for the TOA radiances")
        m = mgrs.MGRS()
        mgrs_tiles = set([])
        
        for lat_i, lon_i in zip(lat.flatten(), lon.flatten()):
            mgrs_code = m.toMGRS(lat_i, lon_i, MGRSPrecision=self.l1cConfig.mgrs_tile_precision)
            mgrs_tiles.add(str(mgrs_code))
        self.logger.info("Create a unique set of all the MGRS tiles in the image")
        self.logger.debug(f"{len(mgrs_tiles)} MGRS tiles found: {', '.join(mgrs_tiles)}")

        mgrs_tiles = list(mgrs_tiles)
        lat_l1c = np.zeros(len(mgrs_tiles))
        lon_l1c = np.zeros(len(mgrs_tiles))
        toa_l1c = np.zeros(len(mgrs_tiles))
        for i, mgrs_tile in enumerate(mgrs_tiles):
            lat_l1c[i], lon_l1c[i] = m.toLatLon(mgrs_tile)
            toa_l1c[i] = bisplev(lat_l1c[i], lon_l1c[i], tck)
        self.logger.info("Iterate for each MGRS tile found")

        toa_l1c = np.clip(toa_l1c, np.nanmin(toa), np.nanmax(toa)) # Clip the TOA values to the min and max of the original TOA to avoid extrapolation artifacts

        return lat_l1c, lon_l1c, toa_l1c

    def checkSize(self, lat,toa):
        '''
        Check the sizes of the input radiances and geodetic coordinates.
        If they don't match, exit.
        :param lat: Latitude 2D matrix
        :param toa: Radiance 2D matrix
        :return: NA
        '''
        if lat.shape != toa.shape:
            self.logger.error("The size of the geodetic coordinates and the radiances do not match. Exiting.")
            self.logger.error("Size of the geodetic coordinates: " + str(lat.shape))
            self.logger.error("Size of the radiances: " + str(toa.shape))
            exit(1)

    def plotProjGrid(self, lat_l1b, lon_l1b, lat_l1c, lon_l1c, band):
        '''
        Plot the L1B boundary and L1C grids for a given band.
        :param lat_l1b: L1B latitudes
        :param lon_l1b: L1B longitudes
        :param lat_l1c: L1C latitudes
        :param lon_l1c: L1C longitudes
        :param band: band
        :return: NA
        '''
        lon_l1b_boundary = np.concatenate([lon_l1b[0, :], lon_l1b[:, -1], lon_l1b[-1, ::-1], lon_l1b[::-1, 0]])
        lat_l1b_boundary = np.concatenate([lat_l1b[0, :], lat_l1b[:, -1], lat_l1b[-1, ::-1], lat_l1b[::-1, 0]])

        plt.figure(figsize=(16, 8))
        plt.plot(lon_l1b_boundary, lat_l1b_boundary, color='black', linewidth=1.5, label='L1B')
        plt.scatter(lon_l1c, lat_l1c, color='red', s=5, label='L1C MGRS')
        plt.title('Projection on ground')
        plt.title('Projection on ground', fontsize=20)
        plt.xlabel('Longitude [deg]', fontsize=16)
        plt.ylabel('Latitude [deg]', fontsize=16)
        plt.xlim([np.min(lon_l1b_boundary)-0.02, np.max(lon_l1b_boundary)+0.02])
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.savefig(self.outdir + '/footprint_' + band + '.png')

    def plotProjtoa(self, lat_l1c, lon_l1c, toa_l1c, band):
        '''
        Plot the L1C radiances for a given band.
        :param lat_l1c: L1C latitudes
        :param lon_l1c: L1C longitudes
        :param toa_l1c: L1C radiances
        :param band: band
        :return: NA
        '''
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor('#808080')
        sc = ax.scatter(lon_l1c, lat_l1c, c=toa_l1c, cmap='jet', s=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('toa', fontsize=12, pad=10)
        cbar = fig.colorbar(sc, ax=ax, orientation='horizontal', pad=0.08, shrink=0.7, extend='both')
        cbar.ax.set_title('toa (mW/m2/sr)', fontsize=10, pad=5)
        min_val = np.nanmin(toa_l1c)
        max_val = np.nanmax(toa_l1c)
        cbar.ax.set_xlabel(f'Data Min = {min_val:.1f}, Max = {max_val:.1f}', fontsize=8, labelpad=5)
        fig.text(0.5, 0.01, 'Figure 9-4: L1C TOA.', ha='center', fontsize=12, fontweight='bold', color='#1B365D')
        plt.tight_layout()
        plt.savefig(self.outdir + '/toa_' + band + '.png')
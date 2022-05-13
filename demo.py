import numpy as np
from datetime import datetime
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation
from astropy.coordinates.angles import Angle


def load_data(filename):
    # with open(filename) as f:
    #     line = f.readline()
    # [x, y, z] = [float(a) for a in line.split()]
    # site = EarthLocation.from_geocentric(x, y, z, unit='meter').to('kilometer')

    yr, mon, day, hr, min = np.loadtxt(filename, dtype=int,
                                       usecols=(0, 1, 2, 3, 4),
                                       skiprows=0,
                                       unpack=True)
    sec, ra, de = np.loadtxt(filename,
                             dtype=float,
                             usecols=(5, 6, 7),
                             skiprows=0,
                             unpack=True)
    pos_x, pos_y, pox_z = np.loadtxt(filename,
                             dtype=float,
                             usecols=(8, 9, 10),
                             skiprows=0,
                             unpack=True)

    # target_x, target_y, target_z,target_vx,target_vy,target_vz = np.loadtxt(filename,
    #                          dtype=float,
    #                          usecols=(11, 12, 13, 14, 15, 16),
    #                          skiprows=0,
    #                          unpack=True)

    pos = np.array([pos_x, pos_y, pox_z])
    pos = pos.transpose()

    angles = np.array([ra, de])
    angles = angles.transpose()

    # target = np.array([target_x, target_y, target_z,target_vx,target_vy,target_vz])
    # target = target.transpose()

    epochs = []
    for i in np.arange(np.size(yr)):
        s = int(sec[i])
        ms = int((sec[i]-s)*1e6)
        epoch = Time(datetime(yr[i], mon[i], day[i], hr[i], min[i], s, ms),
                     scale='utc')
        epochs = np.append(epochs, epoch)
    return pos, epochs, Angle(angles * u.deg)


def demo():

    import nessan_iod as iod
    print()
    print('obs1:')
    # site, epochs, angles = load_data('./Week2/obs1.dat')
    # site, epochs, angles = load_data('obs_simu_2.txt')
    site, epochs, angles = load_data('obs_simu_fin_00.txt')

    print(site)
    print(epochs)
    print(angles)
    print(site.shape)

    indices = np.arange(0, epochs.size, 1)

    orbit, residual = iod.iod_with_angles(epochs[indices],
                                          angles[indices],
                                          site
                                          )

    a, ecc, inc, raan, argp, nu = orbit.classical()
    a = a.to(u.m).to_value()
    ecc = ecc.to_value()
    inc = inc.to(u.deg).to_value()
    raan = raan.to(u.deg).to_value()
    argp = argp.to(u.deg).to_value()
    nu = nu.to(u.deg).to_value()

    print('Orbital elements:')
    print('   a: ', a)
    print(' ecc: ', ecc)
    print(' inc: ', inc)
    print('raan: ', raan)
    print('   w: ', argp)
    print('   M: ', nu)
    # print('mean and std of residuals:')
    # print([np.mean(residual).to_value('deg'),
    #       np.std(residual).to_value('deg')] * u.deg)

    import scipy.stats as stats
    import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 4))
    # ax = plt.subplot(1, 2, 1)
    # ax.scatter(residual[:, 0], residual[:, 1], marker='.', edgecolor=None)
    # bound = 1.1 * np.max(abs(residual.to_value()))
    # ax.set_xlim([-bound, bound])
    # ax.set_ylim([-bound, bound])
    # ax.grid(True)
    #
    # x = np.arange(-bound, bound, bound/500.0)
    # m, s = stats.norm.fit(residual[:, 0])
    # y = stats.norm.pdf(x, m, s)
    # ax = plt.subplot(2, 2, 2)
    # ax.plot(x, y, '-.')
    # ax.hist(residual[:, 0], bins=60, normed=True)
    # ax.grid(True)
    #
    # m, s = stats.norm.fit(residual[:, 1])
    # y = stats.norm.pdf(x, m, s)
    # ax = plt.subplot(2, 2, 4)
    # ax.plot(x, y, '-.')
    # ax.hist(residual[:, 1], bins=60, normed=True)
    # ax.grid(True)
    # plt.show()
    #

if __name__ == "__main__":
    demo()



#!/usr/bin/env python
import numpy as np
from astropy.table import Table
from vega.utils import find_file

'''
# DESI specific code used to generate the table
# of angular coordinates of the fiber positioners
#
# Code is commented out to avoid dependencies
# but is kept for reference
from desimeter.io import load_metrology
from desimodel.io import load_platescale
spots    = load_metrology()

print(spots.dtype.names)
print(np.unique(spots["DEVICE_TYPE"]))
print(np.unique(spots["PETAL_ID"]))

selection=(spots['DEVICE_TYPE']=="POS")&(spots['PETAL_LOC']==3)
xp=spots["X_FP"][selection]
yp=spots["Y_FP"][selection]
rp=np.sqrt(xp**2+yp**2)
platescale = load_platescale()
theta=np.interp(rp,platescale['radius'], platescale['theta'])
xp *= (theta/rp)
yp *= (theta/rp)
rpatrol = 6.*(theta/rp) # patrol radius of positioners (6mm)
thetamax=np.max(theta)
print("max radius = {:.3f} deg".format(thetamax))

t=Table()
t["FOCAL_PLANE_X_DEG"]=xp
t["FOCAL_PLANE_Y_DEG"]=yp
t["PATROL_RADIUS_DEG"]=rpatrol
t.write("desi-positioners.csv")
print("wrote desi-positioners.csv")
'''


def main():
    path = "instrumental_systematics/desi-positioners.csv"
    file = find_file(path)
    print(f"Reading {file}")
    positioner_table = Table.read(file)

    xp = positioner_table["FOCAL_PLANE_X_DEG"]
    yp = positioner_table["FOCAL_PLANE_Y_DEG"]
    rpatrol = positioner_table["PATROL_RADIUS_DEG"]

    '''
    # Commented out picca code to avoid dependencies
    # but  kept for reference

    from picca.constants import Cosmo
    OM=0.315
    OR=7.963e-5
    cosmo = Cosmo(Om=OM,Or=OR)
    Z=2.4
    comoving_distance=cosmo.get_r_comov(Z)
    print("comoving_distance=",comoving_distance)
    # picca: 3941.861037247279 Mpc/h
    '''

    comoving_distance = 3941.86  # Mpc/h
    print(f"Use a comoving distance of {comoving_distance} Mpc/h to convert angles to distance")

    print("Compute randoms...")
    nr = 50000
    x = np.random.uniform(size=nr) * (np.max(xp + rpatrol))
    y = np.random.uniform(size=nr) * (np.max(yp + rpatrol))
    ok = np.repeat(False, nr)
    for xxp, yyp, rrp in zip(xp, yp, rpatrol) :
        ok |= ((x - xxp)**2 + (y - yyp)**2) < rrp**2
    x = x[ok]
    y = y[ok]

    print("Compute correlation...")
    deg2mpc = comoving_distance * np.pi / 180.
    bins = np.linspace(0, 200, 51)
    nbins = bins.size - 1
    h0 = np.zeros(nbins)
    for xx, yy in zip(x, y):
        d = np.sqrt((xx - x)**2 + (yy - y)**2) * deg2mpc
        t, _ = np.histogram(d, bins=bins)
        h0 += t
    ok = (h0 > 0)
    rt = (bins[:-1] + (bins[1] - bins[0]) / 2)
    rt = rt[ok]
    xi = h0[ok] / rt  # number of random pairs scales as rt

    # add a value at 0, last measured bin + 1 step, and 1000 Mpc to avoid extrapolations
    xi_at_0 = (xi[0] - xi[1]) / (rt[0] - rt[1]) * (0 - rt[0]) + xi[0]  # linearly extrapolated to r=0
    rt = np.append(0, rt)
    xi = np.append(xi_at_0, xi)
    rt = np.append(rt, [rt[-1] + bins[1] - bins[0], 1000.])
    xi = np.append(xi, [0, 0])
    xi /= np.max(xi)  # norm

    t = Table()
    t["RT"] = rt
    t["XI"] = xi
    filename = "desi-instrument-syst-for-forest-auto-correlation.csv"
    t.write(filename, overwrite=True)
    print("wrote ", filename)

    '''
    # plotting
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(xp,yp,".",label="positioners")
    plt.plot(x,y,".",alpha=0.2,label="randoms")
    plt.xlabel("X (deg)")
    plt.ylabel("Y (deg)")

    plt.figure()
    title="comoving_distance={:.2f} Mpc/h".format(comoving_distance)
    plt.subplot(111,title=title)
    plt.plot(rt,xi,"-")

    m=(rt/80.-1)**2*(rt<80)
    a=np.sum(m*xi)/np.sum(m**2)
    m*=a
    plt.plot(rt,m,"--",color="gray")

    plt.xlabel("rt (Mpc/h)")
    plt.ylabel("xi")
    plt.grid()

    plt.show()
    '''


if __name__ == '__main__':
    main()

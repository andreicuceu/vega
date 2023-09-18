#!/usr/bin/env python


import sys
import numpy as np
from scipy.interpolate import interp1d
from astropy.table import Table

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

positioner_table=Table.read("desi-positioners.csv")
xp=positioner_table["FOCAL_PLANE_X_DEG"]
yp=positioner_table["FOCAL_PLANE_Y_DEG"]
rpatrol=positioner_table["PATROL_RADIUS_DEG"]

'''
# Commented out picca code to avoid dependencies
# but  kept for reference

from picca.constants import Cosmo
OM=0.315
OR=7.963e-5
cosmo = Cosmo(Om=OM,Or=OR)
Z=2.4
rcom=cosmo.get_r_comov(Z)
print("rcom=",rcom)
# picca: 3941.861037247279 Mpc/h
'''

rcom=3941.86 # Mpc/h

print("compute randoms...")
nr=50000
x=np.random.uniform(size=nr)*(np.max(xp+rpatrol))
y=np.random.uniform(size=nr)*(np.max(yp+rpatrol))
ok=np.repeat(False,nr)
for xxp,yyp,rrp in zip(xp,yp,rpatrol) :
    ok |= ((x-xxp)**2+(y-yyp)**2)<rrp**2
x=x[ok]
y=y[ok]

print("compute correlation...")

deg2mpc = rcom*np.pi/180.
bins=np.linspace(0,200,51)
nbins=bins.size-1
h0=np.zeros(nbins)
for xx,yy in zip(x,y) :
    d=np.sqrt((xx-x)**2+(yy-y)**2)*deg2mpc
    t,_=np.histogram(d,bins=bins)
    h0 += t
ok=(h0>0)
rt=(bins[:-1]+(bins[1]-bins[0])/2)
rt=rt[ok]
xi=h0[ok]/rt # number of random pairs scales as rt
xi /= xi[0] # norm

t=Table()
t["RT"]=rt
t["XI"]=xi
filename="desi-instrument-syst-for-forest-auto-correlation.csv".format(int(rcom))
t.write(filename,overwrite=True)

'''
# plotting
import matplotlib.pyplot as plt

plt.figure()
plt.plot(xp,yp,".",label="positioners")
plt.plot(x,y,".",alpha=0.2,label="randoms")
plt.xlabel("X (deg)")
plt.ylabel("Y (deg)")

plt.figure()
title="rcom={:.2f} Mpc/h".format(rcom)
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

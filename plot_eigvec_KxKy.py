#import commands
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')

import shutil
import os

import h5py

from pylab import *
import numpy as np
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import json
import pylab

from pylab import *
from scipy import *

from scipy          import optimize
from scipy.optimize import curve_fit

from numpy.linalg import inv
import numpy.ma as ma
from numpy.random import uniform
from scipy.interpolate import griddata

####################################################
#  parameters
####################################################
ch   = "PARTICLE_PARTICLE_SUPERCONDUCTING"         # vertex channel
U   = 7                                            # Hubbard U
dens  = 0.85                                       # density of electron in one of the bands
T  = 0.1                                         # temperature, as low as possible
Vs = [0.0, 0.5,1.0, 1.5, 2.0]           # parameter for cycling 

# Please check the parameters below before running this program in case of extracting wrong lines of data. 
Nw   = 32                  # four-point-fermionic-frequencies
Norb = 2                   # numbers of orbitals
Nc   = 12                  # number of k-points in the cluster

orbs= [[orb1,orb2] for orb1 in range(Norb) for orb2 in range(Norb)]
leading=['1st', '2nd', '3rd', '4th', '5th']
    
def plot_contourf(min, max, x, y, z, pic_name, pic_title):

    # define grid.
    xi = np.linspace(min,max,200)
    yi = np.linspace(min,max,200)

    # grid the data.
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')

    figure(num=None)    

    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

    #CS0 = plt.contour (xi,yi,zi,10,linewidths=1,colors='k')   # draw contour lines(optional)
    CS = plt.contourf(xi,yi,zi,100,cmap=plt.cm.jet)            # draw contour plot
    plt.colorbar()                                             # draw colorbar

    #define the region of 1st Brillouin Zone(-pi<Kx<pi, -pi<Ky<pi)
    plt.xlim(min,max)
    plt.ylim(min,max)
    
    xticks([min, min/2, 0, max/2, max], [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"],fontsize=12)
    yticks([min, min/2, 0, max/2, max], [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"],fontsize=12)

    plt.title(pic_title, fontsize=18)

    plot(x, y, "ko", ms=6)                      # mark the site of k-points
    
    xlabel('$\mathbf{K_{x}}$',fontsize=14)
    ylabel('$\mathbf{K_{y}}$',fontsize=14)
    
    savefig(pic_name)
    
clf()

for t in Vs:
    for orb_ind,orb in enumerate(orbs):
        for odr_ind,order in enumerate(leading):
        
 
            print ("==========================================================================")

            # search for "dca_tp.hdf5" and "leading_Evec_vs_Kiwn_T.txt"
            filename = "./"+"e1.0v"+str(t)+"/T="+str(T)+"/dca_tp.hdf5"            
            filename1= "./"+"e1.0v"+str(t)+"/leading_Evec_vs_Kiwn_T"+str(T)+".txt"
            
            print(filename)
            if not os.path.isfile(filename1):
                continue
            
            # extract data for cluster from dca_tp.hdf5
            data = h5py.File(filename,'r')
            k_dmn    = data["domains"]["LATTICE_TP"]["MOMENTUM_SPACE"]["elements"]["data"]      # site of k-points inside the cluster 
            k_basis  = data["domains"]["LATTICE_TP"]["MOMENTUM_SPACE"]["basis"]   ["data"]      # basis of the cluster
            k_super_basis  = data["domains"]["LATTICE_TP"]["MOMENTUM_SPACE"]["super-basis"]["data"]  # translation-invariant period of the cluster 
            freq_dmn = data["domains"]["vertex-frequency-domain (COMPACT)"]["elements"][:]      # Matsubara frequencies   
                        
            print(k_basis[:,:])
            print(k_super_basis[:,:])

            # read data line by line from leading_Evec_vs_Kiwn.txt
            evec=open(filename1, 'r') 
            Kxywev=evec.readlines()
            Kxy_w_ev=[]
            for line in Kxywev:
                Kxy_w_ev.append(line.split())
            evec.close()
            
            N=len(Kxy_w_ev) 
            Nodr=round(N/(Nc*2*Nw*len(orbs)))   #numbers of leading eigenvectors that output to the txt
            
            print(order)
            if odr_ind>=Nodr:
                continue

            # define data points in the three-dimensional contour plot
            x=[]
            y=[]
            z=[]
            
            for k_ind in range(Nc):
                # The L_fmin'th line of data has the lowest frequency
                L_fmin = Nw+k_ind*2*Nw+odr_ind*Nc*2*Nw+orb_ind*Nodr*Nc*2*Nw      
                print(Kxy_w_ev[L_fmin])
                
                for l0 in [-1, 0, 1]:
                    for l1 in [-1, 0, 1]:
                        # extend the area of k-point using periodic boundary condition, meanwhiles fill the frame of square 1st BZ
                        x.append(k_dmn[k_ind,0] + l0*k_super_basis[0,0] + l1*k_super_basis[1,0])
                        y.append(k_dmn[k_ind,1] + l0*k_super_basis[0,1] + l1*k_super_basis[1,1])
                        z.append(float(Kxy_w_ev[L_fmin][5]))



            #entitle the picture(file name and title) with parameters
            pic_name = "Eigvec_K_"+"_"+'V'+str(t)+'_orb'+str(orb[0])+str(orb[1])+'_'+order+"_T"+str(T)+"_Nc"+str(Nc)+".pdf"
            title = 'orb'+str(orb[0])+str(orb[1])+"_T="+str(T)+"_Nc="+str(Nc)+"_U="+str(U)+"_$V/t_d$="+str(t)+'_'+order

            #call the plotting function ahead
            L = pi
            plot_contourf(-L, L, x, y, z, pic_name, title)
            

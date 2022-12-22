'''
Adapted from Peizhi Mai's ipynb script for calculating Pd and Pd0
Ref: https://github.com/JohnstonResearchGroup/Mai_etal_3bandPairingCorr_2021 

The calculation at https://github.com/JohnstonResearchGroup/Mai_etal_3bandPairs_2021 
is INCORRECT !!! Need use the following code

See also check DCA code in Mac Notes
'''

import solveBSE_fromG4_threeband5new

###################################################################################
Ts = [1, 0.75, 0.5, 0.44, 0.4, 0.34, 0.3, 0.25, 0.24, 0.225, 0.2, 0.175, 0.17, 0.15, 0.125, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.035, 0.03, 0.025]
Ts = [0.0625]
channels = ['phcharge','phmag']
channels = ['phmag']
qs = ['00']#,'pi20','pi0','pipi2','pipi','pi2pi2']
#qs = ['pipi']
Nv = [0,1,2,3,4,5,6,7,8]
Nv = [0]

for T_ind, T in enumerate(Ts):
    #for ch in channels:
    for v in Nv:
        for q in qs:
            #file_tp = './T='+str(Ts[T_ind])+'/dca_tp_'+ch+'_q'+q+'.hdf5'
            file_tp = './T='+str(Ts[T_ind])+'/dca_tp_mag_q'+str(q)+'.hdf5'
            file_tp = './T='+str(Ts[T_ind])+'/dca_tp.hdf5'
            #file_tp = './T='+str(Ts[T_ind])+'/dca_tp_mag_qpipi_iw'+str(v)+'.hdf5'
            file_sp = './T='+str(Ts[T_ind])+'/dca_sp.hdf5'

            if(os.path.exists(file_tp)):
                print ("\n =================================\n")
                print ("T =", T)

                b = solveBSE_fromG4_threeband5new.BSE(file_tp,fileG=file_sp,\
                                                      draw=False,\
                                                      useG0=False,\
                                                      symmetrize_G4=False,\
                                                      phSymmetry=False,\
                                                      calcRedVertex=False,\
                                                      calcCluster=False,\
                                                      nkfine=100)

'''
Follow the procedure in Eq.(13) of PRB 103, 144514 (2021) by Peizhi Mai
'''
import numpy as np
import math
from numpy import *
import matplotlib.pyplot as mpl
import h5py
import sys
import os
from matplotlib.pyplot import *
import matplotlib as mpll

nt = b.nt; nc=b.Nc; nw=b.NwG4; nOrb=b.nOrb;
Kvecsnonneg=b.Kvecs.copy()

# switch all K points to be positive within (0,2*pi)
for i in range(b.Nc):
    if (b.Kvecs[i,0]<0):
        Kvecsnonneg[i,0] = Kvecsnonneg[i,0] +2*np.pi
    if (b.Kvecs[i,1]<0):
        Kvecsnonneg[i,1] = Kvecsnonneg[i,1] +2*np.pi
        
rightmatrix = zeros((nc,nOrb,nOrb),dtype='complex')
leftmatrix = zeros((nc,nOrb,nOrb),dtype='complex')

leftmatrix[:,0,0]=rightmatrix[:,0,0]=1.0

for k in range(b.Nc):
    gammak = sqrt(sin(b.Kvecs[k,0]/2)**2+sin(b.Kvecs[k,1]/2)**2)
    if (abs(gammak) < 1.e-10):
        rightmatrix[k,1,1]= 1j/sqrt(2)
        rightmatrix[k,1,2]= -1j/sqrt(2)
        rightmatrix[k,2,1]= -1j/sqrt(2)
        rightmatrix[k,2,2]= -1j/sqrt(2)
        leftmatrix[k,1,1]= -1j/sqrt(2)
        leftmatrix[k,1,2]= 1j/sqrt(2)
        leftmatrix[k,2,1]= 1j/sqrt(2)
        leftmatrix[k,2,2]= 1j/sqrt(2)
    else:
        kx = b.Kvecs[k,0]; ky = b.Kvecs[k,1];
        if (kx < 0):
            kx = kx +2*np.pi
        if (ky < 0):
            ky = ky +2*np.pi
        rightmatrix[k,1,1]= 1j * sin(kx/2)/gammak
        rightmatrix[k,1,2]= -1j * sin(ky/2)/gammak
        rightmatrix[k,2,1]= -1j * sin(ky/2)/gammak
        rightmatrix[k,2,2]= -1j * sin(kx/2)/gammak
        leftmatrix[k,1,1]= -1j * sin(kx/2)/gammak
        leftmatrix[k,1,2]= 1j * sin(ky/2)/gammak
        leftmatrix[k,2,1]= 1j * sin(ky/2)/gammak
        leftmatrix[k,2,2]= 1j * sin(kx/2)/gammak
                            
def changeG4LLbar(Tchi0LLbar,Tchi0):
    Tchi0LLbartemp1 = zeros((b.NwG4,b.Nc,b.nOrb,b.nOrb,b.NwG4,b.Nc,b.nOrb,b.nOrb),dtype='complex')
    for l2 in range(nOrb):
        for l1 in range(nOrb):
            for k1 in range(b.Nc):
                for k2 in range(b.Nc):
                    for iw1 in range(b.NwG4):
                        for iw2 in range(b.NwG4):
                            Tchi0LLbartemp1[iw1,k1,:,:,iw2,k2,l1,l2]= \
                            np.dot(np.dot(leftmatrix[k1,:,:],Tchi0[iw1,k1,:,:,iw2,k2,l1,l2]),\
                                   rightmatrix[k1,:,:])

    for l4 in range(nOrb):
        for l3 in range(nOrb):
            for k1 in range(b.Nc):
                for k2 in range(b.Nc):
                    for iw1 in range(b.NwG4):
                        for iw2 in range(b.NwG4):
                            Tchi0LLbar[iw1,k1,l3,l4,iw2,k2,:,:]= \
                            np.dot(np.dot(rightmatrix[k2,:,:],Tchi0LLbartemp1[iw1,k1,l3,l4,iw2,k2,:,:]),\
                                   leftmatrix[k2,:,:])
        
gkd = cos(Kvecsnonneg[:,0]) - cos(Kvecsnonneg[:,1])
        
# lambdas2 is leading eigenvalues of pm2 matrix in solveBSE*.py
eigval = b.lambdas2
Lambda = np.diag(1./(1-eigval))

# Dkk = zeros((nt,nt), dtype=real)
phit =zeros((nw,nc,nOrb,nOrb,nt), dtype=complex)
phit2=zeros((nw,nc,nOrb,nOrb,nt), dtype=complex)
Dkk =zeros((nw,nc,nOrb,nOrb,nw,nc,nOrb,nOrb), dtype=complex)
Dkk2=zeros((nw,nc,nOrb,nOrb,nw,nc,nOrb,nOrb), dtype=complex)
Pd =zeros((nOrb,nOrb,nOrb,nOrb), dtype=complex)
PdIa =zeros((nOrb,nOrb,nOrb,nOrb), dtype=complex)

chi0 = b.chi0
for ialpha in range(nt):
    phit[:,:,:,:,ialpha] = b.evecs2[:,:,:,:,ialpha]
    
# leading eigenvector
phit2[:,:,:,:,0] = b.evecs2[:,:,:,:,0]

#phi = phit.reshape(nt,nt)
#phi2 = phit2.reshape(nt,nt)

evecscom = b.evecs2.reshape(nt,nt)
Dkktemp = dot(phit,dot(Lambda,linalg.inv(evecscom)))
Dkktemp2 = Dkktemp.reshape(nt,nt)
Dkk = dot(b.chi0M,Dkktemp2)
#Dkk = dot(phi,dot(Lambda,phi.T))
Dkk = Dkk.reshape(nw,nc,nOrb,nOrb,nw,nc,nOrb,nOrb)
#Dkk2 = dot(phi2,dot(Lambda,phi2.T))
Lambda2 = Lambda.copy()
for i in range(len(Lambda)):
    Lambda2[i,i] = 0.0
Lambda2[0,0] = Lambda[0,0]
Dkk2temp = dot(phit2,dot(Lambda2,linalg.inv(evecscom)))
Dkk2temp2 = Dkk2temp.reshape(nt,nt)
Dkk2 = dot(b.chiMasqrt,dot(Dkk2temp2,b.chiMasqrt))
Dkk2 = Dkk2.reshape(nw,nc,nOrb,nOrb,nw,nc,nOrb,nOrb)
DkkLLbar = zeros((b.NwG4,b.Nc,b.nOrb,b.nOrb,b.NwG4,b.Nc,b.nOrb,b.nOrb),dtype='complex')
Dkk2LLbar = zeros((b.NwG4,b.Nc,b.nOrb,b.nOrb,b.NwG4,b.Nc,b.nOrb,b.nOrb),dtype='complex')
changeG4LLbar(DkkLLbar,Dkk)
changeG4LLbar(Dkk2LLbar,Dkk2)
LkkLLbar = sum(sum(DkkLLbar,axis=0),axis=3)
Lkk2LLbar = sum(sum(Dkk2LLbar,axis=0),axis=3)
for l1 in range(nOrb):
    for l2 in range(nOrb):
        for l3 in range(nOrb):
            for l4 in range(nOrb):
                Pd[l1,l2,l3,l4] = dot(gkd,dot(LkkLLbar[:,l1,l2,:,l3,l4],gkd)) * b.temp/b.Nc
                PdIa[l1,l2,l3,l4] = dot(gkd,dot(Lkk2LLbar[:,l1,l2,:,l3,l4],gkd)) * b.temp/b.Nc
                
Pd0 =zeros((nOrb,nOrb,nOrb,nOrb), dtype=complex)
Pd0Ia =zeros((nOrb,nOrb,nOrb,nOrb), dtype=complex)
for i in range(len(Lambda)):
    Lambda[i,i] = 1.0
for ialpha in range(nt):
    phit[:,:,:,:,ialpha] = b.evecs2[:,:,:,:,ialpha]
phit2[:,:,:,:,0] = b.evecs2[:,:,:,:,0]
        #phi = phit.reshape(nt,nt)
        #phi2 = phit2.reshape(nt,nt)
evecscom = b.evecs2.reshape(nt,nt)
Dkktemp = dot(phit,dot(Lambda,linalg.inv(evecscom)))
Dkktemp2 = Dkktemp.reshape(nt,nt)
Dkk = dot(b.chi0M,Dkktemp2)
#Dkk = dot(phi,dot(Lambda,phi.T))
Dkk = Dkk.reshape(nw,nc,nOrb,nOrb,nw,nc,nOrb,nOrb)
#Dkk2 = dot(phi2,dot(Lambda,phi2.T))
Dkk2temp = dot(phit2,linalg.inv(evecscom))
Dkk2temp2 = Dkk2temp.reshape(nt,nt)
Dkk2 = dot(b.chiMasqrt,dot(Dkk2temp2,b.chiMasqrt))
Dkk2 = Dkk2.reshape(nw,nc,nOrb,nOrb,nw,nc,nOrb,nOrb)
DkkLLbar = zeros((b.NwG4,b.Nc,b.nOrb,b.nOrb,b.NwG4,b.Nc,b.nOrb,b.nOrb),dtype='complex')
Dkk2LLbar = zeros((b.NwG4,b.Nc,b.nOrb,b.nOrb,b.NwG4,b.Nc,b.nOrb,b.nOrb),dtype='complex')
changeG4LLbar(DkkLLbar,Dkk)
changeG4LLbar(Dkk2LLbar,Dkk2)
LkkLLbar = sum(sum(DkkLLbar,axis=0),axis=3)
Lkk2LLbar = sum(sum(Dkk2LLbar,axis=0),axis=3)
for l1 in range(nOrb):
    for l2 in range(nOrb):
        for l3 in range(nOrb):
            for l4 in range(nOrb):
                Pd0[l1,l2,l3,l4] = dot(gkd,dot(LkkLLbar[:,l1,l2,:,l3,l4],gkd)) * b.temp/b.Nc
                Pd0Ia[l1,l2,l3,l4] = dot(gkd,dot(Lkk2LLbar[:,l1,l2,:,l3,l4],gkd)) * b.temp/b.Nc
print("Pd=",np.sum(Pd))
print("PdIa=",np.sum(PdIa))
print("Pd0=",np.sum(Pd0))
print("Pd0/(1-lambda)=",np.sum(Pd0)/(1-b.lambdas2[0].real))
print("Pd0Ia=",np.sum(Pd0Ia))
print("Pd0Ia/(1-lambda)=",np.sum(Pd0Ia)/(1-b.lambdas2[0].real))


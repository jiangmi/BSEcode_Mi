# Adapted from Peizhi Mai's email on May.22, 2021
# From his https://github.com/JohnstonResearchGroup/Mai_etal_3bandPairs_2021
# in fact, this program was finished before Apr. 2021

# Note:
# This program is for Mai's modification on an older version of DCA:
# https://github.com/cosdis/DCA
# which lags behind the current master version, but G4 of threeband model is correct

import numpy as np
import math
from numpy import *
import matplotlib.pyplot as mpl
import h5py
import sys
import os
from matplotlib.pyplot import *
import matplotlib as mpll

class BSE:
    def __init__(self,model,Tval,file_analysis_hdf5):
        self.vertex_channels = ["PARTICLE_PARTICLE_UP_DOWN",          \
                                "PARTICLE_HOLE_CHARGE",               \
                                "PARTICLE_HOLE_MAGNETIC",             \
                                "PARTICLE_HOLE_LONGITUDINAL_UP_UP",   \
                                "PARTICLE_HOLE_LONGITUDINAL_UP_DOWN", \
                                "PARTICLE_HOLE_TRANSVERSE"]
        self.model = model
        self.Tval = Tval
        self.file_analysis_hdf5 = file_analysis_hdf5
        
        data = h5py.File(self.file_analysis_hdf5,'r')
        self.Kvecs = array(data['domains']['CLUSTER']['MOMENTUM_SPACE']['elements']['data'])

        self.eigvals = data["analysis-functions"]["leading-eigenvalues"]["data"]
        self.eigvecs = data["analysis-functions"]["leading-eigenvectors"]["data"]
        
        print "analysis-functions/leading-eigenvalues",  self.eigvals.shape
        print "analysis-functions/leading-eigenvectors", self.eigvecs.shape
        
        self.Nw = self.eigvecs.shape[0]
        self.Nc = self.eigvecs.shape[1]
        self.nOrb = self.eigvecs.shape[2]
        
        self.determine_specialK()
        print ("Index of (pi,pi): ",self.iKPiPi)
        print ("Index of (pi,0): " ,self.iKPi0)
        
        self.AnalyzeEigenValues()  
        
    def AnalyzeEigenValues(self):
        iw0 = int(self.Nw/2)
        
        print "Leading 10 eigenvalues of lattice Bethe-salpeter equation"
        for i in range(10):
            print self.eigvals[i,0]
        
        print "Analyze eigenval and eigvec:"
        for io in range(self.nOrb):
            print '================='
            print 'orb ',io
            for inr in range(10):
                imax = argmax(self.eigvecs[iw0,:,io,io,inr,0])
                if (abs(self.eigvecs[iw0-1,imax,io,io,inr,0]-self.eigvecs[iw0,imax,io,io,inr,0]) <= 1.0e-2):
                    print "Eigenval is ", real(self.eigvals[inr]), "even frequency"
                else:
                    print "Eigenval is ", real(self.eigvals[inr]), "odd frequency"
                    
                print "Eigenvec(pi*T) =",self.eigvecs[iw0-1,imax,io,io,inr,0], self.eigvecs[iw0,imax,io,io,inr,0]

        #if self.vertex_channel in ("PARTICLE_PARTICLE_SUPERCONDUCTING","PARTICLE_PARTICLE_UP_DOWN"):
            #Now find d-wave eigenvalue
        gk = cos(self.Kvecs[:,0]) - cos(self.Kvecs[:,1]) # dwave form factor

        for io in range(self.nOrb):
            print '================='
            print 'For d-wave'
            print 'orb ',io
            self.found_d=False
            self.ind_d=0
            for ia in range(10):
                # first term check if Phi has d-wave in k space; 2nd term check if even frequency:
                r1 = dot(gk, self.eigvecs[int(self.Nw/2),:,io,io,ia,0]) * sum(self.eigvecs[:,self.iKPi0,io,io,ia,0])
                if abs(r1) >= 2.0e-1: 
                    self.eigval_d = self.eigvals[ia,0]
                    self.ind_d   = ia
                    self.found_d = True
                    break
            if self.found_d: 
                print "d-wave eigenvalue", self.Tval, ' ', self.eigval_d

        #Now find sx-wave eigenvalue
        gk = cos(self.Kvecs[:,0]) + cos(self.Kvecs[:,1]) # sxwave form factor
        for io in range(self.nOrb):
            print '================='
            print 'For sx-wave'
            print 'orb ',io
            self.found_d=False
            self.ind_d=0
            for ia in range(10):
                # first term check if Phi has d-wave in k space; 2nd term check if even frequency:
                r1 = dot(gk, self.eigvecs[int(self.Nw/2),:,io,io,ia,0]) * sum(self.eigvecs[:,self.iKPi0,io,io,ia,0])
                if abs(r1) >= 2.0e-1: 
                    self.eigval_d = self.eigvals[ia,0]
                    self.ind_d   = ia
                    self.found_d = True
                    break
            if self.found_d: 
                print "sx-wave eigenvalue", self.Tval, ' ', self.eigval_d
                
    def determine_specialK(self):
        self.iKPiPi = 0
        self.iKPi0  = 0
        Nc = self.Nc
        for iK in range(Nc):
            kx = abs(self.Kvecs[iK,0] - np.pi)
            ky = abs(self.Kvecs[iK,1] - np.pi)
            kx2 = abs(self.Kvecs[iK,0])
            ky2 = abs(self.Kvecs[iK,1])
            if kx >= 2*np.pi: kx-=2.*pi
            if ky >= 2*np.pi: ky-=2.*pi
            if ky2 >= 2*np.pi: ky-=2.*pi
            if kx**2+ky**2 <= 1.0e-5:
                self.iKPiPi = iK
            if kx**2+ky2**2 <= 1.0e-5:
                self.iKPi0 = iK
            if kx2**2+ky**2 <= 1.0e-5:
                self.iK0Pi = iK
                
###################################################################################
Ts = [1, 0.75, 0.5, 0.4, 0.3, 0.2, 0.15, 0.125, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04]
Ts = [0.1]
channels = ['phcharge']#,'phmag']
channels = ['phmag']
qs = ['00']#,'pi20','pi0','pipi2','pipi','pi2pi2']
qs = ['pipi']

for T_ind, T in enumerate(Ts):
    for ch in channels:
        for q in qs:
            file_analysis_hdf5 = './sc/T='+str(Ts[T_ind])+'/analysis.hdf5'
            #file_analysis_hdf5 = './Nc4/T='+str(Ts[T_ind])+'/analysis.hdf5'

            if(os.path.exists(file_analysis_hdf5)):
                print "\n =================================\n"
                print "T =", T
                # model='square','bilayer','Emery'
                BSE('square',Ts[T_ind],file_analysis_hdf5)
                

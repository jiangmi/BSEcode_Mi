# Adapted from T.Maier's email on May.17, 2017
# Only works for DCA and multi-orbital models, e.g. bilayer

import numpy as np
from numpy import *
#import matplotlib.pyplot as mpl
import h5py
import sys
import os
#from matplotlib.pyplot import *


class BSE:


    def __init__(self,fileG4,fileG="dca_sp.hdf5",draw=False,useG0=False,symmetrize_G4=False,phSymmetry=False,calcRedVertex=False,calcCluster=False,nkfine=100):
        self.fileG4 = fileG4
        self.fileG = fileG
        self.draw = draw
        self.useG0 = useG0
        self.symmetrize_G4 = symmetrize_G4
        self.calcCluster = calcCluster
        self.calcRedVertex = calcRedVertex
        self.phSymmetry = phSymmetry
        self.readData()
        self.reorderG4()   # new compared to single-orbital solveBSE_fromG4_DCA.py
        self.setupMomentumTables()
        self.determine_iKPiPi()
        # self.symmetrizeG4()
        print ("Index of (pi,pi): ",self.iKPiPi)
        self.calcChi0Cluster()
        self.calcGammaIrr()
        if calcCluster == False: self.buildChi0Lattice(nkfine)
        self.calcFillingBilayer() # new compared to single-orbital solveBSE_fromG4_DCA.py
        self.buildKernelMatrix()
        self.calcKernelEigenValues()
        if self.draw: self.plotLeadingSolutions(self.Kvecs,self.lambdas,self.evecs[:,:,:],title)
        self.transformEvecsToKz() # new compared to single-orbital solveBSE_fromG4_DCA.py
        #title = "Leading eigensolutions of BSE for U="+str(self.U)+", t'="+str(self.tp)+r", $\langle n\rangle$="+str(round(self.fill,4))+", T="+str(round(self.temp,4))
        if self.vertex_channel == "PARTICLE_HOLE_TRANSVERSE":
             self.calcAFLatticeSus()
             print("AF cluster susceptibility: ",sum(self.G4)/(float(self.Nc)*self.invT))
        if calcRedVertex: self.calcReducibleLatticeVertex()
        if self.vertex_channel == "PARTICLE_PARTICLE_UP_DOWN":
            self.calcPairSus()
            self.calcDwaveSCSus()
            self.calcDwaveSCClusterSus()
        # self.calcReducibleClusterVertex()
        # if self.calcRedVertex & (self.calcCluster==False):
        #     self.determineFS()
        #     # FSpoints = array([16,12,9,5,2,29,25,20,23,27,30,6,11,15])
        #     FSpoints = array(self.FSpoints)
        #     iwG40 = int(self.NwG4/2); nFs = int(FSpoints.shape[0])
        #     GRFS = np.sum(self.GammaRed[iwG40-1:iwG40+1,:,iwG40-1:iwG40+1,:],axis=(0,2))[FSpoints,:][:,FSpoints]/4.
        #     print ("s-wave projection of GammaRed: ", real(np.sum(GRFS)/float(nFs*nFs)))
        #     gkd = self.dwave(self.Kvecs[FSpoints,0],self.Kvecs[FSpoints,1])
        #     r1 = real(np.dot(gkd,np.dot(GRFS,gkd)))
        #     print ("d-wave projection of GammaRed: ", r1/np.dot(gkd, gkd)/float(nFs))
        #     GRFSpp = self.GammaRed[iwG40,:,iwG40,:][FSpoints,:][:,FSpoints]
        #     GRFSpm = self.GammaRed[iwG40,:,iwG40-1,:][FSpoints,:][:,FSpoints]
        #     print ("s-wave projection of GammaRed(piT,piT): ", real(np.sum(GRFSpp)/float(nFs*nFs)))
        #     print ("s-wave projection of GammaRed(piT,-piT): ", real(np.sum(GRFSpm)/float(nFs*nFs)))
        #     r1 = real(np.dot(gkd,np.dot(GRFSpp,gkd)))/np.dot(gkd, gkd)/float(nFs)
        #     r2 = real(np.dot(gkd,np.dot(GRFSpm,gkd)))/np.dot(gkd, gkd)/float(nFs)
        #     print ("d-wave projection of GammaRed(piT,piT): " , r1)
        #     print ("d-wave projection of GammaRed(piT,-piT): ", r2)
        #     GRFSpp = self.GammaRed[iwG40+1,:,iwG40+1,:][FSpoints,:][:,FSpoints]
        #     GRFSpm = self.GammaRed[iwG40+1,:,iwG40-2,:][FSpoints,:][:,FSpoints]
        #     r1 = real(np.dot(gkd,np.dot(GRFSpp,gkd)))/np.dot(gkd, gkd)/float(nFs)
        #     r2 = real(np.dot(gkd,np.dot(GRFSpm,gkd)))/np.dot(gkd, gkd)/float(nFs)
        #     print ("d-wave projection of GammaRed(3piT,3piT): " , r1)
        #     print ("d-wave projection of GammaRed(3piT,-3piT): ", r2)
        #     GRFSpp = self.GammaRed[iwG40+2,:,iwG40+2,:][FSpoints,:][:,FSpoints]
        #     GRFSpm = self.GammaRed[iwG40+2,:,iwG40-3,:][FSpoints,:][:,FSpoints]
        #     r1 = real(np.dot(gkd,np.dot(GRFSpp,gkd)))/np.dot(gkd, gkd)/float(nFs)
        #     r2 = real(np.dot(gkd,np.dot(GRFSpm,gkd)))/np.dot(gkd, gkd)/float(nFs)
        #     print ("d-wave projection of GammaRed(5piT,5piT): " , r1)
        #     print ("d-wave projection of GammaRed(5piT,-5piT): ", r2)
        #     self.calcPd0(FSpoints)
        #



    def readData(self):
        f = h5py.File(self.fileG4,'r')
        self.iwm = array(f['parameters']['four-point']['frequency-transfer'])[0] # transferred frequency in units of 2*pi*temp
        print("Transferred frequency iwm = ",self.iwm)
        self.qchannel = array(f['parameters']['four-point']['momentum-transfer'])
        print("Transferred momentum q = ",self.qchannel)
        a = array(f['parameters']['four-point']['type'])[:]
        self.vertex_channel = ''.join(chr(i) for i in a)
        print("Vertex channel = ",self.vertex_channel)
        self.invT = array(f['parameters']['physics']['beta'])[0]
        print("Inverse temperature = ",self.invT)
        self.temp = 1.0/self.invT
        self.Uc = array(f['parameters']['PAM-model']['Uc'])[0]
        print("Uc = ",self.Uc)
        self.Uf = array(f['parameters']['PAM-model']['Uf'])[0]
        print("Uf = ",self.Uf)
        self.tc = np.array(f['parameters']['PAM-model']['tc'])[0]
        print("tc = ",self.tc)
        self.tcp = array(f['parameters']['PAM-model']['tc_prime'])[0]
        print("tc-prime = ",self.tcp)
        self.tf = np.array(f['parameters']['PAM-model']['tf'])[0]
        print("tf = ",self.tf)
        self.tfp = array(f['parameters']['PAM-model']['tf_prime'])[0]
        print("tf-prime = ",self.tfp)
        self.tz = array(f['parameters']['PAM-model']['t_perp'])[0]
        print("t-perp = ",self.tz)
        self.fill = array(f['parameters']['physics']['density'])[0]
        print("filling = ",self.fill)
        self.dens = array(f['DCA-loop-functions']['density']['data'])
        print("actual filling:",self.dens)
        self.orbdens = array(f['DCA-loop-functions']['orbital-occupancies']['data'])
        print("orbital-occupancies:",self.orbdens)
        self.nk = array(f['DCA-loop-functions']['n_k']['data'])
        #print("nk:",self.nk)

        # Now read the 4-point Green's function
        G4Re  = array(f['functions']['G4_k_k_w_w']['data'])[:,:,:,:,:,:,:,:,0]
        G4Im  = array(f['functions']['G4_k_k_w_w']['data'])[:,:,:,:,:,:,:,:,1]
        self.G4 = G4Re+1j*G4Im
        # G4[iw1,iw2,ik1,ik2]

        # Now read the cluster Green's function
        GRe = array(f['functions']['cluster_greens_function_G_k_w']['data'])[:,:,0,:,0,:,0]
        GIm = array(f['functions']['cluster_greens_function_G_k_w']['data'])[:,:,0,:,0,:,1]
        self.Green = GRe + 1j * GIm

        # f2 = h5py.File(self.fileG,'r')

        # bare cluster Green's function G0(k,t)
        # self.G0kt = array(f2['functions']['free_cluster_greens_function_G0_k_t']['data'][:,:,0,:,0,:])
        # self.ntau = G0kt.shape[0]

        # bare cluster Green's function G0(k,w)
        # G0kwRe = array(f2['functions']['free_cluster_greens_function_G0_k_w']['data'][:,:,0,:,0,:,0])
        # G0kwIm = array(f2['functions']['free_cluster_greens_function_G0_k_w']['data'][:,:,0,:,0,:,1])
        # self.G0kw = G0kwRe + 1j*G0kwIm

        # Now read the self-energy
        s  = np.array(f['functions']['Self_Energy']['data'])
        self.sigma = s[:,:,0,:,0,:,0] + 1j *s[:,:,0,:,0,:,1]

        # Now load frequency data
        self.wn = np.array(f['domains']['frequency-domain']['elements'])
        self.wnSet = np.array(f['domains']['vertex-frequency-domain (COMPACT)']['elements'])

        # Now read the K-vectors
        self.Kvecs = array(f['domains']['CLUSTER']['MOMENTUM_SPACE']['elements']['data'])
        print ("K-vectors: ",self.Kvecs)

        # Now read other Hubbard parameters
        self.mu = np.array(f['DCA-loop-functions']['chemical-potential']['data'])[0]


        self.NwG4 = self.G4.shape[0]
        self.Nc  = self.Green.shape[1]
        self.NwG = self.Green.shape[0]
        self.nOrb = self.Green.shape[2]
        self.nt = self.Nc*self.NwG4*self.nOrb*self.nOrb

        print ("NwG4: ",self.NwG4)
        print ("NwG : ",self.NwG)
        print ("Nc  : ",self.Nc)
        print ("nOrb: ",self.nOrb)

        self.NwTP = 2*np.array(f['parameters']['domains']['imaginary-frequency']['four-point-fermionic-frequencies'])[0]
        self.iQ = self.K_2_iK(self.qchannel[0],self.qchannel[1])
        print ("Index of transferred momentum: ", self.iQ)


        self.iwG40 = self.NwG4/2
        self.iwG0 = self.NwG/2

        f.close()
        # f2.close()

        # Now remove vacuum part of charge G4
        if (self.vertex_channel=="PARTICLE_HOLE_CHARGE"):
            if (self.qchannel[0] == 0) & (self.qchannel[1] == 0):
                for ik1 in range(self.Nc):
                    for ik2 in range(self.Nc):
                        for iw1 in range(self.NwG4):
                            for iw2 in range(self.NwG4):
                                for l1 in range(self.nOrb):
                                    for l2 in range(self.nOrb):
                                        for l3 in range(self.nOrb):
                                            for l4 in range(self.nOrb):
                                                iw1Green = iw1 - self.iwG40 + self.iwG0
                                                iw2Green = iw2 - self.iwG40 + self.iwG0
                                                self.G4[iw1,iw2,ik1,ik2,l1,l2,l3,l4] -= 2.0 * self.Green[iw1Green,ik1,l1,l2] * self.Green[iw2Green,ik2,l4,l3]

    def reorderG4(self):
        Nc=self.Nc; NwG4=self.NwG4; NwG=self.NwG; nOrb = self.nOrb
        self.G4r=np.zeros((NwG4,Nc,nOrb,nOrb,NwG4,Nc,nOrb,nOrb),dtype='complex')
        for ik1 in range(self.Nc):
            for ik2 in range(self.Nc):
                for iw1 in range(self.NwG4):
                    for iw2 in range(self.NwG4):
                         for l1 in range(self.nOrb):
                             for l2 in range(self.nOrb):
                                 for l3 in range(self.nOrb):
                                     for l4 in range(self.nOrb):
                                         self.G4r[iw1,ik1,l1,l2,iw2,ik2,l3,l4] = self.G4[iw1,iw2,ik1,ik2,l1,l2,l3,l4]
        self.G4M = self.G4r.reshape(self.nt,self.nt)

    def fermi(self,energy):
        beta=self.invT
        xx = beta*energy
        if xx < 0:
            return 1./(1.+exp(xx))
        else:
            return exp(-xx)/(1.+exp(-xx))

    def calcFillingBilayer(self):
        Nc = self.Nc; beta=self.invT

        Gkw0   = 0.5*(self.cG[:,:,0,0] + self.cG[:,:,1,1] + self.cG[:,:,0,1] + self.cG[:,:,1,0])
        Gkwpi  = 0.5*(self.cG[:,:,0,0] + self.cG[:,:,1,1] - self.cG[:,:,0,1] - self.cG[:,:,1,0])
        G0kw0  = 0.5*(self.cG0[:,:,0,0] + self.cG0[:,:,1,1] + self.cG0[:,:,0,1] + self.cG0[:,:,1,0])
        G0kwpi = 0.5*(self.cG0[:,:,0,0] + self.cG0[:,:,1,1] - self.cG0[:,:,0,1] - self.cG0[:,:,1,0])

        a0  = sum(Gkw0-G0kw0)/(Nc*beta)
        api = sum(Gkwpi-G0kwpi)/(Nc*beta)

        fk0 = 0.0; fkpi = 0.0
        for iK,K in enumerate(self.Kset):
            for k in self.kPatch:
                kx = K[0]+k[0]; ky = K[1]+k[1]
                ek = self.dispersion(kx,ky)
                ek0  = ek[0,0] + ek[1,0]
                ekpi = ek[0,0] - ek[1,0]
                fk0  += self.fermi(ek0-self.mu+self.U*(self.dens/4.-0.5))
                fkpi += self.fermi(ekpi-self.mu+self.U*(self.dens/4.-0.5))
        fk0  /= self.Nc*self.kPatch.shape[0]
        fkpi /= self.Nc*self.kPatch.shape[0]

        c0 = a0 + fk0; cpi = api + fkpi

        print("Filling kz=0  bare : ",real(fk0))
        print("Filling kz=pi bare: ",real(fkpi))

        print("Filling kz=0 : ",real(c0))
        print("Filling kz=pi: ",real(cpi))

        print("Total filling: ",2.*real(c0+cpi))

    def determineFS(self):
        self.FSpoints=[]
        Kset = self.Kvecs.copy()
        for iK in range(self.Nc):
            if Kset[iK,0] > np.pi: Kset[iK,0] -= 2*np.pi
            if Kset[iK,1] > np.pi: Kset[iK,1] -= 2*np.pi

        for iK in range(self.Nc):
            if abs(abs(self.Kset[iK,0])+abs(self.Kset[iK,1]) - np.pi) <= 1.0e-4:
                self.FSpoints.append(iK)

    def symmetrizeG4(self):
        if self.iwm==0 & self.symmetrize_G4:
            self.apply_symmetry_in_wn(self.G4)
            # self.apply_transpose_symmetry(self.G4)
            # if self.phSymmetry: self.apply_ph_symmetry_pp(self.G4)



    def calcChi0Cluster(self):
        print ("Now calculating chi0 on cluster")
        self.chic0  = zeros((self.NwG4,self.Nc,self.nOrb,self.nOrb,self.NwG4,self.Nc,self.nOrb,self.nOrb),dtype='complex')
        Nc=self.Nc; NwG4=self.NwG4; NwG=self.NwG; nOrb = self.nOrb

        if (self.vertex_channel == "PARTICLE_PARTICLE_UP_DOWN"):
            for iw in range(0,NwG4):
                for ik in range(Nc):
                    for l1 in range(nOrb):
                        for l2 in range(nOrb):
                            for l3 in range(nOrb):
                                for l4 in range(nOrb):
                                    iw1  = int(iw - NwG4/2 + NwG/2)
                                    ikPlusQ = int(self.iKSum[self.iKDiff[0,ik],self.iQ]) # -k+Q
                                    minusiwPlusiwm = int(min(max(NwG-iw1-1 + self.iwm,0),NwG-1)) # -iwn + iwm
                                    c1 = self.Green[iw1,ik,l1,l3] * self.Green[minusiwPlusiwm,ikPlusQ,l2,l4]
                                    self.chic0[iw,ik,l1,l2,iw,ik,l3,l4] = c1
        else:
            for iw in range(NwG4):
                for ik in range(Nc):
                    for l1 in range(nOrb):
                        for l2 in range(nOrb):
                            for l3 in range(nOrb):
                                for l4 in range(nOrb):
                                    iw1  = int(iw - NwG4/2 + NwG/2)
                                    ikPlusQ = int(self.iKSum[ik,self.iQ]) # k+Q
                                    iwPlusiwm = int(min(max(iw1 + self.iwm,0),NwG-1))  # iwn+iwm
                                    #print("iw1,ik,iwPlusiwm,ikPlusQ",iw1,ik,iwPlusiwm,ikPlusQ)
                                    c1 = - self.Green[iw1,ik,l1,l3] * self.Green[iwPlusiwm,ikPlusQ,l2,l4]
                                    self.chic0[iw,ik,l1,l2,iw,ik,l3,l4] = c1
        self.chic0M = self.chic0.reshape(self.nt,self.nt)


    def calcGammaIrr(self):
        print ("Calculate the irr. GammaIrr")
        Nc=self.Nc; NwG4=self.NwG4; NwG=self.NwG; nt = self.nt; nOrb = self.nOrb
        G4M = linalg.inv(self.G4M)
        chic0M = linalg.inv(self.chic0M)
        self.GammaM = chic0M - G4M
        self.GammaM *= float(Nc)*self.invT
        self.Gamma = self.GammaM.reshape(NwG4,Nc,nOrb,nOrb,NwG4,Nc,nOrb,nOrb)


    def buildKernelMatrix(self):
        print ("Build kernel matrix Gamma*chi0")
        if (self.calcCluster):
            self.chiM = self.chic0M
        else:
            self.chiM = self.chi0M
        self.pm = np.dot(self.GammaM, self.chiM)
        print ("The matrix size of Gamma*chi0",self.pm.shape)
        self.pm *= 1.0/(self.invT*float(self.Nc))


    def calcKernelEigenValues(self):
        nt = self.nt; Nc = self.Nc; NwG4=self.NwG4; nOrb = self.nOrb
        w,v = linalg.eig(self.pm)
        wt = abs(w-1)
        ilead = argsort(wt)[0:16]
        self.lambdas = w[ilead]
        self.evecs = v[:,ilead]
        self.evecs = self.evecs.reshape(NwG4,Nc,nOrb,nOrb,16)
        print ("Leading eigenvalues of lattice Bethe-salpeter equation",self.lambdas)

    def transformEvecsToKz(self):
        self.phi0  = 1./sqrt(2.)*(self.evecs[:,:,0,0,:] + self.evecs[:,:,1,0,:])
        self.phipi = 1./sqrt(2.)*(self.evecs[:,:,0,0,:] - self.evecs[:,:,1,0,:])

    def calcReducibleLatticeVertex(self):
        # see Eq.(22-30) in PRB 64, 195130(2001), GammaRed is that bar(T2)
        pm = self.pm; Gamma=self.GammaM
        nt = self.nt; Nc = self.Nc; NwG4=self.NwG4; nOrb = self.nOrb
        self.pminv = np.linalg.inv(np.identity(nt)-pm)
        # self.pminv = np.linalg.inv(np.identity(nt)+pm)
        self.GammaRed = dot(self.pminv, Gamma)
        self.GammaRed = self.GammaRed.reshape(NwG4,Nc,nOrb,nOrb,NwG4,Nc,nOrb,nOrb)

    # def calcReducibleClusterVertex(self):
    #     # Calculate cluster vertex from Gamma(q,k,k') = [ G4(q,k,k')-G(k)G(k+q) ] / [ G(k)G(k+q)G(k')G(k'+q) ]
    #     nt = self.nt; Nc = self.Nc; NwG4=self.NwG4
    #     self.GammaCluster = np.zeros((NwG4,Nc,NwG4,Nc),dtype=complex)
    #     for iK1 in range(Nc):
    #         for iK2 in range(Nc):
    #             for iw1 in range(NwG4):
    #                 for iw2 in range(NwG4):
    #
    #                     iwG1 = int(iw1 - self.iwG40 + self.iwG0)
    #                     iwG2 = int(iw2 - self.iwG40 + self.iwG0)
    #
    #                     if self.vertex_channel=="PARTICLE_PARTICLE_SUPERCONDUCTING":
    #
    #                         imk1pq = int(self.iKSum[self.iKDiff[0,iK1],self.iQ])
    #                         imk2pq = int(self.iKSum[self.iKDiff[0,iK2],self.iQ])
    #                         numerator  = self.G4[iw1,iw2,iK1,iK2]
    #                         if (iK1==iK2) & (iw1==iw2): numerator -= self.Green[iwG1,iK1] * self.Green[self.NwG-iwG1-1+self.iwm,imk1pq]
    #                         denominator = self.Green[iwG1,iK1]*self.Green[self.NwG-iwG1-1+self.iwm,imk1pq] * self.Green[iwG2,iK2]*self.Green[self.NwG-iwG2-1+self.iwm,imk2pq]
    #
    #                     else:
    #
    #                         ik1pq = int(self.iKSum[iK1,self.iQ])
    #                         ik2pq = int(self.iKSum[iK2,self.iQ])
    #                         numerator  = self.G4[iw1,iw2,iK1,iK2]
    #                         if (iK1==iK2) & (iw1==iw2): numerator += self.Green[iwG1,iK1] * self.Green[iwG1+self.iwm,ik1pq]
    #                         denominator = self.Green[iwG1,iK1]*self.Green[iwG1+self.iwm,ik1pq] * self.Green[iwG2,iK2]*self.Green[iwG2+self.iwm,ik2pq]
    #                     self.GammaCluster[iw1,iK1,iw2,iK2] = numerator/denominator
    #
    #     self.GammaCluster *= self.invT*self.Nc
    #


    # def calcEigenSolution(self,matrix,title):
    #   w,v = linalg.eig(matrix)
    #   ilead = argsort(w)[::-1][0:10]
    #   self.lambdasMatrix = w[ilead]
    #   self.evecsMatrix = v[:,ilead]
    #   print title,self.lambdasMatrix
    #   if self.draw: self.plotLeadingSolutions(self.Kvecs[self.FSpoints,:], real(self.lambdasMatrix), real(self.evecsMatrix), title)

    def buildChi0Lattice(self,nkfine):
        print ("Now calculating chi0 on lattice")

        NwG=self.NwG
        # Cluster K-grid
        Kset = self.Kvecs.copy() # copy() since Kset will be modified

        # Fine mesh
        klin = np.arange(-pi,pi,2*pi/nkfine)
        kx,ky = np.meshgrid(klin,klin)
        kset = np.column_stack((kx.flatten(),ky.flatten()))

        kPatch = []

        # shift to 1. BZ
        Nc = Kset.shape[0]
        for iK in range(Nc):
            if Kset[iK,0] > np.pi: Kset[iK,0] -= 2*np.pi
            if Kset[iK,1] > np.pi: Kset[iK,1] -= 2*np.pi

        self.Kset = Kset

        #Determine k-points in patch
        for k in kset:
            distance0 = k[0]**2 + k[1]**2
            newPoint = True
            for K in Kset:
                distanceKx = k[0] - K[0]; distanceKy = k[1] - K[1]
                if distanceKx >=  pi: distanceKx -= 2*pi
                if distanceKy >=  pi: distanceKy -= 2*pi
                if distanceKx <= -pi: distanceKx += 2*pi
                if distanceKy <= -pi: distanceKy += 2*pi
                distanceK = distanceKx**2 + distanceKy**2
                if distanceK < distance0:
                    newPoint = False
                    break
            if newPoint: kPatch.append(k.tolist())

        kPatch = np.array(kPatch)
        self.kPatch = kPatch

        # Load frequency domain
        wnSet = self.wnSet

        # Now coarse-grain G*G to build chi0(K) = Nc/N sum_k Gc(K+k')Gc(-K-k')
        nOrb = self.nOrb; nw = wnSet.shape[0]; nk=Kset.shape[0]
        self.chi0  = np.zeros((nw,nk,nOrb,nOrb,nw,nk,nOrb,nOrb),dtype='complex')
        self.chi0D = np.zeros((nw,nk,nOrb,nOrb,nw,nk,nOrb,nOrb),dtype='complex')
        self.chi0D2= np.zeros((nw,nk,nOrb,nOrb,nw,nk,nOrb,nOrb),dtype='complex')
        self.cG    = np.zeros((nw,nk,nOrb,nOrb),dtype='complex')
        self.cG0   = np.zeros((nw,nk,nOrb,nOrb),dtype='complex')
        for iwn,wn in enumerate(wnSet): # reduced tp frequencies !!
            # print "iwn = ",iwn
            iwG = int(iwn - self.iwG40 + self.iwG0)
            for iK,K in enumerate(Kset):
                c1 = np.zeros((nOrb,nOrb,nOrb,nOrb),dtype='complex')
                c2 = np.zeros((nOrb,nOrb,nOrb,nOrb),dtype='complex')
                c3 = np.zeros((nOrb,nOrb,nOrb,nOrb),dtype='complex')
                cG = np.zeros((nOrb,nOrb),dtype='complex')
                cG0 = np.zeros((nOrb,nOrb),dtype='complex')
                for k in kPatch:
                    kx = K[0]+k[0]; ky = K[1]+k[1]
                    ek = self.dispersion(kx,ky)
                    gkd = cos(kx) - cos(ky)
                    Uterm = np.identity(nOrb)
                    Uterm[0,0] = self.Uc*(self.orbdens[0,0,0]-0.5)
                    Uterm[1,1] = self.Uf*(self.orbdens[0,0,1]-0.5) 
                    G0inv = (1j*wn+self.mu)* np.identity(nOrb) -Uterm - ek
                    G0 = linalg.inv(G0inv)
                    if (self.vertex_channel == "PARTICLE_PARTICLE_UP_DOWN"):
                        Qx = self.qchannel[0]; Qy = self.qchannel[1]
                        emkpq = self.dispersion(-kx+Qx, -ky+Qy)
                        iKQ = self.iKSum[self.iKDiff[0,iK],self.iQ]
                        minusiwPlusiwm = min(max(NwG-iwG-1 + self.iwm,0),NwG-1) # -iwn + iwm

                        G1inv = (1j*wn+self.mu) * np.identity(nOrb)-ek-self.sigma[iwG,iK,:,:]
                        G2inv = (-1j*wn+self.mu)* np.identity(nOrb)-emkpq-self.sigma[minusiwPlusiwm,iKQ,:,:]
                        G1 = linalg.inv(G1inv); G2 = linalg.inv(G2inv)

                    else:
                        Qx = self.qchannel[0]; Qy = self.qchannel[1]
                        ekpq = self.dispersion(kx+Qx, ky+Qy)
                        iKQ = int(self.iKSum[iK,self.iQ])
                        iwPlusiwm = int(min(max(iwG + self.iwm,0),NwG-1))  # iwn+iwm

                        G1inv = (1j*wn+self.mu)*np.identity(nOrb)-ek-self.sigma[iwG,iK,:,:]
                        G2inv = (1j*wn+self.mu)*np.identity(nOrb)-ekpq-self.sigma[iwPlusiwm,iKQ,:,:]
                        G1 = linalg.inv(G1inv); G2 = -linalg.inv(G2inv)

                    for l1 in range(nOrb):
                        for l2 in range(nOrb):
                            cG[l1,l2] += G1[l1,l2]
                            cG0[l1,l2] += G0[l1,l2]
                            for l3 in range(nOrb):
                                for l4 in range(nOrb):
                                    c0 = G1[l1,l3]*G2[l2,l4]
                                    c1[l1,l2,l3,l4] += c0
                                    if (self.vertex_channel == "PARTICLE_PARTICLE_UP_DOWN"):
                                        c2[l1,l2,l3,l4] += c0 * gkd
                                        c3[l1,l2,l3,l4] += c0 * gkd**2
                                    
                self.chi0[iwn,iK,:,:,iwn,iK,:,:] = c1[:,:,:,:]/kPatch.shape[0]
                if (self.vertex_channel == "PARTICLE_PARTICLE_UP_DOWN"):
                    self.chi0D [iwn,iK,:,:,iwn,iK,:,:] = c2[:,:,:,:]/kPatch.shape[0]
                    self.chi0D2[iwn,iK,:,:,iwn,iK,:,:] = c3[:,:,:,:]/kPatch.shape[0]

                self.cG[iwn,iK,:,:] = cG[:,:]/kPatch.shape[0]
                self.cG0[iwn,iK,:,:] = cG0[:,:]/kPatch.shape[0]
        self.chi0M = self.chi0.reshape(self.nt,self.nt)
        if (self.vertex_channel == "PARTICLE_PARTICLE_UP_DOWN"):
            self.chi0DM  = self.chi0D.reshape(self.nt,self.nt)
            self.chi0D2M = self.chi0D2.reshape(self.nt,self.nt)

    def calcPairSus(self):
        print ("Calculate pairing chi in simpler way")
        NwG4=self.NwG4; Nc=self.Nc; nOrb=self.nOrb

        gkd = cos(self.Kvecs[:,0]) - cos(self.Kvecs[:,1])
        
        # calculate lattice G4 Green's function
        # from inv( inv(chi0)-inv(chic0)+inv(G4) ), see Eq.(20) in PRB 64, 195130(2001)
        G2L = linalg.inv(linalg.inv(self.chi0M) - self.GammaM/(float(self.Nc)*self.invT))
        G2 = G2L.reshape(NwG4,Nc,nOrb,nOrb,NwG4,Nc,nOrb,nOrb)
        
        # for inter-band susceptibility
        G2spm = 0.25*(G2[:,:,0,1,:,:,0,1]+G2[:,:,0,1,:,:,1,0]+G2[:,:,1,0,:,:,0,1]+G2[:,:,1,0,:,:,1,0])
        Ps = sum(G2spm)/(float(self.Nc)*self.invT)

        # for dwave susceptibility
        # cos kx - cos ky form-factor corresponds to inter- and intra-layer d-wave susceptibility
        G2band1 = G2[:,:,0,0,:,:,0,0]
        G2band2 = G2[:,:,1,1,:,:,1,1]
        G2band12 = 0.5*(G2[:,:,1,1,:,:,0,0]+G2[:,:,0,0,:,:,1,1])
        
        # axis=0 sum over first wn, then G2 becomes (iK,iwn,iK) so that axis=1 sum over second wn 
        G2t0b1 = sum(sum(G2band1,axis=0),axis=1)
        Pd_b1 = dot(gkd,dot(G2t0b1,gkd))/(float(self.Nc)*self.invT)
        G2t0b2 = sum(sum(G2band2,axis=0),axis=1)
        Pd_b2 = dot(gkd,dot(G2t0b2,gkd))/(float(self.Nc)*self.invT)
        G2t0b12 = sum(sum(G2band12,axis=0),axis=1)
        Pd_b12 = dot(gkd,dot(G2t0b12,gkd))/(float(self.Nc)*self.invT)

        print "spm   susceptibility: ", real(Ps)
        print "dwave susceptibility for intra-band 1:  ", real(Pd_b1)
        print "dwave susceptibility for intra-band 2:  ", real(Pd_b2)
        print "dwave susceptibility for inter-band 12: ", real(Pd_b12)

    # def calcPd0(self,FSpoints):
    #     c1=0.0
    #     NwG=self.NwG
    #     for iwn,wn in enumerate(self.wn):
    #     #for iwn,wn in enumerate(self.wnSet):
    #         iwG = iwn
    #         #iwG = int(iwn - self.iwG40 + self.iwG0)
    #         for iK in FSpoints:
    #             Kx = self.Kvecs[iK,0]
    #             Ky = self.Kvecs[iK,1]
    #             #for k in self.kPatch:
    #             for k in [[0,0]]:
    #                 kx = Kx+k[0]; ky = Ky+k[1]
    #                 ek = self.dispersion(kx,ky)
    #                 gkd = cos(kx)-cos(ky)
    #                 emk = self.dispersion(-kx, -ky)
    #                 iKQ = self.iKDiff[0,iK]
    #                 minusiw = min(max(NwG-iwG-1,0),NwG-1) # -iwn + iwm
    #                 c1 += gkd**2/(1j*wn+self.mu-ek-self.sigma[iwG,iK]) * 1./(-1j*wn+self.mu-emk-self.sigma[minusiw,iKQ])
    #     #print("Pd0 = T/N sum_kF (coskx-cosky)^2 G(k,iwn)*G(-k,-iwn)",c1 / (FSpoints.shape[0]*self.kPatch.shape[0]*self.invT))
    #     print("Pd0 = T/N sum_kF (coskx-cosky)^2 G(k,iwn)*G(-k,-iwn)",c1 / (FSpoints.shape[0]*self.invT))
    #
    def calcAFLatticeSus(self):
        # need to modify to consider inter and intra band
        G2L = linalg.inv(linalg.inv(self.chi0M) - self.GammaM/(float(self.Nc)*self.invT))
        print("AF lattice susceptibility: ",sum(G2L)/(float(self.Nc)*self.invT))

    def calcDwaveSCSus(self):
        # Calculate from gd*G4*gd = gd*GG*gd + gd*GG*GammaRed*GG*gd
        # GammaRed = self.GammaRed.reshape(self.NwG4*self.Nc,self.NwG4*self.Nc)
        # GammaRed is bar(T2) in Eq.(22-30) in PRB 64, 195130(2001)
        # chi0D2 and chi0D is Eq.(27-28) in PRB 64, 195130(2001)
        # csum : Eq.(29)'s second term of RHS
        # csum1: Eq.(29)'s first term of RHS

        print ("==============================================================================")
        print ("Calculate lattice d-wave susceptibility with Eq.(22-30) in PRB 64,195130(2001)")
        nOrb = self.nOrb
        gkd = cos(self.Kvecs[:,0]) - cos(self.Kvecs[:,1])
        csum   = np.zeros((nOrb,nOrb,nOrb,nOrb),dtype='complex')
        ccsum  = np.zeros((nOrb,nOrb,nOrb,nOrb),dtype='complex')
        csum1  = np.zeros((nOrb,nOrb,nOrb,nOrb),dtype='complex')
        ccsum1 = np.zeros((nOrb,nOrb,nOrb,nOrb),dtype='complex')
        
        for iw1 in range(self.NwG4):
            for iK1 in range(self.Nc):
                for l1 in range(nOrb):
                    for l2 in range(nOrb):
                        for iw2 in range(self.NwG4):
                            for iK2 in range(self.Nc):
                                for l3 in range(nOrb):
                                    for l4 in range(nOrb):
                                        csum[l1,l2,l3,l4]  += self.chi0D[iw1,iK1,l1,l2,iw2,iK2,l3,l4]*self.GammaRed[iw1,iK1,l1,l2,iw2,iK2,l3,l4]*self.chi0D[iw1,iK1,l1,l2,iw2,iK2,l3,l4]
                                        ccsum[l1,l2,l3,l4] += gkd[iK1]*self.chi0[iw1,iK1,l1,l2,iw2,iK2,l3,l4]*self.GammaRed[iw1,iK1,l1,l2,iw2,iK2,l3,l4]*self.chi0[iw1,iK1,l1,l2,iw2,iK2,l3,l4]*gkd[iK2]

        csum /= (self.Nc*self.invT)**2
        ccsum /= (self.Nc*self.invT)**2

        for l1 in range(nOrb):
            for l2 in range(nOrb):
                for l3 in range(nOrb):
                    for l4 in range(nOrb):
                        csum1[l1,l2,l3,l4] = sum(self.chi0D2[:,:,l1,l2,:,:,l3,l4])/(self.Nc*self.invT)
                        ccsum1[l1,l2,l3,l4] = sum(np.dot(self.chi0[:,:,l1,l2,:,:,l3,l4],gkd**2))/(self.Nc*self.invT)
        csum += csum1
        ccsum += ccsum1

        print ("-------------------------------------------------------------------")
        print ("bare d-wave SC susceptibility:")
        for l1 in range(0,nOrb):
            for l2 in range(l1,nOrb):
                for l3 in range(l2,nOrb):
                    for l4 in range(l3,nOrb):
                        print "(bbbb) = "+str(l1)+str(l2)+str(l3)+str(l4)+"\t",real(csum1[l1,l2,l3,l4])
        print ("-------------------------------------------------------------------")
        print ("bare d-wave SC lattice (with cluster form-factors) susceptibility:")
        for l1 in range(0,nOrb):
            for l2 in range(l1,nOrb):
                for l3 in range(l2,nOrb):
                    for l4 in range(l3,nOrb):
                        print "(bbbb) = "+str(l1)+str(l2)+str(l3)+str(l4)+"\t",real(ccsum1[l1,l2,l3,l4])
        print ("-------------------------------------------------------------------")
        print ("d-wave SC susceptibility:")
        for l1 in range(0,nOrb):
            for l2 in range(l1,nOrb):
                for l3 in range(l2,nOrb):
                    for l4 in range(l3,nOrb):
                        print "(bbbb) = "+str(l1)+str(l2)+str(l3)+str(l4)+"\t",real(csum[l1,l2,l3,l4])
        print ("-------------------------------------------------------------------")
        print ("d-wave SC lattice (with cluster form-factors) susceptibility:")
        for l1 in range(0,nOrb):
            for l2 in range(l1,nOrb):
                for l3 in range(l2,nOrb):
                    for l4 in range(l3,nOrb):
                        print "(bbbb) = "+str(l1)+str(l2)+str(l3)+str(l4)+"\t",real(ccsum[l1,l2,l3,l4])


    def calcDwaveSCClusterSus(self):
        gkd = cos(self.Kvecs[:,0]) - cos(self.Kvecs[:,1])
        csum = 0.0
        for iK1 in range(self.Nc):
            for iK2 in range(self.Nc):
                csum += gkd[iK1]*sum(self.G4[:,:,iK1,iK2])*gkd[iK2]
        csum /= self.Nc*self.invT
        print ("d-wave SC cluster susceptibility: ",csum)

####### Momentum domain functions

    def determine_iKPiPi(self):
        self.iKPiPi = 0
        Nc=self.Nc
        for iK in range(Nc):
            kx = abs(self.Kvecs[iK,0] - np.pi)
            ky = abs(self.Kvecs[iK,1] - np.pi)
            if kx >= 2*np.pi: kx-=2.*pi
            if ky >= 2*np.pi: ky-=2.*pi
            if kx**2+ky**2 <= 1.0e-5:
                self.iKPiPi = iK
                break

    def K_2_iK(self,Kx,Ky):
        delta=1.0e-4
        # First map (Kx,Ky) into [0...2pi,0...2pi] region where Kvecs are defined
        if Kx < -delta      : Kx += 2*pi
        if Ky < -delta      : Ky += 2*pi
        if Kx > 2.*pi-delta : Kx -= 2*pi
        if Ky > 2.*pi-delta : Ky -= 2*pi
        # Now find index of Kvec = (Kx,Ky)
        for iK in range(0,self.Nc):
            if (abs(float(self.Kvecs[iK,0]-Kx)) < delta) & (abs(float(self.Kvecs[iK,1]-Ky)) < delta): return iK
        print("No Kvec found!!!")

    def setupMomentumTables(self):
        # build tables for K+K' and K-K'
        self.iKDiff = zeros((self.Nc,self.Nc),dtype='int')
        self.iKSum  = zeros((self.Nc,self.Nc),dtype='int')
        Nc = self.Nc
        for iK1 in range(Nc):
            Kx1 = self.Kvecs[iK1,0]; Ky1 = self.Kvecs[iK1,1]
            for iK2 in range(0,Nc):
                Kx2 = self.Kvecs[iK2,0]; Ky2 = self.Kvecs[iK2,1]
                iKS = self.K_2_iK(Kx1+Kx2,Ky1+Ky2)
                iKD = self.K_2_iK(Kx1-Kx2,Ky1-Ky2)
                self.iKDiff[iK1,iK2] = iKD
                self.iKSum[iK1,iK2]  = iKS

    def dwave(self,kx,ky):
        return cos(kx)-cos(ky)

    def projectOnDwave(self,Ks,matrix):
        gk = self.dwave(Ks[:,0], Ks[:,1])
        c1 = dot(gk, dot(matrix,gk) ) / dot(gk,gk)
        return c1


    def dispersion(self,kx,ky):
        ek = np.zeros((self.nOrb,self.nOrb))
        rc = -2.*self.tc*(cos(kx)+cos(ky)) - 4.0*self.tcp*cos(kx)*cos(ky)
        rf = -2.*self.tf*(cos(kx)+cos(ky)) - 4.0*self.tfp*cos(kx)*cos(ky)
        ek[0,0] = rc
        ek[1,1] = rf
        ek[0,1] = -self.tz
        ek[1,0] = -self.tz
        return ek

    def selectFS(self,G4,FSpoints):
        NFs=FSpoints.shape[0]
        NwG4 = self.NwG4
        GammaFS = zeros((NFs,NFs),dtype='complex')
        for i1,iK1 in enumerate(FSpoints):
            for i2,iK2 in enumerate(FSpoints):
                GammaFS[i1,i2] = sum(G4[NwG4/2-1:NwG4/2+1,iK1,NwG4/2-1::NwG4/2+1,iK2])/float(4.*NFs)
        return GammaFS

######### Plotting functions


    def plotLeadingSolutions(self,Kvecs,lambdas,evecs,title):
        mpl.style.use("ggplot")

        Nc  = Kvecs.shape[0]
        for ic in range(Nc):
            if Kvecs[ic,0] > pi: Kvecs[ic,0] -=2.*pi
            if Kvecs[ic,1] > pi: Kvecs[ic,1] -=2.*pi

        fig, axes = mpl.subplots(nrows=4,ncols=4, sharex=True,sharey=True,figsize=(16,16))
        inr=0
        for ax in axes.flat:
            self.plotEV(ax,Kvecs,lambdas,evecs,inr)
            inr += 1
            ax.set(adjustable='box-forced', aspect='equal')
        fig.suptitle(title, fontsize=20)

    def plotEV(self,ax,Kvecs,lambdas,evecs,inr):
        Nc = evecs.shape[1]; Nw = self.evecs.shape[0]
        iw0=Nw/2
        imax = argmax(evecs[iw0,:,inr])
        if (abs(evecs[iw0-1,imax,inr]-evecs[iw0,imax,inr]) <= 1.0e-5):
            freqString = "; even frequency"
        else:
            freqString = "; odd frequency"

        colVec = Nc*[rcParams['axes.color_cycle'][0]]
        for ic in range(Nc):
            if real(evecs[iw0,ic,inr])*real(evecs[iw0,imax,inr]) < 0.0: colVec[ic] = rcParams['axes.color_cycle'][1]
        # print "colVec=",colVec
        ax.scatter(Kvecs[:,0]/pi,Kvecs[:,1]/pi,s=abs(real(evecs[iw0,:,inr]))*2500,c=colVec)
        ax.set(aspect=1)
        ax.set_xlim(-1.0,1.2); ax.set_ylim(-1.0,1.2)
        ax.set_title("$\lambda=$"+str(round(lambdas[inr],4))+freqString)
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        ax.grid(True)
        for tic in ax.xaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
        for tic in ax.yaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False


    def apply_symmetry_in_wn(self,G4):
        # for G4[w1,w2,K1,K2]
        # apply symmetry G4(wn,wn',K,K') = G4*(-wn,-wn',K,K')
        Nc = G4.shape[2]
        nwn = G4.shape[0]
        for iw1 in range(nwn):
            for iw2 in range(nwn):
                for iK1 in range(Nc):
                    for iK2 in range(Nc):
                        imw1 = nwn-1-iw1
                        imw2 = nwn-1-iw2
                        tmp1 = G4[iw1,iw2,iK1,iK2]
                        tmp2 = G4[imw1,imw2,iK1,iK2]
                        G4[iw1,iw2,iK1,iK2]   = 0.5*(tmp1+conj(tmp2))
                        G4[imw1,imw2,iK1,iK2] = 0.5*(conj(tmp1)+tmp2)

    def apply_transpose_symmetry(self,G4):
        # Apply symmetry Gamma(K,K') = Gamma(K',K)
        Nc = G4.shape[2]; nwn = G4.shape[0]; nt =Nc*nwn
        GP = np.swapaxes(G4,1,2).reshape(nt,nt)
        GP = 0.5*(GP + GP.transpose())
        G4 = GP.reshape(nwn,nwn,Nc,Nc)

    def apply_ph_symmetry_pp(self,G4):
        # G4pp(k,wn,k',wn') = G4pp(k+Q,wn,k'+Q,wn')
        Nc = G4.shape[2]
        nwn = G4.shape[0]
        for iw1 in range(nwn):
            for iw2 in range(nwn):
                for iK1 in range(Nc):
                    iK1q = self.iKSum[iK1,self.iKPiPi]
                    for iK2 in range(Nc):
                        iK2q = self.iKSum[iK2,self.iKPiPi]
                        tmp1 = G4[iw1,iw2,iK1,iK2]
                        tmp2 = G4[iw1,iw2,iK1q,iK2q]
                        G4[iw1,iw2,iK1,iK2]   = 0.5*(tmp1+tmp2)
                        G4[iw1,iw2,iK1q,iK2q] = 0.5*(tmp1+tmp2)

temps = [0.2, 0.15, 0.1, 0.08, 0.07, 0.06, 0.05]
for T_ind, T in enumerate(temps):
    print "T =", T
    file = "./T="+str(temps[T_ind])+"/dca_tp.hdf5"
    if(os.path.exists(file)):
      p = BSE(file,draw=False,useG0=False,symmetrize_G4=False,phSymmetry=False,calcRedVertex=True,calcCluster=False,nkfine=100)



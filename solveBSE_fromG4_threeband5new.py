import numpy as np
import math
from numpy import *
import matplotlib.pyplot as mpl
import h5py
import sys
import os
from matplotlib.pyplot import *
import matplotlib as mpll
#import symmetrize_Nc4x4


class BSE:

    def __init__(self,fileG4,fileG="dca_tp.hdf5",draw=True,useG0=False,symmetrize_G4=False,phSymmetry=False,calcRedVertex=False,calcCluster=True,nkfine=100):
        self.fileG4 = fileG4
        self.fileG = fileG
        self.draw = draw
        self.useG0 = useG0
        self.symmetrize_G4 = symmetrize_G4
        print("self.symmetrize_G4=", self.symmetrize_G4)
        self.calcCluster = calcCluster
        self.calcRedVertex = calcRedVertex
        self.phSymmetry = phSymmetry
        self.readData()
        #self.calcPS()
        self.reorderG4()
        self.setupMomentumTables()
        self.determine_iKPiPi()
        print ("Index of (pi,pi): ",self.iKPiPi)
        self.determine_specialK()
        print ("Index of (pi,pi): ",self.iKPiPi)
        print ("Index of (pi,0): ",self.iKPi0)
        self.calcDwaveSCClusterSus()
        self.calcChi0Cluster()
        self.calcGammaIrr()
        if calcCluster == False: self.buildChi0Lattice(nkfine)
        self.buildKernelMatrix()
        self.calcKernelEigenValues()            
        if self.draw: self.plotLeadingSolutions(self.Kvecs,self.lambdas,self.evecs[:,:,0,0,:],"Cu-Cu")
        #if calcRedVertex: self.calcReducibleLatticeVertex()
        #if self.vertex_channel in ("PARTICLE_PARTICLE_SUPERCONDUCTING","PARTICLE_PARTICLE_UP_DOWN"):
        #    if calcCluster == False: self.calcSCSus()
        #    self.calcDwaveSCClusterSus()

    # read basic parameters from the data and the cluster one and two particle Green's function
    def readData(self):
        f = h5py.File(self.fileG4,'r')
        # self.iwm = array(f['parameters']['vertex-channel']['w-channel'])[0] # transferred frequency in units of 2*pi*temp
        self.iwm = array(f['parameters']['four-point']['frequency-transfer'])[0] # transferred frequency in units of 2*pi*temp
        print("Transferred frequency iwm = ",self.iwm)
        # self.qchannel = array(f['parameters']['vertex-channel']['q-channel'])
        self.qchannel = array(f['parameters']['four-point']['momentum-transfer'])
        print("Transferred momentum q = ",self.qchannel)
        # a = array(f['parameters']['vertex-channel']['vertex-measurement-type'])[:]
        #a = array(f['parameters']['four-point']['channels']['data'])[0]
        #self.vertex_channel = ''.join(chr(i) for i in a)
        self.vertex_channel = 'PARTICLE_PARTICLE_UP_DOWN'
        print("Vertex channel = ",self.vertex_channel)
        self.invT = array(f['parameters']['physics']['beta'])[0]
        print("Inverse temperature = ",self.invT)
        self.temp = 1.0/self.invT
        self.Upp = array(f['parameters']['threebands-Hubbard-model']['U_pp'])[0]
        print("U_pp = ",self.Upp)
        self.Udd = array(f['parameters']['threebands-Hubbard-model']['U_dd'])[0]
        print("U_dd = ",self.Udd)
        self.tpp = array(f['parameters']['threebands-Hubbard-model']['t_pp'])[0]
        print("t_pp = ",self.tpp)
        self.tpd = array(f['parameters']['threebands-Hubbard-model']['t_pd'])[0]
        print("t_pd = ",self.tpd)
        self.epp = array(f['parameters']['threebands-Hubbard-model']['ep_p'])[0]
        print("ep_p = ",self.epp)
        self.epd = array(f['parameters']['threebands-Hubbard-model']['ep_d'])[0]
        print("ep_d = ",self.epd)
        self.fill = array(f['parameters']['physics']['density'])[0]
        print("filling = ",self.fill)
        self.dens = array(f['DCA-loop-functions']['density']['data'])
        print("actual filling:",self.dens)
        self.nk = array(f['DCA-loop-functions']['n_k']['data'])
        self.sign = array(f['DCA-loop-functions']['sign']['data'])
        print("sign:",self.sign)
        self.orbital=array(f['DCA-loop-functions']['orbital-occupancies']['data'])
        print("orbital occupancy:",self.orbital[self.orbital.shape[0]-1])
        print("Cu filling =", self.orbital[self.orbital.shape[0]-1,0,0]+self.orbital[self.orbital.shape[0]-1,1,0])
        print("Ox filling =", self.orbital[self.orbital.shape[0]-1,0,1]+self.orbital[self.orbital.shape[0]-1,1,1])
        print("Oy filling =", self.orbital[self.orbital.shape[0]-1,0,2]+self.orbital[self.orbital.shape[0]-1,1,2])
        self.sigmaarray=array(f['DCA-loop-functions']['L2_Sigma_difference']['data'])
        print("L2_Sigma_difference =", self.sigmaarray)
        # Now read the 4-point Green's function
        # G4Re  = array(f['functions']['G4_k_k_w_w']['data'])[:,:,:,:,:,:,:,:,0]
        # G4Im  = array(f['functions']['G4_k_k_w_w']['data'])[:,:,:,:,:,:,:,:,1]
        
        G4Re  = array(f['functions']['G4_PARTICLE_PARTICLE_UP_DOWN']['data'])[0,0,:,:,:,:,:,:,:,:,0]
        G4Im  = array(f['functions']['G4_PARTICLE_PARTICLE_UP_DOWN']['data'])[0,0,:,:,:,:,:,:,:,:,1]
        #G4Re  = array(f['functions']['G4']['data'])[0,0,:,:,:,:,:,:,:,:,0]
        #G4Im  = array(f['functions']['G4']['data'])[0,0,:,:,:,:,:,:,:,:,1]
        self.G4 = G4Re+1j*G4Im
        # Now read the cluster Green's function
        GRe = array(f['functions']['cluster_greens_function_G_k_w']['data'])[:,:,0,:,0,:,0]
        GIm = array(f['functions']['cluster_greens_function_G_k_w']['data'])[:,:,0,:,0,:,1]
        #GRe = array(f['functions']['free_cluster_greens_function_G0_k_w']['data'])[:,:,0,:,0,:,0]
        #GIm = array(f['functions']['free_cluster_greens_function_G0_k_w']['data'])[:,:,0,:,0,:,1]
        self.Green = GRe + 1j * GIm
        #print("self.Green =",self.Green[512:1024,0,0,0])
        
        GRe = array(f['functions']['cluster_greens_function_G_k_t']['data'])[:,:,0,:,0,:,0]
        GIm = array(f['functions']['cluster_greens_function_G_k_t']['data'])[:,:,0,:,0,:,1]
        #GRe = array(f['functions']['free_cluster_greens_function_G0_k_t']['data'])[:,:,0,:,0,:,0]
        #GIm = array(f['functions']['free_cluster_greens_function_G0_k_t']['data'])[:,:,0,:,0,:,1]
        self.Greenkt = GRe + 1j * GIm
        #print("self.Greenkt =",self.Greenkt[120:137,0,0,0])
        self.Greenrt = array(f['functions']['cluster_greens_function_G_r_t']['data'])[:,:,0,:,0,:]
        #self.Greenrt = array(f['functions']['free_cluster_greens_function_G0_r_t']['data'])[:,:,0,:,0,:]
        #GIm = array(f['functions']['free_cluster_greens_function_G0_k_t']['data'])[:,:,0,:,0,:]
        #self.Greenrt = GRe + 1j * GIm
        # f2 = h5py.File(self.fileG,'r')
        # bare cluster Green's function G0(k,t)
        # self.G0kt = array(f2['functions']['free_cluster_greens_function_G0_k_t']['data'][:,:,0,:,0,:])
        # self.ntau = G0kt.shape[0]

        # bare cluster Green's function G0(k,w)
        # G0kwRe = array(f2['functions']['free_cluster_greens_function_G0_k_w']['data'][:,:,0,:,0,:,0])
        # G0kwIm = array(f2['functions']['free_cluster_greens_function_G0_k_w']['data'][:,:,0,:,0,:,1])
        # self.G0kw = G0kwRe + 1j*G0kwIm

        # Now read the self-energy
        s = np.array(f['functions']['Self_Energy']['data'])
        self.sigmaoriginal = s[:,:,0,:,0,:,0] + 1j *s[:,:,0,:,0,:,1]
        print("Imsimga=",s[127:138,1,0,0,0,0,1])
        nOrb = self.Green.shape[2]
        nw = self.Green.shape[0]
        nk  = self.Green.shape[1]
        

        # Now load frequency data
        self.wn = np.array(f['domains']['frequency-domain']['elements'])
        self.wnSet = np.array(f['domains']['vertex-frequency-domain (COMPACT)']['elements'])

        # Now read the K-vectors
        self.Kvecs = array(f['domains']['CLUSTER']['MOMENTUM_SPACE']['elements']['data'])
        print ("K-vectors: ",self.Kvecs)

        # Now read other Hubbard parameters
        self.mu = np.array(f['DCA-loop-functions']['chemical-potential']['data'])[0]
        self.nOrb = self.Green.shape[2]
        self.NwG4 = self.G4.shape[0]
        self.Nc  = self.Green.shape[1]
        self.NwG = self.Green.shape[0]
        self.NtG = self.Greenkt.shape[0]
        self.nt = self.Nc*self.NwG4*self.nOrb*self.nOrb

        print ("NwG4: ",self.NwG4)
        print ("NtG: ",self.NtG)
        print ("NwG : ",self.NwG)
        print ("Nc  : ",self.Nc)
        print ("nOrb: ",self.nOrb)
        print ("G4shape0 = ", self.G4.shape[0], ", G4shape1 = ", self.G4.shape[1], ", G4shape2 = ", self.G4.shape[2], ", G4shape3 = ", self.G4.shape[3], ", G4shape4 = ", self.G4.shape[4], ", G4shape5 = ", self.G4.shape[5], ", G4shape6 = ", self.G4.shape[6], ", G4shape7 = ", self.G4.shape[7])
        self.NwTP = 2*np.array(f['parameters']['domains']['imaginary-frequency']['four-point-fermionic-frequencies'])[0]
        self.iQ = self.K_2_iK(self.qchannel[0],self.qchannel[1])
        print ("Index of transferred momentum: ", self.iQ)
        #self.ddwave = array(f['CT-AUX-SOLVER-functions']['dwave-pp-correlator']['data'])
        #print('shape0 of ddwave=',self.ddwave.shape[0])
        #print('shape1 of ddwave=',self.ddwave.shape[1])
        #print('Cu-Cu ddwave=',self.ddwave[:,0])
        #print('Ox-Ox ddwave=',self.ddwave[:,1])
        #print('Oy-Oy ddwave=',self.ddwave[:,2])
        self.iwG40 = self.NwG4/2
        self.iwG0 = self.NwG/2

        f.close()

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
                                                self.G4[iw1,ik1,iw2,ik2,l1,l2,l3,l4] -= 2.0 * self.Green[iw1Green,ik1,l1,l2] * self.Green[iw2Green,ik2,l4,l3]


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


    def reorderG4(self):
        # In Peter's code:
        # PARTICLE_HOLE_MAGNETIC:
        #
        #       k1,l1           k2,l4                      k1,l1             k2,l3
        #     ----->------------->------                 ----->------------->------
        #             |     |                                      |     |
        #             |  G4 |              mapped onto             |  G4 |
        #     -----<-------------<------                 -----<-------------<------
        #     k1+q,l3         k2+q,l2                     k1+q,l2          k2+q,l4
        #
        
        Nc=self.Nc; NwG4=self.NwG4; NwG=self.NwG; nOrb = self.nOrb
        self.G4r=np.zeros((NwG4,Nc,nOrb,nOrb,NwG4,Nc,nOrb,nOrb),dtype='complex')
        G4susQz0 = 0.0
        G4susQzPi = 0.0
        for ik1 in range(self.Nc):
            for ik2 in range(self.Nc):
                for iw1 in range(self.NwG4):
                    for iw2 in range(self.NwG4):
                        for l1 in range(self.nOrb):
                            for l2 in range(self.nOrb):
                                for l3 in range(self.nOrb):
                                    for l4 in range(self.nOrb):
                                        if self.vertex_channel=="PARTICLE_HOLE_MAGNETIC":
                                            c1= self.G4[iw1,iw2,ik1,ik2,l1,l2,l3,l4]
                                            self.G4r[iw1,ik1,l1,l3,iw2,ik2,l4,l2]  = c1
                                            if (l1==l3) & (l4==l2):
                                                G4susQz0 += c1
                                                G4susQzPi += c1*exp(1j*np.pi*(l2-l3))
                                        elif self.vertex_channel=="PARTICLE_PARTICLE_UP_DOWN":
                                            c1 = self.G4[iw2,ik2,iw1,ik1,l4,l3,l2,l1]
                                            self.G4r[iw1,ik1,l1,l2,iw2,ik2,l3,l4] = c1                                  
                                            if (l1!=l2) & (l3!=l4):
                                                G4susQz0 += c1
                                       
        G4rtemp = self.G4r.copy()
            
       
                                                    
                                                                           
                                    
        PS=0.0; PScc=0.0; PSoxox=0.0; PSoyoy=0.0; PScox=0.0; PScoy=0.0; PSoxoy=0.0; 
        for iw1 in range(NwG4):
            for ik1 in range(Nc):
                for iw2 in range(NwG4):
                    for ik2 in range(Nc):
                        PScc += self.G4r[iw1,ik1,0,0,iw2,ik2,0,0]
                        PSoxox += self.G4r[iw1,ik1,1,1,iw2,ik2,1,1]
                        PSoyoy += self.G4r[iw1,ik1,2,2,iw2,ik2,2,2]
                        PScox += self.G4r[iw1,ik1,0,0,iw2,ik2,1,1] + self.G4r[iw1,ik1,1,1,iw2,ik2,0,0]
                        PScoy += self.G4r[iw1,ik1,0,0,iw2,ik2,2,2] + self.G4r[iw1,ik1,2,2,iw2,ik2,0,0]
                        PSoxoy += self.G4r[iw1,ik1,1,1,iw2,ik2,2,2] + self.G4r[iw1,ik1,2,2,iw2,ik2,1,1]

        PScc /= self.Nc*self.invT
        print("G4 s-wave Pairfield susceptibility for Cu-Cu is ",PScc)
        #print("chi0kiw s-wave Pairfield susceptibility error2 for Cu-Cu is ",PSccerror2)
        PSoxox /= self.Nc*self.invT
        print("G4 s-wave Pairfield susceptibility for Ox-Ox is ",PSoxox)
        PSoyoy /= self.Nc*self.invT 
        print("G4 s-wave Pairfield susceptibility for Oy-Oy is ",PSoyoy)
        PScox /= self.Nc*self.invT 
        print("G4 s-wave Pairfield susceptibility for Cu-Ox is ",PScox)
        PScoy /= self.Nc*self.invT 
        print("G4 s-wave Pairfield susceptibility for Cu-Oy is ",PScoy)
        PSoxoy /= self.Nc*self.invT 
        print("G4 s-wave Pairfield susceptibility for Ox-Oy is ",PSoxoy)
        PS = PScc+PSoxox+PSoyoy+PScox+PScoy+PSoxoy
        print("G4 s-wave Pairfield susceptibility is ",PS)
            
        
     
        
    
        #PS=0.0; PScc=0.0; PSoxox=0.0; PSoyoy=0.0; PScox=0.0; PScoy=0.0; PSoxoy=0.0; testG4susQz0=0.0; testG4susQz1=0.0;ep=0.0;
        #for iw in range(0,1024):
        #for iw in range(NwG):
        #        for ik in range(self.Nc):
        #            PScc += self.chic0kiw[iw,ik,0,0,iw,ik,0,0]
        #            PSoxox += self.chic0kiw[iw,ik,1,1,iw,ik,1,1]
        #            PSoyoy += self.chic0kiw[iw,ik,2,2,iw,ik,2,2]
        #            PScox += self.chic0kiw[iw,ik,0,0,iw,ik,1,1] + self.chic0kiw[iw,ik,1,1,iw,ik,0,0]
        #            PScoy += self.chic0kiw[iw,ik,0,0,iw,ik,2,2] + self.chic0kiw[iw,ik,2,2,iw,ik,0,0]
        #            PSoxoy += self.chic0kiw[iw,ik,1,1,iw,ik,2,2] + self.chic0kiw[iw,ik,2,2,iw,ik,1,1]
        #PSccerror1=0.0;PSccerror2=0.0;
        #for iw in range(-512,512):
        #    PSccerror1+=1.0/((2*iw+1)*1j*3.1415926/self.invT-4.25-0.9)/(-(2*iw+1)*1j*3.1415926/self.invT-4.25-0.9)
        #PSccerror1*=1.0/(self.invT)
        #PSccerror2=(1-2.0/(math.exp(self.invT*(4.25+0.9))+1))/2/(4.25+0.9)
        #PScc = PScc*(self.invT-ep)/(float(Nc)*self.invT*self.invT)
        #print("chi0kiw (sum over 64 w) s-wave Pairfield susceptibility for Cu-Cu is ",PScc)
        #print("chi0kiw s-wave Pairfield susceptibility error1 for Cu-Cu is ",PSccerror2)
        #print("chi0kiw s-wave Pairfield susceptibility error2 for Cu-Cu is ",PSccerror2)
        #PSoxox = PSoxox*(self.invT-ep)/(float(Nc)*self.invT*self.invT) 
        #print("chi0kiw (sum over 512 w) s-wave Pairfield susceptibility for Ox-Ox is ",PSoxox)
        #PSoyoy = PSoyoy*(self.invT-ep)/(float(Nc)*self.invT*self.invT) 
        #print("chi0kiw (sum over 512 w) s-wave Pairfield susceptibility for Oy-Oy is ",PSoyoy)
        #PScox = PScox*(self.invT-ep)/(float(Nc)*self.invT*self.invT) 
        #print("chi0kiw (sum over 512 w) s-wave Pairfield susceptibility for Cu-Ox is ",PScox)
        #PScoy = PScoy*(self.invT-ep)/(float(Nc)*self.invT*self.invT) 
        #print("chi0kiw (sum over 512 w) s-wave Pairfield susceptibility for Cu-Oy is ",PScoy)
        #PSoxoy = PSoxoy*(self.invT-ep)/(float(Nc)*self.invT*self.invT) 
        #print("chi0kiw (sum over 512 w) s-wave Pairfield susceptibility for Ox-Oy is ",PSoxoy)
        #PS = PScc+PSoxox+PSoyoy+PScox+PScoy+PSoxoy
        #print("chi0kiw (sum over 512 w) s-wave Pairfield susceptibility is ",PS)        
        
        if self.symmetrize_G4:
            nwn = self.G4r.shape[0]
            
            
            sym=symmetrize_Nc4x4.symmetrize()
            type=dtype(self.G4r[0,0,0,0,0,0,0,0])
            for iK1 in range(0,Nc):
                for iK2 in range(0,Nc):
                    #if (iK1==iK2):
                    tmp = zeros((nwn,nwn),dtype=type)
                    for iSym in [0]: # Apply every point-group symmetry operation
                        iK1Trans = sym.symmTrans_of_iK(iK1,iSym)
                        iK2Trans = sym.symmTrans_of_iK(iK2,iSym)
                        tmp += self.G4r[:,iK1Trans,0,0,:,iK2Trans,0,0]        
                    for iSym in [0]:
                        iK1Trans = sym.symmTrans_of_iK(iK1,iSym)
                        iK2Trans = sym.symmTrans_of_iK(iK2,iSym)
                        self.G4r[:,iK1Trans,0,0,:,iK2Trans,0,0] = tmp
        
            #for iw1 in range(nwn):
            #    for iw2 in range(nwn):
            #        imw1 = nwn-1-iw1
            #        imw2 = nwn-1-iw2
            #        tmp1 = self.G4r[iw1,:,:,:,iw2,:,:,:]
            #        tmp2 = self.G4r[imw1,:,:,:,imw2,:,:,:]
            #        self.G4r[iw1,:,:,:,iw2,:,:,:]   = 0.5*(tmp1+conj(tmp2))
            #        self.G4r[imw1,:,:,:,imw2,:,:,:] = 0.5*(conj(tmp1)+tmp2)

            #GP = self.G4r.reshape(self.nt,self.nt)
            #GP = 0.5*(GP + GP.transpose())
            #self.G4r = GP.reshape(nwn,Nc,nOrb,nOrb,nwn,Nc,nOrb,nOrb)

        self.G4M = self.G4r.reshape(self.nt,self.nt)
        
        if self.vertex_channel=="PARTICLE_HOLE_MAGNETIC":
            print("Cluster Chi(q,qz=0) :", G4susQz0/(self.invT*self.Nc*2.0))
            print("Cluster Chi(q,qz=pi):", G4susQzPi/(self.invT*self.Nc*2.0))        
        
        if self.vertex_channel=="PARTICLE_PARTICLE_UP_DOWN":
            print("Cluster inter-orbital Chi(q=0):", G4susQz0/(self.invT*self.Nc*4.0))

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

    def calcChi0Cluster(self):
        print ("Now calculating chi0 on cluster")
        self.chic0  = zeros((self.NwG4,self.Nc,self.nOrb,self.nOrb,self.NwG4,self.Nc,self.nOrb,self.nOrb),dtype='complex')
        #self.chic0ktau  = zeros((self.NtG,self.Nc,self.nOrb,self.nOrb,self.NtG,self.Nc,self.nOrb,self.nOrb),dtype='complex')
        #self.chic0kiw  = zeros((self.NwG,self.Nc,self.nOrb,self.nOrb,self.NwG,self.Nc,self.nOrb,self.nOrb),dtype='complex')

        Nc=self.Nc; NwG4=self.NwG4; NwG=self.NwG; nOrb = self.nOrb; NtG=self.NtG; c2=0.0;
        
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
                                    #if (l1==l2==l3==l4==0) & (iw==0): print("ik=",ik," ikPlusQ=",ikPlusQ)
                                    if (l2 != l4 and l2 == 0) or (l2 != l4 and l4 == 0): 
                                        c1 = -self.Green[iw1,ik,l3,l1] * self.Green[minusiwPlusiwm,ikPlusQ,l4,l2]
                                    else:
                                        c1 = self.Green[iw1,ik,l3,l1] * self.Green[minusiwPlusiwm,ikPlusQ,l4,l2]
                              
                                    self.chic0[iw,ik,l1,l2,iw,ik,l3,l4] = c1
        else:
            G4susQz0 = 0.0
            G4susQzPi = 0.0
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
                                    c1 = - self.Green[iw1,ik,l1,l3] * self.Green[iwPlusiwm,ikPlusQ,l4,l2]
                                    self.chic0[iw,ik,l1,l2,iw,ik,l3,l4] = c1
                                    if (l1==l2) & (l3==l4):
                                        G4susQz0 += c1
                                        G4susQzPi += c1*exp(1j*np.pi*(l2-l3))

        self.chic0M = self.chic0.reshape(self.nt,self.nt)
        

        if self.vertex_channel=="PARTICLE_HOLE_MAGNETIC":
            print("Cluster Chi0(q,qz=0) :", G4susQz0/(self.invT*self.Nc*2.0))
            print("Cluster Chi0(q,qz=pi):", G4susQzPi/(self.invT*self.Nc*2.0))
        
        
    def calcDwaveSCClusterSus(self):
        print (" ")
        print (" ")
        gksx = cos(self.Kvecs[:,0]) + cos(self.Kvecs[:,1])
        csumCuCu = 0.0; csumOxOx = 0.0; csumOyOy = 0.0; csumCuOx = 0.0; csumCuOy = 0.0; csumOxOy = 0.0;
        

        for iK1 in range(self.Nc):
            for iK2 in range(self.Nc):
                csumCuCu += gksx[iK1]*sum(self.G4r[:,iK1,0,0,:,iK2,0,0])*gksx[iK2]
                csumOxOx += gksx[iK1]*sum(self.G4r[:,iK1,1,1,:,iK2,1,1])*gksx[iK2]
                csumOyOy += gksx[iK1]*sum(self.G4r[:,iK1,2,2,:,iK2,2,2])*gksx[iK2]
                csumCuOx += gksx[iK1]*sum(self.G4r[:,iK1,0,0,:,iK2,1,1])*gksx[iK2] + gksx[iK1]*sum(self.G4r[:,iK1,1,1,:,iK2,0,0])*gksx[iK2]
                csumCuOy += gksx[iK1]*sum(self.G4r[:,iK1,0,0,:,iK2,2,2])*gksx[iK2] + gksx[iK1]*sum(self.G4r[:,iK1,2,2,:,iK2,0,0])*gksx[iK2]
                csumOxOy += gksx[iK1]*sum(self.G4r[:,iK1,1,1,:,iK2,2,2])*gksx[iK2] + gksx[iK1]*sum(self.G4r[:,iK1,2,2,:,iK2,1,1])*gksx[iK2]

        csumCuCu /= self.Nc*self.invT
        csumOxOx /= self.Nc*self.invT
        csumOyOy /= self.Nc*self.invT
        csumCuOx /= self.Nc*self.invT
        csumCuOy /= self.Nc*self.invT
        csumOxOy /= self.Nc*self.invT        
        #self.Pdc = real(csum)
        print ("G4 sx-wave Cu-Cu Pairfield susceptibility: ",csumCuCu)
        print ("G4 sx-wave Ox-Ox Pairfield susceptibility: ",csumOxOx)
        print ("G4 sx-wave Oy-Oy Pairfield susceptibility: ",csumOyOy)
        print ("G4 sx-wave Cu-Ox Pairfield susceptibility: ",csumCuOx)
        print ("G4 sx-wave Cu-Oy Pairfield susceptibility: ",csumCuOy)
        print ("G4 sx-wave Ox-Oy Pairfield susceptibility: ",csumOxOy)
        
        print (" ")
        print (" ")
        gkd = cos(self.Kvecs[:,0]) - cos(self.Kvecs[:,1])
        csumCuCu = 0.0; csumOxOx = 0.0; csumOyOy = 0.0; csumCuOx = 0.0; csumCuOy = 0.0; csumOxOy = 0.0;
        

        for iK1 in range(self.Nc):
            for iK2 in range(self.Nc):
                csumCuCu += gkd[iK1]*sum(self.G4r[:,iK1,0,0,:,iK2,0,0])*gkd[iK2]
                csumOxOx += gkd[iK1]*sum(self.G4r[:,iK1,1,1,:,iK2,1,1])*gkd[iK2]
                csumOyOy += gkd[iK1]*sum(self.G4r[:,iK1,2,2,:,iK2,2,2])*gkd[iK2]
                csumCuOx += gkd[iK1]*sum(self.G4r[:,iK1,0,0,:,iK2,1,1])*gkd[iK2] + gkd[iK1]*sum(self.G4r[:,iK1,1,1,:,iK2,0,0])*gkd[iK2]
                csumCuOy += gkd[iK1]*sum(self.G4r[:,iK1,0,0,:,iK2,2,2])*gkd[iK2] + gkd[iK1]*sum(self.G4r[:,iK1,2,2,:,iK2,0,0])*gkd[iK2]
                csumOxOy += gkd[iK1]*sum(self.G4r[:,iK1,1,1,:,iK2,2,2])*gkd[iK2] + gkd[iK1]*sum(self.G4r[:,iK1,2,2,:,iK2,1,1])*gkd[iK2]

        csumCuCu /= self.Nc*self.invT
        csumOxOx /= self.Nc*self.invT
        csumOyOy /= self.Nc*self.invT
        csumCuOx /= self.Nc*self.invT
        csumCuOy /= self.Nc*self.invT
        csumOxOy /= self.Nc*self.invT        
        #self.Pdc = real(csum)
        print ("G4 d-wave Cu-Cu Pairfield susceptibility: ",csumCuCu)
        print ("G4 d-wave Ox-Ox Pairfield susceptibility: ",csumOxOx)
        print ("G4 d-wave Oy-Oy Pairfield susceptibility: ",csumOyOy)
        print ("G4 d-wave Cu-Ox Pairfield susceptibility: ",csumCuOx)
        print ("G4 d-wave Cu-Oy Pairfield susceptibility: ",csumCuOy)
        print ("G4 d-wave Ox-Oy Pairfield susceptibility: ",csumOxOy)
        
        print (" ")
        print (" ")
        
        
            
    def calcPS(self):
        # Calculate the S wave pairfield susp
        Nc=self.Nc; NwG4=self.NwG4; NwG=self.NwG; nt = self.nt; nOrb = self.nOrb; PS=0.0; PScc=0.0;
        PSoxox=0.0; PSoyoy=0.0; PScox=0.0; PScoy=0.0; PSoxoy=0.0; testG4susQz0=0.0; testG4susQz1=0.0;
        for ik1 in range(self.Nc):
            for ik2 in range(self.Nc):
                for iw1 in range(self.NwG4):
                    for iw2 in range(self.NwG4):
                        for l1 in range(nOrb):
                            for l2 in range(nOrb):
                                for l3 in range(nOrb):
                                    for l4 in range(nOrb):
                                        if (l1!=l2) & (l3!=l4):
                                            testG4susQz0 += self.G4[iw2,ik2,iw1,ik1,l4,l3,l2,l1]
                                        if (l1==l2) & (l3==l4) & (l1==l3):
                                            testG4susQz1 += self.G4[iw2,ik2,iw1,ik1,l4,l3,l2,l1]
        print("Test Cluster inter-orbital Chi(q=0):", testG4susQz0/(self.invT*self.Nc*4.0))
        print("Test Cluster intra-orbital Chi(q=0):", testG4susQz1/(self.invT*self.Nc*4.0))
              
        
        
              
    
                                        
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

        # Load parameters: t,mu
        # t1 = self.t1
        # t2 = self.t2
        mu = self.mu          

        self.sigma = np.zeros((self.NwG,self.Nc,self.nOrb,self.nOrb),dtype='complex')
        self.sigma = self.sigmaoriginal
        

        self.sigmanegk = np.zeros((self.NwG,self.Nc,self.nOrb,self.nOrb),dtype='complex')
        for l2 in range(self.nOrb):
            for l4 in range(self.nOrb):
                if (l2 != l4 and l2 == 0) or (l2 != l4 and l4 == 0): 
                    self.sigmanegk[:,:,l2,l4] = -self.sigma[:,:,l2,l4]
                else:
                    self.sigmanegk[:,:,l2,l4] = self.sigma[:,:,l2,l4]

        # Now coarse-grain G*G to build chi0(K) = Nc/N sum_k Gc(K+k')Gc(-K-k')
        nOrb = self.nOrb; nw = wnSet.shape[0]; nk=Kset.shape[0]
        self.chi0  = np.zeros((nw,nk,nOrb,nOrb,nw,nk,nOrb,nOrb),dtype='complex')
        self.chi0D  = np.zeros((nw,nk,nOrb,nOrb,nw,nk,nOrb,nOrb),dtype='complex')
        self.chi0D2  = np.zeros((nw,nk,nOrb,nOrb,nw,nk,nOrb,nOrb),dtype='complex')
        self.chi0XS  = np.zeros((nw,nk,nOrb,nOrb,nw,nk,nOrb,nOrb),dtype='complex')
        self.chi0XS2  = np.zeros((nw,nk,nOrb,nOrb,nw,nk,nOrb,nOrb),dtype='complex')
        self.gkdNorm = 0.0
        #self.cG    = np.zeros((nw,nk,nOrb,nOrb),dtype='complex')
        #self.cG0   = np.zeros((nw,nk,nOrb,nOrb),dtype='complex')
        for iwn,wn in enumerate(wnSet): # reduced tp frequencies !!
            print("iwn = ",iwn)
            iwG = int(iwn - self.iwG40 + self.iwG0)
            for iK,K in enumerate(Kset):
                c0 = np.zeros((nOrb,nOrb,nOrb,nOrb),dtype='complex')
                c1 = np.zeros((nOrb,nOrb,nOrb,nOrb),dtype='complex')
                c2 = np.zeros((nOrb,nOrb,nOrb,nOrb),dtype='complex')
                c3 = np.zeros((nOrb,nOrb,nOrb,nOrb),dtype='complex')
                c4 = np.zeros((nOrb,nOrb,nOrb,nOrb),dtype='complex')
                c5 = np.zeros((nOrb,nOrb,nOrb,nOrb),dtype='complex')
                #cG = np.zeros((nOrb,nOrb),dtype='complex')
                #cG0 = np.zeros((nOrb,nOrb),dtype='complex')
                for k in kPatch:
                    kx = K[0]+k[0]; ky = K[1]+k[1]
                    if (K[0]<-0.01):
                        kx += 2*np.pi
                    if (K[1]<-0.01):
                        ky += 2*np.pi
                    ek = self.dispersion(kx,ky)
                    gkd = cos(kx) - cos(ky)
                    gkxs= cos(kx) + cos(ky)
                    #G0inv = (1j*wn+self.mu-self.U1*(self.dens[-1]/4.-0.5))*np.identity(nOrb) - ek
                    #G0 = linalg.inv(G0inv)
                    Qx = self.qchannel[0]; Qy = self.qchannel[1]
                    if (self.vertex_channel == "PARTICLE_PARTICLE_UP_DOWN"):
                        emkpq = self.dispersion(-kx+Qx, -ky+Qy)
                        iKQ = self.iKSum[self.iKDiff[0,iK],self.iQ]
                        #emkpq = self.dispersion(Kset[iKQ,0], Kset[iKQ,1])
                        #print("iK=",iK,"iKQ=",iKQ,"iKQx=",Kset[iKQ,0],"iKQy=",Kset[iKQ,1])
                        minusiwPlusiwm = min(max(NwG-iwG-1 + self.iwm,0),NwG-1) # -iwn + iwm
                        #minusiwPlusiwm = int(min(max(NwG-iw1-1 + self.iwm,0),NwG-1))
                        G1inv = (1j*wn+self.mu) * np.identity(nOrb)-ek-self.sigma[iwG,iK,:,:]
                        G2inv = (-1j*wn+self.mu)* np.identity(nOrb)-emkpq-self.sigmanegk[minusiwPlusiwm,iK,:,:]
                        G1 = linalg.inv(G1inv); G2 = linalg.inv(G2inv)

                    else:
                        ekpq = self.dispersion(kx+Qx, ky+Qy)
                        iKQ = int(self.iKSum[iK,self.iQ])
                        iwPlusiwm = int(min(max(iwG + self.iwm,0),NwG-1))  # iwn+iwm

                        G1inv = (1j*wn+self.mu)*np.identity(nOrb)-ek-self.sigma[iwG,iK,:,:]
                        G2inv = (1j*wn+self.mu)*np.identity(nOrb)-ekpq-self.sigmanegk[iwPlusiwm,iKQ,:,:]
                        G1 = linalg.inv(G1inv); G2 = -linalg.inv(G2inv)

                    for l1 in range(nOrb):
                        for l2 in range(nOrb):
                            for l3 in range(nOrb):
                                for l4 in range(nOrb):
                                        c0[l1,l2,l3,l4] = G1[l3,l1]*G2[l4,l2]
                                    
                    c1[:,:,:,:] += c0[:,:,:,:]
                    c2[:,:,:,:] += c0[:,:,:,:] * gkd
                    c3[:,:,:,:] += c0[:,:,:,:] * gkd**2
                    c4[:,:,:,:] += c0[:,:,:,:] * gkxs
                    c5[:,:,:,:] += c0[:,:,:,:] * gkxs**2
                    if (iwn==0): self.gkdNorm += gkd**2

                self.chi0[iwn,iK,:,:,iwn,iK,:,:]  = c1[:,:,:,:]/kPatch.shape[0]
                self.chi0D[iwn,iK,:,:,iwn,iK,:,:]  = c2[:,:,:,:]/kPatch.shape[0]
                self.chi0D2[iwn,iK,:,:,iwn,iK,:,:]  = c3[:,:,:,:]/kPatch.shape[0]
                self.chi0XS[iwn,iK,:,:,iwn,iK,:,:]  = c4[:,:,:,:]/kPatch.shape[0]
                self.chi0XS2[iwn,iK,:,:,iwn,iK,:,:]  = c5[:,:,:,:]/kPatch.shape[0]
                #self.cG[iwn,iK,:,:] = cG[:,:]/kPatch.shape[0]
                #self.cG0[iwn,iK,:,:] = cG0[:,:]/kPatch.shape[0]

        
        #sym=symmetrize_Nc4x4.symmetrize()
        #nwn = self.chi0.shape[0]
        #type=dtype(self.chi0[0,0,0,0,0,0,0,0])
        #for iK1 in range(0,Nc):
        #    for iK2 in range(0,Nc):
        #        tmp = zeros((nwn,nwn),dtype=type)
        #        for iSym in range(8): # Apply every point-group symmetry operation
        #            iK1Trans = sym.symmTrans_of_iK(iK1,iSym)
        #            iK2Trans = sym.symmTrans_of_iK(iK2,iSym)
        #            tmp += self.chi0[:,iK1Trans,0,0,:,iK2Trans,0,0]
        #            print("isym=",iSym,"iK1=",iK1,"iK1Trans=",iK1Trans,"iK2=",iK2,"iK2Trans=",iK2Trans)
        #            print("chi0[15,iK1Trans,0,0,15,iK2Trans,0,0]=",self.chi0[15,iK1Trans,0,0,15,iK2Trans,0,0])

        #        for iSym in range(8):
        #            iK1Trans = sym.symmTrans_of_iK(iK1,iSym)
        #            iK2Trans = sym.symmTrans_of_iK(iK2,iSym)
        #            self.chi0[:,iK1Trans,0,0,:,iK2Trans,0,0] = tmp/8.
        
        
        self.chi0M = self.chi0.reshape(self.nt,self.nt)
        self.gkdNorm /= kPatch.shape[0]


        if self.vertex_channel=="PARTICLE_HOLE_MAGNETIC":
            chi0Loc = sum(sum(sum(sum(self.chi0,axis=0),axis=0),axis=2),axis=2)
            chi00 = 0.0; chi0Pi = 0.0
            for l1 in range(0,nOrb):
                for l3 in range(0,nOrb):
                    chi00  += chi0Loc[l1,l1,l3,l3]
                    chi0Pi += chi0Loc[l1,l1,l3,l3] * exp(1j*np.pi*(l1-l3))

            print("Lattice Chi0(q,qz=0) :", chi00 /(self.invT*self.Nc*2.0))
            print("Lattice Chi0(q,qz=pi):", chi0Pi/(self.invT*self.Nc*2.0))

            
            
    def calcSCSus(self):
        # Calculate from gd*G4*gd = gd*GG*gd + gd*GG*GammaRed*GG*gd
        #GammaRed = self.GammaRed.reshape(self.NwG4*self.Nc,self.NwG4*self.Nc)
        print("")
        gkd = cos(self.Kvecs[:,0]) - cos(self.Kvecs[:,1])
        nOrb = self.nOrb;Nc=self.Nc;NwG4=self.NwG4;
        csum = np.zeros((nOrb,nOrb,nOrb,nOrb),dtype='complex')
        csumxs = np.zeros((nOrb,nOrb,nOrb,nOrb),dtype='complex')
        ccsum = np.zeros((nOrb,nOrb,nOrb,nOrb),dtype='complex')
        csum3 = np.zeros((nOrb,nOrb,nOrb,nOrb),dtype='complex')
        csum1 = np.zeros((nOrb,nOrb,nOrb,nOrb),dtype='complex')
        csum1xs = np.zeros((nOrb,nOrb,nOrb,nOrb),dtype='complex')
        ccsum1 = np.zeros((nOrb,nOrb,nOrb,nOrb),dtype='complex')
        tempchi  = np.zeros((NwG4,Nc),dtype='complex')



        for iw1 in range(NwG4):
            for iK1 in range(Nc):
                for iw2 in range(NwG4):
                    for iK2 in range(Nc):
                        for l1 in range(nOrb):
                            for l2 in range(nOrb):
                                for l3 in range(nOrb):
                                    for l4 in range(nOrb):
                                        for l5 in range(nOrb):
                                            for l6 in range(nOrb):
                                                for l7 in range(nOrb):
                                                    for l8 in range(nOrb):
                                                        csum[l1,l2,l3,l4]  += self.chi0D[iw1,iK1,l1,l2,iw1,iK1,l5,l6] * self.GammaRed[iw1,iK1,l5,l6,iw2,iK2,l7,l8] * self.chi0D[iw2,iK2,l7,l8,iw2,iK2,l3,l4]
                                                        csumxs[l1,l2,l3,l4] += self.chi0XS[iw1,iK1,l1,l2,iw1,iK1,l5,l6] * self.GammaRed[iw1,iK1,l5,l6,iw2,iK2,l7,l8] * self.chi0XS[iw2,iK2,l7,l8,iw2,iK2,l3,l4]
                                                        ccsum[l1,l2,l3,l4] += gkd[iK1] * self.chi0[iw1,iK1,l1,l2,iw1,iK1,l5,l6] * self.GammaRed[iw1,iK1,l5,l6,iw2,iK2,l7,l8] * self.chi0[iw2,iK2,l7,l8,iw2,iK2,l3,l4] * gkd[iK2]
        csum[:,:,:,:] /= (self.Nc*self.invT)**2
        csumxs[:,:,:,:] /= (self.Nc*self.invT)**2
        ccsum[:,:,:,:] /= (self.Nc*self.invT)**2
        self.chi0D2sim = np.zeros((NwG4,Nc,nOrb,nOrb,nOrb,nOrb),dtype='complex')
        self.chi0XS2sim = np.zeros((NwG4,Nc,nOrb,nOrb,nOrb,nOrb),dtype='complex')
        self.chi0sim = np.zeros((NwG4,Nc,nOrb,nOrb,nOrb,nOrb),dtype='complex')
        for iw in range(NwG4):
            for iK in range(Nc):
                self.chi0D2sim[iw,iK,:,:,:,:] = self.chi0D2[iw,iK,:,:,iw,iK,:,:]
                self.chi0XS2sim[iw,iK,:,:,:,:] = self.chi0XS2[iw,iK,:,:,iw,iK,:,:]
                self.chi0sim[iw,iK,:,:,:,:] = self.chi0[iw,iK,:,:,iw,iK,:,:]
        for l1 in range(nOrb):
            for l2 in range(nOrb):
                for l3 in range(nOrb):
                    for l4 in range(nOrb):
                        csum1[l1,l2,l3,l4]  = sum(self.chi0D2sim[:,:,l1,l2,l3,l4])/(self.Nc*self.invT)
                        csum1xs[l1,l2,l3,l4] = sum(self.chi0XS2sim[:,:,l1,l2,l3,l4])/(self.Nc*self.invT)
                        ccsum1[l1,l2,l3,l4] = sum(np.dot(self.chi0sim[:,:,l1,l2,l3,l4],gkd**2))/(self.Nc*self.invT)
        csum[:,:,:,:] += csum1[:,:,:,:]
        csumxs[:,:,:,:] += csum1xs[:,:,:,:]
        ccsum[:,:,:,:] += ccsum1[:,:,:,:]
        self.Pd = real(csum)
        self.csum = csum
        self.ccsum = ccsum
        self.Pxs = real(csumxs)
        self.Pdgkc = real(ccsum)
        #csum3 = sum(real(self.chi0D2[abs(self.wnSet) <= 2.*4*self.t**2/self.U,:]))/(self.Nc*self.invT)
        print ("Calculations from GammaRed:")
        print ("d-wave  SC susceptibility: ",csum[0,0,0,0],csum[1,1,1,1],csum[2,2,2,2])
        print ("xs-wave SC susceptibility: ",csumxs[0,0,0,0],csumxs[1,1,1,1],csumxs[2,2,2,2])
        print("")
        print ("bare d-wave SC susceptibility:  ",csum1[0,0,0,0],csum1[1,1,1,1],csum1[2,2,2,2])
        print ("bare xs-wave SC susceptibility: ",csum1xs[0,0,0,0],csum1xs[1,1,1,1],csum1xs[2,2,2,2])
        #print ("bare d-wave SC susceptibility with cutoff wc = J: ",csum3)
        print ("bare d-wave SC lattice (with cluster form-factors) susceptibility: ",ccsum1[0,0,0,0],ccsum1[1,1,1,1],ccsum1[2,2,2,2])
        print ("d-wave SC lattice (with cluster form-factors) susceptibility: ",ccsum[0,0,0,0],ccsum[1,1,1,1],ccsum[2,2,2,2])
        print("")

            

    def calcGammaIrr(self):
        # Calculate the irr. GammaIrr
        Nc=self.Nc; NwG4=self.NwG4; NwG=self.NwG; nt = self.nt; nOrb = self.nOrb
        
        #print("chic0M(105,106)=",self.chic0M[105,106],"chic0(2,3,2,0,2,3,2,1)=",self.chic0[2,3,2,0,2,3,2,1])
        #print("chic0M(106,105)=",self.chic0M[106,105],"chic0(2,3,2,1,2,3,2,0)=",self.chic0[2,3,2,1,2,3,2,0])
        #for i in range(0,nt):
        #    for j in range(i,nt):
        #        c1 = 0.5*(self.GammaM[i,j]+self.GammaM[j,i])
        #        self.GammaM[i,j] = c1
        #        self.GammaM[j,i] = c1

        G4M = linalg.inv(self.G4M)
        chic0M = linalg.inv(self.chic0M)
        
        
        self.GammaM = chic0M - G4M        
        
        #self.GammaM *= float(Nc)*self.invT*float(self.nOrb)
        self.GammaM *= float(Nc)*self.invT

                
        self.Gamma = self.GammaM.reshape(NwG4,Nc,nOrb,nOrb,NwG4,Nc,nOrb,nOrb)
                
        Gamma1 = self.Gamma.copy()
        for iw2 in range(NwG4):
            self.Gamma[:,:,:,:,iw2,:,:,:]=(Gamma1[:,:,:,:,iw2,:,:,:]+Gamma1[:,:,:,:,NwG4-iw2-1,:,:,:])/2
        Gamma1 = self.Gamma.copy()
        for iw1 in range(NwG4):
            self.Gamma[iw1,:,:,:,:,:,:,:]=(Gamma1[iw1,:,:,:,:,:,:,:]+Gamma1[NwG4-iw1-1,:,:,:,:,:,:,:])/2
            
            
    def buildKernelMatrix(self):
        Nc=self.Nc; NwG4=self.NwG4; NwG=self.NwG; nt = self.nt; nOrb = self.nOrb
        # Build kernel matrix Gamma*chi0
              
        if (self.calcCluster):
            self.chiM = self.chic0M
        else:
            self.chiM = self.chi0M
            
        self.pm = np.dot(self.GammaM, self.chiM)
        #self.pm *= 1.0/(self.invT*float(self.Nc)*float(self.nOrb))
        self.pm *= 1.0/(self.invT*float(self.Nc))

        
        wtemp,vtemp = linalg.eig(self.chiM)
        wttemp = abs(wtemp-1)
        ileadtemp = argsort(wttemp)
        self.lambdastemp = wtemp[ileadtemp]
        self.evecstemp = vtemp[:,ileadtemp]
        
        self.Lambdatemp = sqrt(np.diag(self.lambdastemp))
        self.chiMasqrt = np.dot(self.evecstemp,np.dot(self.Lambdatemp,linalg.inv(self.evecstemp)))

        self.pm2 =  np.dot(self.chiMasqrt,np.dot(self.GammaM, self.chiMasqrt))
        self.pm2 *= 1.0/(self.invT*float(self.Nc))
        
        #self.pm2m = self.pm2.reshape(NwG4,Nc,nOrb,nOrb,NwG4,Nc,nOrb,nOrb)
                
        #pm2m1 = self.pm2m.copy()
        #for iw2 in range(NwG4):
        #    self.pm2m[:,:,:,:,iw2,:,:,:]=(pm2m1[:,:,:,:,iw2,:,:,:]+pm2m1[:,:,:,:,NwG4-iw2-1,:,:,:])/2
        #pm2m1 = self.pm2m.copy()
        #for iw1 in range(NwG4):
        #    self.pm2m[iw1,:,:,:,:,:,:,:]=(pm2m1[iw1,:,:,:,:,:,:,:]+pm2m1[NwG4-iw1-1,:,:,:,:,:,:,:])/2
        
        
        
        #self.Gamma = self.GammaM.reshape(NwG4,Nc,nOrb,nOrb,NwG4,Nc,nOrb,nOrb)
        
        #Gamma1 = self.Gamma.copy()
        #for iw2 in range(NwG4):
        #    self.Gamma[:,:,:,:,iw2,:,:,:]=(Gamma1[:,:,:,:,iw2,:,:,:]+Gamma1[:,:,:,:,NwG4-iw2-1,:,:,:])/2
        #Gamma1 = self.Gamma.copy()
        #for iw1 in range(NwG4):
        #    self.Gamma[iw1,:,:,:,:,:,:,:]=(Gamma1[iw1,:,:,:,:,:,:,:]+Gamma1[NwG4-iw1-1,:,:,:,:,:,:,:])/2

    def calcKernelEigenValuesnew(self):
        nt = self.nt; Nc = self.Nc; NwG4=self.NwG4; nOrb = self.nOrb
        w,v = linalg.eig(self.pm2)
        wt = abs(w-1)
        ilead = argsort(wt)
        self.lambdas = w[ilead]
        self.evecs = v[:,ilead]
        self.evecs = self.evecs.reshape(NwG4,Nc,nOrb,nOrb,nt)
        
        
        iw0=int(NwG4/2)
        for inr in range(16):
            imax = argmax(self.evecs[iw0,:,0,0,inr])
            if (abs(self.evecs[iw0-1,imax,0,0,inr]-self.evecs[iw0,imax,0,0,inr]) <= 1.0e-2):
                print("Eigenvalue is ", self.lambdas[inr], "even frequency")
                print("Eigenvector=",self.evecs[iw0-1,imax,0,0,inr],self.evecs[iw0,imax,0,0,inr])
            else:
                print("Eigenvalue is ", self.lambdas[inr], "odd frequency")
                print("Eigenvector=",self.evecs[iw0-1,imax,0,0,inr],self.evecs[iw0,imax,0,0,inr])

        print ("Leading 16 eigenvalues of lattice Bethe-salpeter equation",self.lambdas[0:16])
        if self.vertex_channel in ("PARTICLE_PARTICLE_SUPERCONDUCTING","PARTICLE_PARTICLE_UP_DOWN"):
            #Now find d-wave eigenvalue
            gk = cos(self.Kvecs[:,0]) - cos(self.Kvecs[:,1]) # dwave form factor
            self.found_d=False
            self.ind_d=0
            for ia in range(nt):
                r1 = dot(gk,self.evecs[int(self.NwG4/2),:,0,0,ia]) * sum(self.evecs[:,1,0,0,ia])
                if abs(r1) >= 2.0e-1:
                    self.lambdad = self.lambdas[ia]
                    self.ind_d = ia
                    self.found_d=True
                    break
            if self.found_d:
                print("Cu-Cu d-wave eigenvalue",self.lambdad)
                #self.calcPdFromEigenFull(self.ind_d)
                #self.calcPdFromEigenFull2(self.ind_d)
            #Now find sx-wave eigenvalue
            gk = cos(self.Kvecs[:,0]) + cos(self.Kvecs[:,1]) # sxwave form factor
            self.found_d=False
            self.ind_d=0
            for ia in range(nt):
                r1 = dot(gk,self.evecs[int(self.NwG4/2),:,0,0,ia]) * sum(self.evecs[:,self.iKPi0,0,0,ia])
                if abs(r1) >= 2.0e-1:
                    self.lambdad = self.lambdas[ia]
                    self.ind_d = ia
                    self.found_d=True
                    break
            if self.found_d:
                print("Cu-Cu sx-wave eigenvalue",self.lambdad)

        
    def calcKernelEigenValues(self):
        nt = self.nt; Nc = self.Nc; NwG4=self.NwG4; nOrb = self.nOrb
        w,v = linalg.eig(self.pm2)
        wt = abs(w-1)
        ilead = argsort(wt)
        self.lambdas = w[ilead]
        self.evecs = v[:,ilead]
        self.evecs = self.evecs.reshape(NwG4,Nc,nOrb,nOrb,nt)

        w2,v2 = linalg.eig(self.pm2)
        wt2 = abs(w2-1)
        ilead2 = argsort(wt2)
        self.lambdas2 = w2[ilead2]
        self.evecs2 = v2[:,ilead2]
        self.evecs2 = self.evecs2.reshape(NwG4,Nc,nOrb,nOrb,nt)
        
        iw0=int(NwG4/2)
        for inr in range(16):
            imax = argmax(self.evecs[iw0,:,0,0,inr])
            if (abs(self.evecs[iw0-1,imax,0,0,inr]-self.evecs[iw0,imax,0,0,inr]) <= 1.0e-2):
                print("Eigenvalue is ", self.lambdas[inr], "even frequency")
                print("Eigenvector=",self.evecs[iw0-1,imax,0,0,inr],self.evecs[iw0,imax,0,0,inr])
            else:
                print("Eigenvalue is ", self.lambdas[inr], "odd frequency")
                print("Eigenvector=",self.evecs[iw0-1,imax,0,0,inr],self.evecs[iw0,imax,0,0,inr])

        print ("Leading 16 eigenvalues of lattice Bethe-salpeter equation",self.lambdas[0:16])
        if self.vertex_channel in ("PARTICLE_PARTICLE_SUPERCONDUCTING","PARTICLE_PARTICLE_UP_DOWN"):
            #Now find d-wave eigenvalue
            gk = cos(self.Kvecs[:,0]) - cos(self.Kvecs[:,1]) # dwave form factor
            self.found_d=False
            self.ind_d=0
            for ia in range(nt):
                r1 = dot(gk,self.evecs[int(self.NwG4/2),:,0,0,ia]) * sum(self.evecs[:,1,0,0,ia])
                if abs(r1) >= 2.0e-1: 
                    self.lambdad = self.lambdas[ia]
                    self.ind_d = ia
                    self.found_d=True
                    break
            if self.found_d: 
                print("Cu-Cu d-wave eigenvalue",self.lambdad)
                #self.calcPdFromEigenFull(self.ind_d)
                #self.calcPdFromEigenFull2(self.ind_d)
            #self.calcPdFromEigenFull(self.ind_d)
            #Now find sx-wave eigenvalue
            gk = cos(self.Kvecs[:,0]) + cos(self.Kvecs[:,1]) # sxwave form factor
            self.found_d=False
            self.ind_d=0
            for ia in range(nt):
                r1 = dot(gk,self.evecs[int(self.NwG4/2),:,0,0,ia]) * sum(self.evecs[:,self.iKPi0,0,0,ia])
                if abs(r1) >= 2.0e-1: 
                    self.lambdad = self.lambdas[ia]
                    self.ind_d = ia
                    self.found_d=True
                    break
            if self.found_d: 
                print("Cu-Cu sx-wave eigenvalue",self.lambdad)
                
    def calcReducibleLatticeVertex(self):
        pm = self.pm; Gamma=self.GammaM
        nt = self.nt; Nc = self.Nc; NwG4=self.NwG4; nOrb=self.nOrb
        self.pminv = np.linalg.inv(np.identity(nt)-pm)
        # self.pminv = np.linalg.inv(np.identity(nt)+pm)
        self.GammaRed = dot(self.pminv, Gamma)
        self.GammaRed = self.GammaRed.reshape(NwG4,Nc,nOrb,nOrb,NwG4,Nc,nOrb,nOrb)

    def determine_specialK(self):
        self.iKPiPi = 0
        self.iKPi0  = 0
        Nc=self.Nc
        for iK in range(Nc):
            kx = abs(self.Kvecs[iK,0] - np.pi)
            ky = abs(self.Kvecs[iK,1] - np.pi)
            ky2 = abs(self.Kvecs[iK,1])
            if kx >= 2*np.pi: kx-=2.*pi
            if ky >= 2*np.pi: ky-=2.*pi
            if ky2 >= 2*np.pi: ky-=2.*pi
            if kx**2+ky**2 <= 1.0e-5:
                self.iKPiPi = iK
            if kx**2+ky2**2 <= 1.0e-5:
                self.iKPi0 = iK
            
    def dwave(self,kx,ky):
        return cos(kx)-cos(ky)

        #if self.vertex_channel in ("PARTICLE_PARTICLE_SUPERCONDUCTING","PARTICLE_PARTICLE_UP_DOWN"):
        #    w2,v2 = linalg.eigh(self.pm2)

        #    wt2 = abs(w2-1)
        #    ilead2 = argsort(wt2)
        #    self.lambdas2 = w2[ilead2]
        #    self.evecs2 = v2[:,ilead2]
        #    self.evecs2 = self.evecs2.reshape(NwG4,Nc,nOrb,nOrb,nt)
        #    print ("10 leading eigenvalues of symmetrized Bethe-salpeter equation",self.lambdas2[0:10])

            #Now find d-wave eigenvalue
        #    gk = cos(self.Kvecs[:,0]) - cos(self.Kvecs[:,1]) # dwave form factor
        #    self.found_d=False
        #    for ia in range(0,10):
        #        r1 = dot(gk,self.evecs[int(self.NwG4/2),:,0,0,ia]) * sum(self.evecs[:,2,0,0,ia])
        #        if abs(r1) >= 2.0e-1: 
        #            self.lambdad = self.lambdas[ia]
        #            self.ind_d = ia
        #            self.found_d=True
        #            break
        #    if self.found_d: print("d-wave eigenvalue",self.lambdad)

        
    def calcPdFromEigenFull(self,ia=0):
        gk = self.dwave(self.Kvecs[:,0],self.Kvecs[:,1])
        nt = self.nt; nc=self.Nc; nw=self.NwG4; nOrb=self.nOrb;
        
        eval = self.lambdas
        Lambda = np.diag(1./(1-eval))
        # Dkk = zeros((nt,nt), dtype=real)
        phit =zeros((nw,nc,nOrb,nOrb,nt), dtype=complex)
        phit2=zeros((nw,nc,nOrb,nOrb,nt), dtype=complex)
        Dkk =zeros((nw,nc,nOrb,nOrb,nw,nc,nOrb,nOrb), dtype=complex)
        Dkk2=zeros((nw,nc,nOrb,nOrb,nw,nc,nOrb,nOrb), dtype=complex)
        Pd =zeros((nOrb,nOrb,nOrb,nOrb), dtype=complex)
        Pdk =zeros((nc,nOrb,nOrb,nc,nOrb,nOrb), dtype=complex)
        PdIa =zeros((nOrb,nOrb,nOrb,nOrb), dtype=complex)
        
        if self.calcCluster: 
            chi0 = self.chic0
        else:
            chi0 = self.chi0
        for ialpha in range(nt):
            phit[:,:,:,:,ialpha] = self.evecs[:,:,:,:,ialpha]
        phit2[:,:,:,:,ia] = self.evecs[:,:,:,:,ia]
        #phi = phit.reshape(nt,nt)
        #phi2 = phit2.reshape(nt,nt)
        evecscom = self.evecs.reshape(nt,nt)
        Dkktemp = dot(phit,dot(Lambda,linalg.inv(evecscom)))
        Dkktemp2 = Dkktemp.reshape(nt,nt)
        Dkk = dot(self.chic0M,Dkktemp2)
        #Dkk = dot(phi,dot(Lambda,phi.T))
        Dkk = Dkk.reshape(nw,nc,nOrb,nOrb,nw,nc,nOrb,nOrb)
        #Dkk2 = dot(phi2,dot(Lambda,phi2.T))
        Dkk2temp = dot(phit2,dot(Lambda,linalg.inv(evecscom)))
        Dkk2temp2 = Dkk2temp.reshape(nt,nt)
        Dkk2 = dot(self.chic0M,Dkk2temp2)
        Dkk2 = Dkk2.reshape(nw,nc,nOrb,nOrb,nw,nc,nOrb,nOrb)
        Lkk = sum(sum(Dkk,axis=0),axis=3)
        Lkk2 = sum(sum(Dkk2,axis=0),axis=3)
        
        for ki in range(self.Nc):
            for kj in range(self.Nc):
                Pdk[ki,:,:,kj,:,:] = gk[ki]*Lkk[ki,:,:,kj,:,:]*gk[kj]
        
        Pd[0,0,0,0] = dot(gk,dot(Lkk[:,0,0,:,0,0],gk)) * self.temp/self.Nc
        PdIa[0,0,0,0] = dot(gk,dot(Lkk2[:,0,0,:,0,0],gk)) * self.temp/self.Nc
        Pd[1,1,1,1] = dot(gk,dot(Lkk[:,1,1,:,1,1],gk)) * self.temp/self.Nc
        PdIa[1,1,1,1] = dot(gk,dot(Lkk2[:,1,1,:,1,1],gk)) * self.temp/self.Nc
        Pd[2,2,2,2] = dot(gk,dot(Lkk[:,2,2,:,2,2],gk)) * self.temp/self.Nc
        PdIa[2,2,2,2] = dot(gk,dot(Lkk2[:,2,2,:,2,2],gk)) * self.temp/self.Nc
        Pd[0,0,1,1] = dot(gk,dot(Lkk[:,0,0,:,1,1],gk)) * self.temp/self.Nc
        PdIa[0,0,1,1] = dot(gk,dot(Lkk2[:,0,0,:,1,1],gk)) * self.temp/self.Nc
        Pd[1,1,0,0] = dot(gk,dot(Lkk[:,1,1,:,0,0],gk)) * self.temp/self.Nc
        PdIa[1,1,0,0] = dot(gk,dot(Lkk2[:,1,1,:,0,0],gk)) * self.temp/self.Nc
        Pd[0,0,2,2] = dot(gk,dot(Lkk[:,0,0,:,2,2],gk)) * self.temp/self.Nc
        PdIa[0,0,2,2] = dot(gk,dot(Lkk2[:,0,0,:,2,2],gk)) * self.temp/self.Nc
        Pd[2,2,0,0] = dot(gk,dot(Lkk[:,2,2,:,0,0],gk)) * self.temp/self.Nc
        PdIa[2,2,0,0] = dot(gk,dot(Lkk2[:,2,2,:,0,0],gk)) * self.temp/self.Nc
        Pd[1,1,2,2] = dot(gk,dot(Lkk[:,1,1,:,2,2],gk)) * self.temp/self.Nc
        PdIa[1,1,2,2] = dot(gk,dot(Lkk2[:,1,1,:,2,2],gk)) * self.temp/self.Nc
        Pd[2,2,1,1] = dot(gk,dot(Lkk[:,2,2,:,1,1],gk)) * self.temp/self.Nc
        PdIa[2,2,1,1] = dot(gk,dot(Lkk2[:,2,2,:,1,1],gk)) * self.temp/self.Nc
        

        self.PdEigen = Pd
        self.PdIa = PdIa
        self.Pdk = Pdk
        print ("Calculations from BSE eigenvalues and eigenvectors:")
        print("Cu-Cu Pd from eigensystem (all eigenvalues): ",Pd[0,0,0,0])
        print("Ox-Ox Pd from eigensystem (all eigenvalues): ",Pd[1,1,1,1])
        print("Oy-Oy Pd from eigensystem (all eigenvalues): ",Pd[2,2,2,2])
        print("Cu-Ox Pd from eigensystem (all eigenvalues): ",Pd[0,0,1,1]+Pd[1,1,0,0])
        print("Cu-Oy Pd from eigensystem (all eigenvalues): ",Pd[0,0,2,2]+Pd[2,2,0,0])
        print("Ox-Oy Pd from eigensystem (all eigenvalues): ",Pd[1,1,2,2]+Pd[2,2,1,1])
        
    def calcPdFromEigenFull2(self,ia=0):
        gk = self.dwave(self.Kvecs[:,0],self.Kvecs[:,1])
        nt = self.nt; nc=self.Nc; nw=self.NwG4; nOrb=self.nOrb;
        
        eval = self.lambdas2
        Lambda = np.diag(1./(1-eval))
        # Dkk = zeros((nt,nt), dtype=real)
        phit =zeros((nw,nc,nOrb,nOrb,nt), dtype=complex)
        phit2=zeros((nw,nc,nOrb,nOrb,nt), dtype=complex)
        Dkk =zeros((nw,nc,nOrb,nOrb,nw,nc,nOrb,nOrb), dtype=complex)
        Dkk2=zeros((nw,nc,nOrb,nOrb,nw,nc,nOrb,nOrb), dtype=complex)
        Pd =zeros((nOrb,nOrb,nOrb,nOrb), dtype=complex)
        PdIa =zeros((nOrb,nOrb,nOrb,nOrb), dtype=complex)
        
        if self.calcCluster: 
            chi0 = self.chic0
        else:
            chi0 = self.chi0
        for ialpha in range(nt):
            phit[:,:,:,:,ialpha] = self.evecs2[:,:,:,:,ialpha]
        phit2[:,:,:,:,ia] = self.evecs2[:,:,:,:,ia]
        #phi = phit.reshape(nt,nt)
        #phi2 = phit2.reshape(nt,nt)
        evecscom = self.evecs2.reshape(nt,nt)
        Dkktemp = dot(phit,dot(Lambda,linalg.inv(evecscom)))
        Dkktemp2 = Dkktemp.reshape(nt,nt)
        Dkk = dot(self.chi0M,Dkktemp2)
        #Dkk = dot(phi,dot(Lambda,phi.T))
        Dkk = Dkk.reshape(nw,nc,nOrb,nOrb,nw,nc,nOrb,nOrb)
        #Dkk2 = dot(phi2,dot(Lambda,phi2.T))
        Dkk2temp = dot(phit2,dot(Lambda,linalg.inv(evecscom)))
        Dkk2temp2 = Dkk2temp.reshape(nt,nt)
        Dkk2 = dot(self.chiMasqrt,dot(Dkk2temp2,self.chiMasqrt))
        Dkk2 = Dkk2.reshape(nw,nc,nOrb,nOrb,nw,nc,nOrb,nOrb)
        Lkk = sum(sum(Dkk,axis=0),axis=3)
        Lkk2 = sum(sum(Dkk2,axis=0),axis=3)
        Pd[0,0,0,0] = dot(gk,dot(Lkk[:,0,0,:,0,0],gk)) * self.temp/self.Nc
        PdIa[0,0,0,0] = dot(gk,dot(Lkk2[:,0,0,:,0,0],gk)) * self.temp/self.Nc
        Pd[1,1,1,1] = dot(gk,dot(Lkk[:,1,1,:,1,1],gk)) * self.temp/self.Nc
        PdIa[1,1,1,1] = dot(gk,dot(Lkk2[:,1,1,:,1,1],gk)) * self.temp/self.Nc
        Pd[2,2,2,2] = dot(gk,dot(Lkk[:,2,2,:,2,2],gk)) * self.temp/self.Nc
        PdIa[2,2,2,2] = dot(gk,dot(Lkk2[:,2,2,:,2,2],gk)) * self.temp/self.Nc
        Pd[0,0,1,1] = dot(gk,dot(Lkk[:,0,0,:,1,1],gk)) * self.temp/self.Nc
        PdIa[0,0,1,1] = dot(gk,dot(Lkk2[:,0,0,:,1,1],gk)) * self.temp/self.Nc
        Pd[1,1,0,0] = dot(gk,dot(Lkk[:,1,1,:,0,0],gk)) * self.temp/self.Nc
        PdIa[1,1,0,0] = dot(gk,dot(Lkk2[:,1,1,:,0,0],gk)) * self.temp/self.Nc
        Pd[0,0,2,2] = dot(gk,dot(Lkk[:,0,0,:,2,2],gk)) * self.temp/self.Nc
        PdIa[0,0,2,2] = dot(gk,dot(Lkk2[:,0,0,:,2,2],gk)) * self.temp/self.Nc
        Pd[2,2,0,0] = dot(gk,dot(Lkk[:,2,2,:,0,0],gk)) * self.temp/self.Nc
        PdIa[2,2,0,0] = dot(gk,dot(Lkk2[:,2,2,:,0,0],gk)) * self.temp/self.Nc
        Pd[1,1,2,2] = dot(gk,dot(Lkk[:,1,1,:,2,2],gk)) * self.temp/self.Nc
        PdIa[1,1,2,2] = dot(gk,dot(Lkk2[:,1,1,:,2,2],gk)) * self.temp/self.Nc
        Pd[2,2,1,1] = dot(gk,dot(Lkk[:,2,2,:,1,1],gk)) * self.temp/self.Nc
        PdIa[2,2,1,1] = dot(gk,dot(Lkk2[:,2,2,:,1,1],gk)) * self.temp/self.Nc

        self.PdEigen = Pd
        self.PdIa = PdIa
        print ("Calculations from BSE2 eigenvalues and eigenvectors:")
        print("Cu-Cu Pd from eigensystem (all eigenvalues): ",Pd[0,0,0,0])
        print("Ox-Ox Pd from eigensystem (all eigenvalues): ",Pd[1,1,1,1])
        print("Oy-Oy Pd from eigensystem (all eigenvalues): ",Pd[2,2,2,2])
        print("Cu-Ox Pd from eigensystem (all eigenvalues): ",Pd[0,0,1,1]+Pd[1,1,0,0])
        print("Cu-Oy Pd from eigensystem (all eigenvalues): ",Pd[0,0,2,2]+Pd[2,2,0,0])
        print("Ox-Oy Pd from eigensystem (all eigenvalues): ",Pd[1,1,2,2]+Pd[2,2,1,1])

        

    def transformEvecsToKz(self):
        self.phi0  = 1./sqrt(2.)*(self.evecs[:,:,0,0,:] + self.evecs[:,:,1,0,:])
        self.phipi = 1./sqrt(2.)*(self.evecs[:,:,0,0,:] - self.evecs[:,:,1,0,:])

    def projectOnDwave(self,Ks,matrix):
        gk = self.dwave(Ks[:,0], Ks[:,1])
        c1 = dot(gk, dot(matrix,gk) ) / dot(gk,gk)
        return c1


    def dispersion(self,kx,ky):
        ek = np.zeros((self.nOrb,self.nOrb),dtype='complex')
        r1 = -2.* 1j *self.tpd*sin(kx/2.)
        r2 = 2.* 1j *self.tpd*sin(ky/2.)
        r3 = 4.*self.tpp*sin(kx/2.)*sin(ky/2.)
        ek[0,0] = self.epd
        ek[1,1] = self.epp
        ek[2,2] = self.epp
        ek[0,1] = r1
        ek[1,0] = -r1
        ek[0,2] = r2
        ek[2,0] = -r2
        ek[1,2] = r3
        ek[2,1] = r3
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

    def plotLeadingSolutions(self,Kvecs,lambdas,evecs,title=None):
        mpl.style.use(["ggplot"])

        Nc  = Kvecs.shape[0]
        for ic in range(Nc):
            if Kvecs[ic,0] > pi: Kvecs[ic,0] -=2.*pi
            if Kvecs[ic,1] > pi: Kvecs[ic,1] -=2.*pi

        fig, axes = mpl.subplots(nrows=4,ncols=4, sharex=True,sharey=True,figsize=(16,16))
        inr=0
        for ax in axes.flat:
            self.plotEV(ax,Kvecs,lambdas,evecs,inr)
            inr += 1
            ax.set(adjustable='box', aspect='equal')
        if title==None:
            title = r"Leading eigensolutions of BSE for $Upp=$" + str(self.Upp) + r", $t\prime=$" + str(self.tp1) + r", $\langle n\rangle=$" + str(self.fill) + r", $T=$" + str(self.temp)
        fig.suptitle(title, fontsize=10)
        mpl.show()

    def plotEV(self,ax,Kvecs,lambdas,evecs,inr):
        prop_cycle = rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        Nc = evecs.shape[1]; Nw = self.evecs.shape[0]
        iw0=int(Nw/2)
        imax = argmax(evecs[iw0,:,inr])
        if (abs(evecs[iw0-1,imax,inr]-evecs[iw0,imax,inr]) <= 1.0e-2):
            freqString = "; even frequency"
        else:
            freqString = "; odd frequency"

        colVec = Nc*[colors[0]]
        for ic in range(Nc):
            #if real(evecs[iw0,ic,inr])*real(evecs[iw0,imax,inr]) < 0.0: colVec[ic] = colors[1]
            if real(evecs[iw0,ic,inr])*10 < 0.0: colVec[ic] = colors[1]
        # print "colVec=",colVec
        ax.scatter(Kvecs[:,0]/pi,Kvecs[:,1]/pi,s=abs(real(evecs[iw0,:,inr]))*2500,c=colVec)
        ax.set(aspect=1)
        ax.set_xlim(-0.75,1.25); ax.set_ylim(-0.75,1.25)
        ax.set_title(r"$\lambda=$"+str(round(lambdas[inr].real,4))+freqString)
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


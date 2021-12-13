import numpy as np
from numpy import *
import matplotlib.pyplot as mpl
import h5py
import sys
import os
from matplotlib.pyplot import *


class BSE:


    def __init__(self,fileG4,fileG="data.DCA_sp.hdf5",draw=False,useG0=False,symmetrize_G4=False,phSymmetry=False,calcRedVertex=False,calcCluster=False,nkfine=100):
        self.vertex_channels = ["PARTICLE_PARTICLE_UP_DOWN",          \
                                "PARTICLE_HOLE_CHARGE",               \
                                "PARTICLE_HOLE_MAGNETIC",             \
                                "PARTICLE_HOLE_LONGITUDINAL_UP_UP",   \
                                "PARTICLE_HOLE_LONGITUDINAL_UP_DOWN", \
                                "PARTICLE_HOLE_TRANSVERSE"]
        
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
        self.reorderG4()
        self.setupMomentumTables()
        self.determine_iKPiPi()
        #if symmetrize_G4: self.symmetrizeG4()
        print ("Index of (pi,pi): ",self.iKPiPi)
        self.calcChi0Cluster()
        self.calcGammaIrr()
        if calcCluster == False: self.buildChi0Lattice(nkfine)
        # self.calcFillingBilayer()
        # self.buildKernelMatrix()
        self.buildSymmetricKernelMatrix()
        self.calcKernelEigenValues()
        self.transformEvecsToKz()
        # self.calcSus()


        # title = "Leading eigensolutions of BSE for U="+str(self.U)+", t'="+str(self.tp)+r", $\langle n\rangle$="+str(round(self.fill,4))+", T="+str(round(self.temp,4))
        # if self.vertex_channel == "PARTICLE_HOLE_TRANSVERSE":
        #     self.calcAFSus()
        #     print("AF cluster susceptibility: ",sum(self.G4)/(float(self.Nc)*self.invT))
        # if self.draw: self.plotLeadingSolutions(self.Kvecs,self.lambdas,self.evecs[:,:,:],title)
        # if calcRedVertex: self.calcReducibleLatticeVertex()
        # if self.vertex_channel == "PARTICLE_PARTICLE_SUPERCONDUCTING":
        #     self.calcDwaveSCSus()
        #     self.calcDwaveSCClusterSus()
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

        self.cluster = array(f["domains"]["CLUSTER"]["REAL_SPACE"]["super-basis"]["data"])
        print("Cluster vectors:",self.cluster)

        # self.iwm = array(f['parameters']['vertex-channel']['w-channel'])[0] # transferred frequency in units of 2*pi*temp
        self.iwm = array(f['parameters']['four-point']['frequency-transfer'])[0] # transferred frequency in units of 2*pi*temp
        print("Transferred frequency iwm = ",self.iwm)
        # self.qchannel = array(f['parameters']['vertex-channel']['q-channel'])
        self.qchannel = array(f['parameters']['four-point']['momentum-transfer'])
        print("Transferred momentum q = ",self.qchannel)
        
        for ver in self.vertex_channels:
            if 'G4_'+ver in f['functions'].keys():
                self.vertex_channel = ver
                print "Vertex channel = ",self.vertex_channel,'\n'
                    
        self.invT = array(f['parameters']['physics']['beta'])[0]
        print("Inverse temperature = ",self.invT)
        self.temp = 1.0/self.invT

        self.e1 = array(f['parameters']['bilayer-Hubbard-model']['e1'])[0]
        print"e1 = ",self.e1
        self.e2 = array(f['parameters']['bilayer-Hubbard-model']['e2'])[0]
        print"e2 = ",self.e2
        self.U1 = array(f['parameters']['bilayer-Hubbard-model']['U1'])[0]
        print"U1 = ",self.U1
        self.U2 = array(f['parameters']['bilayer-Hubbard-model']['U2'])[0]
        print"U2 = ",self.U2
        self.t1 = array(f['parameters']['bilayer-Hubbard-model']['t1'])[0]
        print"t1 = ",self.t1
        self.t2 = array(f['parameters']['bilayer-Hubbard-model']['t2'])[0]
        print"t2 = ",self.t2
        self.t1p = array(f['parameters']['bilayer-Hubbard-model']['t1-prime'])[0]
        print"t1-prime = ",self.t1p
        self.t2p = array(f['parameters']['bilayer-Hubbard-model']['t2-prime'])[0]
        print"t2-prime = ",self.t2p
        self.tperp = array(f['parameters']['bilayer-Hubbard-model']['t-perp'])[0]
        print"tperp = ",self.tperp
        self.V = array(f['parameters']['bilayer-Hubbard-model']['V'])[0]
        print"V = ",self.V
        self.Vp = array(f['parameters']['bilayer-Hubbard-model']['V-prime'])[0]
        print"V-prime = ",self.Vp

        self.fill = array(f['parameters']['physics']['density'])[0]
        print("filling = ",self.fill)
        self.dens = array(f['DCA-loop-functions']['density']['data'])
        print("actual filling:",self.dens)
        self.nk = array(f['DCA-loop-functions']['n_k']['data'])

        # Now read the 4-point Green's function
        G4Re  = array(f['functions']['G4_'+self.vertex_channel]['data'])[0,0,:,:,:,:,:,:,:,:,0]
        G4Im  = array(f['functions']['G4_'+self.vertex_channel]['data'])[0,0,:,:,:,:,:,:,:,:,1]
        self.G4 = G4Re+1j*G4Im
        # G4[iw1,iw2,ik1,ik2,l1,l2,l3,l4]
        print "Extracted G4.shape=", self.G4.shape ,'\n'

        # Now read the cluster Green's function
        GRe = array(f['functions']['cluster_greens_function_G_k_w']['data'])[:,:,0,:,0,:,0]
        GIm = array(f['functions']['cluster_greens_function_G_k_w']['data'])[:,:,0,:,0,:,1]
        self.Green = GRe + 1j * GIm
        print "Extracted G.shape=", self.Green.shape,'\n'

        self.Nc  = self.Green.shape[1]
        self.NwG = self.Green.shape[0]
        self.nOrb = self.Green.shape[2]

        if self.symmetrize_G4: self.Green = self.symmetrize_single_particle_function(self.Green)

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

        if self.symmetrize_G4: self.sigma = self.symmetrize_single_particle_function(self.sigma)


        # Now load frequency data
        self.wn = np.array(f['domains']['frequency-domain']['elements'])
        self.wnSet = np.array(f['domains']['vertex-frequency-domain (COMPACT)']['elements'])

        # Now read the K-vectors
        self.Kvecs = array(f['domains']['CLUSTER']['MOMENTUM_SPACE']['elements']['data'])
        print ("K-vectors: ",self.Kvecs)

        self.mu = np.array(f['DCA-loop-functions']['chemical-potential']['data'])[0]


        self.NwG4 = self.G4.shape[0]
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

        self.qmcSign = list(f['DCA-loop-functions']['sign']['data'])
        print("QMC sign:",self.qmcSign)


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
                                            c1= self.G4[iw1,ik1,iw2,ik2,l1,l2,l3,l4]
                                            self.G4r[iw1,ik1,l1,l3,iw2,ik2,l4,l2]  = c1
                                            if (l1==l3) & (l4==l2):
                                                G4susQz0 += c1
                                                G4susQzPi += c1*exp(1j*np.pi*(l2-l3))
                                        elif self.vertex_channel=="PARTICLE_PARTICLE_UP_DOWN":
                                            c1 = self.G4[iw2,ik2,iw1,ik1,l4,l3,l2,l1]
                                            self.G4r[iw1,ik1,l1,l2,iw2,ik2,l3,l4] = c1
                                            if (l1!=l2) & (l3!=l4):
                                                G4susQz0 += c1

        if self.symmetrize_G4:
            if (self.cluster[0,0] == 4 and self.cluster[0,1] == 0 and self.cluster[1,0] == 0 and self.cluster[1,1] == 4):  # 4x4 cluster
                import symmetrize_Nc4x4
                sym=symmetrize_Nc4x4.symmetrize()
                print("symmetrizing 16B (4x4) cluster")
                # Note: We also need to know the effect of the symmetry operations on the orbital component of the operators
                # Orbital 0 is a d_{x^2-y^2} orbital, and orbital 1 a d_{3z^2-r^2} orbital.
                # symmTrans_of_orb = zeros((2,8))
                # symmTrans_of_orb[1,:] = 1
                # for iSym in range(0,8):
                #     if sym.iex[iSym] == 0:
                #         symmTrans_of_orb[0,iSym] =  1
                #     else:
                #         symmTrans_of_orb[0,iSym] = -1 # x <--> y switched --> d_{x^2-y^2} orbital gets a -1 phase factor

                nwn = self.G4r.shape[0]
                G4New = zeros_like(self.G4r)

                for iK1 in range(0,Nc):
                    for iK2 in range(0,Nc):

                        for l1 in range(self.nOrb):
                            for l2 in range(self.nOrb):
                                for l3 in range(self.nOrb):
                                    for l4 in range(self.nOrb):

                                        for iSym in range(0,8): # Apply every point-group symmetry operation

                                            iK1Trans = sym.symmTrans_of_iK(iK1,iSym)
                                            iK2Trans = sym.symmTrans_of_iK(iK2,iSym)
                                            # sgn = symmTrans_of_orb[l1,iSym]*symmTrans_of_orb[l2,iSym]*symmTrans_of_orb[l3,iSym]*symmTrans_of_orb[l4,iSym]
                                            sgn = 1.0 - 2.0 * mod(l1+l2+l3+l4,2) * sym.iex[iSym]
                                            # print("l1,l2,l3,l4,iSym,sgn: ",l1,l2,l3,l4,iSym,sgn)
                                            G4New[:,iK1,l1,l2,:,iK2,l3,l4] += sgn*self.G4r[:,iK1Trans,l1,l2,:,iK2Trans,l3,l4]
            
                self.G4r = G4New / 8.

            # symmetrize in wn,wn' assuming that G4 is symmetric under k --> -k, k' --> -k'
            for iw1 in range(NwG4):
                for iw2 in range(NwG4):
                    imw1 = NwG4-1-iw1
                    imw2 = NwG4-1-iw2
                    tmp1 = self.G4r[iw1,:,:,:,iw2,:,:,:]
                    tmp2 = self.G4r[imw1,:,:,:,imw2,:,:,:]
                    self.G4r[iw1,:,:,:,iw2,:,:,:]   = 0.5*(tmp1+conj(tmp2))
                    self.G4r[imw1,:,:,:,imw2,:,:,:] = 0.5*(conj(tmp1)+tmp2)



        self.G4M = self.G4r.reshape(self.nt,self.nt)

        if self.vertex_channel=="PARTICLE_HOLE_MAGNETIC":
            print("Cluster Chi(q,qz=0) :", G4susQz0/(self.invT*self.Nc*2.0))
            print("Cluster Chi(q,qz=pi):", G4susQzPi/(self.invT*self.Nc*2.0))


        if self.vertex_channel=="PARTICLE_PARTICLE_UP_DOWN":
            print("Cluster inter-orbital Chi(q=0):", G4susQz0/(self.invT*self.Nc*4.0))

    def symmetrize_single_particle_function(self,G):
        if (self.cluster[0,0] == 4 and self.cluster[0,1] == 0 and self.cluster[1,0] == 0 and self.cluster[1,1] == 4):  # 4x4 cluster
            import symmetrize_Nc4x4
            sym=symmetrize_Nc4x4.symmetrize()
            print("symmetrizing 16B (4x4) cluster")

            nwn = G.shape[0] # G[w,k,l1,l2]
            GNew = zeros_like(G)

            for iK in range(self.Nc):
                for l1 in range(self.nOrb):
                    for l2 in range(self.nOrb):

                        for iSym in range(0,8):
                            iKTrans = sym.symmTrans_of_iK(iK,iSym)
                            sgn = 1.0 - 2.0 * mod(l1+l2,2) * sym.iex[iSym]

                            GNew[:,iK,l1,l2] += sgn * G[:,iKTrans,l1,l2]

            G = GNew / 8.
        return G
        

    def fermi(self,energy):
        beta=self.invT
        xx = beta*energy
        if xx < 0:
            return 1./(1.+exp(xx))
        else:
            return exp(-xx)/(1.+exp(-xx))

    # def calcFillingBilayer(self):
    #     Nc = self.Nc; beta=self.invT

    #     Gkw0   = 0.5*(self.cG[:,:,0,0] + self.cG[:,:,1,1] + self.cG[:,:,0,1] + self.cG[:,:,1,0])
    #     Gkwpi  = 0.5*(self.cG[:,:,0,0] + self.cG[:,:,1,1] - self.cG[:,:,0,1] - self.cG[:,:,1,0])
    #     G0kw0  = 0.5*(self.cG0[:,:,0,0] + self.cG0[:,:,1,1] + self.cG0[:,:,0,1] + self.cG0[:,:,1,0])
    #     G0kwpi = 0.5*(self.cG0[:,:,0,0] + self.cG0[:,:,1,1] - self.cG0[:,:,0,1] - self.cG0[:,:,1,0])

    #     a0  = sum(Gkw0-G0kw0)/(Nc*beta)
    #     api = sum(Gkwpi-G0kwpi)/(Nc*beta)

    #     fk0 = 0.0; fkpi = 0.0
    #     for iK,K in enumerate(self.Kset):
    #         for k in self.kPatch:
    #             kx = K[0]+k[0]; ky = K[1]+k[1]
    #             ek = self.dispersion(kx,ky)
    #             ek0  = ek[0,0] + ek[1,0]
    #             ekpi = ek[0,0] - ek[1,0]
    #             fk0  += self.fermi(ek0-self.mu+self.U*(self.dens/4.-0.5))
    #             fkpi += self.fermi(ekpi-self.mu+self.U*(self.dens/4.-0.5))
    #     fk0  /= self.Nc*self.kPatch.shape[0]
    #     fkpi /= self.Nc*self.kPatch.shape[0]

    #     c0 = a0 + fk0; cpi = api + fkpi

    #     print("Filling kz=0  bare : ",real(fk0))
    #     print("Filling kz=pi bare: ",real(fkpi))

    #     print("Filling kz=0 : ",real(c0))
    #     print("Filling kz=pi: ",real(cpi))

    #     print("Total filling: ",2.*real(c0+cpi))


    # def calcFilling(self):


    def determineFS(self):
        self.FSpoints=[]
        Kset = self.Kvecs.copy()
        for iK in range(self.Nc):
            if Kset[iK,0] > np.pi: Kset[iK,0] -= 2*np.pi
            if Kset[iK,1] > np.pi: Kset[iK,1] -= 2*np.pi

        for iK in range(self.Nc):
            if abs(abs(self.Kset[iK,0])+abs(self.Kset[iK,1]) - np.pi) <= 1.0e-4:
                self.FSpoints.append(iK)

    # def symmetrizeG4(self):
        # if self.iwm==0 & self.symmetrize_G4:
            # self.apply_symmetry_in_wn(self.G4)
    #         sym=symmetrize_Nc4x4.symmetrize()
    #         sym.apply_point_group_symmetries_Q0(GammaIrr.reshape(Nc,nwG4,Nc,nwG4))
    #         # self.apply_transpose_symmetry(self.G4)
            # if self.phSymmetry: self.apply_ph_symmetry_pp(self.G4)



    def calcChi0Cluster(self):
        print ("Now calculating chi0 on cluster")
        Nc=self.Nc; NwG4=self.NwG4; NwG=self.NwG; nOrb=self.nOrb
        self.chic0 = zeros((NwG4,Nc,nOrb,nOrb,NwG4,Nc,nOrb,nOrb),dtype='complex')

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
                                    c1 = self.Green[iw1,ik,l1,l3] \
                                       * self.Green[minusiwPlusiwm,ikPlusQ,l2,l4]
                                    # c1 = self.Green[iw1,ik,l2,l4] \
                                    #     * self.Green[minusiwPlusiwm,ikPlusQ,l1,l3]
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




    def calcGammaIrr(self):
        # Calculate the irr. GammaIrr
        Nc=self.Nc; NwG4=self.NwG4; NwG=self.NwG; nt = self.nt; nOrb = self.nOrb
        G4M = linalg.inv(self.G4M)
        chic0M = linalg.inv(self.chic0M)
        self.GammaM = chic0M - G4M
        # self.GammaM *= float(Nc)*self.invT*float(self.nOrb)
        self.GammaM *= float(Nc)*self.invT
        self.Gamma = self.GammaM.reshape(NwG4,Nc,nOrb,nOrb,NwG4,Nc,nOrb,nOrb)


    def buildKernelMatrix(self):
        # Build kernel matrix Gamma*chi0
        if (self.calcCluster):
            self.chiM = self.chic0M
        else:
            self.chiM = self.chi0M
        self.pm = np.dot(self.GammaM, self.chiM)
        # self.pm *= 1.0/(self.invT*float(self.Nc)*float(self.nOrb))
        self.pm *= 1.0/(self.invT*float(self.Nc))

    def buildSymmetricKernelMatrix(self):
        # Build symmetric kernel matrix M = 0.5*(Gamma(wn,wn')*chi0(wn')+Gamma(wn,-wn')*chi0(-wn')
        if (self.calcCluster):
            self.chiM = self.chic0M
        else:
            self.chiM = self.chi0M
        GammaTemp = self.Gamma.reshape(self.NwG4,self.Nc*self.nOrb*self.nOrb,self.NwG4,self.Nc*self.nOrb*self.nOrb)
        Gamma2 = np.zeros_like(GammaTemp)
        for iw1 in range(self.NwG4):
            for iw2 in range(self.NwG4):
                Gamma2[iw1,:,iw2,:] = GammaTemp[iw1,:,self.NwG4-iw2-1]
        Gamma2 = Gamma2.reshape(self.NwG4*self.Nc*self.nOrb*self.nOrb,self.NwG4*self.Nc*self.nOrb*self.nOrb)
        self.pm = 0.5*(np.dot(self.GammaM, self.chiM) + np.dot(Gamma2,conj(self.chiM)))
        # self.pm *= 1.0/(self.invT*float(self.Nc)*float(self.nOrb))
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

    # def calcReducibleLatticeVertex(self):
    #     pm = self.pm; Gamma=self.GammaM
    #     nt = self.nt; Nc = self.Nc; NwG4=self.NwG4
    #     self.pminv = np.linalg.inv(np.identity(nt)-pm)
    #     # self.pminv = np.linalg.inv(np.identity(nt)+pm)
    #     self.GammaRed = dot(self.pminv, Gamma)
    #     self.GammaRed = self.GammaRed.reshape(NwG4,Nc,NwG4,Nc)
    #
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

        # Load parameters: t,mu
        # t1 = self.t1
        # t2 = self.t2
        mu = self.mu

        # Now coarse-grain G*G to build chi0(K) = Nc/N sum_k Gc(K+k')Gc(-K-k')
        nOrb = self.nOrb; nw = wnSet.shape[0]; nk=Kset.shape[0]
        self.chi0  = np.zeros((nw,nk,nOrb,nOrb,nw,nk,nOrb,nOrb),dtype='complex')
        self.cG    = np.zeros((nw,nk,nOrb,nOrb),dtype='complex')
        self.cG0   = np.zeros((nw,nk,nOrb,nOrb),dtype='complex')
        for iwn,wn in enumerate(wnSet): # reduced tp frequencies !!
            # print "iwn = ",iwn
            iwG = int(iwn - self.iwG40 + self.iwG0)
            for iK,K in enumerate(Kset):
                c1 = np.zeros((nOrb,nOrb,nOrb,nOrb),dtype='complex')
                cG = np.zeros((nOrb,nOrb),dtype='complex')
                cG0 = np.zeros((nOrb,nOrb),dtype='complex')
                for k in kPatch:
                    kx = K[0]+k[0]; ky = K[1]+k[1]
                    ek = self.dispersion(kx,ky)
                    #G0inv = (1j*wn+self.mu-self.U*(self.dens[-1]/4.-0.5))* np.identity(nOrb) - ek
                    #G0 = linalg.inv(G0inv)
                    Qx = self.qchannel[0]; Qy = self.qchannel[1]
                    if (self.vertex_channel == "PARTICLE_PARTICLE_UP_DOWN"):
                        emkpq = self.dispersion(-kx+Qx, -ky+Qy)
                        iKQ = self.iKSum[self.iKDiff[0,iK],self.iQ]
                        minusiwPlusiwm = min(max(NwG-iwG-1 + self.iwm,0),NwG-1) # -iwn + iwm

                        G1inv = (1j*wn+self.mu) * np.identity(nOrb)-ek-self.sigma[iwG,iK,:,:]
                        G2inv = (-1j*wn+self.mu)* np.identity(nOrb)-emkpq-self.sigma[minusiwPlusiwm,iKQ,:,:]
                        G1 = linalg.inv(G1inv); G2 = linalg.inv(G2inv)

                    else:
                        ekpq = self.dispersion(kx+Qx, ky+Qy)
                        iKQ = int(self.iKSum[iK,self.iQ])
                        iwPlusiwm = int(min(max(iwG + self.iwm,0),NwG-1))  # iwn+iwm

                        G1inv = (1j*wn+self.mu)*np.identity(nOrb)-ek-self.sigma[iwG,iK,:,:]
                        G2inv = (1j*wn+self.mu)*np.identity(nOrb)-ekpq-self.sigma[iwPlusiwm,iKQ,:,:]
                        G1 = linalg.inv(G1inv); G2 = -linalg.inv(G2inv)

                    for l1 in range(nOrb):
                        for l2 in range(nOrb):
                            cG[l1,l2] += G1[l1,l2]
                            #cG0[l1,l2] += G0[l1,l2]
                            for l3 in range(nOrb):
                                for l4 in range(nOrb):
                                    c1[l1,l2,l3,l4] += G1[l1,l3]*G2[l2,l4]


                self.chi0[iwn,iK,:,:,iwn,iK,:,:]  = c1[:,:,:,:]/kPatch.shape[0]
                self.cG[iwn,iK,:,:] = cG[:,:]/kPatch.shape[0]
                #self.cG0[iwn,iK,:,:] = cG0[:,:]/kPatch.shape[0]
        self.chi0M = self.chi0.reshape(self.nt,self.nt)

        if self.vertex_channel=="PARTICLE_HOLE_MAGNETIC":
            chi0Loc = sum(sum(sum(sum(self.chi0,axis=0),axis=0),axis=2),axis=2)
            chi00 = 0.0; chi0Pi = 0.0
            for l1 in range(0,nOrb):
                for l3 in range(0,nOrb):
                    chi00  += chi0Loc[l1,l1,l3,l3]
                    chi0Pi += chi0Loc[l1,l1,l3,l3] * exp(1j*np.pi*(l1-l3))

            print("Lattice Chi0(q,qz=0) :", chi00 /(self.invT*self.Nc*2.0))
            print("Lattice Chi0(q,qz=pi):", chi0Pi/(self.invT*self.Nc*2.0))

    def calcSus(self):
        NwG4=self.NwG4; Nc=self.Nc; nOrb=self.nOrb

        # calculate lattice G4 Green's function
        # G2L = linalg.inv(linalg.inv(self.chi0M) - self.GammaM/(float(self.Nc)*self.invT*float(self.nOrb)))
        G2L = linalg.inv(linalg.inv(self.chi0M) - self.GammaM/(float(self.Nc)*self.invT))
        G2 = G2L.reshape(NwG4,Nc,nOrb,nOrb,NwG4,Nc,nOrb,nOrb)
        G2intra = 0.5*(G2[:,:,0,0,:,:,0,0]+G2[:,:,1,1,:,:,0,0]+G2[:,:,0,0,:,:,1,1]+G2[:,:,1,1,:,:,1,1])
        G2t0 = sum(sum(G2intra,axis=0),axis=1)/self.invT

        if self.vertex_channel=="PARTICLE_PARTICLE_UP_DOWN":

            # for spm susceptibility
            # cos kz form-factor corresponds to inter-layer susceptibility
            # Note: This is for qz=0
            G2spm = 0.5*(G2[:,:,0,1,:,:,0,1]+G2[:,:,0,1,:,:,1,0]+G2[:,:,1,0,:,:,0,1]+G2[:,:,1,0,:,:,1,0])
            Ps = sum(G2spm)/(float(self.Nc)*self.invT)
            # for dwave susceptibility
            # cos kx - cos ky form-factor corresponds to intra-layer d-wave susceptibility
            # Note: This is for qz=0
            # intra-layer form-factor = 1 --> use G2intra and G2t0
            gkd = cos(self.Kvecs[:,0]) - cos(self.Kvecs[:,1])
            Pd = 0.5*dot(gkd,dot(G2t0,gkd))/self.Nc
            print("spm   susceptibility: ",Ps)
            print("dwave susceptibility: ",Pd)

        elif (self.vertex_channel=="PARTICLE_HOLE_MAGNETIC") | (self.vertex_channel=="PARTICLE_HOLE_TRANSVERSE"):

            # intra-layer form-factor = 1 --> use G2intra and G2t0 for qz=0
            # For qz = 0:
            chiQz0 = sum(G2t0)/self.Nc
            # For qz=pi we need:
            G2QzPi = 0.5*(G2[:,:,0,0,:,:,0,0]-G2[:,:,1,1,:,:,0,0]-G2[:,:,0,0,:,:,1,1]+G2[:,:,1,1,:,:,1,1])
            G2QzPit0 = sum(sum(G2QzPi,axis=0),axis=1)/self.invT
            chiQzPi = sum(G2QzPit0)/self.Nc

            print("ChiAF(Qz= 0) susceptibility: ",chiQz0)
            print("ChiAF(Qz=pi) susceptibility: ",chiQzPi)





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
    # def calcAFSus(self):
    #     G2L = linalg.inv(linalg.inv(self.chi0M) - self.GammaM/(float(self.Nc)*self.invT))
    #     print("AF susceptibility: ",sum(G2L)/(float(self.Nc)*self.invT))
    #
    # def calcDwaveSCSus(self):
    #     # Calculate from gd*G4*gd = gd*GG*gd + gd*GG*GammaRed*GG*gd
    #     #GammaRed = self.GammaRed.reshape(self.NwG4*self.Nc,self.NwG4*self.Nc)
    #     gkd = cos(self.Kvecs[:,0]) - cos(self.Kvecs[:,1])
    #     csum = 0.0
    #     ccsum = 0.0
    #     for iw1 in range(self.NwG4):
    #         for iK1 in range(self.Nc):
    #             for iw2 in range(self.NwG4):
    #                 for iK2 in range(self.Nc):
    #                     csum  += self.chi0D[iw1,iK1]*self.GammaRed[iw1,iK1,iw2,iK2]*self.chi0D[iw2,iK2]
    #                     ccsum += gkd[iK1]*self.chi0[iw1,iK1]*self.GammaRed[iw1,iK1,iw2,iK2]*self.chi0[iw2,iK2]*gkd[iK2]
    #     csum /= (self.Nc*self.invT)**2
    #     ccsum /= (self.Nc*self.invT)**2
    #     csum1 = sum(self.chi0D2)/(self.Nc*self.invT)
    #     ccsum1 = sum(np.dot(self.chi0,gkd**2))/(self.Nc*self.invT)
    #     csum += csum1
    #     ccsum += ccsum1
    #     print ("bare d-wave SC susceptibility: ",csum1)
    #     print ("bare d-wave SC lattice (with cluster form-factors) susceptibility: ",ccsum1)
    #     print ("d-wave SC susceptibility: ",csum)
    #     print ("d-wave SC lattice (with cluster form-factors) susceptibility: ",ccsum)
    #
    # def calcDwaveSCClusterSus(self):
    #     gkd = cos(self.Kvecs[:,0]) - cos(self.Kvecs[:,1])
    #     csum = 0.0
    #     for iK1 in range(self.Nc):
    #         for iK2 in range(self.Nc):
    #             csum += gkd[iK1]*sum(self.G4[:,:,iK1,iK2])*gkd[iK2]
    #     csum /= self.Nc*self.invT
    #     print ("d-wave SC cluster susceptibility: ",csum)

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


    # def dispersion(self,kx,ky):
    #     ek = np.zeros((self.nOrb,self.nOrb))
    #     r1 = -2.*self.t1*(cos(kx)+cos(ky)) - 4.0*self.tp1*cos(kx)*cos(ky)
    #     r2 = -2.*self.t2*(cos(kx)+cos(ky)) - 4.0*self.tp2*cos(kx)*cos(ky) + self.DeltaE
    #     # r2 = -2.*self.t2*(cos(kx)+cos(ky)) - 4.0*self.tp2*cos(kx)*cos(ky)
    #     ek[0,0] = r1
    #     ek[1,1] = r2
    #     ek[0,1] = -self.tperp
    #     ek[1,0] = -self.tperp
    #     # ek[0,1] = -self.tperp - self.tperpp*(cos(kx)+cos(ky))
    #     # ek[1,0] = -self.tperp - self.tperpp*(cos(kx)+cos(ky))
    #     return ek


    def dispersion(self,kx,ky):
        ek = np.zeros((self.nOrb,self.nOrb),dtype='complex')
        r11  = self.e1 -2.*self.t1*(cos(kx)+cos(ky)) - 4.0*self.t1p*cos(kx)*cos(ky)
        r22  = self.e2 -2.*self.t2*(cos(kx)+cos(ky)) - 4.0*self.t2p*cos(kx)*cos(ky)
        r12  = -self.tperp

        ek[0,0] = r11
        ek[1,1] = r22
        ek[0,1] = r12
        ek[1,0] = r12
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


    # def plotLeadingSolutions(self,Kvecs,lambdas,evecs,title=None):
    #     mpl.style.use("ggplot")

    #     Nc  = Kvecs.shape[0]
    #     for ic in range(Nc):
    #         if Kvecs[ic,0] > pi: Kvecs[ic,0] -=2.*pi
    #         if Kvecs[ic,1] > pi: Kvecs[ic,1] -=2.*pi

    #     fig, axes = mpl.subplots(nrows=4,ncols=4, sharex=True,sharey=True,figsize=(16,16))
    #     inr=0
    #     for ax in axes.flat:
    #         self.plotEV(ax,Kvecs,lambdas,evecs,inr)
    #         inr += 1
    #         ax.set(adjustable='box-forced', aspect='equal')
    #     # if title==None:
    #         # title = r"Leading eigensolutions of BSE for $U=$" + str(self.U) + r", $t\prime=$" + str(self.tp) + r", $\langle n\rangle=$" + str(self.fill) + r", $T=$" + str(self.temp)

    #     fig.suptitle(title, fontsize=20)

    def plotEV(self,inr):
        # expects evecs[iw,iK,inr]
        

        Kvecs = self.Kset
        lambdas = self.lambdas
        evecs = self.evecs 
        prop_cycle = rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        Nc = evecs.shape[1]; Nw = self.evecs.shape[0]; nOrb = self.evecs.shape[2]
        iw0=int(Nw/2)
        imax = argmax(evecs[iw0,:,0,0,inr])
        # if (abs(evecs[iw0-1,imax,inr]-evecs[iw0,imax,inr]) <= 1.0e-2):
        #     freqString = "; even frequency"
        # else:
        #     freqString = "; odd frequency"


        fig, ax = mpl.subplots(nrows=nOrb,ncols=nOrb, sharex=True,sharey=True,figsize=(10,10))

        for l1 in range(nOrb):
            for l2 in range(nOrb):

                colVec = Nc*[colors[0]]
                for ic in range(Nc):
                        if real(evecs[iw0,ic,l1,l2,inr])*real(evecs[iw0,imax,0,0,inr]) < 0.0: colVec[ic] = colors[1]

                ax[l1,l2].scatter(Kvecs[:,0]/pi,Kvecs[:,1]/pi,s=abs(real(evecs[iw0,:,l1,l2,inr]))*2500,c=colVec)
                ax[l1,l2].set(aspect=1)
                ax[l1,l2].set_xlim(-1.0,1.2); ax[l1,l2].set_ylim(-1.0,1.2)
                # ax.set_title(r"$\lambda=$"+str(round(lambdas[inr],4))+freqString)
                ax[l1,l2].set_title(r"$\ell_1=$"+str(l1)+r", $\ell_2=$"+str(l2))
                # ax.get_xaxis().set_visible(False)
                # ax.get_yaxis().set_visible(False)
                ax[l1,l2].grid(True)
                for tic in ax[l1,l2].xaxis.get_major_ticks():
                    tic.tick1On = tic.tick2On = False
                for tic in ax[l1,l2].yaxis.get_major_ticks():
                    tic.tick1On = tic.tick2On = False
        suptitle = r"Eigenvalue #"+str(inr)+r":  $\lambda=$"+str(round(lambdas[inr],4))
        fig.suptitle(suptitle)
        print(suptitle)


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

###################################################################################
Ts = [1, 0.75, 0.5, 0.4, 0.3, 0.2, 0.15, 0.125, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04]
Ts = [0.4]
channels = ['phcharge']#,'phmag']
channels = ['phmag']
qs = ['00']#,'pi20','pi0','pipi2','pipi','pi2pi2']
qs = ['pipi']

for T_ind, T in enumerate(Ts):
    for ch in channels:
        for q in qs:
            file_tp = './T='+str(Ts[T_ind])+'/dca_tp_'+ch+'_q'+q+'.hdf5'
            file_tp = './T='+str(Ts[T_ind])+'/dca_tp.hdf5'
            #file_tp = './sc/T='+str(Ts[T_ind])+'/dca_tp.hdf5'
            #file_tp = './Nc4/T='+str(Ts[T_ind])+'/dca_tp.hdf5'
            file_sp = './T='+str(Ts[T_ind])+'/dca_sp.hdf5'
            file_analysis_hdf5 = './T='+str(Ts[T_ind])+'/analysis.hdf5'
            #file_analysis_hdf5 = './Nc4/T='+str(Ts[T_ind])+'/analysis.hdf5'

            if(os.path.exists(file_tp)):
                print "\n =================================\n"
                print "T =", T
                # model='square','bilayer','Emery'
                BSE(file_tp,\
                    draw=False,\
                    useG0=False,\
                    symmetrize_G4=True,\
                    phSymmetry=True,\
                    calcRedVertex=True,\
                    calcCluster=False,\
                    nkfine=100)


import numpy as np
from numpy import *
import matplotlib.pyplot as mpl
import h5py
import sys
import os
from matplotlib.pyplot import *
from plotnine import *
import pandas as p
# import myggplottheme



class BSE:


    def __init__(self,fileG4,draw=False,useG0=False,symmetrize_G4=False,phSymmetry=False,calcRedVertex=False,calcCluster=False,nkfine=100,oldFormat=False,shiftedK=False,newMaster=False,allq=False,iq=0,evenFreqOnly=True,wCutOff=1):
        self.fileG4 = fileG4
        self.draw = draw
        self.useG0 = useG0
        self.symmetrize_G4 = symmetrize_G4
        self.calcCluster = calcCluster
        self.calcRedVertex = calcRedVertex
        self.phSymmetry = phSymmetry
        self.oldFormat = oldFormat
        self.shiftedK = shiftedK
        self.newMaster = newMaster
        self.allq = allq
        self.iq=iq
        self.evenFreqOnly = evenFreqOnly
        self.wCutOff = wCutOff
        self.readData()
        self.setupMomentumTables()
        self.iK0 = self.K_2_iK(0.0, 0.0)
        self.determine_specialK()
        print ("Index of (pi,pi): ",self.iKPiPi)
        print ("Index of (pi,0): ",self.iKPi0)

        if self.symmetrize_G4: self.symmetrizeG4()


        if self.vertex_channel in ("PARTICLE_PARTICLE_UP_DOWN", "PARTICLE_PARTICLE_SUPERCONDUCTING", "PARTICLE_PARTICLE_SINGLET"):
            self.calcSCClusterSus()

        if self.vertex_channel in ("PARTICLE_PARTICLE_SINGLET"): sys.exit("PARTICLE_PARTICLE_SINGLET channel has singular chi0");

        self.calcChi0Cluster()
        self.calcGammaIrr()
        if self.vertex_channel in ("PARTICLE_PARTICLE_SUPERCONDUCTING","PARTICLE_PARTICLE_UP_DOWN"):
            self.symmetrizeGamma()
        if calcCluster == False: self.buildChi0Lattice(nkfine)
        self.buildKernelMatrix()
        self.calcKernelEigenValues()
        title = "Leading eigensolutions of BSE for U="+str(self.U)+", t'="+str(self.tp)+r", $\langle n\rangle$="+str(round(self.fill,4))+", T="+str(round(self.temp,4))
        if self.vertex_channel in ("PARTICLE_HOLE_TRANSVERSE","PARTICLE_HOLE_MAGNETIC"):
            self.calcSWaveSus()
            # print("Cluster spin susceptibility: ",sum(self.G4)/(float(self.Nc)*self.invT))
        if self.draw: self.plotLeadingSolutions(self.Kvecs,self.lambdas,self.evecs[:,:,:],title)
        if calcRedVertex: self.calcReducibleLatticeVertex()
        if self.vertex_channel in ("PARTICLE_PARTICLE_SUPERCONDUCTING","PARTICLE_PARTICLE_UP_DOWN"):
            if calcCluster == False: 
                self.calcReducibleLatticeVertex()
                self.calcSCSus()
                self.determineFS(); FSpoints = array(self.FSpoints)
                self.calcPd0FS(FSpoints)
            self.calcSCClusterSus()
            self.calcPd0(wCutOff=1)

            # # self.Pd0 = self.calcPd0_LorentzianCutoff()
            # self.Pd0 = self.calcProjectionsKwn(self.chi0M,self.dwave,0)/(self.invT*self.Nc)
            # print("\nPd0(T) = ",self.Pd0)
            # self.Vd = self.calcProjectionsKwn(self.GammaM,self.dwave,2)
            # print("Vd(T)  = ",self.Vd)
            # # Pd0 = self.calcProjections(self.chi0M,self.dwave)
            # prod=self.Vd*self.Pd0
            # print("Vd(T)*Pd0(T) = ",prod)

            # print("PMd(T)  = ",self.calcProjectionsKwn(self.pm,self.dwave,1))

            # print("\nPpp0(T) = ",self.calcProjections(self.chi0M,self.pxpwave)/(self.invT*self.Nc))
            # print("Vpp(T)  = ",self.calcProjections(self.GammaM,self.pxpwave))
            # print("PMpp(T)  = ",self.calcProjections(self.pm,self.pxpwave))
            # self.Vd0 = self.calcProjections(self.GammaM,self.dwave,wCutOff=0)/self.Nc
            # print("\nVd0 (piT,piT) = ", self.Vd0)
            # print("Vd0(T)*Pd0(T) = ",self.Vd0*self.Pd0)
            # print("\n")

            self.calcPd0andVd()
            print("Vd01, Pd01 :",self.Vd01,self.Pd01)
            print("Vd02, Pd02 :",self.Vd02,self.Pd02)
            print("Vd03, Pd03 :",self.Vd03,self.Pd03)


            if self.found_d: self.calcPdFromEigenFull(self.ind_d)

        self.calcReducibleClusterVertex()
        if self.calcRedVertex & (self.calcCluster==False):
            # FSpoints = array([16,12,9,5,2,29,25,20,23,27,30,6,11,15])
            iwG40 = int(self.NwG4/2); nFs = int(FSpoints.shape[0])
            GRFS = np.sum(self.GammaRed[iwG40-1:iwG40+1,:,iwG40-1:iwG40+1,:],axis=(0,2))[FSpoints,:][:,FSpoints]/4.
            print ("s-wave projection of GammaRed averaged: ", real(np.sum(GRFS)/float(nFs*nFs)))
            gkd = self.dwave(self.Kvecs[FSpoints,0],self.Kvecs[FSpoints,1])
            r1 = real(np.dot(gkd,np.dot(GRFS,gkd)))
            print ("d-wave projection of GammaRed averaged: ", r1/np.dot(gkd, gkd)/float(nFs))
            GRFSpp = self.GammaRed[iwG40,:,iwG40,:][FSpoints,:][:,FSpoints]
            GRFSpm = self.GammaRed[iwG40,:,iwG40-1,:][FSpoints,:][:,FSpoints]
            print ("s-wave projection of GammaRed(piT,piT): ", real(np.sum(GRFSpp)/float(nFs*nFs)))
            print ("s-wave projection of GammaRed(piT,-piT): ", real(np.sum(GRFSpm)/float(nFs*nFs)))
            r1 = real(np.dot(gkd,np.dot(GRFSpp,gkd)))/np.dot(gkd, gkd)/float(nFs)
            r2 = real(np.dot(gkd,np.dot(GRFSpm,gkd)))/np.dot(gkd, gkd)/float(nFs)
            print ("d-wave projection of GammaRed(piT,piT): " , r1)
            print ("d-wave projection of GammaRed(piT,-piT): ", r2)
            GRFSpp = self.GammaRed[iwG40+1,:,iwG40+1,:][FSpoints,:][:,FSpoints]
            GRFSpm = self.GammaRed[iwG40+1,:,iwG40-2,:][FSpoints,:][:,FSpoints]
            print ("s-wave projection of GammaRed(3piT,3piT): ", real(np.sum(GRFSpp)/float(nFs*nFs)))
            print ("s-wave projection of GammaRed(3piT,-3piT): ", real(np.sum(GRFSpm)/float(nFs*nFs)))
            r1 = real(np.dot(gkd,np.dot(GRFSpp,gkd)))/np.dot(gkd, gkd)/float(nFs)
            r2 = real(np.dot(gkd,np.dot(GRFSpm,gkd)))/np.dot(gkd, gkd)/float(nFs)
            print ("d-wave projection of GammaRed(3piT,3piT): " , r1)
            print ("d-wave projection of GammaRed(3piT,-3piT): ", r2)
            
            GRFSpp = self.GammaCluster[iwG40,:,iwG40,:][FSpoints,:][:,FSpoints]
            GRFSpm = self.GammaCluster[iwG40,:,iwG40-1,:][FSpoints,:][:,FSpoints]
            r1 = real(np.dot(gkd,np.dot(GRFSpp,gkd)))/np.dot(gkd, gkd)/float(nFs)
            r2 = real(np.dot(gkd,np.dot(GRFSpm,gkd)))/np.dot(gkd, gkd)/float(nFs)
            print ("d-wave projection of GammaCluster(piT,piT): " , r1)
            print ("d-wave projection of GammaCluster(piT,-piT): ", r2)
            self.calcGammaRedEvals()


    def readData(self):
        f = h5py.File(self.fileG4,'r')
        if self.oldFormat == True:

            self.cluster = array(f["domains"]["CLUSTER"]["REAL_SPACE"]["super-basis"]["data"])
            print("Cluster vectors:",self.cluster)
            
            self.iwm = array(f['parameters']['vertex-channel']['w-channel'])[0] # transferred frequency in units of 2*pi*temp
            self.qchannel = array(f['parameters']['vertex-channel']['q-channel'])
            a = array(f['parameters']['vertex-channel']['vertex-measurement-type'])[:]
            self.vertex_channel = ''.join(chr(i) for i in a)
            self.invT = array(f['parameters']['physics-parameters']['beta'])[0]
            self.temp = 1.0/self.invT
            self.U = array(f['parameters']['2D-Hubbard-model']['U'])[0]
            self.tp = array(f['parameters']['2D-Hubbard-model']['t-prime'])[0]
            self.fill = array(f['parameters']['physics-parameters']['density'])[0]
            self.dens = array(f['DCA-loop-functions']['density']['data'])

            # Now read the 4-point Green's function
            G4Re  = array(f['functions']['G4_k_k_w_w']['data'])[:,:,:,:,0,0,0,0,0]
            G4Im  = array(f['functions']['G4_k_k_w_w']['data'])[:,:,:,:,0,0,0,0,1]
            self.G4 = G4Re+1j*G4Im
            # G4[iw1,iw2,ik1,ik2]

            # Now read the cluster Green's function
            GRe = array(f['functions']['cluster_greens_function_G_k_w']['data'])[:,:,0,0,0,0,0]
            GIm = array(f['functions']['cluster_greens_function_G_k_w']['data'])[:,:,0,0,0,0,1]
            self.Green = GRe + 1j * GIm

            # Now read the self-energy
            s  = np.array(f['functions']['Self_Energy']['data'])
            self.sigma = s[:,:,0,0,0,0,0] + 1j *s[:,:,0,0,0,0,1]

            # Now load frequency data
            self.wn = np.array(f['domains']['frequency-domain']['elements'])
            self.wnSet = np.array(f['domains']['vertex-frequency-domain (COMPACT)']['elements'])

            # Now read the K-vectors
            self.Kvecs = array(f['domains']['CLUSTER']['MOMENTUM_SPACE']['elements']['data'])

            # Now read other Hubbard parameters
            self.t = np.array(f['parameters']['2D-Hubbard-model']['t'])[0]
            self.mu = np.array(f['DCA-loop-functions']['chemical-potential']['data'])[-1]
            self.NwTP = 2*np.array(f['parameters']['function-parameters']['two-particle-functions']['fermionic-frequencies'])[0]

        else:
            self.cluster = array(f["domains"]["CLUSTER"]["REAL_SPACE"]["super-basis"]["data"])
            print("Cluster vectors:",self.cluster)

            self.iwm = array(f['parameters']['four-point']['frequency-transfer'])[0] # transferred frequency in units of 2*pi*temp
            self.qchannel = array(f['parameters']['four-point']['momentum-transfer'])
            a = array(f['parameters']['four-point']['type'])[:]
            self.vertex_channel = ''.join(chr(i) for i in a)
            self.invT = array(f['parameters']['physics']['beta'])[0]
            self.temp = 1.0/self.invT
            self.U = array(f['parameters']['single-band-Hubbard-model']['U'])[0]
            self.tp = array(f['parameters']['single-band-Hubbard-model']['t-prime'])[0]
            self.fill = array(f['parameters']['physics']['density'])[0]
            self.dens = array(f['DCA-loop-functions']['density']['data'])
            self.nk = array(f['DCA-loop-functions']['n_k']['data'])
            if self.newMaster == False:
                G4Re  = array(f['functions']['G4_k_k_w_w']['data'])[:,:,:,:,0,0,0,0,0]
                G4Im  = array(f['functions']['G4_k_k_w_w']['data'])[:,:,:,:,0,0,0,0,1]
                self.G4 = G4Re+1j*G4Im
            else:
                if self.allq == False:
                    # olf format: order of indices: w1,w2,K1,K2
                    # G4Re  = array(f['functions']['G4']['data'])[0,:,:,0,:,:,0,0,0,0,0]
                    # G4Im  = array(f['functions']['G4']['data'])[0,:,:,0,:,:,0,0,0,0,1]
                    # New format: order of indices: w1,K1,w2,K2
                    G4Re  = array(f['functions']['G4']['data'])[0,0,:,:,:,:,0,0,0,0,0]
                    G4Im  = array(f['functions']['G4']['data'])[0,0,:,:,:,:,0,0,0,0,1]
                    print("G4 shape:",G4Re.shape)
                else:
                    # order of indices: Q,w1,K1,w2,K2
                    G4Re  = array(f['functions']['G4']['data'])[0,self.iq,:,:,:,:,0,0,0,0,0]
                    G4Im  = array(f['functions']['G4']['data'])[0,self.iq,:,:,:,:,0,0,0,0,1]

                self.G4 = G4Re+1j*G4Im
                # Now reorder G4
                self.G4=self.G4.swapaxes(1,2) # Now it G4's shape is w1,w2,K1,K2

            GRe = array(f['functions']['cluster_greens_function_G_k_w']['data'])[:,:,0,0,0,0,0]
            GIm = array(f['functions']['cluster_greens_function_G_k_w']['data'])[:,:,0,0,0,0,1]
            self.Green = GRe + 1j * GIm
            s  = np.array(f['functions']['Self_Energy']['data'])
            self.sigma = s[:,:,0,0,0,0,0] + 1j *s[:,:,0,0,0,0,1]
            # self.sigma = s[:,:,0,:,0,:,0] + 1j *s[:,:,0,:,0,:,1]
            self.wn = np.array(f['domains']['frequency-domain']['elements'])
            self.wnSet = np.array(f['domains']['vertex-frequency-domain (COMPACT)']['elements'])
            self.Kvecs = array(f['domains']['CLUSTER']['MOMENTUM_SPACE']['elements']['data'])
            self.t = np.array(f['parameters']['single-band-Hubbard-model']['t'])[0]
            self.mu = np.array(f['DCA-loop-functions']['chemical-potential']['data'])[-1]
            self.NwTP = 2*np.array(f['parameters']['domains']['imaginary-frequency']['four-point-fermionic-frequencies'])[0]
            self.qmcSign = list(f['DCA-loop-functions']['sign']['data'])
            print("QMC sign:",self.qmcSign)

        self.NwG4 = self.G4.shape[0]
        self.Nc  = self.Green.shape[1]
        self.NwG = self.Green.shape[0]
        self.nt = self.Nc*self.NwG4
        if self.allq == False:
            self.iQ = self.K_2_iK(self.qchannel[0],self.qchannel[1])
        else:
            self.iQ = self.iq;self.qchannel = self.Kvecs[self.iq]

        self.iwG40 = int(self.NwG4/2)
        self.iwG0 = int(self.NwG/2)

        print("Transferred frequency iwm = ",self.iwm)
        # print("Momentum transfer index, Q:",self.iQ,self.Kvecs[self.iq])
        print("Transferred momentum q = ",self.qchannel)
        print ("Index of transferred momentum: ", self.iQ)
        print("Vertex channel = ",self.vertex_channel)
        print("Inverse temperature = ",self.invT)
        print("U = ",self.U)
        print("t-prime = ",self.tp)
        print("target filling = ",self.fill)
        print("actual filling = ",self.dens)
        print ("K-vectors: ",self.Kvecs)
        print ("NwG4: ",self.NwG4)
        print ("NwG : ",self.NwG)
        print ("Nc  : ",self.Nc)




        f.close()

        # Now remove vacuum part of charge G4
        if (self.vertex_channel=="PARTICLE_HOLE_CHARGE"):
            if (self.qchannel[0] == 0) & (self.qchannel[1] == 0):
                for ik1 in range(self.Nc):
                    for ik2 in range(self.Nc):
                        for iw1 in range(self.NwG4):
                            for iw2 in range(self.NwG4):
                                iw1Green = iw1 - self.iwG40 + self.iwG0
                                iw2Green = iw2 - self.iwG40 + self.iwG0
                                self.G4[iw1,iw2,ik1,ik2] -= 2.0 * self.Green[iw1Green,ik1] * self.Green[iw2Green,ik2]

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
        if self.iwm==0:
            self.apply_symmetry_in_wn(self.G4)
            print("Imposing symmetry in wn")
        
        if self.vertex_channel in ("PARTICLE_PARTICLE_SUPERCONDUCTING","PARTICLE_PARTICLE_UP_DOWN"):
            # print("G4.shape:",self.G4.shape)
            self.apply_transpose_symmetry(self.G4)
            print("Imposing transpose symmetry")
            if self.phSymmetry: self.apply_ph_symmetry_pp(self.G4)

        # 16A cluster [[4,2],[0,4]]
        if   (self.cluster[0,0] == 4.0 and self.cluster[0,1] == 2.0 and self.cluster[1,0] == 0.0 and self.cluster[1,1] == 4.0):
            import symmetrize_Nc16A; sym=symmetrize_Nc16A.symmetrize()
            print("symmetrizing 16A cluster")
            sym.apply_point_group_symmetries_Q0(self.G4)
        elif (self.cluster[0,0] == 4 and self.cluster[0,1] == 0 and self.cluster[1,0] == 0 and self.cluster[1,1] == 4):
            import symmetrize_Nc16B; sym=symmetrize_Nc16B.symmetrize()
            print("symmetrizing 16B cluster")
            sym.apply_point_group_symmetries_Q0(self.G4)
        elif (self.cluster[0,0] == 2 and self.cluster[0,1] == 2 and self.cluster[1,0] == -4 and self.cluster[1,1] == 2):
            import symmetrize_Nc12; sym=symmetrize_Nc12.symmetrize()
            print("symmetrizing 12A cluster")
            sym.apply_point_group_symmetries_Q0(self.G4)
        elif (self.cluster[0,0] == 4 and self.cluster[0,1] == 4 and self.cluster[1,0] == 4 and self.cluster[1,1] == -4):
            import symmetrize_Nc32A_v2; sym=symmetrize_Nc32A_v2.symmetrize()
            print("symmetrizing 32A cluster")
            sym.apply_point_group_symmetries_Q0(self.G4)
        elif (self.cluster[0,0] == 2 and self.cluster[0,1] == 2 and self.cluster[1,0] == -2 and self.cluster[1,1] == 2):
            import symmetrize_Nc8; sym=symmetrize_Nc8.symmetrize()
            print("symmetrizing 8-site cluster")
            sym.apply_point_group_symmetries_Q0(self.G4)




    def calcChi0Cluster(self):
        print ("Now calculating chi0 on cluster")
        self.chic0  = zeros((self.NwG4,self.Nc),dtype='complex')
        Nc=self.Nc; NwG4=self.NwG4; NwG=self.NwG
        nt = Nc*NwG4

        if self.vertex_channel in ("PARTICLE_PARTICLE_SUPERCONDUCTING","PARTICLE_PARTICLE_UP_DOWN","PARTICLE_PARTICLE_SINGLET"):
            for iw in range(0,NwG4):
                for ik in range(Nc):
                    iw1  = int(iw - NwG4/2 + NwG/2) # convert to frequency for single-particle G
                    ikPlusQ = int(self.iKSum[self.iKDiff[self.iK0,ik],self.iQ]) # -k+Q
                    minusiwPlusiwm = int(min(max(NwG-iw1-1 + self.iwm,0),NwG-1)) # -iwn + iwm
                    c1 = self.Green[iw1,ik] * self.Green[minusiwPlusiwm,ikPlusQ]
                    self.chic0[iw,ik] = c1 
        else:
            for iw in range(NwG4):
                for ik in range(Nc):
                    iw1  = int(iw - NwG4/2 + NwG/2)
                    ikPlusQ = int(self.iKSum[ik,self.iQ]) # k+Q
                    iwPlusiwm = int(min(max(iw1 + self.iwm,0),NwG-1))  # iwn+iwm
                    #print("iw1,ik,iwPlusiwm,ikPlusQ",iw1,ik,iwPlusiwm,ikPlusQ)
                    c1 = - self.Green[iw1,ik] * self.Green[iwPlusiwm,ikPlusQ]
                    self.chic0[iw,ik] = c1

        self.chic0M = np.diag(self.chic0.reshape(nt))
        # for PARTICLE_PARTICLE_SINGLET, chic0 also appears for k2=q-k1
        if self.vertex_channel in ("PARTICLE_PARTICLE_SINGLET"):
            for iw in range(0,NwG4):
                for ik in range(Nc):
                    i1 = ik + iw * Nc
                    ikPlusQ = int(self.iKSum[self.iKDiff[self.iK0,ik],self.iQ]) # -k+Q
                    minusiwPlusiwm = int(min(max(NwG4-iw-1 + self.iwm,0),NwG-1)) # -iwn + iwm
                    i2 = ikPlusQ + minusiwPlusiwm * Nc # k2 = q-k1
                    self.chic0M[i1,i2] += self.chic0[iw,ik]
                    

    def calcChi0CPPDwave(self):
        print ("Now calculating chi0pp(q=0,iwm) on cluster")
        Nc=self.Nc; NwG4=self.NwG4; NwG=self.NwG
        self.chic0pp = zeros((NwG4),dtype=complex)

        for iwm in range(0,NwG4):
            for iwn in range(0,NwG):
                for iK in range(0,Nc):
                    gK = cos(self.Kvecs[iK,0]) - cos(self.Kvecs[iK,1])
                    minusiwPlusiwm = int(min(max(NwG-iwn-1 + iwm,0),NwG-1)) # -iwn + iwm
                    miK = self.iKDiff[self.iK0,iK]                    
                    self.chic0pp[iwm] += self.Green[iwn,iK] * self.Green[minusiwPlusiwm,miK] * gK**2
        self.chic0pp *= self.temp/self.Nc

    def calcGammaIrr(self):
        # Calculate the irr. GammaIrr
        Nc=self.Nc; NwG4=self.NwG4; NwG=self.NwG; nt = self.nt
        self.G4M = np.swapaxes(self.G4,1,2).reshape(nt,nt)
        G4M = linalg.inv(self.G4M)
        chic0M = linalg.inv(self.chic0M)
        self.GammaM = chic0M - G4M
        self.GammaM *= float(Nc)*self.invT
        self.Gamma = self.GammaM.reshape(NwG4,Nc,NwG4,Nc)
        if (self.evenFreqOnly==True) & (self.vertex_channel in ("PARTICLE_PARTICLE_SUPERCONDUCTING","PARTICLE_PARTICLE_UP_DOWN","PARTICLE_PARTICLE_SINGLET")):
            print("Solutions restricted to even frequency!!!")
            self.Gamma = 0.5*(self.Gamma+self.Gamma[:,:,::-1,:])
            self.GammaM = self.Gamma.reshape(nt,nt)



    def buildKernelMatrix(self):
        # Build kernel matrix Gamma*chi0
        if (self.calcCluster):
            self.chi0Ma = self.chic0M
        else:
            self.chi0Ma = self.chi0M        
        self.pm = np.dot(self.GammaM, self.chi0Ma)/(self.invT*float(self.Nc))
        
        if self.vertex_channel in ("PARTICLE_PARTICLE_SUPERCONDUCTING","PARTICLE_PARTICLE_UP_DOWN","PARTICLE_PARTICLE_SINGLET"):
            self.pm2 = np.dot(sqrt(real(self.chi0Ma)),np.dot(real(self.GammaM), sqrt(real(self.chi0Ma))))
            self.pm2 *= 1.0/(self.invT*float(self.Nc))
        
        # self.pm *= 1.0/(self.invT*float(self.Nc))

    def symmetrizeGamma(self):
        nt = self.GammaM.shape[0]
        for i in range(0,nt):
            for j in range(i,nt):
                c1 = 0.5*(self.GammaM[i,j]+self.GammaM[j,i])
                self.GammaM[i,j] = c1
                self.GammaM[j,i] = c1

    def calcKernelEigenValues(self):
        nt = self.nt; Nc = self.Nc; NwG4=self.NwG4
        w,v = linalg.eig(self.pm)

        wt = abs(w-1)
        ilead = argsort(wt)
        self.lambdas = w[ilead]
        self.evecs = v[:,ilead]
        self.evecs = self.evecs.reshape(NwG4,Nc,nt)
        print ("10 leading eigenvalues of lattice Bethe-salpeter equation",self.lambdas[0:10])

        if self.vertex_channel in ("PARTICLE_PARTICLE_SUPERCONDUCTING","PARTICLE_PARTICLE_UP_DOWN"):
            w2,v2 = linalg.eigh(self.pm2)

            wt2 = abs(w2-1)
            ilead2 = argsort(wt2)
            self.lambdas2 = w2[ilead2]
            self.evecs2 = v2[:,ilead2]
            self.evecs2 = self.evecs2.reshape(NwG4,Nc,nt)
            print ("10 leading eigenvalues of symmetrized Bethe-salpeter equation",self.lambdas2[0:10])

            #Now find d-wave eigenvalue
            gk = cos(self.Kvecs[:,0]) - cos(self.Kvecs[:,1]) # dwave form factor
            gk2 = cos(2.*self.Kvecs[:,0]) - cos(2.*self.Kvecs[:,1]) # dwave nnn form factor
            self.found_d=False
            for ia in range(0,nt):
                r1 = dot(gk,self.evecs2[int(self.NwG4/2),:,ia]) * sum(self.evecs2[:,self.iKPi0,ia])
                if abs(r1) >= 2.0e-1: 
                    self.lambdad = self.lambdas2[ia]
                    self.ind_d = ia
                    self.found_d=True
                    r1 = dot(gk,self.evecs2[int(self.NwG4/2),:,ia])
                    r2 = dot(gk2,self.evecs2[int(self.NwG4/2),:,ia])
                    print("d1 = ", r1)
                    print("d2 = ", r2)
                    break
            if self.found_d: print("d-wave eigenvalue",self.lambdad)



    def calcReducibleLatticeVertex(self):
        pm = self.pm; Gamma=self.GammaM
        nt = self.nt; Nc = self.Nc; NwG4=self.NwG4
        self.pminv = np.linalg.inv(np.identity(nt)-pm)
        # self.pminv = np.linalg.inv(np.identity(nt)+pm)
        self.GammaRed = dot(self.pminv, Gamma)
        self.GammaRed = self.GammaRed.reshape(NwG4,Nc,NwG4,Nc)

    def calcReducibleClusterVertex(self):
        # Calculate cluster vertex from Gamma(q,k,k') = [ G4(q,k,k')-G(k)G(k+q) ] / [ G(k)G(k+q)G(k')G(k'+q) ]
        nt = self.nt; Nc = self.Nc; NwG4=self.NwG4
        self.GammaCluster = np.zeros((NwG4,Nc,NwG4,Nc),dtype=complex)
        for iK1 in range(Nc):
            for iK2 in range(Nc):
                for iw1 in range(NwG4):
                    for iw2 in range(NwG4):

                        iwG1 = int(iw1 - self.iwG40 + self.iwG0)
                        iwG2 = int(iw2 - self.iwG40 + self.iwG0)

                        if self.vertex_channel in ("PARTICLE_PARTICLE_SUPERCONDUCTING","PARTICLE_PARTICLE_UP_DOWN"):

                            imk1pq = int(self.iKSum[self.iKDiff[self.iK0,iK1],self.iQ])
                            imk2pq = int(self.iKSum[self.iKDiff[self.iK0,iK2],self.iQ])
                            numerator  = self.G4[iw1,iw2,iK1,iK2]
                            if (iK1==iK2) & (iw1==iw2): numerator -= self.Green[iwG1,iK1] * self.Green[self.NwG-iwG1-1+self.iwm,imk1pq]
                            denominator = self.Green[iwG1,iK1]*self.Green[self.NwG-iwG1-1+self.iwm,imk1pq] * self.Green[iwG2,iK2]*self.Green[self.NwG-iwG2-1+self.iwm,imk2pq]

                        else:

                            ik1pq = int(self.iKSum[iK1,self.iQ])
                            ik2pq = int(self.iKSum[iK2,self.iQ])
                            numerator  = self.G4[iw1,iw2,iK1,iK2]
                            if (iK1==iK2) & (iw1==iw2): numerator += self.Green[iwG1,iK1] * self.Green[iwG1+self.iwm,ik1pq]
                            denominator = self.Green[iwG1,iK1]*self.Green[iwG1+self.iwm,ik1pq] * self.Green[iwG2,iK2]*self.Green[iwG2+self.iwm,ik2pq]
                        self.GammaCluster[iw1,iK1,iw2,iK2] = numerator/denominator

        self.GammaCluster *= self.invT*self.Nc



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
        t = self.t
        mu = self.mu

        # Now coarse-grain G*G to build chi0(K) = Nc/N sum_k Gc(K+k')Gc(-K-k')
        self.chi0    = np.zeros((wnSet.shape[0],Kset.shape[0]),dtype='complex')
        self.chi0D   = np.zeros((wnSet.shape[0],Kset.shape[0]),dtype='complex')
        self.chi0D2  = np.zeros((wnSet.shape[0],Kset.shape[0]),dtype='complex')
        self.chi0XS  = np.zeros((wnSet.shape[0],Kset.shape[0]),dtype='complex')
        self.chi0XS2 = np.zeros((wnSet.shape[0],Kset.shape[0]),dtype='complex')
        self.chi0PxP = np.zeros((wnSet.shape[0],Kset.shape[0]),dtype='complex')
        self.chi0PxP2 = np.zeros((wnSet.shape[0],Kset.shape[0]),dtype='complex')
        self.gkdNorm = 0.0
    
        for iwn,wn in enumerate(wnSet): # reduced tp frequencies !!
            # print "iwn = ",iwn
            iwG = int(iwn - self.iwG40 + self.iwG0)
            minusiwPlusiwm = min(max(NwG-iwG-1 + self.iwm,0),NwG-1) # -iwn + iwm
            iwPlusiwm = int(min(max(iwG + self.iwm,0),NwG-1))  # iwn+iwm
            for iK,K in enumerate(Kset):
                # c = zeros((8),dtype='complex')
                # c1 = 0.0; c2 = 0.0; c3 = 0.0; c4 = 0.0; c5 = 0.0
                # for k in kPatch:
                #     kx = K[0]+k[0]; ky = K[1]+k[1]
                #     ek = self.dispersion(kx,ky)
                #     gkd = cos(kx) - cos(ky)
                #     gkxs= cos(kx) + cos(ky)
                #     gkpp= gkd * sin(kx)
                #     if self.vertex_channel in ("PARTICLE_PARTICLE_SUPERCONDUCTING","PARTICLE_PARTICLE_UP_DOWN","PARTICLE_PARTICLE_SINGLET"):
                #         Qx = self.qchannel[0]; Qy = self.qchannel[1]
                #         emkpq = self.dispersion(-kx+Qx, -ky+Qy)
                #         iKQ = self.iKSum[self.iKDiff[self.iK0,iK],self.iQ]
                #         minusiwPlusiwm = min(max(NwG-iwG-1 + self.iwm,0),NwG-1) # -iwn + iwm
                #         c[0] = 1./(1j*wn+self.mu-ek-self.sigma[iwG,iK]) * 1./(-1j*wn+self.mu-emkpq-self.sigma[minusiwPlusiwm,iKQ])
                #         c[1] += c[0]
                #         c[2] += c[0] * gkd
                #         c[3] += c[0] * gkd**2
                #         c[4] += c[0] * gkxs
                #         c[5] += c[0] * gkxs**2
                #         c[6] += c[0] * gkpp
                #         c[7] += c[0] * gkpp**2
                #         if (iwn==0): self.gkdNorm += gkd**2
                #     else:
                #         Qx = self.qchannel[0]; Qy = self.qchannel[1]
                #         ekpq = self.dispersion(kx+Qx, ky+Qy)
                #         iKQ = int(self.iKSum[iK,self.iQ])
                #         iwPlusiwm = int(min(max(iwG + self.iwm,0),NwG-1))  # iwn+iwm
                #         c[1] -= 1./(1j*wn+self.mu-ek-self.sigma[iwG,iK]) * 1./(1j*wn+self.mu-ekpq-self.sigma[iwPlusiwm,iKQ])

                Qx = self.qchannel[0]; Qy = self.qchannel[1]
                kx = K[0]+self.kPatch[:,0]; ky = K[1]+self.kPatch[:,1]
                ek = self.dispersion(kx,ky)
                gkd = cos(kx) - cos(ky)
                gkxs= cos(kx) + cos(ky)
                gkpp= gkd * sin(kx)
                c = zeros((7),dtype='complex')
                if self.vertex_channel in ("PARTICLE_PARTICLE_SUPERCONDUCTING","PARTICLE_PARTICLE_UP_DOWN","PARTICLE_PARTICLE_SINGLET"):                
                    emkpq = self.dispersion(-kx+Qx, -ky+Qy)
                    iKQ = self.iKSum[self.iKDiff[self.iK0,iK],self.iQ]
                    cc = 1./(1j*wn+self.mu-ek-self.sigma[iwG,iK]) * 1./(-1j*wn+self.mu-emkpq-self.sigma[minusiwPlusiwm,iKQ])
                    c[0] = sum(cc)
                    c[1] = sum(cc*gkd)
                    c[2] = sum(cc*gkd**2)
                    c[3] = sum(cc*gkxs)
                    c[4] = sum(cc*gkxs**2)
                    c[5] = sum(cc*gkpp)
                    c[6] = sum(cc*gkpp**2)
                    if (iwn==0): self.gkdNorm = sum(gkd**2)
                else:
                    ekpq = self.dispersion(kx+Qx, ky+Qy)
                    iKQ  = int(self.iKSum[iK,self.iQ])
                    cc   = -1./(1j*wn+self.mu-ek-self.sigma[iwG,iK]) * 1./(1j*wn+self.mu-ekpq-self.sigma[iwPlusiwm,iKQ])
                    c[0] = sum(cc)

                self.chi0[iwn,iK]     = c[0]/kPatch.shape[0]
                self.chi0D[iwn,iK]    = c[1]/kPatch.shape[0]
                self.chi0D2[iwn,iK]   = c[2]/kPatch.shape[0]
                self.chi0XS[iwn,iK]   = c[3]/kPatch.shape[0]
                self.chi0XS2[iwn,iK]  = c[4]/kPatch.shape[0]
                self.chi0PxP[iwn,iK]  = c[5]/kPatch.shape[0]
                self.chi0PxP2[iwn,iK] = c[6]/kPatch.shape[0]

        self.chi0M = np.diag(self.chi0.reshape(self.nt))
        self.gkdNorm /= kPatch.shape[0]

        if self.vertex_channel in ("PARTICLE_PARTICLE_SINGLET"):
            # self.chi0[iwn,ik] also appears for k2=q-k1 from the cross terms
            NwG4 = self.NwG4
            for iwn in range(NwG4):
                for ik in range(Nc):
                    i1 = ik + Nc * iwn
                    ikPlusQ = int(self.iKSum[self.iKDiff[self.iK0,ik],self.iQ]) # -k+Q
                    minusiwPlusiwm = int(min(max(NwG4-iwn-1 + self.iwm,0),NwG4-1)) # -iwn + iwm
                    i2 = ikPlusQ + minusiwPlusiwm * Nc # k2 = q-k1
                    self.chi0M[i1,i2] += self.chi0[iwn,ik]


    def calcPd0FS(self,FSpoints):
        c1=0.0
        NwG=self.NwG
        for iwn,wn in enumerate(self.wn):
        #for iwn,wn in enumerate(self.wnSet):
            iwG = iwn
            #iwG = int(iwn - self.iwG40 + self.iwG0)
            for iK in FSpoints:
                Kx = self.Kvecs[iK,0]
                Ky = self.Kvecs[iK,1]
                #for k in self.kPatch:
                for k in [[0,0]]:
                    kx = Kx+k[0]; ky = Ky+k[1]
                    ek = self.dispersion(kx,ky)
                    gkd = cos(kx)-cos(ky)
                    emk = self.dispersion(-kx, -ky)
                    iKQ = self.iKDiff[self.iK0,iK]
                    minusiw = min(max(NwG-iwG-1,0),NwG-1) # -iwn + iwm
                    c1 += gkd**2/(1j*wn+self.mu-ek-self.sigma[iwG,iK]) * 1./(-1j*wn+self.mu-emk-self.sigma[minusiw,iKQ])
        #print("Pd0 = T/N sum_kF (coskx-cosky)^2 G(k,iwn)*G(-k,-iwn)",c1 / (FSpoints.shape[0]*self.kPatch.shape[0]*self.invT))
        print("Pd0FS = T/N sum_kF (coskx-cosky)^2 G(k,iwn)*G(-k,-iwn)",c1 / (FSpoints.shape[0]*self.invT))

    def calcProjectionsKwn(self,matrix,formFactor,normPower=0):
        gk = formFactor(self.Kvecs[:,0],self.Kvecs[:,1])
        fwn = ((np.pi*self.temp)**2+self.wCutOff**2)/(self.wnSet**2 + self.wCutOff**2)
        ff = np.outer(fwn,gk).reshape(self.NwG4*self.Nc)
        matrixProjected = real(np.dot(ff,np.dot(matrix,ff)))/np.inner(ff,ff)**normPower
        return matrixProjected

    def calcProjection(self,matrix,fofKwn):
        matrixProjected = real(np.dot(fofKwn,np.dot(matrix,fofKwn)))
        return matrixProjected


    def calcPd0andVd(self):
        # Calculate various separable approximations for lambda_d ~ Vd * Pd0 

        # 1. $P_{d,0}  = \frac{T}{N_c}\sum_k g^2(k) G(k)G(-k)$ with $g(k) = (\cos k_x-\cos k_y)\frac{\pi^2T^2+\omega_c^2}{\omega_n^2+\omega_c^2}$.
        gkd = cos(self.Kvecs[:,0]) - cos(self.Kvecs[:,1])
        fwn = ((np.pi*self.temp)**2+self.wCutOff**2)/(self.wnSet**2 + self.wCutOff**2)
        fg = np.outer(fwn,gkd).reshape(self.NwG4*self.Nc)

        self.Vd01 = real( self.calcProjection(self.GammaM,fg)) / np.inner(fg,fg)**2
        self.Pd01 = real( self.calcProjection(self.chi0M,fg) )/ (self.invT*self.Nc)

        if (self.found_d):

            # 2. $P_{d,02} = \frac{T}{N_c}\sum_k \phi_d^2(k) G(k)G(-k)$ with $\phi_d(k)$ the d-wave eigenvector.  
            fg = self.evecs[:,:,self.ind_d].reshape(self.NwG4*self.Nc) / self.evecs[:,:,self.ind_d].max()

            self.Vd02 = real( self.calcProjection(self.GammaM,fg)) / np.inner(fg,fg)**2
            self.Pd02 = real( self.calcProjection(self.chi0M,fg) )/ (self.invT*self.Nc)

            # 3. Now use phid(K,piT) for k-dependence and Lorentzian for frequency dependence
            gkd = self.evecs[int(self.NwG4/2),:,self.ind_d] / self.evecs[int(self.NwG4/2),:,self.ind_d].max()
            fwn = ((np.pi*self.temp)**2+self.wCutOff**2)/(self.wnSet**2 + self.wCutOff**2)
            fg = np.outer(fwn,gkd).reshape(self.NwG4*self.Nc)

            self.Vd03 = real(self.calcProjection(self.GammaM,fg)) / np.inner(fg,fg)**2
            self.Pd03 = real(self.calcProjection(self.chi0M,fg) )/ (self.invT*self.Nc)




    def calcProjections(self,matrix,formFactor,wCutOff=1):
        nCutOff = ceil(wCutOff*self.invT/(2.*pi) - 0.5)
        nCutOff = int(max(nCutOff,1))
        # print("nCutOff=",nCutOff)
        gk = formFactor(self.Kvecs[:,0],self.Kvecs[:,1])
        if wCutOff > 0:
            fwn = ((np.pi*self.temp)**2+wCutOff**2)/(self.wnSet**2 + wCutOff**2)
            fg = np.outer(fwn,gk).reshape(self.NwG4*self.Nc)
            matrixProjected = real(np.dot(fg,np.dot(matrix,fg)))
            # matrixProjected = real(np.dot(fg,np.dot(matrix,fg)))/np.inner(fg,fg)
        else: #only take piT,piT element
            ind0 = int(self.NwG4/2)
            matrix00 = matrix.reshape(self.NwG4,self.Nc,self.NwG4,self.Nc)[ind0,:,ind0,:]
            # mm = sum(sum(matrix00,axis=0),axis=1)/4.0
            matrixProjected = real(np.dot(gk,np.dot(matrix00,gk)))
            # matrixProjected = real(np.dot(gk,np.dot(mm,gk)))
        return matrixProjected


    def calcPd0_LorentzianCutoff(self):
        # P_d0(T,x)= T/N Sum (coskx-cosky)^2G(k,wn)G(-k,-wn) [(piT)^2+wc^2]/[wn^2+wc^2]
        gk = self.dwave(self.Kvecs[:,0],self.Kvecs[:,1])
        fwn = ((np.pi*self.temp)**2+self.wCutOff**2)/(self.wnSet**2 + self.wCutOff**2)
        ff = np.outer(fwn,gk**2).reshape(self.NwG4*self.Nc)
        chi0Projected = real(np.dot(self.chi0.reshape(self.NwG4*self.Nc),ff))/(self.invT*self.Nc)
        return chi0Projected

    def calcPd0(self,wCutOff=0.5714):
        nCutOff = ceil(wCutOff*self.invT/(2.*pi) - 0.5)
        nCutOff = int(max(nCutOff,1))
        print("nCutOff=",nCutOff)
        gkd = cos(self.Kvecs[:,0]) - cos(self.Kvecs[:,1])
        fwn = ((np.pi*self.temp)**2+wCutOff**2)/(self.wnSet**2 + wCutOff**2)
        # cc = real(sum(self.chi0[abs(self.wnSet) <= wc,:],axis=0))
        # cc = 2.0*real(sum(self.chi0[self.iwG40:self.iwG40+nCutOff,:],axis=0))
        # self.Pd0 = np.dot(gkd**2, cc)/(self.invT*self.Nc)/np.inner(gkd,gkd)
        fg = np.outer(fwn,gkd).reshape(self.NwG4*self.Nc)
        self.Pd0a = real(np.dot(fg**2, self.chi0.reshape(self.NwG4*self.Nc))/(self.invT*self.Nc))
        print("Pd0(T) with cutOff=",self.Pd0a)
        # Now compare this with projection onto d-wave eigenvector
        if self.found_d:
            phid = self.evecs[:,:,self.ind_d]
            phid /= phid.max()
            self.phid = phid.reshape(self.NwG4*self.Nc)
            chi0pp = self.chi0.reshape(self.NwG4*self.Nc)
            self.Pd0b = real(np.dot(self.phid**2,chi0pp) / (self.invT*self.Nc) )
            print("Pd0(T) from projection onto eigenvector=",self.Pd0b)
            # Projection onto phid * fg
            self.fg = fg/fg.max()
            self.Pd0c = np.dot(abs(self.phid*self.fg),chi0pp) / (self.invT*self.Nc)
            print("Pd0(T) from projection onto eigenvector * fg =",real(self.Pd0c))

            # Since phid at low T becomes noisy at large wn, lets cut-off phid(k,wn) for |wn| > wc 
            self.phidFixed = np.zeros_like(self.evecs[:,:,self.ind_d])
            nC = 1
            self.phidFixed[self.iwG40-nC:self.iwG40+nC,:] = self.evecs[:,:,self.ind_d][self.iwG40-nC:self.iwG40+nC,:]
            phidF = self.phidFixed.reshape(self.NwG4*self.Nc)
            norm = np.dot(phidF,phidF)
            self.phidFixed /= sqrt(norm)
            phidF = self.phidFixed.reshape(self.NwG4*self.Nc)
            self.Pd0d = real(np.dot(phidF**2,chi0pp) /(self.invT*self.Nc))
            print("Pd0(T) from projection onto fixed eigenvector=",self.Pd0d)
            # Now use self.chi0D2
            cc = 2.0*real(sum(self.chi0D2[self.iwG40:self.iwG40+nCutOff,:]))
            self.Pd0e = cc/(self.invT*self.Nc)/self.gkdNorm
            print("Pd0(T) with lattice gkd and wn cutOff=",real(self.Pd0e))
            # Now use phid(K,piT) for k-dependence and Lorentzian for frequency dependence
            gkd = phid[int(self.NwG4/2),:]
            fg = np.outer(fwn,gkd).reshape(self.NwG4*self.Nc)
            self.Pd05 = real(np.dot(fg**2, self.chi0.reshape(self.NwG4*self.Nc)))/(self.invT*self.Nc)



    # def calcChi0Tilde(self,evec):
    #     gk = self.dwave(self.Kvecs[:,0],self.Kvecs[:,1])
    #     pk = 0.0*np.ones_like(evec)
    #     for ik in range(self.Nc):
    #         pk[:,ik] = evec[:,ik] * sqrt(real(self.chic0[:,ik])) * gk[ik]

    #     chi0Tilde = sum(real(self.chic0*pk))*self.temp/self.Nc * sum(pk)
    #     return chi0Tilde

    # def calcPdFromEigen(self,ia=0):
    #     nt = self.Nc*self.NwG4

    #     c2 = 1./(1.-self.lambdas2[ia]) * self.calcChi0Tilde(self.evecs[:,:,ia])

    #     print("Pd from eigensystem (only eigenvalue ia=",ia,"): ",c2)

    def calcPdFromEigenFull(self,ia=0):
        gk = self.dwave(self.Kvecs[:,0],self.Kvecs[:,1])
        nt = self.nt; nc=self.Nc; nw=self.NwG4
        
        eval = self.lambdas2
        Lambda = np.diag(1./(1-eval))
        # Dkk = zeros((nt,nt), dtype=real)
        phit =zeros((nw,nc,nt), dtype=float)
        phit2=zeros((nw,nc,nt), dtype=float)
        if self.calcCluster: 
            chi0 = self.chic0
        else:
            chi0 = self.chi0
        for ialpha in range(nt):
            phit[:,:,ialpha] = real(self.evecs2[:,:,ialpha])*sqrt(real(chi0[:,:]))
        phit2[:,:,ia] = real(self.evecs2[:,:,ia])*sqrt(real(chi0[:,:]))
        phi = phit.reshape(nt,nt)
        phi2 = phit2.reshape(nt,nt)
        Dkk = dot(phi,dot(Lambda,phi.T))
        Dkk = Dkk.reshape(nw,nc,nw,nc)
        Dkk2 = dot(phi2,dot(Lambda,phi2.T))
        Dkk2 = Dkk2.reshape(nw,nc,nw,nc)
        Lkk = sum(sum(Dkk,axis=0),axis=1)
        Lkk2 = sum(sum(Dkk2,axis=0),axis=1)
        Pd = dot(gk,dot(Lkk,gk)) * self.temp/self.Nc
        PdIa = dot(gk,dot(Lkk2,gk)) * self.temp/self.Nc

        self.PdEigen = Pd
        self.PdIa = PdIa
        print("Pd from eigensystem (all eigenvalues): ",Pd)
        print("Pd from eigensystem (ia   eigenvalue): ",PdIa)


    def calcSWaveSus(self):
        G2L = linalg.inv(linalg.inv(self.chi0M) - self.GammaM/(float(self.Nc)*self.invT))
        self.chiL = sum(G2L)/(float(self.Nc)*self.invT)
        self.chiC = sum(self.G4)/(float(self.Nc)*self.invT)
        print("Lattice spin susceptibility     : ", self.chiL)
        print("Lattice bare spin susceptibility: ",sum(self.chi0)/(float(self.Nc)*self.invT))
        print("Cluster spin susceptibility     : ", self.chiC)
        print("Cluster bare spin susceptibility: ",sum(self.chic0)/(float(self.Nc)*self.invT))

    def calcSCSus(self):
        # Calculate from gd*G4*gd = gd*GG*gd + gd*GG*GammaRed*GG*gd
        #GammaRed = self.GammaRed.reshape(self.NwG4*self.Nc,self.NwG4*self.Nc)
        print("")
        gkd = cos(self.Kvecs[:,0]) - cos(self.Kvecs[:,1])
        csum = 0.0
        csumxs = 0.0
        csumpxp = 0.0
        ccsum = 0.0
        csum3 = 0.0
        for iw1 in range(self.NwG4):
            for iK1 in range(self.Nc):
                for iw2 in range(self.NwG4):
                    for iK2 in range(self.Nc):
                        csum   += self.chi0D[iw1,iK1]*self.GammaRed[iw1,iK1,iw2,iK2]*self.chi0D[iw2,iK2]
                        csumxs += self.chi0XS[iw1,iK1]*self.GammaRed[iw1,iK1,iw2,iK2]*self.chi0XS[iw2,iK2]
                        csumpxp+= self.chi0PxP[iw1,iK1]*self.GammaRed[iw1,iK1,iw2,iK2]*self.chi0PxP[iw2,iK2]
                        ccsum  += gkd[iK1]*self.chi0[iw1,iK1]*self.GammaRed[iw1,iK1,iw2,iK2]*self.chi0[iw2,iK2]*gkd[iK2]
        csum /= (self.Nc*self.invT)**2
        csumxs /= (self.Nc*self.invT)**2
        csumpxp /= (self.Nc*self.invT)**2
        ccsum /= (self.Nc*self.invT)**2
        csum1  = sum(self.chi0D2)/(self.Nc*self.invT)
        csum1xs= sum(self.chi0XS2)/(self.Nc*self.invT)
        csum1pxp= sum(self.chi0PxP2)/(self.Nc*self.invT)
        ccsum1 = sum(np.dot(self.chi0,gkd**2))/(self.Nc*self.invT)
        csum += csum1
        csumxs += csum1xs
        csumpxp += csum1pxp
        ccsum += ccsum1
        self.Pd = real(csum)
        self.Pxs = real(csumxs)
        self.Pdgkc = real(ccsum)
        self.Ppxp = real(csumpxp)
        csum3 = sum(real(self.chi0D2[abs(self.wnSet) <= 2.*4*self.t**2/self.U,:]))/(self.Nc*self.invT)
        print ("d-wave  SC susceptibility: ",csum)
        print ("xs-wave SC susceptibility: ",csumxs)
        print ("px'-wave SC susceptibility: ",csumpxp)
        print("")
        print ("bare d-wave SC susceptibility:  ",csum1)
        print ("bare xs-wave SC susceptibility: ",csum1xs)
        print ("bare d-wave SC susceptibility with cutoff wc = J: ",csum3)
        print ("bare d-wave SC lattice (with cluster form-factors) susceptibility: ",ccsum1)
        print ("d-wave SC lattice (with cluster form-factors) susceptibility: ",ccsum)
        print("")

    def calcSCClusterSus(self):
        gkd  = cos(self.Kvecs[:,0]) - cos(self.Kvecs[:,1])
        gkpx = sin(self.Kvecs[:,0])
        gkpp = gkd*gkpx

        csum=zeros((3),dtype='complex')
        for iK1 in range(self.Nc):
            for iK2 in range(self.Nc):
                csum[0]  += gkd[iK1]*sum(self.G4[:,:,iK1,iK2])*gkd[iK2]
                csum[1]  += gkpx[iK1]*sum(self.G4[:,:,iK1,iK2])*gkpx[iK2]
                csum[2]  += gkpp[iK1]*sum(self.G4[:,:,iK1,iK2])*gkpp[iK2]
        csum /= self.Nc*self.invT
        self.Pdc = real(csum[0])
        print ("d-wave SC cluster susceptibility : ",csum[0])
        print ("px-wave SC cluster susceptibility: ",csum[1])
        print ("p'-wave SC cluster susceptibility: ",csum[2])



    def calcGammaRedEvals(self):
        iw2 = self.iwG40 # = piT
        GR = self.GammaRed[:,self.FSpoints,iw2,:][:,:,self.FSpoints]
        self.eigdSC = zeros((self.NwG4))
        for iw1 in range(self.NwG4):
            w,v = linalg.eig(GR[iw1,:,:])
            # print(iw1,w.max()/self.FSpoints.__len__())
            self.eigdSC[iw1] = real(w.max()/self.FSpoints.__len__())

    def calcGammaProjected(self):
        return 0

####### Momentum domain functions

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

    def K_2_iK(self,Kx,Ky):
        delta=1.0e-4
        shift = 0.0
        if self.shiftedK: shift = -np.pi
        # First map (Kx,Ky) into [0...2pi,0...2pi] region where Kvecs are defined
        if Kx < shift-delta       : Kx += 2*pi
        if Ky < shift-delta       : Ky += 2*pi
        if Kx > shift+2.*pi-delta : Kx -= 2*pi
        if Ky > shift+2.*pi-delta : Ky -= 2*pi
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

    def pxpwave(self,kx,ky):
        return (cos(kx)-cos(ky))*sin(kx)

    def projectOnDwave(self,matrix):
        Ks = self.Kvecs
        gk = self.dwave(Ks[:,0], Ks[:,1])
        c1 = dot(gk, dot(matrix,gk) ) / dot(gk,gk)
        return c1

    def projectVectorOnDwave(self,vec):
        gk = self.dwave(self.Kvecs[:,0],self.Kvecs[:,1])
        nt = self.nt; nc=self.Nc; nw=self.NwG4
        c1 = dot(gk,vec)
        # print("d-wave projection: ", c1)
        return c1

    def dispersion(self,kx,ky):
        return -2.*self.t*(cos(kx)+cos(ky)) - 4.0*self.tp*cos(kx)*cos(ky)

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
            ax.set(adjustable='box-forced', aspect='equal')
        if title==None:
            title = r"Leading eigensolutions of BSE for $U=$" + str(self.U) + r", $t\prime=$" + str(self.tp) + r", $\langle n\rangle=$" + str(self.fill) + r", $T=$" + str(self.temp)
        fig.suptitle(title, fontsize=20)

    def plotEV(self,ax,Kvecs,lambdas,evecs,inr):
        prop_cycle = rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        Nc = evecs.shape[1]; Nw = self.evecs.shape[0]
        iw0=int(Nw/2)
        imax = argmax(evecs[iw0,:,inr])
        if (abs(evecs[iw0-1,imax,inr]-evecs[iw0,imax,inr]) <= 1.0e-5):
            freqString = "; even frequency"
        else:
            freqString = "; odd frequency"

        colVec = Nc*[colors[0]]
        for ic in range(Nc):
            if real(evecs[iw0,ic,inr])*real(evecs[iw0,imax,inr]) < 0.0: colVec[ic] = colors[1]
        # print "colVec=",colVec
        ax.scatter(Kvecs[:,0]/pi,Kvecs[:,1]/pi,s=abs(real(evecs[iw0,:,inr]))*2500,c=colVec)
        ax.set(aspect=1)
        ax.set_xlim(-1.0,1.2); ax.set_ylim(-1.0,1.2)
        ax.set_title(r"$\lambda=$"+str(round(real(lambdas[inr]),4))+freqString)
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        ax.grid(True)
        for tic in ax.xaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
        for tic in ax.yaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False


    def plotEvec(self,index):
        # Make dataframe for eigenvector
        # %%
        df = p.DataFrame([[self.wnSet[j],str("K=(") + str(round(self.Kset[i,0]/pi,2)) + str(",") + str(round(self.Kset[i,1]/pi,2)) + str(")"),real(self.evecs2[j,i,index])] for i in range(self.Nc) for j in range(self.wnSet.shape[0])])
        df.columns = ["$\omega_n$","K","$\phi(K,\omega_n)$"]

        self.plotEv = (ggplot(df,aes(x="$\omega_n$",y="$\phi(K,\omega_n)$")) 
                    + geom_line() 
                    + facet_wrap("K") 
                    + ggtitle("index:"+str(index)+"; $\lambda=$"+str(round(self.lambdas2[index],3)))
        )
        print(self.plotEv)


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


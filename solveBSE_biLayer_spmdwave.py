'''
Modified from Maier's code solveBSE_fromG4_BiLayer_newFormat_220211.py
'''
import numpy as np
from numpy import *
import matplotlib.pyplot as mpl
import h5py
import sys
import os
from matplotlib.pyplot import *

#import myggplottheme
# Similar to solveBSE_fromG4_multiOrbit.py, but here we transform to kz=0,pi for bilayer model immediately
# This currently only works for pp channel

class BSE:
    def __init__(self,Tval,fileG4,fileG,draw=False,symmetrize_G4=False,phSymmetry=False,nkfine=100,oldFormat=False,newMaster=True,allq=False,evenFreqOnly=True,write_data_file=False):

        self.Tval = Tval
        
        self.fileG4 = fileG4

        self.fileG = fileG

        self.draw = draw

        self.symmetrize_G4 = symmetrize_G4

        self.phSymmetry = phSymmetry

        self.oldFormat = oldFormat

        self.newMaster = newMaster

        self.allq = allq

        self.evenFreqOnly = evenFreqOnly
        
        self.write_data_file = write_data_file


        self.readData()

        self.setupMomentumTables() # This is done with Kz !

        self.iQ0  = self.K_2_iK(self.qchannel[0],self.qchannel[1],0.0)

        self.iQPi = self.K_2_iK(self.qchannel[0],self.qchannel[1],np.pi)


        if self.symmetrize_G4: self.symmetrizePG()

        self.transformG()

        self.transformG4()


        if self.symmetrize_G4: self.symmetrizeG4_wn()



        # self.determine_iKPiPi()

        # self.symmetrizeG4()

        # print ("Index of (pi,pi): ",self.iKPiPi)


        self.calcChi0Cluster()

        self.calcGammaIrr()

        self.buildChi0Lattice(nkfine)

        self.calcFillingBilayer()

        self.buildKernelMatrix()

        self.calcKernelEigenValues()

        #self.transformEvecsToKz()

        self.calcSus()

        # self.calcVandP0(2)

        self.calcVandP0(1.)

        self.calcVs()

        self.calcEffectiveSingleBandInteraction()

        #title ="Leading eigensolutions of BSE for U="+str(self.U)+", t'="+str(self.t1)+r", $\langle n\rangle$="+str(round(self.fill,4))+", T="+str(round(self.temp,4))

        # if self.vertex_channel == "PARTICLE_HOLE_TRANSVERSE":

        #     self.calcAFSus()

        #     print("AF cluster susceptibility: ",sum(self.G4)/(float(self.Nc)*self.invT))

        # self.plotLeadingSolutions(self.Kvecs,self.lambdas2,self.evecs2[:,0:16,:],title)

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

        if self.oldFormat == False:

            self.iwm = array(f['parameters']['four-point']['frequency-transfer'])[0] # transferred frequency in units of 2*pi*temp

            print("Transferred frequency iwm = ",self.iwm)

            self.qchannel = array(f['parameters']['four-point']['momentum-transfer'])

            print("Transferred momentum q = ",self.qchannel)

            # if ((self.qchannel[0]!=0) | (self.qchannel[1]!=0)): exit(0)

            #a = array(f['parameters']['four-point']['type'])[:]

            #self.vertex_channel = ''.join(chr(i) for i in a)

            #print("Vertex channel = ",self.vertex_channel)

            self.vertex_channel = 'PARTICLE_PARTICLE_UP_DOWN'

            self.invT = array(f['parameters']['physics']['beta'])[0]

            print("Inverse temperature = ",self.invT)

            self.temp = 1.0/self.invT

            self.U = array(f['parameters']['bilayer-Hubbard-model']['U1'])[0]

            print("U = ",self.U)

            # self.tp1 = array(f['parameters']['bilayer-Hubbard-model']['t-prime1'])[0]

            self.tp1 = array(f['parameters']['bilayer-Hubbard-model']['t1-prime'])[0]

            print("t-prime = ",self.tp1)

            #self.tpp = array(f['parameters']['bilayer-Hubbard-model']['t-pp'])[0]

            #print("t-pp = ",self.tpp)

            # self.t1 = np.array(f['parameters']['bilayer-Hubbard-model']['t1'])[0]

            self.t1 = np.array(f['parameters']['bilayer-Hubbard-model']['t1'])[0]

            # self.t2 = np.array(f['parameters']['bilayer-Hubbard-model']['t2'])[0]

            print("t = ",self.t1)

            # self.tp2 = array(f['parameters']['bilayer-Hubbard-model']['t-prime2'])[0]

            # print("t-prime = ",self.tp2)

            # if (self.tp1!=self.tp2):

                # sys.exit("tp1 not equal to tp2; exiting")

            # self.DeltaE = array(f['parameters']['bilayer-Hubbard-model']['Delta-E'])[0]

            # print("Delta-E = ",self.DeltaE)

            # if self.DeltaE!=0.0:

                # sys.exit("Delta-E = finite; exiting")

            self.tperp = array(f['parameters']['bilayer-Hubbard-model']['t-perp'])[0]

            print("tperp = ",self.tperp)

            self.tperpp = array(f['parameters']['bilayer-Hubbard-model']['t-perp-prime'])[0]

            print("tperp_p = ",self.tperpp)

            #self.tperppp = array(f['parameters']['bilayer-Hubbard-model']['t-perp-pp'])[0]

            #print("tperp_pp = ",self.tperppp)

            # self.tperpp = array(f['parameters']['bilayer-Hubbard-model']['t-perp-p'])[0]

            # print("tperpp = ",self.tperpp)

            self.fill = array(f['parameters']['physics']['density'])[0]

            print("filling = ",self.fill)

            self.dens = array(f['DCA-loop-functions']['density']['data'])[-1]

            print("actual filling:",self.dens)

            self.nk = array(f['DCA-loop-functions']['n_k']['data'])

            self.sigmaarray=array(f['DCA-loop-functions']['L2_Sigma_difference']['data'])
            print("L2_Sigma_difference =", self.sigmaarray,'\n')

            # Now read the 4-point Green's function

            if self.newMaster == False:

                G4Re  = array(f['functions']['G4_k_k_w_w']['data'])[:,:,:,:,:,:,:,:,0]

                G4Im  = array(f['functions']['G4_k_k_w_w']['data'])[:,:,:,:,:,:,:,:,1]

                self.G4 = G4Re+1j*G4Im

            else:

                if self.allq == False:

                    # olf format: order of indices: w1,w2,K1,K2

                    # G4Re  = array(f['functions']['G4']['data'])[0,:,:,0,:,:,0,0,0,0,0]

                    # G4Im  = array(f['functions']['G4']['data'])[0,:,:,0,:,:,0,0,0,0,1]

                    # New format: order of indices: w1,K1,w2,K2

                    G4Re  = array(f['functions']['G4_PARTICLE_PARTICLE_UP_DOWN']['data'])[0,0,:,:,:,:,:,:,:,:,0]

                    G4Im  = array(f['functions']['G4_PARTICLE_PARTICLE_UP_DOWN']['data'])[0,0,:,:,:,:,:,:,:,:,1]

                    print("G4 shape:",G4Re.shape)

                else:

                    # order of indices: Q,w1,K1,w2,K2

                    G4Re  = array(f['functions']['G4_PARTICLE_PARTICLE_UP_DOWN']['data'])[0,self.iq,:,:,:,:,0,0,0,0,0]

                    G4Im  = array(f['functions']['G4_PARTICLE_PARTICLE_UP_DOWN']['data'])[0,self.iq,:,:,:,:,0,0,0,0,1]

                self.G4 = G4Re+1j*G4Im

                # Now reorder G4

                self.G4=self.G4.swapaxes(1,2) # Now G4's shape is w1,w2,K1,K2


            # Now read the cluster Green's function

            GRe = array(f['functions']['cluster_greens_function_G_k_w']['data'])[:,:,0,:,0,:,0]

            GIm = array(f['functions']['cluster_greens_function_G_k_w']['data'])[:,:,0,:,0,:,1]

            self.Green = GRe + 1j * GIm



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

            self.mu = np.array(f['DCA-loop-functions']['chemical-potential']['data'])[-1]

            # if (self.t1!=self.t2):

                # sys.exit("t1 not equal to t2; exiting")


            self.NwTP = 2*np.array(f['parameters']['domains']['imaginary-frequency']['four-point-fermionic-frequencies'])[0]


            # Now read the QMC sign

            self.sign = np.array(f['DCA-loop-functions']['sign']['data'])[:]

            print ("QMC sign:",self.sign)
            
        else:
            self.iwm = array(f['parameters']['vertex-channel']['w-channel'])[0] # transferred frequency in units of 2*pi*temp

            print("Transferred frequency iwm = ",self.iwm)

            self.qchannel = array(f['parameters']['vertex-channel']['q-channel'])

            print("Transferred momentum q = ",self.qchannel)

            # if ((self.qchannel[0]!=0) | (self.qchannel[1]!=0)): exit(0)

            a = array(f['parameters']['vertex-channel']['vertex-measurement-type'])[:]

            self.vertex_channel = ''.join(chr(i) for i in a)

            print("Vertex channel = ",self.vertex_channel)

            self.invT = array(f['parameters']['physics-parameters']['beta'])[0]

            print("Inverse temperature = ",self.invT)

            self.temp = 1.0/self.invT

            self.U = array(f['parameters']['bilayer-model']['U'])[0]

            print("U = ",self.U)

            self.tp = array(f['parameters']['bilayer-model']['t-prime'])[0]

            self.tp1 = self.tp; self.tp2=self.tp

            print("t-prime = ",self.tp)

            self.DeltaE=0.0

            self.tperp = array(f['parameters']['bilayer-model']['tz'])[0]

            print("tperp = ",self.tperp)

            self.fill = array(f['parameters']['physics-parameters']['density'])[0]

            print("filling = ",self.fill)

            self.dens = array(f['DCA-loop-functions']['density']['data'])

            print("actual filling:",self.dens)

            self.nk = array(f['DCA-loop-functions']['n_k']['data'])



            # Now read the 4-point Green's function

            G4Re  = array(f['functions']['G4_k_k_w_w']['data'])[:,:,:,:,:,:,:,:,0]

            G4Im  = array(f['functions']['G4_k_k_w_w']['data'])[:,:,:,:,:,:,:,:,1]

            self.G4 = G4Re+1j*G4Im


            # Now read the cluster Green's function

            GRe = array(f['functions']['cluster_greens_function_G_k_w']['data'])[:,:,0,:,0,:,0]

            GIm = array(f['functions']['cluster_greens_function_G_k_w']['data'])[:,:,0,:,0,:,1]

            self.Green = GRe + 1j * GIm



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

            self.t = np.array(f['parameters']['bilayer-model']['t'])[0]

            self.t1=self.t; self.t2=self.t

            self.mu = np.array(f['DCA-loop-functions']['chemical-potential']['data'])[0]



            self.NwTP = 2*np.array(f['parameters']['function-parameters']['two-particle-functions']['fermionic-frequencies'])[0]

            # self.iQ = self.K_2_iK(self.qchannel[0],self.qchannel[1])

            # print ("Index of transferred momentum: ", self.iQ)



        self.NwG4 = self.G4.shape[0]

        self.Nc  = self.Green.shape[1]

        self.NwG = self.Green.shape[0]

        self.nOrb = self.Green.shape[2]

        self.nt = 2*self.Nc*self.NwG4

        print ("NwG4: ",self.NwG4)
        print ("NwG : ",self.NwG)
        print ("Nc  : ",self.Nc)
        print ("nOrb: ",self.nOrb)

        # self.iQ = self.K_2_iK(self.qchannel[0],self.qchannel[1])

        # print ("Index of transferred momentum: ", self.iQ)

        self.iwG40 = int(self.NwG4/2)

        self.iwG0  = int(self.NwG/2)

        f.close()

        # f2.close()

        # if (self.vertex_channel!="PARTICLE_PARTICLE_SUPERCONDUCTING"):

            # exit(0)


    def symmetrizePG(self):

        if (self.Nc == 16):

            import symmetrize_Nc16B

            sym=symmetrize_Nc16B.symmetrize()

        elif (self.Nc == 8):

            import symmetrize_Nc8

            sym=symmetrize_Nc8.symmetrize()


            for l1 in range(self.nOrb):

                for l2 in range(self.nOrb):

                    for l3 in range(self.nOrb):

                        for l4 in range(self.nOrb):

                            sym.apply_point_group_symmetries_Q0(self.G4[:,:,:,:,l1,l2,l3,l4])



    def transformG(self):
        # Essentially change the basis of the wave function from layer/orbital |1>, |2>
        # to |a> = 1/sqrt(2) (|1> + |2>) and |b> = 1/sqrt(2) (|1> - |2>)
        # So |a> and |b> are for kz=0 and pi separately
        # now G for band a and b are
        # <a|G|a> = 1/2 (...) and <b|G|b> = 1/2 (...)
        
        # 2*Nc to extend (kx,ky) to (kx,ky,kz)
        # see setupMomentumTables
        # self.K[0:Nc,0:2]    = self.Kvecs; self.K[0:Nc,2] = 0
        # self.K[Nc:2*Nc,0:2] = self.Kvecs; self.K[Nc:2*Nc,2] = np.pi

        Nc=self.Nc; NwG4=self.NwG4; NwG=self.NwG; nOrb = self.nOrb

        self.cGK=np.zeros((NwG,2*Nc),dtype='complex')

        self.sigmaK=np.zeros((NwG,2*Nc),dtype='complex')

        self.cGK[:,0:Nc]    = 0.5*(self.Green[:,:,0,0] + self.Green[:,:,1,1] + self.Green[:,:,0,1] + self.Green[:,:,1,0])

        self.cGK[:,Nc:2*Nc] = 0.5*(self.Green[:,:,0,0] + self.Green[:,:,1,1] - self.Green[:,:,0,1] - self.Green[:,:,1,0])

        self.sigmaK[:,0:Nc]    = 0.5*(self.sigma[:,:,0,0] + self.sigma[:,:,1,1] + self.sigma[:,:,0,1] + self.sigma[:,:,1,0])

        self.sigmaK[:,Nc:2*Nc] = 0.5*(self.sigma[:,:,0,0] + self.sigma[:,:,1,1] - self.sigma[:,:,0,1] - self.sigma[:,:,1,0])


    def transformG4(self):
        # In Peter's code:

        # PARTICLE_HOLE_MAGNETIC:

        #

        #       k1,l1           k2,l4

        #     ----->------------->------

        #             |     |

        #             |  G4 |

        #     -----<-------------<------

        #     k1+q,l3         k2+q,l2

        # For how to change basis into bonding and antibonding orbitals, see transformG but now more complicated

        Nc=self.Nc; NwG4=self.NwG4; NwG=self.NwG; nOrb = self.nOrb

        self.G4K=np.zeros((NwG4,2*Nc,NwG4,2*Nc,2),dtype='complex') # 2 for Qz=0 and Qz=pi

        for iw1 in range(self.NwG4):

            for iw2 in range(self.NwG4):

                 for l1 in range(self.nOrb):

                     for l2 in range(self.nOrb):

                         for l3 in range(self.nOrb):

                             for l4 in range(self.nOrb):

                                for iqz in range(0,2):

                                     # print("l1,l2,l3,l4,exp(np.pi*(l1-l2)),exp(np.pi*(l3-l4))",l1,l2,l3,l4,exp(1j*np.pi*(l1-l2)),exp(1j*np.pi*(l3-l4)))

                                    if self.vertex_channel in ("PARTICLE_HOLE_MAGNETIC","PARTICLE_HOLE_TRANSVERSE"):

                                        self.G4K[iw1,0:Nc,iw2,0:Nc,iqz]       += self.G4[iw1,iw2,:,:,l1,l2,l3,l4] * exp(1j*np.pi*iqz*(l2-l3))

                                        self.G4K[iw1,Nc:2*Nc,iw2,0:Nc,iqz]    += self.G4[iw1,iw2,:,:,l1,l2,l3,l4] * exp(1j*np.pi*(l1-l3)) * exp(1j*np.pi*iqz*(l2-l3))

                                        self.G4K[iw1,0:Nc,iw2,Nc:2*Nc,iqz]    += self.G4[iw1,iw2,:,:,l1,l2,l3,l4] * exp(1j*np.pi*(l2-l4)) * exp(1j*np.pi*iqz*(l2-l3))

                                        self.G4K[iw1,Nc:2*Nc,iw2,Nc:2*Nc,iqz] += self.G4[iw1,iw2,:,:,l1,l2,l3,l4] * exp(1j*np.pi*(l1-l3)) * exp(1j*np.pi*(l2-l4)) * exp(1j*np.pi*iqz*(l2-l3))

                                    else:

                                        self.G4K[iw1,0:Nc,iw2,0:Nc,iqz]       += self.G4[iw1,iw2,:,:,l1,l2,l3,l4] * exp(1j*np.pi*iqz*(l2-l4))

                                        self.G4K[iw1,Nc:2*Nc,iw2,0:Nc,iqz]    += self.G4[iw1,iw2,:,:,l1,l2,l3,l4] * exp(1j*np.pi*(l1-l2)) * exp(1j*np.pi*iqz*(l2-l4))

                                        self.G4K[iw1,0:Nc,iw2,Nc:2*Nc,iqz]    += self.G4[iw1,iw2,:,:,l1,l2,l3,l4] * exp(1j*np.pi*(l3-l4)) * exp(1j*np.pi*iqz*(l2-l4))

                                        self.G4K[iw1,Nc:2*Nc,iw2,Nc:2*Nc,iqz] += self.G4[iw1,iw2,:,:,l1,l2,l3,l4] * exp(1j*np.pi*(l1-l2)) * exp(1j*np.pi*(l3-l4)) * exp(1j*np.pi*iqz*(l2-l4))


        self.G4K = self.G4K*0.25

        if self.symmetrize_G4: self.symmetrizeG4()

        self.G4M = self.G4K.reshape(self.nt,self.nt,2)


        if self.vertex_channel in ("PARTICLE_HOLE_MAGNETIC","PARTICLE_HOLE_TRANSVERSE"):

            print("Cluster Chi(q,qz=0) :", sum(self.G4K[:,:,:,:,0]/(self.invT*2.0*self.Nc)))

            print("Cluster Chi(q,qz=pi):", sum(self.G4K[:,:,:,:,1]/(self.invT*2.0*self.Nc)))


    def symmetrizeG4_wn(self):

        if self.vertex_channel in ("PARTICLE_PARTICLE_SUPERCONDUCTING","PARTICLE_PARTICLE_UP_DOWN"):

            # G4K(-wn,-wn',qz=0) = G4K(wn,wn',qz=0)

            for iw1 in range(self.iwG40,self.NwG4):

                for iw2 in range(self.iwG40,self.NwG4):

                    tmp = 0.5 * (self.G4K[iw1,:,iw2,:,0] + self.G4K[self.NwG4-iw1-1,:,self.NwG4-iw2-1,:,0])

                    self.G4K[iw1,:,iw2,:,0] = tmp

                    self.G4K[self.NwG4-iw1-1,:,self.NwG4-iw2-1,:,0] = tmp



    def fermi(self,energy):

        beta=self.invT

        xx = beta*energy

        if xx < 0:
            return 1./(1.+exp(xx))
        else:
            return exp(-xx)/(1.+exp(-xx))



    def calcFillingBilayer(self):

        Nc = self.Nc; beta=self.invT

        Gkw0   = self.cG[:,0:Nc]

        G0kw0  = self.cG0[:,0:Nc]

        Gkwpi  = self.cG[:,Nc:2*Nc]

        G0kwpi = self.cG0[:,Nc:2*Nc]


        a0  = sum(Gkw0-G0kw0)/(Nc*beta)

        api = sum(Gkwpi-G0kwpi)/(Nc*beta)


        fk0 = 0.0; fkpi = 0.0

        for iK,K in enumerate(self.Kset):

            for k in self.kPatch:

                kx = K[0]+k[0]; ky = K[1]+k[1]

                ek0  = self.dispersion(kx,ky,0.0)

                ekpi = self.dispersion(kx,ky,np.pi)

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



    def symmetrizeG4(self):

        if self.iwm==0 & self.symmetrize_G4:

            self.apply_symmetry_in_wn()

            self.apply_transpose_symmetry()

            # if self.phSymmetry: self.apply_ph_symmetry_pp(self.G4)


    def calcChi0Cluster(self):

        print ("Now calculating chi0 on cluster")

        self.chic0  = zeros((self.NwG4,2*self.Nc,2),dtype='complex') # last 2 for Qz=0 and pi

        self.chic0M = zeros((self.NwG4*2*self.Nc,self.NwG4*2*self.Nc,2),dtype='complex') # last 2 for Qz=0 and pi

        Nc=self.Nc; NwG4=self.NwG4; NwG=self.NwG; nOrb = self.nOrb


        if self.vertex_channel in ("PARTICLE_PARTICLE_UP_DOWN","PARTICLE_PARTICLE_SUPERCONDUCTING"):

            for iqz in range(0,2):

                for iw in range(0,NwG4):

                    for ik in range(2*Nc):

                        iw1  = int(iw - NwG4/2 + NwG/2)

                        if iqz==0: iQ = self.iQ0

                        else:      iQ = self.iQPi

                        mikPlusiQ = int(self.iKSum[self.iKDiff[0,ik],iQ]) # -k+Q

                        minusiwPlusiwm = int(min(max(NwG-iw1-1 + self.iwm,0),NwG-1)) # -iwn + iwm

                        c1 = self.cGK[iw1,ik] * self.cGK[minusiwPlusiwm,mikPlusiQ]

                        # print(ik,mikPlusiQ,iw1,minusiwPlusiwm)

                        self.chic0[iw,ik,iqz] = c1

            self.chic0M[:,:,0] = np.diag(self.chic0[:,:,0].reshape(self.nt))

            self.chic0M[:,:,1] = np.diag(self.chic0[:,:,1].reshape(self.nt))

        else:

            for iqz in range(0,2):

                for iw in range(0,NwG4):

                    for ik in range(2*Nc):

                        iw1  = int(iw - NwG4/2 + NwG/2)

                        if iqz==0: iQ = self.iQ0

                        else:      iQ = self.iQPi

                        ikPlusQ = int(self.iKSum[iQ,ik]) # -k

                        iwPlusiwm = int(min(max(iw1 + self.iwm,0),NwG-1))  # iwn+iwm

                        c1 = - self.cGK[iw1,ik] * self.cGK[iwPlusiwm,ikPlusQ]

                        self.chic0[iw,ik,iqz] = c1

            self.chic0M[:,:,0] = np.diag(self.chic0[:,:,0].reshape(self.nt))

            self.chic0M[:,:,1] = np.diag(self.chic0[:,:,1].reshape(self.nt))


        if self.vertex_channel=="PARTICLE_HOLE_MAGNETIC":

            chi00 = sum(self.chic0M[:,:,0]/(self.invT*2.0*self.Nc)); chiRPA0 = chi00/(1.-self.U*chi00)

            chi0Pi = sum(self.chic0M[:,:,1]/(self.invT*2.0*self.Nc)); chiRPAPi = chi0Pi/(1.-self.U*chi0Pi)

            print("Cluster Chi0(q,qz=0) & RPA Chi:", chi00,chiRPA0)

            print("Cluster Chi0(q,qz=pi) & RPA Chi:", chi0Pi,chiRPAPi)


    def calcGammaIrr(self):

        # Calculate the irr. GammaIrr

        Nc=2*self.Nc; NwG4=self.NwG4; NwG=self.NwG; nt = self.nt

        self.GammaM = zeros((nt,nt,2),dtype='complex')

        for iqz in range(0,2):

            G4M = linalg.inv(self.G4M[:,:,iqz])

            chic0M = linalg.inv(self.chic0M[:,:,iqz])

            self.GammaM[:,:,iqz] = (chic0M - G4M) * float(Nc)*self.invT

        self.Gamma = self.GammaM.reshape(NwG4,Nc,NwG4,Nc,2)

        if (self.evenFreqOnly==True) & (self.vertex_channel in ("PARTICLE_PARTICLE_SUPERCONDUCTING","PARTICLE_PARTICLE_UP_DOWN","PARTICLE_PARTICLE_SINGLET")):

            print("Solutions restricted to even frequency!!!")

            self.Gamma = 0.5*(self.Gamma+self.Gamma[:,:,::-1,:,:])

            self.GammaM = self.Gamma.reshape(nt,nt,2)


    def buildKernelMatrix(self):

        # Build kernel matrix Gamma*chi0

        self.pm = zeros((self.nt,self.nt,2),dtype='complex')

        for iqz in range(0,2):

            self.pm[:,:,iqz] = np.dot(self.GammaM[:,:,iqz], self.chi0M[:,:,iqz])/(self.invT*float(2.0*self.Nc))


        if self.vertex_channel in ("PARTICLE_PARTICLE_SUPERCONDUCTING","PARTICLE_PARTICLE_UP_DOWN","PARTICLE_PARTICLE_SINGLET"):

            self.pm2 = zeros((self.nt,self.nt,2),dtype='complex')
            
            for iqz in range(0,2):

                self.pm2[:,:,iqz] = np.dot(sqrt(real(self.chi0M[:,:,iqz])),np.dot(real(self.GammaM[:,:,iqz]), sqrt(real(self.chi0M[:,:,iqz])))) / (self.invT*float(2*self.Nc))


    def calcKernelEigenValues(self):
        # self.nt = 2*self.Nc*self.NwG4

        nt = self.nt; Nc = 2*self.Nc; NwG4=self.NwG4; nOrb = self.nOrb

        self.lambdas = zeros((16,2),dtype='complex')

        self.evecs = zeros((self.nt,16,2),dtype='complex')

        for iqz in range(0,2):

            w,v = linalg.eig(self.pm[:,:,iqz])

            wt = abs(w-1)

            ilead = argsort(wt)[0:16]

            self.lambdas[:,iqz] = w[ilead]

            self.evecs[:,:,iqz] = v[:,ilead]

            print("compare: ", v[:,ilead].shape, self.evecs.shape)

        print ("Leading eigenvalues of lattice Bethe-salpeter equation for qz=0", self.Tval, self.lambdas[:,0])

        print ("Leading eigenvalues of lattice Bethe-salpeter equation for qz=pi", self.Tval, self.lambdas[:,1])

        self.evecs = self.evecs.reshape(NwG4,Nc,16,2)
        
        
        # write data:
        if self.write_data_file:
            fname = 'leading_Evec_vs_Kiwn_T'+str(self.Tval)+'.txt'
            if os.path.isfile(fname):
                os.remove(fname)
        
            # (kx,ky,kz,wn, evec at qz=0, evec at qz=pi)
            # print first two leading Evec to include cases with two degenerate Eval for d+is wave
            for ilam in range(0,2):
                for iNc in range(Nc):
                    kx = self.K[iNc,0]; ky = self.K[iNc,1]; kz = self.K[iNc,2]
                    kxs = np.full((NwG4, 1), kx)
                    kys = np.full((NwG4, 1), ky)
                    kzs = np.full((NwG4, 1), kz)

                    self.write_data_6cols(fname, kxs,kys,kzs, self.wnSet, self.evecs[:,iNc,ilam,0], self.evecs[:,iNc,ilam,1])



        if self.vertex_channel in ("PARTICLE_PARTICLE_SUPERCONDUCTING","PARTICLE_PARTICLE_UP_DOWN"):
            
            self.lambdas2 = zeros((16,2),dtype='complex')

            self.evecs2 = zeros((self.nt,16,2),dtype='complex')

            for iqz in range(0,2):
                
                w2,v2 = linalg.eigh(self.pm2[:,:,iqz])

                wt2 = abs(w2-1)

                ilead2 = argsort(wt2)[0:16]

                self.lambdas2[:,iqz] = w2[ilead2]

                self.evecs2[:,:,iqz] = v2[:,ilead2]

            print ("Leading eigenvalues of symmetrized Bethe-salpeter equation for qz=0", self.Tval, self.lambdas2[:,0])
            
            print ("Leading eigenvalues of symmetrized Bethe-salpeter equation for qz=pi", self.Tval, self.lambdas2[:,1])
            
            self.evecs2 = self.evecs2.reshape(NwG4,Nc,16,2)


            # write data:
            if self.write_data_file:
                fname = 'leading_Evec_sym_vs_Kiwn_T'+str(self.Tval)+'.txt'
                if os.path.isfile(fname):
                    os.remove(fname)
            
                # (kx,ky,kz,wn, evec at qz=0, evec at qz=pi)
                for iNc in range(Nc):
                    kx = self.K[iNc,0]; ky = self.K[iNc,1]; kz = self.K[iNc,2]
                    kxs = np.full((NwG4, 1), kx)
                    kys = np.full((NwG4, 1), ky)
                    kzs = np.full((NwG4, 1), kz)
                    self.write_data_6cols(fname, kxs,kys,kzs, self.wnSet, self.evecs2[:,iNc,0,0], self.evecs2[:,iNc,0,1])
                    
                    
            #Now find d-wave eigenvalue (based on g(k) in band kz=0)

            gk = cos(self.Kvecs[:,0]) - cos(self.Kvecs[:,1]) # dwave form factor

            self.found_d=False

            for ia in range(0,16):

                r1 = dot(gk,self.evecs2[int(self.NwG4/2),0:int(Nc/2),ia,0])

                if abs(r1) >= 0.2: 

                    self.lambdad = self.lambdas2[ia,0]

                    self.ind_d = ia

                    self.found_d=True

                    break

            if self.found_d: print("d-wave eigenvalue for qz=0", self.Tval, real(self.lambdad))

            #self.evecs2 = self.evecs2.reshape(NwG4,16,2,nt)


            #Now find d-wave eigenvalue (based on g(k) in band kz=pi)

            self.found_d=False

            for ia in range(0,16):

                r1 = dot(gk,self.evecs2[int(self.NwG4/2),0:int(Nc/2),ia,1])

                if abs(r1) >= 0.2:

                    self.lambdad = self.lambdas2[ia,1]

                    self.ind_d = ia

                    self.found_d=True

                    break

            if self.found_d: print("d-wave eigenvalue for qz=pi", self.Tval, real(self.lambdad))

                
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

        # t = self.t

        mu = self.mu



        # Now coarse-grain G*G to build chi0(K) = Nc/N sum_k Gc(K+k')Gc(-K-k')

        nOrb = self.nOrb; nw = wnSet.shape[0]; nk=Kset.shape[0]

        self.chi0  = np.zeros((nw,2*nk,nw,2*nk,2),dtype='complex')

        self.cG    = np.zeros((nw,2*nk),dtype='complex')

        self.cG0   = np.zeros((nw,2*nk),dtype='complex')

        for iwn,wn in enumerate(wnSet): # reduced tp frequencies !!

            # print "iwn = ",iwn

            iwG = int(iwn - self.iwG40 + self.iwG0)

            for iK,K in enumerate(self.K):

                for iqz in range(0,2):

                    c1 = 0.0

                    if iqz==0: cG = 0.0; cG0 = 0.0

                    for k in kPatch:

                        kx = K[0]+k[0]; ky = K[1]+k[1]; kz = K[2]

                        ek = self.dispersion(kx,ky,kz)

                        if iqz==0: G0 = 1./(1j*wn+self.mu-self.U*(self.dens/4.-0.5) - ek)



                        Qx = self.qchannel[0]; Qy = self.qchannel[1]; Qz = float(iqz)*np.pi

                        if iqz==0: iQ = self.iQ0

                        else: iQ = self.iQPi



                        if self.vertex_channel in ("PARTICLE_PARTICLE_UP_DOWN","PARTICLE_PARTICLE_SUPERCONDUCTING"):

                            emkpq = self.dispersion(-kx+Qx, -ky+Qy, -kz+Qz)

                            miKPQ = int(self.iKSum[self.iKDiff[0,iK],iQ])

                            minusiwPlusiwm = min(max(NwG-iwG-1 + self.iwm,0),NwG-1) # -iwn + iwm



                            G1 = 1./(1j*wn+self.mu-ek-self.sigmaK[iwG,iK])

                            G2 = 1./(-1j*wn+self.mu-emkpq-self.sigmaK[minusiwPlusiwm,miKPQ])

                        else:

                            ekpq = self.dispersion(kx+Qx, ky+Qy, kz+Qz)

                            iKPQ = int(self.iKSum[iK,iQ])

                            iwPlusiwm = int(min(max(iwG + self.iwm,0),NwG-1))  # iwn+i



                            G1 =  1./(1j*wn+self.mu-ek-self.sigmaK[iwG,iK])

                            G2 = -1./(1j*wn+self.mu-ekpq-self.sigmaK[iwPlusiwm,iKPQ])


                        if iqz==0: cG  += G1; cG0 += G0

                        c1  += G1*G2

                        
                    self.chi0[iwn,iK,iwn,iK,iqz] = c1/kPatch.shape[0]

                self.cG[iwn,iK]  = cG/kPatch.shape[0]

                self.cG0[iwn,iK] = cG0/kPatch.shape[0]

        self.chi0M = self.chi0.reshape(self.nt,self.nt,2)



        if self.vertex_channel in ("PARTICLE_HOLE_MAGNETIC","PARTICLE_HOLE_TRANSVERSE"):

            print("Lattice Chi0(q,qz=0) :", sum(self.chi0M[:,:,0]/(self.invT*2.0*self.Nc)))

            print("Lattice Chi0(q,qz=pi):", sum(self.chi0M[:,:,1]/(self.invT*2.0*self.Nc)))



    def calcSus(self):

        NwG4=self.NwG4; Nc=2*self.Nc; nOrb=self.nOrb

        gkd = cos(self.K[:,0]) - cos(self.K[:,1])

        gspm = cos(self.K[:,2])


        for iqz in range(0,2):

            # calculate lattice G4 Green's function

            G2L = linalg.inv(linalg.inv(self.chi0M[:,:,iqz]) - self.GammaM[:,:,iqz]/(float(Nc)*self.invT))

            G2 = G2L.reshape(NwG4,Nc,NwG4,Nc)


            # for spm susceptibility

            G2t0 = sum(sum(G2,axis=0),axis=1)/self.invT



            if self.vertex_channel in ("PARTICLE_PARTICLE_UP_DOWN","PARTICLE_PARTICLE_SUPERCONDUCTING"):
                Ps = dot(dot(gspm, G2t0), gspm) / float(Nc)

                Pd = dot(dot(gkd , G2t0), gkd ) / float(Nc)


                if iqz==0: self.Ps = Ps

                print("spm   susceptibility for Qz = ",np.pi*iqz,": ", self.Tval, real(Ps))

                print("dwave susceptibility for Qz = ",np.pi*iqz,": ", self.Tval, real(Pd))



            elif self.vertex_channel in ("PARTICLE_HOLE_MAGNETIC","PARTICLE_HOLE_TRANSVERSE"):

                self.ChiQ = sum(G2t0) / float(Nc)

                print("Chis for for Qz = ",round(np.pi*iqz,3),": ",self.ChiQ)

                

    def calcVs(self):

        self.Vs=zeros(int(self.NwG4/2))

        for iw in range(int(self.NwG4/2),int(self.NwG4)):

            self.Vs[iw-int(self.NwG4/2)] = real(self.projectOnSpm(self.K,self.Gamma[iw,:,int(self.NwG4/2),:,0])) #/(self.Nc*self.invT)



    def calcVandP0(self,wCutOff=0.5):

        iwG40 = int(self.NwG4/2)

        nwc = max(int(0.5*wCutOff*self.invT/np.pi - 0.5),0)

        print("cutoff number: ",nwc)

        # GKKP = sum(sum(self.Gamma[iwG40-nwc:iwG40+nwc,:,iwG40-nwc:iwG40+nwc,:,0],axis=0),axis=1) / (2*nwc)

        # Vs1 = real(self.projectOnSpm(self.K,GKKP))

        # Chi0K = sum(sum(self.chi0[iwG40-nwc:iwG40+nwc,:,iwG40-nwc:iwG40+nwc,:,0],axis=0),axis=1) / (2*nwc)

        # Ps01 = real(self.projectOnSpm(self.K,Chi0K)) / (self.invT * 2*self.Nc)

        # Ps01Kz0  = real(sum(Chi0K[0:self.Nc,0:self.Nc])) / (self.invT * self.Nc)

        # Ps01Kzpi = real(sum(Chi0K[self.Nc:2*self.Nc,self.Nc:2*self.Nc])) / (self.invT * self.Nc)



        # print("Vs, Ps0, Vs*Ps0 from Phi(K,wn)~cosKz*(H(wn+wc) - H(wn-wc)):", Vs1,Ps01,Vs1*Ps01)

        # print("Ps01Kz0,Ps01Kzpi:", Ps01Kz0,Ps01Kzpi)



        # Vs2  = real(self.projectOnEigenvector(self.GammaM[:,:,0],self.evecs[:,:,0,0].reshape(self.NwG4*2*self.Nc)))

        # Ps02 = real(self.projectOnEigenvector(self.chi0M[:,:,0],self.evecs[:,:,0,0].reshape(self.NwG4*2*self.Nc)))/(self.invT*2*self.Nc)



        # print("Vs, Ps0, Vs*Ps0 from projection on leading eigenvector:", Vs2,Ps02,Vs2*Ps02)



        # GKKP = self.Gamma[iwG40,:,iwG40,:,0]

        # Vs3 = real(self.projectOnSpm(self.K,GKKP))

        # Chi0K = self.chi0[iwG40,:,iwG40,:,0]

        # Ps03 = real(self.projectOnSpm(self.K,Chi0K)) / (self.invT * 2*self.Nc)



        # print("Vs, Ps0, Vs*Ps0 from Phi(K,wn)~cosKz and wn,wn'=piT:", Vs3,Ps03,Vs3*Ps03)



        # GKKP = sum(sum(self.Gamma[iwG40-nwc:iwG40+nwc,:,iwG40-nwc:iwG40+nwc,:,0],axis=0),axis=1) / (2*nwc)

        # Vs4  = real(self.projectOnEigenvector(GKKP,self.evecs[iwG40,:,0,0]))

        # Chi0K = sum(sum(self.chi0[iwG40-nwc:iwG40+nwc,:,iwG40-nwc:iwG40+nwc,:,0],axis=0),axis=1) / (2*nwc)

        # Ps04 = real(self.projectOnEigenvector(Chi0K,self.evecs[iwG40,:,0,0])) / (self.invT * 2 * self.Nc)



        # print("Vs, Ps0, Vs*Ps0 from proj. onto leading ev for low wn:", Vs4,Ps04,Vs4*Ps04)



        # return Vs1,Ps01,Vs2,Ps02,Vs3,Ps03

        fwn = ((np.pi*self.temp)**2+wCutOff**2)/(self.wnSet**2 + wCutOff**2)

        # fwn = zeros_like(self.wnSet); fwn[iwG40-nwc-1:iwG40+nwc+1] = 1.0

        Nc=self.Nc

        Chi0ww0 = real(sum(sum(self.chi0[:,0:Nc,:,0:Nc,0],axis=1),axis=2)/float(Nc))

        Chi0wwpi= real(sum(sum(self.chi0[:,Nc:2*Nc,:,Nc:2*Nc,0],axis=1),axis=2)/float(Nc))

        Ps01Kz0 = np.dot(fwn, np.dot(Chi0ww0 ,fwn)) / self.invT

        Ps01Kzpi= np.dot(fwn, np.dot(Chi0wwpi,fwn)) / self.invT

        print("Ps01Kz0,Ps01Kzpi:", Ps01Kz0,Ps01Kzpi)



    def calcEffectiveSingleBandInteraction(self):

        # Calculate effective interaction for kz=0 band:

        # Gamma_eff(k,k') = Gamma_00(k,k') - T/Nc \sum_k" Gamma_0pi(k,k")G_pi(k")G_pi(-k")Gamma_pi0(k",k')


        NwG4=self.NwG4; Nc=self.Nc; nt = NwG4*Nc

        Gamma = -self.Gamma[:,:,:,:,0] # pp irr. vertex; has shape (NwG4,2*Nc,NwG4,2*Nc); 0..Nc has kz=0, Nc..2*Nc has kz=pi

        Gamma_00  = Gamma[:,0:Nc,:,0:Nc].reshape(nt,nt)

        Gamma_0pi = Gamma[:,0:Nc,:,Nc:2*Nc].reshape(nt,nt)

        Gamma_pi0 = Gamma[:,Nc:2*Nc,:,0:Nc].reshape(nt,nt)

        # chic0_pi  = np.diag(self.chic0[:,Nc:2*Nc,0].reshape(nt))

        chic0_pi  = self.chi0[:,Nc:2*Nc,:,Nc:2*Nc,0].reshape(nt,nt)



        # Now calculate Gamma_eff(k,k')

        # GammaEff  = zeros((nt,nt, dtype=complex))

        self.GammaEff1 = Gamma_00.reshape(NwG4,Nc,NwG4,Nc)

        self.GammaEff2 =  np.dot(chic0_pi,Gamma_pi0)

        self.GammaEff2 = -np.dot(Gamma_0pi,self.GammaEff2) / (self.invT*2*float(Nc))

        self.GammaEff2 = self.GammaEff2.reshape(NwG4,Nc,NwG4,Nc)

        self.GammaEff  = self.GammaEff1 + self.GammaEff2


        # Now calc. eigenspectrum of GammaEff


        self.pmEff = - np.dot(self.GammaEff.reshape(nt,nt),self.chi0[:,0:Nc,:,0:Nc,0].reshape(nt,nt)) / (self.invT*2*float(Nc))


        self.wEff,self.vEff = linalg.eig(self.pmEff)


        # Now calculate effective interaction including reducible diagrams in (pi,pi) channel

        lam0 = 1.0 # 1 instead of leading eigenvalue

        # lam0 = self.lambdas[0,0] 

        Gamma_pipi = Gamma[:,Nc:2*Nc,:,Nc:2*Nc].reshape(nt,nt)

        GammaEff2 = linalg.inv(lam0*np.eye(nt)+np.dot(Gamma_pipi,chic0_pi)/(self.invT*2.*float(Nc)))

        GammaEff2 = np.dot(GammaEff2,Gamma_pi0)

        self.GammaEff2Exact = -np.dot(np.dot(Gamma_0pi,chic0_pi),GammaEff2)/(self.invT*2.*float(Nc))

        self.GammaEffExact = self.GammaEff1.reshape(nt,nt) + self.GammaEff2Exact

        self.pmEffExact = - np.dot(self.GammaEffExact,self.chi0[:,0:Nc,:,0:Nc,0].reshape(nt,nt)) / (self.invT*2.*float(Nc))



        self.wEffExact,self.vEffExact = linalg.eig(self.pmEffExact)



    def plotDispersion(self,nkSeg=80):

        kx = arange(0,nkSeg)*pi/nkSeg

        kx = append(kx,ones(nkSeg)*pi)

        kx = append(kx,arange(0,nkSeg)[::-1]*pi/nkSeg)

        ky = zeros((nkSeg))

        ky = append(ky,arange(0,nkSeg)*pi/nkSeg)

        ky = append(ky,arange(0,nkSeg)[::-1]*pi/nkSeg)


        self.ek0  = self.dispersion(kx,ky,kz=0) - self.mu

        self.ekpi = self.dispersion(kx,ky,kz=pi) - self.mu


        nk = kx.shape[0]

        df = p.DataFrame([[kx[i],ky[i],i,self.ek0[i],self.ekpi[i]] for i in range(nk) ])

        df.columns = ["$k_x$","$k_y$","k","bonding","antibonding"]

        df = p.melt(df,id_vars=["$k_x$","$k_y$","k"])

        df.columns = ["$k_x$","$k_y$","k","$k_z$","$E(k)$"]


        plot = (ggplot(df,aes(x="k",y="$E(k)$",color="$k_z$")) 

            + geom_line()

            + geom_hline(yintercept=0,color='red',size=0.25)

            + geom_vline(xintercept=[nkSeg,2*nkSeg],linetype="dashed",alpha=0.5)

            + scale_x_continuous(expand=(0,0),breaks=[0,nkSeg,2*nkSeg,3*nkSeg-1],labels=["$\Gamma$","X","M","$\Gamma$"])

            # + scale_y_continuous(expand=(0,0))

            + theme(legend_title=element_blank(),legend_position='top')

        )

        print(plot)



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



    # def determine_iKPiPi(self):

    #     self.iKPiPi = 0

    #     Nc=self.Nc

    #     for iK in range(Nc):

    #         kx = abs(self.Kvecs[iK,0] - np.pi)

    #         ky = abs(self.Kvecs[iK,1] - np.pi)

    #         if kx >= 2*np.pi: kx-=2.*pi

    #         if ky >= 2*np.pi: ky-=2.*pi

    #         if kx**2+ky**2 <= 1.0e-5:

    #             self.iKPiPi = iK

    #             break





    def K_2_iK(self,Kx,Ky,Kz=0):

        delta=1.0e-4

        # First map (Kx,Ky) into [0...2pi,0...2pi] region where Kvecs are defined

        if Kx < -delta      : Kx += 2*pi

        if Ky < -delta      : Ky += 2*pi

        if Kz < -delta      : Kz += 2*pi

        if Kx > 2.*pi-delta : Kx -= 2*pi

        if Ky > 2.*pi-delta : Ky -= 2*pi

        if Kz > 2.*pi-delta : Kz -= 2*pi

        # Now find index of Kvec = (Kx,Ky)

        for iK in range(0,2*self.Nc):

            if (abs(float(self.K[iK,0]-Kx)) < delta) & (abs(float(self.K[iK,1]-Ky)) < delta) & (abs(float(self.K[iK,2]-Kz)) < delta): return iK

        print("No Kvec found!!!")



    def setupMomentumTables(self):

        # Extend tables with Kz=0 and pi

        Nc = self.Nc

        self.K = zeros((2*Nc,3))

        self.K[0:Nc,0:2]    = self.Kvecs; self.K[0:Nc,2] = 0

        self.K[Nc:2*Nc,0:2] = self.Kvecs; self.K[Nc:2*Nc,2] = np.pi

        # build tables for K+K' and K-K'

        self.iKDiff = zeros((2*self.Nc,2*self.Nc),dtype='int')

        self.iKSum  = zeros((2*self.Nc,2*self.Nc),dtype='int')

        Nc = 2*self.Nc

        for iK1 in range(Nc):

            Kx1 = self.K[iK1,0]; Ky1 = self.K[iK1,1]; Kz1 = self.K[iK1,2]

            for iK2 in range(0,Nc):

                Kx2 = self.K[iK2,0]; Ky2 = self.K[iK2,1]; Kz2 = self.K[iK2,2]

                iKS = self.K_2_iK(Kx1+Kx2,Ky1+Ky2,Kz1+Kz2)

                iKD = self.K_2_iK(Kx1-Kx2,Ky1-Ky2,Kz1-Kz2)

                self.iKDiff[iK1,iK2] = iKD

                self.iKSum[iK1,iK2]  = iKS



    def dwave(self,kx,ky):

        return cos(kx)-cos(ky)



    def projectOnDwave(self,Ks,matrix):

        gk = self.dwave(Ks[:,0], Ks[:,1])

        c1 = dot(gk, dot(matrix,gk) ) / dot(gk,gk)

        return c1



    def projectOnSpm(self,Ks,matrix):

        gk = cos(Ks[:,2])

        c1 = dot(gk, dot(matrix,gk) ) / dot(gk,gk)

        return c1



    def projectOnEigenvector(self,matrix,eigenvector):

        return dot(eigenvector.conjugate(),dot(matrix,eigenvector)) / dot(eigenvector.conjugate(),eigenvector)



    def dispersion(self,kx,ky,kz):

        # const auto val1 = -2. * t * (std::cos(k[0]) + std::cos(k[1]));

        # const auto val2 = -4. * t_prime * std::cos(k[0]) * std::cos(k[1]);

        # const auto val3 = -2. * t_pp * ( std::cos(2.*k[0]) + std::cos(2.*k[1]) );

        # const auto ek_intra = val1 + val2 + val3;



        # const auto val4 = - t_perp - 4. * t_perp_p * std::cos(k[0]) * std::cos(k[1]);

        # const auto val5 = - 2. * t_perp_pp * ( std::cos(2.*k[0]) + std::cos(2.*k[1]) );

        # const auto ek_inter = val4 + val5;

        cx = cos(kx); cy = cos(ky); cxy = cx*cy; c2 = cos(2*kx)+cos(2*ky)



        #r1 = -2.*self.t1*(cx+cy) - 4.0*self.tp1*cxy - 2.0*self.tpp*c2

        r1 = -2.*self.t1*(cx+cy) 

        r2 = - (self.tperp + 2.*self.tperpp*(cos(kx)+cos(ky)))* cos(kz)

        #r2 = cos(kz) * (-self.tperp-4.*self.tperpp*cxy-2.*self.tperppp*c2)



        return r1+r2



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

        print('axes.flat',axes.flat)

        for ax in axes.flat:

            self.plotEV(ax,Kvecs,lambdas,evecs,inr)

            inr += 1

            ax.set(adjustable='box', aspect='equal')

        fig.suptitle(title, fontsize=20)



    def plotEV(self,ax,Kvecs,lambdas,evecs,inr):

        print('inr:',inr)

        prop_cycle = rcParams['axes.prop_cycle']

        colors = prop_cycle.by_key()['color']



        Nc = evecs.shape[1]; Nw = self.evecs.shape[0]

        iw0=int(Nw/2)

        imax = argmax(evecs[iw0,:,inr])

        if (abs(evecs[iw0-1,imax,inr]-evecs[iw0,imax,inr]) <= 1.0e-5):

            freqString = "; even"

        else:

            freqString = "; odd"



        colVec = Nc*[colors[0]]

        for ic in range(Nc):

            if real(evecs[iw0,ic,inr])*real(evecs[iw0,imax,inr]) < 0.0: colVec[ic] = colors[1]

        # print "colVec=",colVec

        ax.scatter(Kvecs[:,0]/pi,Kvecs[:,1]/pi,s=abs(real(evecs[iw0,:,inr]))*2500,c=colVec)

        ax.set(aspect=1)

        ax.set_xlim(-1.0,1.2); ax.set_ylim(-1.0,1.2)

        #ax.set_title(r"$\lambda=$"+str(round(lambdas[inr],4))+freqString)

        ax.set_title(r"$\lambda=$"+str(np.round(np.real(lambdas[inr]),4))+freqString)

        # ax.get_xaxis().set_visible(False)

        # ax.get_yaxis().set_visible(False)

        ax.grid(True)

        for tic in ax.xaxis.get_major_ticks():

            # tic.tick1On = tic.tick2On = False

            tic.tick1line.set_visible(False)

            tic.tick2line.set_visible(False)

        for tic in ax.yaxis.get_major_ticks():

            # tic.tick1On = tic.tick2On = False

            tic.tick1line.set_visible(False)

            tic.tick2line.set_visible(False)



    def plotEvec(self,index):

        # Make dataframe for eigenvector

        # %%

        df = p.DataFrame([[self.wnSet[j],str("K=(") + str(round(self.K[i,0]/pi,2)) + str(",") + str(round(self.K[i,1]/pi,2)) + str(",") + str(round(self.K[i,2]/pi,2)) + str(")"),real(self.evecs2[j,i,index])] for i in range(2*self.Nc) for j in range(self.wnSet.shape[0])])

        df.columns = ["$\omega_n$","K","$\phi(K,\omega_n)$"]



        self.plotEv = (ggplot(df,aes(x="$\omega_n$",y="$\phi(K,\omega_n)$")) 

                    + geom_line() 

                    + facet_wrap("K") 

                    + ggtitle("index:"+str(index)+"; $\lambda=$"+str(round(self.lambdas2[index],3)))

        )

        print(self.plotEv)


    # def plotEvec(self,index):

    #     # Make dataframe for eigenvector

    #     # %%

    #     df = p.DataFrame([[self.wnSet[j],str("K=(") + str(round(self.Kset[i,0]/pi,2)) + str(",") + str(round(self.Kset[i,1]/pi,2)) + str(")"),real(self.evecs2[j,i,index])] for i in range(self.Nc) for j in range(self.wnSet.shape[0])])

    #     df.columns = ["$\omega_n$","K","$\phi(K,\omega_n)$"]



    #     self.plotEv = (ggplot(df,aes(x="$\omega_n$",y="$\phi(K,\omega_n)$")) 

    #                 + geom_line() 

    #                 + facet_wrap("K") 

    #                 + ggtitle("index:"+str(index)+"; $\lambda=$"+str(round(self.lambdas2[index],3)))

    #     )

    #     print(self.plotEv)


# Symmetry stuff


    def apply_symmetry_in_wn(self):

        # for G4K[w1,K1,w2,K2] = G4*(-wn,K,-wn',K')

        # apply symmetry G4(wn,wn',K,K') = G4*(-wn,-wn',K,K')

        Nc = self.G4K.shape[1]

        nwn = self.G4K.shape[0]

        for iw1 in range(nwn):

            for iw2 in range(nwn):

                for iK1 in range(Nc):

                    for iK2 in range(Nc):

                        imw1 = nwn-1-iw1

                        imw2 = nwn-1-iw2

                        tmp1 = self.G4K[iw1,iK1,iw2,iK2,0]

                        tmp2 = self.G4K[imw1,iK1,imw2,iK2,0]

                        self.G4K[iw1,iK1,iw2,iK2,0]   = 0.5*(tmp1+conj(tmp2))

                        self.G4K[imw1,iK1,imw2,iK2,0] = 0.5*(conj(tmp1)+tmp2)



    def apply_transpose_symmetry(self):

        # Apply symmetry Gamma(K,K') = Gamma(K',K)

        # G4K(w,K,w',K')

        Nc = self.G4K.shape[1]; nwn = self.G4K.shape[0]; nt =Nc*nwn

        GP = self.G4K[:,:,:,:,0].reshape(nt,nt)

        GP = 0.5*(GP + GP.transpose())

        self.G4K[:,:,:,:,0] = GP.reshape(nwn,Nc,nwn,Nc)
        
        
    ######### writting functions

    def write_data_6cols(self, fname, kx,ky,kz, xs, ys, zs):
        f = open(fname,'a',1) 
        for i in range(len(xs)):
            f.write('{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\n'.format(float(kx[i]),float(ky[i]),float(kz[i]),float(xs[i]),float(ys[i]),float(zs[i])))
            

    # def apply_ph_symmetry_pp(self,G4):

    #     # G4pp(k,wn,k',wn') = G4pp(k+Q,wn,k'+Q,wn')

    #     Nc = G4.shape[2]

    #     nwn = G4.shape[0]

    #     for iw1 in range(nwn):

    #         for iw2 in range(nwn):

    #             for iK1 in range(Nc):

    #                 iK1q = self.iKSum[iK1,self.iKPiPi]

    #                 for iK2 in range(Nc):

    #                     iK2q = self.iKSum[iK2,self.iKPiPi]

    #                     tmp1 = G4[iw1,iw2,iK1,iK2]

    #                     tmp2 = G4[iw1,iw2,iK1q,iK2q]

    #                     G4[iw1,iw2,iK1,iK2]   = 0.5*(tmp1+tmp2)

    #                     G4[iw1,iw2,iK1q,iK2q] = 0.5*(tmp1+tmp2)


    
###################################################################################
Ts = [1, 0.75, 0.5, 0.44, 0.4, 0.34, 0.3, 0.25, 0.24, 0.2, 0.17, 0.15, 0.125, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.035, 0.03, 0.025]
Ts = [0.03]
channels = ['phcharge','phmag']
channels = ['phmag']
qs = ['00','pi20','pi0','pipi2','pipi','pi2pi2']
qs = ['pipi']
Nv = [0]#,1,2,3,4,5,6,7,8]

for T_ind, T in enumerate(Ts):
    #for ch in channels:
    for v in Nv:
        for q in qs:
            #file_tp = './T='+str(Ts[T_ind])+'/dca_tp_'+ch+'_q'+q+'.hdf5'
            file_tp = './T='+str(Ts[T_ind])+'/dca_tp_mag_q'+str(q)+'.hdf5'
            fileG4 = './T='+str(Ts[T_ind])+'/dca_tp.hdf5'
            fileG  = './T='+str(Ts[T_ind])+'/dca_sp.hdf5'
            file_analysis_hdf5 = './T='+str(Ts[T_ind])+'/analysis.hdf5'

            if(os.path.exists(fileG4)):
                print ("\n =================================\n")
                print ("T =", T)
                # model='square','bilayer','Emery'
                BSE(Ts[T_ind],\
                    fileG4,\
                    fileG,\
                    write_data_file=True)
                

# Adapted from Peizhi Mai's email on May.22, 2021
# From his https://github.com/JohnstonResearchGroup/Mai_etal_3bandPairs_2021
# in fact, this program was finished before Apr. 2021

# Note:
# This program is for Mai's modification on an older version of DCA:
# https://github.com/cosdis/DCA
# which lags behind the current master version, but G4 of threeband model is correct

# Code check:
# Nov.12, 2021:
# Pass check with Maier's PRL 2006 for single band Hubbard model (Nc=24)
# for pp-SC, ph-magnetic, ph-charge channels
# All d-wave eigenvalue, mag and charge eigenvalues, cluster and lattice susceptibilities
# are consistent with Maier's BSE code solveBSE_fromG4_200618.py for single band case

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
    def __init__(self,model,Tval,fileG4,fileG,file_analysis_hdf5,draw,useG0,symmetrizeG4,phSymmetry,calcCluster,build_sym_kernel,useGamma_hdf5,nkfine,compare_with_analysishdf5,write_data_file):
        self.vertex_channels = ["PARTICLE_PARTICLE_UP_DOWN",          \
                                "PARTICLE_HOLE_CHARGE",               \
                                "PARTICLE_HOLE_MAGNETIC",             \
                                "PARTICLE_HOLE_LONGITUDINAL_UP_UP",   \
                                "PARTICLE_HOLE_LONGITUDINAL_UP_DOWN", \
                                "PARTICLE_HOLE_TRANSVERSE"]
        self.model = model
        self.Tval = Tval
        self.fileG4 = fileG4
        self.fileG = fileG
        self.file_analysis_hdf5 = file_analysis_hdf5
        self.draw = draw
        self.useG0 = useG0
        self.calcCluster = calcCluster
        #self.calcRedVertex = calcRedVertex
        self.phSymmetry = phSymmetry
        self.build_sym_kernel = build_sym_kernel
        self.useGamma_hdf5 = useGamma_hdf5
        self.compareHDF5 = compare_with_analysishdf5
        self.write_data_file = write_data_file
        
        self.readData()
        self.setupMomentumTables()
        self.iK0 = self.K_2_iK(0.0, 0.0)
        self.determine_specialK()
        print ("Index of (pi,pi): ",self.iKPiPi)
        print ("Index of (pi,0): " ,self.iKPi0)
        #self.calcPS()
        
        self.reorder_G4()
        if symmetrizeG4: self.symmetrize_G4()
        
        self.setupMomentumTables()
        self.determine_specialK()
        print ("Index of (pi,pi): ",self.iKPiPi)
        print ("Index of (pi,0): ",self.iKPi0)

        '''
        Solve BSE below:
        '''
        self.calcChi0Cluster()
        self.calcGammaIrr()
        self.calcGamma_d()
        if calcCluster == False: self.buildChi0Lattice(nkfine)
        self.buildKernelMatrix()
        self.calcKernelEigenValues()      
        self.AnalyzeEigvec()
        if self.draw: self.plotLeadingSolutions(self.Kvecs,self.lambdas,self.evecs[:,:,0,0,:],"Cu-Cu")
            

    # read basic parameters from the data and the cluster one and two particle Green's function
    def readData(self):
        f = h5py.File(self.fileG4,'r')
        self.cluster = array(f["domains"]["CLUSTER"]["REAL_SPACE"]["super-basis"]["data"])
        print("Cluster vectors:",self.cluster,'\n')
            
        # self.iwm = array(f['parameters']['vertex-channel']['w-channel'])[0] # transferred frequency in units of 2*pi*temp
        self.iwm = array(f['parameters']['four-point']['frequency-transfer'])[0] # transferred frequency in units of 2*pi*temp
        print ("Transferred frequency iwm = ",self.iwm,'\n')
        # self.qchannel = array(f['parameters']['vertex-channel']['q-channel'])
        self.qchannel = array(f['parameters']['four-point']['momentum-transfer'])
        print ("Transferred momentum q = ",self.qchannel,'\n')
        # a = array(f['parameters']['vertex-channel']['vertex-measurement-type'])[:]
        #a = array(f['parameters']['four-point']['channels']['data'])[0]
        #self.vertex_channel = ''.join(chr(i) for i in a)
        
        # cannot use self.vertex_channel = array(f['parameters']['four-point']['channels'])[0]
        # because this program works for older version of DCA code, see top:
        # So manually set vertex_channel
        # self.vertex_channel = 'PARTICLE_PARTICLE_UP_DOWN'

        #print "Vertex channel = ",self.vertex_channel
        self.invT = array(f['parameters']['physics']['beta'])[0]
        print ("Inverse temperature = ",self.invT,'\n')
        self.temp = 1.0/self.invT
        assert(abs(self.temp-self.Tval)<1.e-4)
        
        # readin model specific parameters
        self.readin_model(self.model,f)
        
        self.fill = array(f['parameters']['physics']['density'])[0]
        print ("desired filling = ",self.fill,'\n')
        self.dens = array(f['DCA-loop-functions']['density']['data'])
        print ("actual filling:",self.dens,'\n')
        self.nk = array(f['DCA-loop-functions']['n_k']['data'])
        self.sign = array(f['DCA-loop-functions']['sign']['data'])
        print ("sign:",self.sign,'\n')
        self.orbital=array(f['DCA-loop-functions']['orbital-occupancies']['data'])
        print ("orbital.shape:",self.orbital.shape,'\n')
        # ('orbital.shape:', (1, 2, 3))
        
        if self.model=='square':
            print ("orbital occupancy:",self.orbital[0],'\n')
            print ("filling =", self.orbital[0,0,0]+self.orbital[0,1,0],'\n')
        elif self.model=='bilayer':
            print ("orbital occupancy:",self.orbital[self.orbital.shape[0]-1],'\n')
            print ("Layer1 filling =", self.orbital[self.orbital.shape[0]-1,0,0]+self.orbital[self.orbital.shape[0]-1,1,0],'\n')
            print ("Layer2 filling =", self.orbital[self.orbital.shape[0]-1,0,1]+self.orbital[self.orbital.shape[0]-1,1,1],'\n')
        elif self.model=='trilayer':
            print("orbital occupancy:",self.orbital[self.orbital.shape[0]-1],'\n')
            print("Layer1 filling =", self.orbital[self.orbital.shape[0]-1,0,0]+self.orbital[self.orbital.shape[0]-1,1,0],'\n')
            print("Layer2 filling =", self.orbital[self.orbital.shape[0]-1,0,1]+self.orbital[self.orbital.shape[0]-1,1,1],'\n')
            print("Layer3 filling =", self.orbital[self.orbital.shape[0]-1,0,2]+self.orbital[self.orbital.shape[0]-1,1,2],'\n')
        
        self.sigmaarray=array(f['DCA-loop-functions']['L2_Sigma_difference']['data'])
        print ("L2_Sigma_difference =", self.sigmaarray,'\n')
        # Now read the 4-point Green's function
        # G4Re  = array(f['functions']['G4_k_k_w_w']['data'])[:,:,:,:,:,:,:,:,0]
        # G4Im  = array(f['functions']['G4_k_k_w_w']['data'])[:,:,:,:,:,:,:,:,1]
        
        for ver in self.vertex_channels:
            if 'G4_'+ver in f['functions'].keys():
                self.vertex_channel = ver
                print ("Vertex channel = ",self.vertex_channel,'\n')
                GG = array(f['functions']['G4_'+self.vertex_channel]['data'])
                print ("original G4.shape=", GG.shape,'\n')
        
        G4Re  = array(f['functions']['G4_'+self.vertex_channel]['data'])[0,0,:,:,:,:,:,:,:,:,0]
        G4Im  = array(f['functions']['G4_'+self.vertex_channel]['data'])[0,0,:,:,:,:,:,:,:,:,1]
        #G4Re  = array(f['functions']['G4']['data'])[0,0,:,:,:,:,:,:,:,:,0]
        #G4Im  = array(f['functions']['G4']['data'])[0,0,:,:,:,:,:,:,:,:,1]
        self.G4 = G4Re+1j*G4Im
        #print ("Extracted G4.shape=", self.G4.shape ,'\n')
        
        GG = array(f['functions']['cluster_greens_function_G_k_w']['data'])
        #print ("original G.shape=", GG.shape,'\n')
        
        # Now read the cluster Green's function, only need spin up component
        GRe = array(f['functions']['cluster_greens_function_G_k_w']['data'])[:,:,0,:,0,:,0]
        GIm = array(f['functions']['cluster_greens_function_G_k_w']['data'])[:,:,0,:,0,:,1]
        #GRe = array(f['functions']['free_cluster_greens_function_G0_k_w']['data'])[:,:,0,:,0,:,0]
        #GIm = array(f['functions']['free_cluster_greens_function_G0_k_w']['data'])[:,:,0,:,0,:,1]
        self.Green = GRe + 1j * GIm
        #print ("Extracted G.shape=", self.Green.shape,'\n')
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
        #print ("sigma.shape=", s.shape,'\n')
        self.sigmaoriginal = s[:,:,0,:,0,:,0] + 1j *s[:,:,0,:,0,:,1]
        #print "Im sigma=",s[127:138,1,0,0,0,0,1]
        
        # Now load frequency data
        self.wn = np.array(f['domains']['frequency-domain']['elements'])
        self.wnSet = np.array(f['domains']['vertex-frequency-domain (COMPACT)']['elements'])
        print ('G4\'s iwn = ', self.wnSet[24],'\n')

        # Now read the K-vectors
        self.Kvecs = array(f['domains']['CLUSTER']['MOMENTUM_SPACE']['elements']['data'])
        print ("K-vectors: ",self.Kvecs,'\n')

        # Now read other Hubbard parameters
        self.mu = np.array(f['DCA-loop-functions']['chemical-potential']['data'])[0]
        self.nOrb = self.Green.shape[2]
        self.NwG4 = self.G4.shape[0]
        self.Nc  = self.Green.shape[1]
        self.NwG = self.Green.shape[0]
        self.NtG = self.Greenkt.shape[0]
        self.nt = self.Nc*self.NwG4*self.nOrb*self.nOrb

        print('nw:',self.wnSet.shape[0])
        print ("NwG4: ",self.NwG4)
        print ("NtG: ",self.NtG)
        print ("NwG : ",self.NwG)
        print ("Nc  : ",self.Nc)
        print ("nOrb: ",self.nOrb)
        #print ("G4shape0 = ", self.G4.shape[0], ", G4shape1 = ", self.G4.shape[1], ", G4shape2 = ", self.G4.shape[2], ", G4shape3 = ", self.G4.shape[3], ", G4shape4 = ", self.G4.shape[4], ", G4shape5 = ", self.G4.shape[5], ", G4shape6 = ", self.G4.shape[6], ", G4shape7 = ", self.G4.shape[7])
        self.NwTP = 2*np.array(f['parameters']['domains']['imaginary-frequency']['four-point-fermionic-frequencies'])[0]
        self.iQ = self.K_2_iK(self.qchannel[0],self.qchannel[1])
        #print ("Index of transferred momentum: ", self.iQ,'\n')
        #self.ddwave = array(f['CT-AUX-SOLVER-functions']['dwave-pp-correlator']['data'])
        #print('shape0 of ddwave=',self.ddwave.shape[0])
        #print('shape1 of ddwave=',self.ddwave.shape[1])
        #print('Cu-Cu ddwave=',self.ddwave[:,0])
        #print('Ox-Ox ddwave=',self.ddwave[:,1])
        #print('Oy-Oy ddwave=',self.ddwave[:,2])
        
        # indices of G4 and G at iwn=2*pi*T
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
                                                self.G4[iw1,ik1,iw2,ik2,l1,l2,l3,l4] -= 2.0 * self.Green[iw1Green,ik1,l1,l2] \
                                                                                            * self.Green[iw2Green,ik2,l4,l3] 

    def readin_model(self,model,f):
        if model=='square':
            self.U = array(f['parameters']['single-band-Hubbard-model']['U'])[0]
            print ("U = ",self.U)
            self.t = array(f['parameters']['single-band-Hubbard-model']['t'])[0]
            print ("t = ",self.t)
            self.tp = array(f['parameters']['single-band-Hubbard-model']['t-prime'])[0]
            print ("t-prime = ",self.tp)
            self.Vp = array(f['parameters']['single-band-Hubbard-model']['V-prime'])[0]
            print ("V-prime = ",self.Vp)

        if model=='square_txty':
            self.U = array(f['parameters']['single-band-Hubbard-model']['U'])[0]
            print ("U = ",self.U)
            self.tx = array(f['parameters']['single-band-Hubbard-model']['tx'])[0]
            print ("tx = ",self.tx)
            self.ty = array(f['parameters']['single-band-Hubbard-model']['ty'])[0]
            print ("ty = ",self.ty)
            self.tx_nn = array(f['parameters']['single-band-Hubbard-model']['tx_nn'])[0]
            print ("tx_nn = ",self.tx_nn)
            self.ty_nn = array(f['parameters']['single-band-Hubbard-model']['ty_nn'])[0]
            print ("ty_nn = ",self.ty_nn)
            self.tp = array(f['parameters']['single-band-Hubbard-model']['t-prime'])[0]
            print ("t-prime = ",self.tp)
            self.Vp = array(f['parameters']['single-band-Hubbard-model']['V-prime'])[0]
            print ("V-prime = ",self.Vp)

        if model == 'Eg_bilayer':  # effectively 4 layers because of its 3D lattice nature
            self.e1 = array(f['parameters']['bilayer-eg-model']['e1'])[0]
            print("e1 = ", self.e1)
            self.e2 = array(f['parameters']['bilayer-eg-model']['e2'])[0]
            print("e2 = ", self.e2)
            self.U1 = array(f['parameters']['bilayer-eg-model']['U1'])[0]
            print("U1 = ", self.U1)
            self.U2 = array(f['parameters']['bilayer-eg-model']['U2'])[0]
            print("U2 = ", self.U2)
            self.t1 = array(f['parameters']['bilayer-eg-model']['t1'])[0]
            print("t1 = ", self.t1)
            self.t2 = array(f['parameters']['bilayer-eg-model']['t2'])[0]
            print("t2 = ", self.t2)
            self.t1p = array(f['parameters']['bilayer-eg-model']['t1-prime'])[0]
            print("t1-prime = ", self.t1p)
            self.t2p = array(f['parameters']['bilayer-eg-model']['t2-prime'])[0]
            print("t2-prime = ", self.t2p)
            self.thybx = array(f['parameters']['bilayer-eg-model']['t-hyb-x'])[0]
            print("thyb-x = ", self.thybx)
            self.thyby = array(f['parameters']['bilayer-eg-model']['t-hyb-y'])[0]
            print("thyb-y = ", self.thyby)
            self.tperp1 = array(f['parameters']['bilayer-eg-model']['t-perp1'])[0]
            print("tperp1 = ", self.tperp1)
            self.tperp2 = array(f['parameters']['bilayer-eg-model']['t-perp2'])[0]
            print("tperp2 = ", self.tperp2)
            self.V = array(f['parameters']['bilayer-eg-model']['V'])[0]
            print("V = ", self.V)
            self.Vp = array(f['parameters']['bilayer-eg-model']['V-prime'])[0]
            print("V-prime = ", self.Vp)

        if model == 'dsmodel':
            self.e1 = array(f['parameters']['ds-model']['e1'])[0]
            print("e1 = ", self.e1)
            self.e2 = array(f['parameters']['ds-model']['e2'])[0]
            print("e2 = ", self.e2)
            self.U1 = array(f['parameters']['ds-model']['U1'])[0]
            print("U1 = ", self.U1)
            self.U2 = array(f['parameters']['ds-model']['U2'])[0]
            print("U2 = ", self.U2)
            self.t1 = array(f['parameters']['ds-model']['t1'])[0]
            print("t1 = ", self.t1)
            self.t2 = array(f['parameters']['ds-model']['t2'])[0]
            print("t2 = ", self.t2)
            self.t1p = array(f['parameters']['ds-model']['t1-prime'])[0]
            print("t1-prime = ", self.t1p)
            self.t2p = array(f['parameters']['ds-model']['t2-prime'])[0]
            print("t2-prime = ", self.t2p)
            self.tperp = array(f['parameters']['ds-model']['t-perp'])[0]
            print("tperp = ", self.tperp)
            self.tperp_p = array(f['parameters']['ds-model']['t-perp-prime'])[0]
            print("tperp-prime = ", self.tperp_p)
            self.V = array(f['parameters']['ds-model']['V'])[0]
            print("V = ", self.V)
            self.Vp = array(f['parameters']['ds-model']['V-prime'])[0]
            print("V-prime = ", self.Vp)

        if model == 'Emery':
            self.Udd = array(f['parameters']['threebands-Hubbard-model']['U_dd'])[0]
            print("Udd = ", self.Udd)
            self.Upp = array(f['parameters']['threebands-Hubbard-model']['U_pp'])[0]
            print("Upp = ", self.Upp)
            self.tpd = np.array(f['parameters']['threebands-Hubbard-model']['t_pd'])[0]
            print("tpd = ", self.tpd)
            self.tpp = np.array(f['parameters']['threebands-Hubbard-model']['t_pp'])[0]
            print("tpp = ", self.tpp)
            self.epd = np.array(f['parameters']['threebands-Hubbard-model']['ep_d'])[0]
            print("ep_d = ", self.epd)
            self.epp = np.array(f['parameters']['threebands-Hubbard-model']['ep_p'])[0]
            print("ep_p = ", self.epp)

        if model == 'ddpmodel':
            self.Udd = array(f['parameters']['ddp-model']['U_dd'])[0]
            print("Udd = ", self.Udd)
            self.Upp = array(f['parameters']['ddp-model']['U_pp'])[0]
            print("Upp = ", self.Upp)
            self.tpd = np.array(f['parameters']['ddp-model']['t_pd'])[0]
            print("tpd = ", self.tpd)
            self.tpdz2 = np.array(f['parameters']['ddp-model']['t_pdz2'])[0]
            print("tpdz2 = ", self.tpdz2)
            self.tpp = np.array(f['parameters']['ddp-model']['t_pp'])[0]
            print("tpp = ", self.tpp)
            self.ep_d = np.array(f['parameters']['ddp-model']['ep_d'])[0]
            print("ep_d = ", self.ep_d)
            self.edz2 = np.array(f['parameters']['ddp-model']['edz2'])[0]
            print("edz2 = ", self.edz2)
            self.ep_p = np.array(f['parameters']['ddp-model']['ep_p'])[0]
            print("ep_p = ", self.ep_p)

        if model=='bilayer':
            self.e1 = array(f['parameters']['bilayer-Hubbard-model']['e1'])[0]
            print ("e1 = ",self.e1)
            self.e2 = array(f['parameters']['bilayer-Hubbard-model']['e2'])[0]
            print ("e2 = ",self.e2)
            self.U1 = array(f['parameters']['bilayer-Hubbard-model']['U1'])[0]
            print ("U1 = ",self.U1)
            self.U2 = array(f['parameters']['bilayer-Hubbard-model']['U2'])[0]
            print ("U2 = ",self.U2)
            self.t1 = array(f['parameters']['bilayer-Hubbard-model']['t1'])[0]
            print ("t1 = ",self.t1)
            self.t2 = array(f['parameters']['bilayer-Hubbard-model']['t2'])[0]
            print ("t2 = ",self.t2)
            self.t1p = array(f['parameters']['bilayer-Hubbard-model']['t1-prime'])[0]
            print ("t1-prime = ",self.t1p)
            self.t2p = array(f['parameters']['bilayer-Hubbard-model']['t2-prime'])[0]
            print ("t2-prime = ",self.t2p)
            self.tperp = array(f['parameters']['bilayer-Hubbard-model']['t-perp'])[0]
            print ("tperp = ",self.tperp)
            self.V = array(f['parameters']['bilayer-Hubbard-model']['V'])[0]
            print ("V = ",self.V)
            self.Vp = array(f['parameters']['bilayer-Hubbard-model']['V-prime'])[0]
            print ("V-prime = ",self.Vp)

        if model=='sdsmodel':
            self.e1 = array(f['parameters']['trilayer-Hubbard-model']['e1'])[0]
            print ("e1 = ",self.e1)
            self.e2 = array(f['parameters']['trilayer-Hubbard-model']['e2'])[0]
            print ("e2 = ",self.e2)
            self.e3 = array(f['parameters']['trilayer-Hubbard-model']['e3'])[0]
            print ("e3 = ",self.e3)
            self.U1 = array(f['parameters']['trilayer-Hubbard-model']['U1'])[0]
            print ("U1 = ",self.U1)
            self.U2 = array(f['parameters']['trilayer-Hubbard-model']['U2'])[0]
            print ("U2 = ",self.U2)
            self.U3 = array(f['parameters']['trilayer-Hubbard-model']['U3'])[0]
            print ("U3 = ",self.U3)
            self.t1 = array(f['parameters']['trilayer-Hubbard-model']['t1'])[0]
            print ("t1 = ",self.t1)
            self.t2 = array(f['parameters']['trilayer-Hubbard-model']['t2'])[0]
            print ("t2 = ",self.t2)
            self.t3 = array(f['parameters']['trilayer-Hubbard-model']['t3'])[0]
            print ("t3 = ",self.t3)
            self.t1p = array(f['parameters']['trilayer-Hubbard-model']['t1-prime'])[0]
            print ("t1-prime = ",self.t1p)
            self.t2p = array(f['parameters']['trilayer-Hubbard-model']['t2-prime'])[0]
            print ("t2-prime = ",self.t2p)
            self.t3p = array(f['parameters']['trilayer-Hubbard-model']['t3-prime'])[0]
            print ("t3-prime = ",self.t3p)
            self.tperp = array(f['parameters']['trilayer-Hubbard-model']['t-perp'])[0]
            print ("tperp = ",self.tperp)
            self.tperp_p = array(f['parameters']['trilayer-Hubbard-model']['t-perp-prime'])[0]
            print ("tperp_p = ",self.tperp_p)
            self.V = array(f['parameters']['trilayer-Hubbard-model']['V'])[0]
            print ("V = ",self.V)
            self.Vp = array(f['parameters']['trilayer-Hubbard-model']['V-prime'])[0]
            print ("V-prime = ",self.Vp)

        if model=='trilayer':
            self.e1 = array(f['parameters']['trilayer-Hubbard-model']['e1'])[0]
            print ("e1 = ",self.e1)
            self.e2 = array(f['parameters']['trilayer-Hubbard-model']['e2'])[0]
            print ("e2 = ",self.e2)
            self.e3 = array(f['parameters']['trilayer-Hubbard-model']['e3'])[0]
            print ("e3 = ",self.e3)
            self.U1 = array(f['parameters']['trilayer-Hubbard-model']['U1'])[0]
            print("U1 = ",self.U1)
            self.U2 = array(f['parameters']['trilayer-Hubbard-model']['U2'])[0]
            print("U2 = ", self.U2)
            self.U3 = array(f['parameters']['trilayer-Hubbard-model']['U3'])[0]
            print("U3 = ", self.U3)
            self.tp1 = array(f['parameters']['trilayer-Hubbard-model']['t1-prime'])[0]
            print("t-prime = ",self.tp1)
            self.t1 = np.array(f['parameters']['trilayer-Hubbard-model']['t1'])[0]
            print("t = ",self.t1)
            self.tperp12 = array(f['parameters']['trilayer-Hubbard-model']['t-perp12'])[0]
            print("tperp12 = ",self.tperp12)
            self.tperpp12 = array(f['parameters']['trilayer-Hubbard-model']['t-perp-prime12'])[0]
            print("tperp_p12 = ",self.tperpp12)
            self.tperp23 = array(f['parameters']['trilayer-Hubbard-model']['t-perp23'])[0]
            print("tperp23 = ", self.tperp23)
            self.tperpp23 = array(f['parameters']['trilayer-Hubbard-model']['t-perp-prime23'])[0]
            print("tperp_p23 = ", self.tperpp23)
            self.coefp = array(f['parameters']['trilayer-Hubbard-model']['coefp'])[0]
            print("coefp = ", self.coefp)
            self.coefm = array(f['parameters']['trilayer-Hubbard-model']['coefm'])[0]
            print("coefm = ", self.coefm)

        if model=='Emery':
            self.Udd = array(f['parameters']['threebands-Hubbard-model']['U_dd'])[0]
            print ("Udd = ",self.Udd)
            self.Upp = array(f['parameters']['threebands-Hubbard-model']['U_pp'])[0]
            print ("Upp = ",self.Upp)
            self.tpd = np.array(f['parameters']['threebands-Hubbard-model']['t_pd'])[0]
            print ("tpd = ",self.tpd)
            self.tpp = np.array(f['parameters']['threebands-Hubbard-model']['t_pp'])[0]
            print ("tpp = ",self.tpp)
            self.ep_d = np.array(f['parameters']['threebands-Hubbard-model']['ep_d'])[0]
            print ("ep_d = ",self.ep_d)
            self.ep_p = np.array(f['parameters']['threebands-Hubbard-model']['ep_p'])[0]
            print ("ep_p = ",self.ep_p)
            
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
        print ("No Kvec found!!!")
 
    def reorder_G4(self):
        print ("reorder G4",'\n')
        Nc=self.Nc; NwG4=self.NwG4; NwG=self.NwG; nOrb = self.nOrb
        self.G4r=np.zeros((NwG4,Nc,nOrb,nOrb,NwG4,Nc,nOrb,nOrb),dtype='complex')
        G4sus0Qz0 = 0.0;G4sus1Qz0 = 0.0;G4sus2Qz0 = 0.0;G4susQz0 = 0.0;G4susinQz0 = 0.0
        G4susQzPi = 0.0
        for ik1 in range(self.Nc):
            for ik2 in range(self.Nc):
                for iw1 in range(self.NwG4):
                    for iw2 in range(self.NwG4):
                        for l1 in range(self.nOrb):
                            for l2 in range(self.nOrb):
                                for l3 in range(self.nOrb):
                                    for l4 in range(self.nOrb):
                                        if self.vertex_channel in ("PARTICLE_HOLE_MAGNETIC","PARTICLE_HOLE_CHARGE"):                                      
                                            c1= self.G4[iw1,ik1,iw2,ik2,l1,l2,l3,l4] 
                                            self.G4r[iw1,ik1,l1,l3,iw2,ik2,l4,l2]  = c1                                        
                                            if l1==l3==l4==l2==0:                                                            
                                                G4sus0Qz0 += self.G4r[iw1,ik1,0,0,iw2,ik2,0,0]
                                            if l1==l3==l4==l2==1:
                                                G4sus1Qz0 += self.G4r[iw1,ik1,1,1,iw2,ik2,1,1]
                                            if l1==l3==l4==l2==2:
                                                G4sus2Qz0 += self.G4r[iw1,ik1,2,2,iw2,ik2,2,2]
                                            if (l1==l3) & (l4==l2) & (l1!=l2):
                                                G4susinQz0 += c1
                                            if (l1==l3) & (l4==l2):
                                                G4susQz0 += c1
                                                #G4susQzPi += c1*exp(1j*np.pi*(l2-l3))
                                        elif self.vertex_channel=="PARTICLE_PARTICLE_UP_DOWN":                                 
                                            c1 = self.G4[iw2,ik2,iw1,ik1,l4,l3,l2,l1]
                                            self.G4r[iw1,ik1,l1,l2,iw2,ik2,l3,l4] = c1                                         
                                            if (l1!=l2) & (l3!=l4):                                                            
                                                G4susQz0 += c1  

        G4rtemp = self.G4r.copy()
        if self.model == 'bilayer' or self.model == 'trilayer':
            self.apply_symmetry_in_layer(self.G4r);print('symmetry G4r')
            self.G4M = self.G4rt.reshape(self.nt,self.nt)
        else:
            self.G4M = self.G4r.reshape(self.nt, self.nt)

        if self.vertex_channel=="PARTICLE_HOLE_MAGNETIC":
            print ("Cluster Chi_total(q,qz=0) :", G4susQz0/(self.invT*self.Nc*2.0))
            print ("Cluster Chi_OL(q,qz=0) :", (G4sus0Qz0+G4sus2Qz0)/2.0/(self.invT*self.Nc*2.0))
            print ("Cluster Chi_IL(q,qz=0) :", G4sus1Qz0/(self.invT*self.Nc*2.0))
            print ("Cluster Chi_inter(q,qz=0) :", G4susinQz0/(self.invT*self.Nc*2.0))
            #print ("Cluster Chi(q,qz=pi):", G4susQzPi/(self.invT*self.Nc*2.0))

        if self.vertex_channel=="PARTICLE_PARTICLE_UP_DOWN":
            print ("Cluster inter-orbital Chi(q=0):", G4susQz0/(self.invT*self.Nc*4.0))
            
    def symmetrize_G4(self):
        print ("symmetrize G4",'\n')
        # for iv=0; see Maier's note on symmetries of G4
        if self.iwm==0:
            print ("Imposing symmetry in wn")
            self.apply_symmetry_in_wn(self.G4r)

           # 2021.12.13:
           # Not sure why T.Maier's original code solveBSE_fromG4_multiOrbit_200622.py does not include below:
           # To get the same lambda's as in some papers, may need comment out the following tempororily
            if self.vertex_channel in ("PARTICLE_PARTICLE_SUPERCONDUCTING","PARTICLE_PARTICLE_UP_DOWN"):
                # print("G4.shape:",self.G4.shape)
                print ("Imposing transpose symmetry (for iv=0!)")
                self.apply_transpose_symmetry()  
                
                # 2022.2.21:
                # Maier's latest code solveBSE_fromG4_BiLayer_newFormat_220211.py comment this out
                # See apply_ph_symmetry_pp below for details
                if self.phSymmetry: self.apply_ph_symmetry_pp(self.G4)

        # 16A cluster [[4,2],[0,4]]
       # if (self.cluster[0,0] == 4.0 and self.cluster[0,1] == 2.0 and self.cluster[1,0] == 0.0 and self.cluster[1,1] == 4.0):
       #     import symmetrize_Nc16A; sym=symmetrize_Nc16A.symmetrize()
       #     print("symmetrizing 16A cluster")
       #     sym.apply_point_group_symmetries_Q0(self.G4)
       # elif (self.cluster[0,0] == 4 and self.cluster[0,1] == 0 and self.cluster[1,0] == 0 and self.cluster[1,1] == 4):
       #     import symmetrize_Nc4x4; sym=symmetrize_Nc4x4.symmetrize()
       #     print("symmetrizing 16B cluster")
       #     sym.apply_point_group_symmetries_Q0(self.G4)
       # elif (self.cluster[0,0] == 2 and self.cluster[0,1] == 2 and self.cluster[1,0] == -4 and self.cluster[1,1] == 2):
       #     import symmetrize_Nc12; sym=symmetrize_Nc12.symmetrize()
       #     print("symmetrizing 12A cluster")
       #     sym.apply_point_group_symmetries_Q0(self.G4)
       # elif (self.cluster[0,0] == 4 and self.cluster[0,1] == 4 and self.cluster[1,0] == 4 and self.cluster[1,1] == -4):
       #     import symmetrize_Nc32A_v2; sym=symmetrize_Nc32A_v2.symmetrize()
       #     print("symmetrizing 32A cluster")
       #     sym.apply_point_group_symmetries_Q0(self.G4)
       # elif (self.cluster[0,0] == 2 and self.cluster[0,1] == 2 and self.cluster[1,0] == -2 and self.cluster[1,1] == 2):
       #     import symmetrize_Nc8; sym=symmetrize_Nc8.symmetrize()
       #     print("symmetrizing 8-site cluster")
       #     sym.apply_point_group_symmetries_Q0(self.G4)
    
    ##############################################################
    def trans01(self,X):  ##for bilayer
        L = []
        for i in range(len(X)):
            x = str(X[i])
            l = x.replace('0','temp').replace('1','0').replace('temp','1')
            l1 = int(l)
            L.append(l1)
        return L

    def trans02(self,X):  ##for trilayer
        L = []
        for i in range(len(X)):
            x = str(X[i])
            l = x.replace('0','temp').replace('2','0').replace('temp','2')
            l1 = int(l)
            L.append(l1)
        return L

    def apply_symmetry_in_wn(self,G4r):
        # for G4[w1,K1,w2,K2]
        # apply symmetry G4(K,wn,K',wn') = G4*(K,-wn,K',-wn')
        # see 2-particle symmetries.pdf's final equation
        # similar to apply_transpose_symmetry
        # Be careful for multi-orbital case, where k=(K,iwn,b) with b index included
        Nc=self.Nc; NwG4=self.NwG4; NwG=self.NwG; nOrb = self.nOrb
        for l1 in range(nOrb):
            for l2 in range(nOrb):
                for l3 in range(nOrb):
                    for l4 in range(nOrb):
                        for iw1 in range(NwG4):
                            for iw2 in range(NwG4):
                                for iK1 in range(Nc):
                                    for iK2 in range(Nc):
                                        imw1 = NwG4-1-iw1
                                        imw2 = NwG4-1-iw2
                                        tmp1 = G4r[iw1,iK1, l1,l2,iw2,iK2, l3,l4]
                                        tmp2 = G4r[imw1,iK1,l1,l2,imw2,iK2,l3,l4]
                                        G4r[iw1,iK1, l1,l2,iw2,iK2, l3,l4] = 0.5*(tmp1+conj(tmp2))
                                        G4r[imw1,iK1,l1,l2,imw2,iK2,l3,l4] = 0.5*(conj(tmp1)+tmp2)

    def apply_symmetry_in_layer(self,G4r):  ##symmtrize the G4r
        Nc=self.Nc; NwG4=self.NwG4; NwG=self.NwG; nOrb = self.nOrb
        self.G4rt=np.zeros((NwG4,Nc,nOrb,nOrb,NwG4,Nc,nOrb,nOrb),dtype='complex')
        for l1 in range(nOrb):
            for l2 in range(nOrb):
                for l3 in range(nOrb):
                    for l4 in range(nOrb):
                        X = [l1,l2,l3,l4]
                        if self.model == 'trilayer':
                            L = self.trans02(X)
                        elif self.model == 'bilayer':
                            L = self.trans01(X)
                        #print(X,L)
                        for iw1 in range(NwG4):
                            for iw2 in range(NwG4):
                                for iK1 in range(Nc):
                                    for iK2 in range(Nc):

                                        if l1==l2 and l3==l4:                 
                                            self.G4rt[iw1,iK1,l1,l2,iw2,iK2,l3,l4] = 0.5*(G4r[iw1,iK1,X[0],X[1],iw2,iK2,X[2],X[3]]+G4r[iw1,iK1,L[0],L[1],iw2,iK2,L[2],L[3]])
                                        else:
                                            self.G4rt[iw1,iK1,l1,l2,iw2,iK2,l3,l4] = 0.5*(G4r[iw1,iK1,X[0],X[1],iw2,iK2,X[2],X[3]]+G4r[iw1,iK1,L[0],L[1],iw2,iK2,L[2],L[3]])

    def apply_symmetry_in_chic_layer(self,chic0):  ##symmetrize the cluster chi
        Nc=self.Nc; NwG4=self.NwG4; NwG=self.NwG; nOrb = self.nOrb
        self.chic0t=np.zeros((NwG4,Nc,nOrb,nOrb,NwG4,Nc,nOrb,nOrb),dtype='complex')
        for l1 in range(nOrb):
            for l2 in range(nOrb):
                for l3 in range(nOrb):
                    for l4 in range(nOrb):
                        X = [l1,l2,l3,l4]
                        if self.model == 'trilayer':
                            L = self.trans02(X)
                        elif self.model == 'bilayer':
                            L = self.trans01(X)
                            #print(X,L)
                        for iw1 in range(NwG4):
                            for iK1 in range(Nc):
                                for iK2 in range(Nc):

                                    if l1==l2 and l3==l4:
                                        self.chic0t[iw1,iK1,l1,l2,iw1,iK2,l3,l4] = 0.5*(chic0[iw1,iK1,X[0],X[1],iw1,iK2,X[2],X[3]]+chic0[iw1,iK1,L[0],L[1],iw1,iK2,L[2],L[3]])
                                    else:
                                        self.chic0t[iw1,iK1,l1,l2,iw1,iK2,l3,l4] = 0.5*(chic0[iw1,iK1,X[0],X[1],iw1,iK2,X[2],X[3]]+chic0[iw1,iK1,L[0],L[1],iw1,iK2,L[2],L[3]])

    def apply_symmetry_in_chi_layer(self,chi0): ##symmetrize the lattice chi
        Nc=self.Nc; NwG4=self.NwG4; NwG=self.NwG; nOrb = self.nOrb
        self.chi0t=np.zeros((NwG4,Nc,nOrb,nOrb,NwG4,Nc,nOrb,nOrb),dtype='complex')
        for l1 in range(nOrb):
            for l2 in range(nOrb):
                for l3 in range(nOrb):
                    for l4 in range(nOrb):
                        X = [l1,l2,l3,l4]
                        if self.model == 'trilayer':
                            L = self.trans02(X)
                        elif self.model == 'bilayer':
                            L = self.trans01(X)
                            #print(X,L)
                        for iw1 in range(NwG4):
                            for iK1 in range(Nc):
                                for iK2 in range(Nc):

                                    if l1==l2 and l3==l4:
                                        self.chi0t[iw1,iK1,l1,l2,iw1,iK2,l3,l4] = 0.5*(chi0[iw1,iK1,X[0],X[1],iw1,iK2,X[2],X[3]]+chi0[iw1,iK1,L[0],L[1],iw1,iK2,L[2],L[3]])
                                    else:
                                        self.chi0t[iw1,iK1,l1,l2,iw1,iK2,l3,l4] = 0.5*(chi0[iw1,iK1,X[0],X[1],iw1,iK2,X[2],X[3]]+chi0[iw1,iK1,L[0],L[1],iw1,iK2,L[2],L[3]])

    def apply_transpose_symmetry(self):
        # Apply symmetry Gamma(K,K') = Gamma(K',K)
        # Be careful for multi-orbital case, where k=(K,iwn,b) with b index included
        Nc=self.Nc; NwG4=self.NwG4; NwG=self.NwG; nOrb = self.nOrb
        GP = 0.5*(self.G4M + self.G4M.transpose())
        self.G4r = GP.reshape(NwG4,Nc,nOrb,nOrb,NwG4,Nc,nOrb,nOrb)
        self.G4M = GP
        
    def apply_ph_symmetry_pp(self,G4):
        # G4pp(k,wn,k',wn') = G4pp(k+Q,wn,k'+Q,wn'), with Q=(pi,pi)
        # 2022.2.21:
        # From Maier's notes on symmetries of G4
        # the simplied notation G4(K,K') = G4(K+Q,K'+Q) means that
        # At (q,iv)=0, G4(-K,K,K',-K') = G4(-K-Q,K+Q,K'+Q,-K'-Q)
        #
        # so that PARTICLE_PARTICLE_UP_DOWN:
        #
        #       -K,l1          -K',l4         -K-Q,l1       -K'-Q,l4
        #     -----<-------------<------    -----<-------------<------
        #             |     |                         |     |
        #             |  G4 |            =            |  G4 |
        #     -----<-------------<------    -----<-------------<------
        #        K,l3           K',l2          K+Q,l3        K'+Q,l2
        #
        # This is not necessarily correct for all Q
        # only when ph_symmetry is satisfied? (probably at half-filling, Q=(pi,pi)
        # is nesting wavevector so that here use Q=(pi,pi))
        
        Nc  = G4.shape[1]
        nwn = G4.shape[0]
        
        for l1 in range(self.nOrb):
            for l2 in range(self.nOrb):
                for l3 in range(self.nOrb):
                    for l4 in range(self.nOrb):
                        for iw1 in range(nwn):
                            for iw2 in range(nwn):
                                for iK1 in range(Nc):
                                    iK1q = self.iKSum[iK1,self.iKPiPi]
                                    for iK2 in range(Nc):
                                        iK2q = self.iKSum[iK2,self.iKPiPi]
                                        tmp1 = G4[iw1,iK1, iw2,iK2,l1,l2,l3,l4]
                                        tmp2 = G4[iw1,iK1q,iw2,iK2q,l1,l2,l3,l4]
                                        G4[iw1,iK1, iw2,iK2,l1,l2,l3,l4]  = 0.5*(tmp1+tmp2)
                                        G4[iw1,iK1q,iw2,iK2q,l1,l2,l3,l4] = 0.5*(tmp1+tmp2)

    ########################################################################
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
        
    ########################################################################
    def calcChi0Cluster(self):
        print ("Now calculating chi0 on cluster chic0 = Gc*Gc, chi^0_c of Eq.(22) in PRB 64, 195130(2001)",'\n')
        Nc=self.Nc; NwG4=self.NwG4; NwG=self.NwG; nOrb = self.nOrb; NtG=self.NtG; c2=0.0
        self.chic0 = zeros((NwG4,Nc,nOrb,nOrb,NwG4,Nc,nOrb,nOrb),dtype='complex')
        
        #self.chic0ktau  = zeros((self.NtG,self.Nc,self.nOrb,self.nOrb,self.NtG,self.Nc,self.nOrb,self.nOrb),dtype='complex')
        #self.chic0kiw  = zeros((self.NwG,self.Nc,self.nOrb,self.nOrb,self.NwG,self.Nc,self.nOrb,self.nOrb),dtype='complex')
        
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
                                    
                                    # change sign below if one band is Cu
                                    # Because of the special dipersion relation of d-p model
                                    # for Q=0, e_-k = -e_k
                                    if self.model=='threeband':
                                        if (l2 != l4 and l2 == 0) or (l2 != l4 and l4 == 0): 
                                            c1 = -self.Green[iw1,ik,l3,l1] * self.Green[minusiwPlusiwm,ikPlusQ,l4,l2]
                                        else:
                                            c1 = self.Green[iw1,ik,l3,l1] * self.Green[minusiwPlusiwm,ikPlusQ,l4,l2]
                                    else:
                                        c1 = self.Green[iw1,ik,l3,l1] * self.Green[minusiwPlusiwm,ikPlusQ,l4,l2] 
                              
                                    self.chic0[iw,ik,l1,l2,iw,ik,l3,l4] = c1
        else:
            G4sus0Qz0 = 0.0;G4sus1Qz0 = 0.0;G4sus2Qz0 = 0.0
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
                                    c1 = -self.Green[iw1,ik,l1,l3] * self.Green[iwPlusiwm,ikPlusQ,l4,l2]
                                    self.chic0[iw,ik,l1,l2,iw,ik,l3,l4] = c1
                                    if l1==l2==l3==l4==0:
                                        G4sus0Qz0 += self.chic0[iw,ik,0,0,iw,ik,0,0]
                                    if l1==l2==l3==l4==1:
                                        G4sus1Qz0 += self.chic0[iw,ik,1,1,iw,ik,1,1]
                                    if l1==l2==l3==l4==2:
                                        G4sus2Qz0 += self.chic0[iw,ik,2,2,iw,ik,2,2]
                                        G4susQzPi += c1*exp(1j*np.pi*(l2-l3))

        if self.model == 'bilayer' or self.model == 'trilayer':
            self.apply_symmetry_in_chic_layer(self.chic0);print('symmetry chic0')
            self.chic0M = self.chic0t.reshape(self.nt,self.nt) ##apply layer symmetry
        else:
            self.chic0M = self.chic0.reshape(self.nt, self.nt)
        #print('chic0:',self.chic0[:,1,0,0,24,1,0,0])
        #print('chic0:',self.chic0[:,1,2,2,24,1,2,2])
        
        # compare with data obtained by analysis code
        # if using Gamma from hdf5, it does not matter chi0_cluster's difference
        if self.compareHDF5 and self.useGamma_hdf5==False:
            data = h5py.File(self.file_analysis_hdf5,'r')

            datafile = data["analysis-functions"]["G_II_0_function"]["data"]
            print ('analysis-functions/G_II_0_function', datafile.shape,'\n')
            Nw = datafile.shape[0]
            Nk = datafile.shape[1]
            for iK in range(0,Nk):
                for iw in range(0,Nw):
                    difference = datafile[iw,iK,0,0,0,0,0] - real(self.chic0[iw,ik,0,0,iw,ik,0,0])
                    if abs(difference)>1.e-2:
                        print ('chi0c diffrence !')
                        print (iK, iw, datafile[iw,iK,0,0,0,0,0], real(self.chic0[iw,ik,0,0,iw,ik,0,0]))

 
        if self.vertex_channel=="PARTICLE_HOLE_MAGNETIC":
            print ("Cluster Chi0_OL(q,qz=0) :", (G4sus0Qz0+G4sus2Qz0)/2.0/(self.invT*self.Nc*2.0))
            print ("Cluster Chi0_IL(q,qz=0) :", G4sus1Qz0/(self.invT*self.Nc*2.0))
            #print ("Cluster Chi0(q,qz=pi):", G4susQzPi/(self.invT*self.Nc*2.0))
              
    def calcGammaIrr(self):
        '''
        On Page 7 of PRB 64, 195130(2001): "We see that again it is the irreducible quantity, i.e., the vertex
        function, for which cluster and lattice correspond."
        '''
        Nc=self.Nc; NwG4=self.NwG4; NwG=self.NwG; nOrb = self.nOrb
        self.Gamma = zeros((NwG4,Nc,nOrb,nOrb,NwG4,Nc,nOrb,nOrb),dtype='complex')
        
        print ("Now calculating the irr. Gamma on cluster, which is used as approximation of that on lattice",'\n')
        
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
        #print('-:',self.Gamma[:,1,0,0,23,1,0,0])
        #print('+:',self.Gamma[:,1,0,0,24,1,0,0])
        
        #Gamma1 = self.Gamma.copy()
        #for iw2 in range(NwG4):
        #    self.Gamma[:,:,:,:,iw2,:,:,:]=(Gamma1[:,:,:,:,iw2,:,:,:]+Gamma1[:,:,:,:,NwG4-iw2-1,:,:,:])/2
        #Gamma1 = self.Gamma.copy()
        #for iw1 in range(NwG4):
        #    self.Gamma[iw1,:,:,:,:,:,:,:]=(Gamma1[iw1,:,:,:,:,:,:,:]+Gamma1[NwG4-iw1-1,:,:,:,:,:,:,:])/2
            
        # compare with data obtained by analysis code
        if self.useGamma_hdf5:
            print ("Use Gamma from HDF5 generated in analysis run instead of computing it via inv(chi_0_cluster) - inv(G4) !!!",'\n')
            data = h5py.File(self.file_analysis_hdf5,'r')
            dataRe = array(data["analysis-functions"]["Gamma_lattice"]["data"])[:,:,:,:,:,:,:,:,0]
            dataIm = array(data["analysis-functions"]["Gamma_lattice"]["data"])[:,:,:,:,:,:,:,:,1]
            print ("analysis-functions/Gamma_lattice", dataRe.shape,'\n')
            Ga = dataRe+1j*dataIm
            
            for l1 in range(nOrb):
                for l2 in range(nOrb):
                    for l3 in range(nOrb):
                        for l4 in range(nOrb):
                            for iK1 in range(0,Nc):
                                for iK2 in range(0,Nc):
                                    for iw1 in range(0,NwG4):
                                        for iw2 in range(0,NwG4):
                                            self.Gamma[iw1,iK1,l1,l2,iw2,iK2,l3,l4] = Ga[iw1,iK1,l1,l2,iw2,iK2,l3,l4] 
                                            
            self.GammaM = self.Gamma.reshape(self.nt,self.nt)
            #print(self.GammaM)
            
            
    def calcGamma_d(self):
        '''
        See MJ's paper on extended Hubbard model in 2018 PRB
        '''                            
        print ("Now calculating frequency dependence of the d-wave-projected pairing interaction",'\n')
        
        Nc=self.Nc; NwG4=self.NwG4; nt = self.nt; nOrb = self.nOrb
        NwG4_2 = int(NwG4/2)

        Gamma = -self.Gamma  # neg sign is for convention of Gamma
        gkd = cos(self.Kvecs[:,0]) - cos(self.Kvecs[:,1])
        Gd = zeros((NwG4_2,nOrb))
        norm = 0.
        
        for iK1 in range(Nc):
            norm += gkd[iK1]*gkd[iK1]
        
        for io in range(nOrb):
            for iw1 in range(NwG4_2):
                for iK1 in range(Nc):
                    for iK2 in range(Nc):
                        Gd[iw1,io] += gkd[iK1] * real(Gamma[iw1+NwG4_2,iK1,io,io,NwG4_2,iK2,io,io]) * gkd[iK2]
        Gd /= norm
        
        # write data:
     #   if self.write_data_file:
     #       if self.model=='square':
     #           fname = 'Gamma_vs_iwm_T'+str(self.Tval)+'.txt'
     #           self.write_data_2cols(fname, self.wnSet[NwG4_2:NwG4]-self.wnSet[NwG4_2], Gd[:,0])
     #       elif self.model=='bilayer' or self.model=='trilayer':
     #           for io in range(nOrb):
     #               fname = 'Gamma_vs_iwm_T'+str(self.Tval)+'_orb'+str(io)+'.txt'
     #               self.write_data_2cols(fname, self.wnSet[NwG4_2:NwG4]-self.wnSet[NwG4_2], Gd[:,io])
          
    ########################################################################
    def buildChi0Lattice(self,nkfine):
        print ("Now calculating coarsed-grained chi0, Eq.(18) in PRB 64, 195130(2001)",'\n')

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
        
        # for calculation of G2inv below
        # change sign below if one band is Cu
        # Because of the special dipersion relation: e_-k = -e_k for d-p hopping

        for l2 in range(self.nOrb):
            for l4 in range(self.nOrb):
                self.sigmanegk[:,:,l2,l4] = self.sigma[:,:,l2,l4] ##not Emery

        if self.model == 'Emery':
            for l2 in range(self.nOrb):
                for l4 in range(self.nOrb):
                    if (l2 != l4 and l2 == 0) or (l2 != l4 and l4 == 0):
                        self.sigmanegk[:, :, l2, l4] = -self.sigma[:, :, l2, l4]
                    else:
                        self.sigmanegk[:, :, l2, l4] = self.sigma[:, :, l2, l4]

        # Now coarse-grain G*G to build chi0(K) = Nc/N sum_k Gc(K+k')Gc(-K-k')
        nOrb = self.nOrb; nw = wnSet.shape[0]; nk=Kset.shape[0]; 
        self.chi0  = np.zeros((nw,nk,nOrb,nOrb,nw,nk,nOrb,nOrb),dtype='complex')
        self.chi0D  = np.zeros((nw,nk,nOrb,nOrb,nw,nk,nOrb,nOrb),dtype='complex')
        self.chi0D2  = np.zeros((nw,nk,nOrb,nOrb,nw,nk,nOrb,nOrb),dtype='complex')
        self.chi0XS  = np.zeros((nw,nk,nOrb,nOrb,nw,nk,nOrb,nOrb),dtype='complex')
        self.chi0XS2  = np.zeros((nw,nk,nOrb,nOrb,nw,nk,nOrb,nOrb),dtype='complex')
        self.gkdNorm = 0.0
        #self.cG    = np.zeros((nw,nk,nOrb,nOrb),dtype='complex')
        #self.cG0   = np.zeros((nw,nk,nOrb,nOrb),dtype='complex')
        for iwn,wn in enumerate(wnSet): # reduced tp frequencies !!
           # print("iwn = ",iwn)
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

        if self.model == 'bilayer' or self.model == 'trilayer':
            self.apply_symmetry_in_chi_layer(self.chi0);print('symmetry chi0')
            self.chi0M = self.chi0t.reshape(self.nt, self.nt)  ##apply layer symmetry
        else:
            self.chic0M = self.chi0.reshape(self.nt, self.nt)
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
        
        # write chi0_lattice at K=(pi,0):
        NwG4=self.NwG4
        NwG4_2 = int(NwG4/2)

        chi0print  = np.zeros((nw,nk,nOrb,nOrb,nOrb,nOrb),dtype='complex')
        for l1 in range(nOrb):
            for l2 in range(nOrb):
                for l3 in range(nOrb):
                    for l4 in range(nOrb):
                        for iw in range(nw):
                            for iK in range(nk):
                                chi0print[iw,iK,l1,l2,l3,l4] = self.chi0[iw,iK,l1,l2,iw,iK,l3,l4]
              
      #  if self.write_data_file:
      #      for io in range(nOrb):
      #          fname = 'chi0_lattice_vs_iwn_T'+str(self.Tval)+'_orb'+str(io)+'.txt'
      #          self.write_data_4cols(fname, self.wnSet[NwG4_2:NwG4],\
      #                                chi0print[NwG4_2:NwG4,0,         io,io,io,io],\
      #                                chi0print[NwG4_2:NwG4,self.iKPi0,io,io,io,io],\
      #                                chi0print[NwG4_2:NwG4,self.iKPiPi,io,io,io,io])
                

        #print('chi0:',self.chi0[:,1,0,0,24,1,0,0])
        #print('chi0:',self.chi0[:,1,2,2,24,1,2,2])
        self.gkdNorm /= kPatch.shape[0]

        # compare with data obtained by analysis code
        if self.compareHDF5:
            data = h5py.File(self.file_analysis_hdf5,'r')

            datafile = data["analysis-functions"]["chi_0_lattice"]["data"]
            print ("analysis-functions/chi_0_lattice", datafile.shape,'\n')
            Nw = datafile.shape[0]
            Nk = datafile.shape[1]
            for iK in range(0,Nk):
                for iw in range(0,Nw):
                    difference = datafile[iw,iK,0,0,0,0,0] - real(self.chi0[iw,iK,0,0,iw,iK,0,0])/(self.invT*float(self.Nc))
                    if abs(difference)>1.e-3:
                        print ('chi_0_lattice difference !')
                        print (iK, iw, datafile[iw,iK,0,0,0,0,0],\
                                      real(self.chi0[iw,iK,0,0,iw,iK,0,0])/(self.invT*float(self.Nc)))


        #if self.vertex_channel=="PARTICLE_HOLE_MAGNETIC":
        if self.vertex_channel in ("PARTICLE_HOLE_MAGNETIC","PARTICLE_HOLE_CHARGE"):
            chi0Loc = sum(sum(sum(sum(self.chi0,axis=0),axis=0),axis=2),axis=2)
            chi00 = 0.0; chi0Pi = 0.0
            for l1 in range(0,nOrb):
                for l3 in range(0,nOrb):
                    chi00  += chi0Loc[l1,l1,l3,l3]
                    chi0Pi += chi0Loc[l1,l1,l3,l3] * exp(1j*np.pi*(l1-l3))

            print ("Lattice Chi0_OL(q,qz=0) :", (chi0Loc[0,0,0,0]+chi0Loc[2,2,2,2])/2.0/(self.invT*self.Nc*2.0))
            print ("Lattice Chi0_IL(q,qz=0) :", chi0Loc[1,1,1,1]/(self.invT*self.Nc*2.0))
            #print ("Lattice Chi0(q,qz=pi):", chi0Pi/(self.invT*self.Nc*2.0))
            
    def buildKernelMatrix(self):
        print ("Build kernel matrix GammaIrrCluster(=GammaIrrLattice) * coarse_grained chi0",'\n')
            
        Nc=self.Nc; NwG4=self.NwG4; NwG=self.NwG; nt = self.nt; nOrb = self.nOrb
        # Build kernel matrix Gamma*chi0
              
        if (self.calcCluster):
            self.chiM = self.chic0M
        else:
            self.chiM = self.chi0M
            
        self.pm = np.dot(self.GammaM, self.chiM)
            
        #self.pm *= 1.0/(self.invT*float(self.Nc)*float(self.nOrb))
        self.pm *= 1.0/(self.invT*float(self.Nc))
        
        if self.vertex_channel in ("PARTICLE_PARTICLE_SUPERCONDUCTING",\
                                   "PARTICLE_PARTICLE_UP_DOWN",\
                                   "PARTICLE_PARTICLE_SINGLET") and self.build_sym_kernel==True:
            # 1. one way to symmetrize the pairing kernel (note that abs might be wrong, but otherwise NaN error)
            self.pm2 = np.dot(sqrt(abs(real(self.chiM))),np.dot(real(self.GammaM), sqrt(abs(real(self.chiM)))))
            self.pm2 *= 1.0/(self.invT*float(self.Nc))
            
            # 2. another way to symmetrize the pairing kernel
            # see PRB 103, 144514 (2021) Eq.(8)
            wtemp,vtemp = linalg.eig(self.chiM)
            wttemp = abs(wtemp-1)
            ileadtemp = argsort(wttemp)
            self.lambdastemp = wtemp[ileadtemp]
            self.evecstemp = vtemp[:,ileadtemp]

            self.Lambdatemp = sqrt(np.diag(self.lambdastemp))
            self.chiMasqrt = np.dot(self.evecstemp,np.dot(self.Lambdatemp,linalg.inv(self.evecstemp)))

            self.pm3 =  np.dot(self.chiMasqrt,np.dot(self.GammaM, self.chiMasqrt))
            self.pm3 *= 1.0/(self.invT*float(self.Nc))

            # 3. Build symmetric kernel matrix M = 0.5*(Gamma(wn,wn')*chi0(wn')+Gamma(wn,-wn')*chi0(-wn')
            GammaTemp = self.Gamma.reshape(NwG4, Nc*nOrb*nOrb, NwG4, Nc*nOrb*nOrb)
            Gamma2 = np.zeros_like(GammaTemp)
            for iw1 in range(self.NwG4):
                for iw2 in range(self.NwG4):
                    Gamma2[iw1,:,iw2,:] = GammaTemp[iw1,:,self.NwG4-iw2-1]
            Gamma2 = Gamma2.reshape(NwG4*Nc*nOrb*nOrb, NwG4*Nc*nOrb*nOrb)
            self.pm4 = 0.5*(np.dot(self.GammaM, self.chiM) + np.dot(Gamma2,conj(self.chiM)))
            # self.pm *= 1.0/(self.invT*float(self.Nc)*float(self.nOrb))
            self.pm4 *= 1.0/(self.invT*float(self.Nc))

    def trans_basis(self,evecs):  ### transform the orbital-basis eigenvector into band basis
        #for trilayer
        Nc = self.Nc; NwG4=self.NwG4; N = self.nOrb; nt = self.nt
        targetVec = np.zeros((NwG4,Nc,N,N,nt),dtype='complex')

        if self.model == 'trilayer':
            tmat=np.zeros((3,3),dtype=np.float64)
            tmat[0,0] = 0.5;         tmat[0,1] = -0.5*sqrt(2);  tmat[0,2] = 0.5        #Bond
            tmat[1,0] = 0.5*sqrt(2); tmat[1,1] = 0;            tmat[1,2] = -0.5*sqrt(2) #Non
            tmat[2,0] = 0.5;         tmat[2,1] = 0.5*sqrt(2); tmat[2,2] = 0.5       #Anti

        elif self.model == 'bilayer':
            tmat = np.zeros((2, 2), dtype=np.float64)
            tmat[0, 0] = 1;  tmat[0, 1] = 1  # Bond
            tmat[1, 0] = 1;  tmat[1, 1] = -1  # Anti

        tmatinv = np.linalg.inv(tmat)

        for ia in range(0,1): 
            for iwn in range(NwG4):
                for iNc in range(Nc):
                     targetVec[iwn,iNc,:,:,ia] = np.matmul(tmatinv,np.matmul(evecs[iwn,iNc,:,:,ia],tmat)).astype('complex') #refer to note

        if self.write_data_file:
            fname = 'leadingspm_Evec_vs_K_T'+str(self.Tval)+'.txt'
            if os.path.isfile(fname):
                os.remove(fname)    
  
            for ilam in range(0,1):
                    for iNc in range(Nc):
                        kx = self.Kvecs[iNc,0]; ky = self.Kvecs[iNc,1]
                        kxs = np.full((1, 1), kx)
                        kys = np.full((1, 1), ky)
                        ## wirte the data of band-basis eigenvector at lowest frequency
                        if self.model == 'trilayer':
                             self.write_data_6cols(fname, kxs, kys, self.wnSet[int(NwG4/2)], targetVec[int(NwG4/2):int(NwG4/2)+1,iNc,0,0,ilam], targetVec[int(NwG4/2):int(NwG4/2)+1,iNc,1,1,ilam],targetVec[int(NwG4/2):int(NwG4/2)+1,iNc,2,2,ilam])
                        elif self.model == 'bilayer':
                             self.write_data_5cols(fname, kxs, kys, self.wnSet[int(NwG4/2)], targetVec[int(NwG4/2):int(NwG4/2)+1,iNc,0,0,ilam], targetVec[int(NwG4/2):int(NwG4/2)+1,iNc,1,1,ilam])

    def calcKernelEigenValues(self):
        nt = self.nt; Nc = self.Nc; NwG4=self.NwG4; nOrb = self.nOrb
        w,v = linalg.eig(self.pm)
        wt = abs(w-1)
        ilead = argsort(wt)
        self.lambdas = w[ilead]
        self.evecs = v[:,ilead]
        self.evecs = self.evecs.reshape(NwG4,Nc,nOrb,nOrb,nt)
        if self.model == 'bilayer' or self.model == 'trilayer':
            self.trans_basis(self.evecs)
        #print('wn:',self.wnSet)
        #print('Eigenvector:',self.evecs[:,0,0,0,0])
        #print ("Leading 16 eigenvalues of BSE (no symmetrization)",'\n')
        #for i in range(16):
        #    print (real(self.lambdas[i]))
                
        print ('\n',"Leading 16 eigenvalues of BSE (no symmetrization)",self.Tval,self.lambdas[0:16])
  
        if self.write_data_file:
            fname = 'leading_Evec_vs_K_T'+str(self.Tval)+'.txt'
            if os.path.isfile(fname):
                os.remove(fname)
                        
            for io1 in range(nOrb):
                for io2 in range(nOrb):
                    o1s = np.full((1, 1), io1)
                    o2s = np.full((1, 1), io2)
                    #print(o1s,o2s)
                    # print first two leading Evec to include cases
                    for ilam in range(0,16):
                        for iNc in range(Nc):
                            kx = self.Kvecs[iNc,0]; ky = self.Kvecs[iNc,1]
                            kxs = np.full((1, 1), kx)
                            kys = np.full((1, 1), ky)
                            #print(kxs,kys)
                            ## wirte the data of orbital-basis eigenvector at lowest frequency
                            self.write_data_6cols(fname, o1s, o2s, kxs, kys, self.wnSet[int(NwG4/2)], self.evecs[int(NwG4/2):int(NwG4/2)+1,iNc,io1,io2,ilam])           

        # compare with data obtained by analysis code
        if self.compareHDF5:
            data = h5py.File(self.file_analysis_hdf5,'r')

            datafile = data["analysis-functions"]["leading-eigenvalues"]["data"]
            print ("analysis-functions/leading-eigenvalues", datafile.shape,'\n')
            for ii in range(0,10):
                difference = datafile[ii,0] - real(self.lambdas[ii])
                if abs(difference)>1.e-2:
                    print ('Leading eigenvalue difference !')
                    print (ii, datafile[ii,0], real(self.lambdas[ii]))

    def AnalyzeEigvec(self):
        # only concerned about Cu-Cu eigenvalue by setting orb index to be 0
        Nc=self.Nc; NwG4=self.NwG4; NwG=self.NwG; nt = self.nt; nOrb = self.nOrb

        if self.vertex_channel in ("PARTICLE_PARTICLE_SUPERCONDUCTING",\
                                   "PARTICLE_PARTICLE_UP_DOWN",\
                                   "PARTICLE_PARTICLE_SINGLET"):
            print ('\n', "Analyze eigenval and eigvec (no symmetrization):",'\n')
            self.AnalyzeEigvec_execute(self.evecs, self.lambdas, 'BSE (no symmetrization)')

            if self.build_sym_kernel==True:
                print ('\n', "Analyze eigvec for BSE (sqrt(chi)*Gamma*sqrt(chi)):")
                self.AnalyzeEigvec_execute(self.evecs2, self.lambdas2, 'BSE (sqrt(chi)*Gamma*sqrt(chi))')
            
                print ('\n', "Analyze eigvec for BSE (Peizhi Mai's PRB 103, 144514 (2021) Eq.(8)):")
                self.AnalyzeEigvec_execute(self.evecs3, self.lambdas3, 'BSE (Peizhi Mai PRB 103, 144514 (2021) Eq.(8))')
            
                print ('\n', "Analyze eigvec for BSE (Maier's buildSymmetricKernelMatrix):")
                self.AnalyzeEigvec_execute(self.evecs4, self.lambdas4, 'BSE (Maier buildSymmetricKernelMatrix)')
        else:
            print ('\n', "leading eigenvalue", self.Tval, ' ', real(self.lambdas[0]))
        
    def AnalyzeEigvec_execute(self,evecs,lambdas,label):
        #Now find d-wave eigenvalue
        Nc=self.Nc; NwG4=self.NwG4; NwG=self.NwG; nt = self.nt; nOrb = self.nOrb
        gk = cos(self.Kvecs[:,0]) - cos(self.Kvecs[:,1]) # dwave form factor

        iw0=int(NwG4/2)

        for io in range(nOrb):
            print ('=================')
            print ('orb ',io)
            for inr in range(10):
                imax = argmax(self.evecs[iw0,:,io,io,inr])

                if (abs(self.evecs[iw0-1,imax,io,io,inr]-self.evecs[iw0,imax,io,io,inr]) <= 1.0e-1):
                    print (label, " Eigenval is ", real(self.lambdas[inr]), "even frequency")
                else:
                    print (label, " Eigenval is ", real(self.lambdas[inr]), "odd frequency")

                print (label, " Eigenvec(pi*T) =",self.evecs[iw0-1,imax,io,io,inr], self.evecs[iw0,imax,io,io,inr])

        # d-wave even frequency
        for io in range(nOrb):
            print ('=================')
            print ('For d-wave')
            self.found_d=False
            self.ind_d=0
            for ia in range(10):
                # first term check if Phi has d-wave in k space; 2nd term check if even frequency:
                r1 = dot(gk,evecs[iw0,:,io,io,ia]) * sum(evecs[:,self.iKPi0,io,io,ia])
                if abs(r1) >= 2.0e-1: 
                    self.lambdad = lambdas[ia]
                    self.ind_d   = ia
                    self.found_d = True
                    break
            if self.found_d: 
                print (label, " orb ",io," d-wave eigenvalue", self.Tval, ' ', real(self.lambdad))

                # write data:
                #if self.write_data_file:
                    #fname = 'Eigenvec_dwave_vs_iwn_T'+str(self.Tval)+'_orb'+str(io)+'.txt'
                    #self.write_data_3cols(fname, self.wnSet[iw0:NwG4],\
                    #                      evecs[iw0:NwG4,self.iKPi0,io,io,ia],\
                    #                      evecs[iw0:NwG4,self.iK0Pi,io,io,ia])


########################################################################
    # Below for cluster susceptibilities
########################################################################
    def calcClusterSus(self):
        '''
        For non-SC channels
        '''
        print(" ")
        print("Calculate cluster susceptibility via G4 summation:")
        Nc = self.Nc;
        NwG4 = self.NwG4;
        NwG = self.NwG;
        nOrb = self.nOrb
        chi_c = np.zeros((nOrb, nOrb), dtype='complex')

        for io1 in range(nOrb):
            for io2 in range(nOrb):
                chi_c[io1, io2] = sum(self.G4r[:, :, io1, io1, :, :, io2, io2])
                chi_c[io1, io2] /= self.Nc * self.invT

                print("cluster susceptibility for layer/orb ", io1, io2, ": T = ", self.Tval, ' ',
                      real(chi_c[io1, io2]))

        print("Total cluster susceptibility: T = ", self.Tval, ' ', sum(chi_c))

    def calcPsCluster(self):
        '''
        For PARTICLE_PARTICLE_UP_DOWN channel
        '''
        print(" ")
        print("Calculate s-wave cluster pair-field susceptibility:")
        Nc = self.Nc;
        NwG4 = self.NwG4;
        NwG = self.NwG;
        nOrb = self.nOrb

        Ps = np.zeros((nOrb, nOrb), dtype='complex')

        for io1 in range(nOrb):
            for io2 in range(nOrb):
                Ps[io1, io2] = sum(self.G4r[:, :, io1, io1, :, :, io2, io2])
                Ps[io1, io2] /= self.Nc * self.invT

                print("G4 s-wave cluster pair-field susceptibility for layer/orb ", io1, io2, \
                      ": T = ", self.Tval, ' ', real(Ps[io1, io2]))

        print("Total G4 s-wave cluster pair-field susceptibility: T = ", self.Tval, ' ', sum(Ps))

        # PS=0.0; PScc=0.0; PSoxox=0.0; PSoyoy=0.0; PScox=0.0; PScoy=0.0; PSoxoy=0.0; testG4susQz0=0.0; testG4susQz1=0.0;ep=0.0;
        # for iw in range(0,1024):
        # for iw in range(NwG):
        #        for ik in range(self.Nc):
        #            PScc += self.chic0kiw[iw,ik,0,0,iw,ik,0,0]
        #            PSoxox += self.chic0kiw[iw,ik,1,1,iw,ik,1,1]
        #            PSoyoy += self.chic0kiw[iw,ik,2,2,iw,ik,2,2]
        #            PScox += self.chic0kiw[iw,ik,0,0,iw,ik,1,1] + self.chic0kiw[iw,ik,1,1,iw,ik,0,0]
        #            PScoy += self.chic0kiw[iw,ik,0,0,iw,ik,2,2] + self.chic0kiw[iw,ik,2,2,iw,ik,0,0]
        #            PSoxoy += self.chic0kiw[iw,ik,1,1,iw,ik,2,2] + self.chic0kiw[iw,ik,2,2,iw,ik,1,1]
        # PSccerror1=0.0;PSccerror2=0.0;
        # for iw in range(-512,512):
        #    PSccerror1+=1.0/((2*iw+1)*1j*3.1415926/self.invT-4.25-0.9)/(-(2*iw+1)*1j*3.1415926/self.invT-4.25-0.9)
        # PSccerror1*=1.0/(self.invT)
        # PSccerror2=(1-2.0/(math.exp(self.invT*(4.25+0.9))+1))/2/(4.25+0.9)
        # PScc = PScc*(self.invT-ep)/(float(Nc)*self.invT*self.invT)
        # print("chi0kiw (sum over 64 w) s-wave Pairfield susceptibility for Cu-Cu is ",PScc)
        # print("chi0kiw s-wave Pairfield susceptibility error1 for Cu-Cu is ",PSccerror2)
        # print("chi0kiw s-wave Pairfield susceptibility error2 for Cu-Cu is ",PSccerror2)
        # PSoxox = PSoxox*(self.invT-ep)/(float(Nc)*self.invT*self.invT)
        # print("chi0kiw (sum over 512 w) s-wave Pairfield susceptibility for Ox-Ox is ",PSoxox)
        # PSoyoy = PSoyoy*(self.invT-ep)/(float(Nc)*self.invT*self.invT)
        # print("chi0kiw (sum over 512 w) s-wave Pairfield susceptibility for Oy-Oy is ",PSoyoy)
        # PScox = PScox*(self.invT-ep)/(float(Nc)*self.invT*self.invT)
        # print("chi0kiw (sum over 512 w) s-wave Pairfield susceptibility for Cu-Ox is ",PScox)
        # PScoy = PScoy*(self.invT-ep)/(float(Nc)*self.invT*self.invT)
        # print("chi0kiw (sum over 512 w) s-wave Pairfield susceptibility for Cu-Oy is ",PScoy)
        # PSoxoy = PSoxoy*(self.invT-ep)/(float(Nc)*self.invT*self.invT)
        # print("chi0kiw (sum over 512 w) s-wave Pairfield susceptibility for Ox-Oy is ",PSoxoy)
        # PS = PScc+PSoxox+PSoyoy+PScox+PScoy+PSoxoy
        # print("chi0kiw (sum over 512 w) s-wave Pairfield susceptibility is ",PS)

    def calcSCClusterSus(self):
        '''
        For PARTICLE_PARTICLE_UP_DOWN channel
        '''
        print(" ")
        print(" ")
        gksx = cos(self.Kvecs[:, 0]) + cos(self.Kvecs[:, 1])
        gkd = cos(self.Kvecs[:, 0]) - cos(self.Kvecs[:, 1])

        nOrb = self.nOrb
        Pspm = np.zeros((nOrb, nOrb), dtype='complex')
        Pd = np.zeros((nOrb, nOrb), dtype='complex')

        for io1 in range(nOrb):
            for io2 in range(nOrb):
                for iK1 in range(self.Nc):
                    for iK2 in range(self.Nc):
                        Pspm[io1, io2] += gksx[iK1] * sum(self.G4r[:, iK1, io1, io1, :, iK2, io2, io2]) * gksx[iK2]
                        Pd[io1, io2] += gkd[iK1] * sum(self.G4r[:, iK1, io1, io1, :, iK2, io2, io2]) * gkd[iK2]

                Pspm[io1, io2] /= self.Nc * self.invT
                Pd[io1, io2] /= self.Nc * self.invT

                print("G4 sx-wave cluster pair-field susceptibility for layer/orb ", io1, io2, \
                      ": T = ", self.Tval, ' ', real(Pspm[io1, io2]))
                print("G4 d-wave cluster pair-field susceptibility for layer/orb ", io1, io2, \
                      ": T = ", self.Tval, ' ', real(Pd[io1, io2]))

        if self.model == 'bilayer':
            G4spm = 0.25 * (self.G4r[:, :, 0, 1, :, :, 0, 1] + self.G4r[:, :, 0, 1, :, :, 1, 0] \
                            + self.G4r[:, :, 1, 0, :, :, 0, 1] + self.G4r[:, :, 1, 0, :, :, 1, 0])
            Ps = sum(G4spm) / (float(self.Nc) * self.invT)
            print("G4 spm cluster Pairfield susceptibility: ", real(Ps))

    ########################################################################
    # Below for lattice susceptibilities
    ########################################################################
    def calcReducibleLatticeVertex(self):
        # see Eq.(22-30) in PRB 64, 195130(2001), GammaRed is that bar(T2)
        pm = self.pm;
        Gamma = self.GammaM
        nt = self.nt;
        Nc = self.Nc;
        NwG4 = self.NwG4;
        nOrb = self.nOrb
        self.pminv = np.linalg.inv(np.identity(nt) - pm)
        # self.pminv = np.linalg.inv(np.identity(nt)+pm)
        self.GammaRed = dot(self.pminv, Gamma)
        self.GammaRed = self.GammaRed.reshape(NwG4, Nc, nOrb, nOrb, NwG4, Nc, nOrb, nOrb)

    def calcLatticeSCSus(self):
        '''
        For PARTICLE_PARTICLE_UP_DOWN channel
        Calculate lattice d-wave susceptibility via Eq.(22-30) in PRB 64, 195130(2001)
        # Calculate from gd*G4*gd = gd*GG*gd + gd*GG*GammaRed*GG*gd
        # GammaRed = self.GammaRed.reshape(self.NwG4*self.Nc,self.NwG4*self.Nc)
        # GammaRed is bar(T2) in Eq.(22-30) in PRB 64, 195130(2001)
        # chi0D2 and chi0D is Eq.(27-28) in PRB 64, 195130(2001)
        '''
        print("==============================================================================")
        print("Calculate lattice d-wave susceptibility with Eq.(22-30) in PRB 64,195130(2001)", '\n')
        gkd = cos(self.Kvecs[:, 0]) - cos(self.Kvecs[:, 1])
        nOrb = self.nOrb;
        Nc = self.Nc;
        NwG4 = self.NwG4;
        csum = np.zeros((nOrb, nOrb, nOrb, nOrb), dtype='complex')
        csumxs = np.zeros((nOrb, nOrb, nOrb, nOrb), dtype='complex')
        ccsum = np.zeros((nOrb, nOrb, nOrb, nOrb), dtype='complex')
        csumspm = np.zeros((nOrb, nOrb, nOrb, nOrb), dtype='complex')
        csum1 = np.zeros((nOrb, nOrb, nOrb, nOrb), dtype='complex')
        csum1xs = np.zeros((nOrb, nOrb, nOrb, nOrb), dtype='complex')
        ccsum1 = np.zeros((nOrb, nOrb, nOrb, nOrb), dtype='complex')
        csum1spm = np.zeros((nOrb, nOrb, nOrb, nOrb), dtype='complex')
        tempchi = np.zeros((NwG4, Nc), dtype='complex')

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
                                                        csum[l1, l2, l3, l4] += self.chi0D[iw1, iK1, l1, l2, iw1, iK1, l5, l6] * \
                                                                                self.GammaRed[iw1, iK1, l5, l6, iw2, iK2, l7, l8] * \
                                                                                self.chi0D[iw2, iK2, l7, l8, iw2, iK2, l3, l4]
                                                        csumxs[l1, l2, l3, l4] += self.chi0XS[iw1, iK1, l1, l2, iw1, iK1, l5, l6] * \
                                                                                  self.GammaRed[iw1, iK1, l5, l6, iw2, iK2, l7, l8] * \
                                                                                  self.chi0XS[iw2, iK2, l7, l8, iw2, iK2, l3, l4]
                                                        ccsum[l1, l2, l3, l4] += gkd[iK1] * self.chi0[iw1, iK1, l1, l2, iw1, iK1, l5, l6] * \
                                                                                 self.GammaRed[iw1, iK1, l5, l6, iw2, iK2, l7, l8] * \
                                                                                 self.chi0[iw2, iK2, l7, l8, iw2, iK2, l3, l4] * \
                                                                                 gkd[iK2]
                                                        csumspm[l1, l2, l3, l4] += self.chi0[ iw1, iK1, l1, l2, iw1, iK1, l5, l6] * \
                                                                                   self.GammaRed[iw1, iK1, l5, l6, iw2, iK2, l7, l8] * \
                                                                                   self.chi0[iw2, iK2, l7, l8, iw2, iK2, l3, l4]
        csum[:, :, :, :] /= (self.Nc * self.invT) ** 2
        csumxs[:, :, :, :] /= (self.Nc * self.invT) ** 2
        ccsum[:, :, :, :] /= (self.Nc * self.invT) ** 2
        csumspm[:, :, :, :] /= (self.Nc * self.invT) ** 2

        self.chi0D2sim = np.zeros((NwG4, Nc, nOrb, nOrb, nOrb, nOrb), dtype='complex')
        self.chi0XS2sim = np.zeros((NwG4, Nc, nOrb, nOrb, nOrb, nOrb), dtype='complex')
        self.chi0sim = np.zeros((NwG4, Nc, nOrb, nOrb, nOrb, nOrb), dtype='complex')
        for iw in range(NwG4):
            for iK in range(Nc):
                self.chi0D2sim[iw, iK, :, :, :, :] = self.chi0D2[iw, iK, :, :, iw, iK, :, :]
                self.chi0XS2sim[iw, iK, :, :, :, :] = self.chi0XS2[iw, iK, :, :, iw, iK, :, :]
                self.chi0sim[iw, iK, :, :, :, :] = self.chi0[iw, iK, :, :, iw, iK, :, :]
        for l1 in range(nOrb):
            for l2 in range(nOrb):
                for l3 in range(nOrb):
                    for l4 in range(nOrb):
                        csum1[l1, l2, l3, l4] = sum(self.chi0D2sim[:, :, l1, l2, l3, l4]) / (self.Nc * self.invT)
                        csum1xs[l1, l2, l3, l4] = sum(self.chi0XS2sim[:, :, l1, l2, l3, l4]) / (self.Nc * self.invT)
                        ccsum1[l1, l2, l3, l4] = sum(np.dot(self.chi0sim[:, :, l1, l2, l3, l4], gkd ** 2)) / (self.Nc * self.invT)
                        csum1spm[l1, l2, l3, l4] = sum(self.chi0sim[:, :, l1, l2, l3, l4]) / (self.Nc * self.invT)

        csum[:, :, :, :] += csum1[:, :, :, :]
        csumxs[:, :, :, :] += csum1xs[:, :, :, :]
        ccsum[:, :, :, :] += ccsum1[:, :, :, :]
        csumspm[:, :, :, :] += csum1spm[:, :, :, :]

        self.Pd = real(csum)
        self.csum = csum
        self.ccsum = ccsum
        self.Pxs = real(csumxs)
        self.Pdgkc = real(ccsum)

        if self.model == 'bilayer':
            Pspm = 0.25 * (csum[0, 1, 0, 1] + csum[0, 1, 1, 0] + csum[1, 0, 0, 1] + csum[1, 0, 1, 0])
            print("spm SC susceptibility: ", self.Tval, ' ', real(Pspm))

        # csum3 = sum(real(self.chi0D2[abs(self.wnSet) <= 2.*4*self.t**2/self.U,:]))/(self.Nc*self.invT)
        print("Calculations from GammaRed:", '\n')

        print('=================')
        for io in range(nOrb):
            print('=================')
            print("orb ", io, " d-wave  SC susceptibility: ", self.Tval, ' ', real(csum[io, io, io, io]))
            print("orb ", io, " xs-wave SC susceptibility: ", self.Tval, ' ', real(csumxs[io, io, io, io]))
            print("")
            print("orb ", io, " bare d-wave SC susceptibility:  ", self.Tval, ' ', real(csum1[io, io, io, io]))
            print("orb ", io, " bare xs-wave SC susceptibility: ", self.Tval, ' ', real(csum1xs[io, io, io, io]))
            # print ("bare d-wave SC susceptibility with cutoff wc = J: ",csum3)
            print("orb ", io, " bare d-wave SC lattice (with cluster form-factors) susceptibility: ", \
                  self.Tval, ' ', real(ccsum1[io, io, io, io]))
            print("orb ", io, " d-wave SC lattice (with cluster form-factors) susceptibility: ", \
                  self.Tval, ' ', real(ccsum[io, io, io, io]))
        print("")

    def calcLatticeSus(self):
        '''
        For non-PARTICLE_PARTICLE_UP_DOWN channels
        Eq.(20) is simply inv(chi) = inv(chi0) - Gamma_c
        Note the DCA substitution from Gamma_cluster to Gamma_lattice
        '''
        print("==============================================================================")
        print("Calculate lattice magnetic and/or charge susceptibility with Eq.(16-21) in PRB 64,195130(2001)")
        print("==============================================================================")

        NwG4 = self.NwG4;
        Nc = self.Nc;
        nOrb = self.nOrb
        csum = np.zeros((nOrb, nOrb, nOrb, nOrb), dtype='complex')

        G2L = linalg.inv(linalg.inv(self.chi0M) - self.GammaM / (float(Nc) * self.invT))
        G2L /= (float(self.Nc) * self.invT)
        G2 = G2L.reshape(NwG4, Nc, nOrb, nOrb, NwG4, Nc, nOrb, nOrb)

        for l1 in range(nOrb):
            for l2 in range(nOrb):
                for l3 in range(nOrb):
                    for l4 in range(nOrb):
                        csum[l1, l2, l3, l4] = sum(G2[:, :, l1, l2, :, :, l3, l4])  # /= (self.Nc*self.invT)#**2
                        print('orb ', l1, l2, l3, l4, " lattice susceptibility = ", self.Tval, ' ',
                              real(csum[l1, l2, l3, l4]))

    def calcPairSus_bilayer(self):
        '''
        Copied from solveBSE_fromG4_DCA_bilayer_Maier2017.py on Feb.2, 2022
        '''
        print("Calculate pairing chi in simpler way, similar to above calcLatticeSus")
        NwG4 = self.NwG4;
        Nc = self.Nc;
        nOrb = self.nOrb

        gkd = cos(self.Kvecs[:, 0]) - cos(self.Kvecs[:, 1])

        norm = dot(gkd, gkd)

        # calculate lattice G4 Green's function
        # from inv( inv(chi0)-inv(chic0)+inv(G4) ), see Eq.(20) in PRB 64, 195130(2001)
        G2L = linalg.inv(linalg.inv(self.chi0M) - self.GammaM / (float(self.Nc) * self.invT))
        G2 = G2L.reshape(NwG4, Nc, nOrb, nOrb, NwG4, Nc, nOrb, nOrb)

        # for inter-band susceptibility
        G2spm = 0.25 * (G2[:, :, 0, 1, :, :, 0, 1] + G2[:, :, 0, 1, :, :, 1, 0] + G2[:, :, 1, 0, :, :, 0, 1] + G2[:, :, 1,0, :, :, 1, 0])
        Ps = sum(G2spm) / (float(self.Nc) * self.invT)

        # for dwave susceptibility
        # cos kx - cos ky form-factor corresponds to inter- and intra-layer d-wave susceptibility
        G2band1 = G2[:, :, 0, 0, :, :, 0, 0]
        G2band2 = G2[:, :, 1, 1, :, :, 1, 1]
        G2band12 = 0.5 * (G2[:, :, 1, 1, :, :, 0, 0] + G2[:, :, 0, 0, :, :, 1, 1])

        # axis=0 sum over first wn, then G2 becomes (iK,iwn,iK) so that axis=1 sum over second wn
        G2t0b1 = sum(sum(G2band1, axis=0), axis=1)
        Pd_b1 = dot(gkd, dot(G2t0b1, gkd)) / (float(self.Nc) * self.invT)
        G2t0b2 = sum(sum(G2band2, axis=0), axis=1)
        Pd_b2 = dot(gkd, dot(G2t0b2, gkd)) / (float(self.Nc) * self.invT)
        G2t0b12 = sum(sum(G2band12, axis=0), axis=1)
        Pd_b12 = dot(gkd, dot(G2t0b12, gkd)) / (float(self.Nc) * self.invT)

        # *2 to agree with Maier's 2011 PRB, which probably comes from the 0.25* factor above over multiplication
        print("spm   susceptibility: ", self.Tval, '  ', real(Ps) * 2.)
        print("dwave susceptibility for intra-band 1:  ", self.Tval, '  ', real(Pd_b1))  # /norm
        print("dwave susceptibility for intra-band 2:  ", self.Tval, '  ', real(Pd_b2))  # /norm
        print("dwave susceptibility for inter-band 12: ", self.Tval, '  ', real(Pd_b12))  # /norm

    def determine_specialK(self):
        self.iKPiPi = 0
        self.iKPi0  = 0
        Nc=self.Nc
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
            
    def dwave(self,kx,ky):
        return cos(kx)-cos(ky)

    def projectOnDwave(self,Ks,matrix):
        gk = self.dwave(Ks[:,0], Ks[:,1])
        c1 = dot(gk, dot(matrix,gk) ) / dot(gk,gk)
        return c1

    def dispersion(self,kx,ky):
        if self.model=='square':
            ek  = -2.*self.t*(cos(kx)+cos(ky)) - 4.0*self.tp*cos(kx)*cos(ky)

        elif self.model=='square_txty':
            ek  = -2.*(self.tx*cos(kx) + self.ty*cos(ky)) -2.*(self.tx_nn*cos(2.*kx) + self.ty_nn*cos(2.*ky)) - 4.0*self.tp*cos(kx)*cos(ky)

        elif self.model == 'Eg_bilayer':
            ek = np.zeros((self.nOrb, self.nOrb), dtype='complex')
            r11 = self.e1 + 2. * self.t1 * (cos(kx) + cos(ky)) + 4.0 * self.t1p * cos(kx) * cos(ky)
            r22 = self.e2 + 2. * self.t2 * (cos(kx) + cos(ky)) + 4.0 * self.t2p * cos(kx) * cos(ky)
            r12 = 2. * (self.thybx * std::cos(kx) + self.thyby * std::cos(ky))

            ek[0, 0] = r11
            ek[1, 1] = r22
            ek[2, 2] = r11
            ek[3, 3] = r22
            ek[0, 1] = r12
            ek[1, 0] = r12
            ek[2, 3] = r12
            ek[3, 2] = r12
            ek[0, 2] = self.tperp1
            ek[2, 0] = self.tperp1
            ek[1, 3] = self.tperp2
            ek[3, 1] = self.tperp2

        elif self.model=='dsmodel':
            ek = np.zeros((self.nOrb,self.nOrb),dtype='complex')
            r11  = self.e1 -2.*self.t1*(cos(kx)+cos(ky)) - 4.0*self.t1p*cos(kx)*cos(ky)
            r22  = self.e2 -2.*self.t2*(cos(kx)+cos(ky)) - 4.0*self.t2p*cos(kx)*cos(ky)
            r12  = -self.tperp -2. * self.tperp_p * (cos(kx) - cos(ky))
            ek[0,0] = r11; ek[1,1] = r22
            ek[0,1] = r12; ek[1,0] = r12

        elif self.model=='sdsmodel':
            ek = np.zeros((self.nOrb,self.nOrb),dtype='complex')
            r11  = self.e1 -2.*self.t1*(cos(kx)+cos(ky)) - 4.0*self.t1p*cos(kx)*cos(ky)
            r22  = self.e2 -2.*self.t2*(cos(kx)+cos(ky)) - 4.0*self.t2p*cos(kx)*cos(ky)
            r33  = self.e3 -2.*self.t3*(cos(kx)+cos(ky)) - 4.0*self.t3p*cos(kx)*cos(ky)
            r12  = -self.tperp -2. * self.tperp_p * (cos(kx) - cos(ky))

            ek[0,0] = r11; ek[1,1] = r22; ek[2,2] = r33
            ek[0,1] = r12; ek[1,0] = r12
            ek[1,2] = r12; ek[2,1] = r12

        elif self.model=='Emery':
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

        elif self.model=='bilayer':            
            ek = np.zeros((self.nOrb,self.nOrb),dtype='complex')
            r11  = self.e1 -2.*self.t1*(cos(kx)+cos(ky)) - 4.0*self.t1p*cos(kx)*cos(ky)
            r22  = self.e2 -2.*self.t2*(cos(kx)+cos(ky)) - 4.0*self.t2p*cos(kx)*cos(ky)
            r12  = -self.tperp

            ek[0,0] = r11
            ek[1,1] = r22
            ek[0,1] = r12
            ek[1,0] = r12

        elif self.model=='ddpmodel':
            ek = np.zeros((self.nOrb,self.nOrb),dtype='complex')
            r1  =  2.* 1j *self.tpd*sin(kx/2.)
            r2  = -2.* 1j *self.tpd*sin(ky/2.)
            r1z =  2.* 1j *self.tpdz2*sin(kx/2.)
            r2z = -2.* 1j *self.tpdz2*sin(ky/2.)
            r3  =  4.*self.tpp*sin(kx/2.)*sin(ky/2.)
            ek[0,0] = self.epd
            ek[1,1] = self.epp
            ek[2,2] = self.epp
            ek[3,3] = self.edz2
            ek[0,1] = r1
            ek[1,0] = -r1
            ek[0,2] = r2
            ek[2,0] = -r2
            ek[1,2] = r3
            ek[2,1] = r3
            ek[3,1] = -r1z
            ek[1,3] = r1z
            ek[3,2] = r2z
            ek[2,3] = -r2z
            
        elif self.model=='trilayer':
            ek = np.zeros((self.nOrb,self.nOrb),dtype='complex')
            r11  = self.e1 -2.*self.t1*(cos(kx)+cos(ky)) - 4.0*self.tp1*cos(kx)*cos(ky)
            r22  = self.e2 -2.*self.t1*(cos(kx)+cos(ky)) - 4.0*self.tp1*cos(kx)*cos(ky)
            r33  = self.e3 -2.*self.t1*(cos(kx)+cos(ky)) - 4.0*self.tp1*cos(kx)*cos(ky)
            r12  = -self.tperp12-2.*self.coefp*self.tperpp12*(cos(kx)+cos(ky))-2.*self.coefm*self.tperpp12*(cos(kx)-cos(ky))
            r23  = -self.tperp23-2.*self.coefp*self.tperpp23*(cos(kx)+cos(ky))-2.*self.coefm*self.tperpp23*(cos(kx)-cos(ky))

            ek[0,0] = r11
            ek[1,1] = r22
            ek[2,2] = r33
            ek[0,1] = r12
            ek[1,2] = r23
            ek[1,0] = r12
            ek[2,1] = r23
            
        return ek

###################################################################### writting functions

    def write_data_2cols(self, fname, xs, ys):
        f = open(fname,'w',1) 
        for i in range(len(xs)):
            f.write('{:.6e}\t{:.6e}\n'.format(float(xs[i]),float(ys[i])))

    def write_data_3cols(self, fname, xs, ys, zs):
        f = open(fname,'w',1) 
        for i in range(len(xs)):
            f.write('{:.6e}\t{:.6e}\t{:.6e}\n'.format(float(xs[i]),float(ys[i]),float(zs[i])))
            
    def write_data_4cols(self, fname, xs, ys, zs, ws):
        f = open(fname,'a',1) 
        for i in range(len(xs)):
            f.write('{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\n'.format(float(xs),float(ys),float(zs),float(ws.real)))

    def write_data_5cols(self, fname, xs, ys, zs, ws, gs):
        f = open(fname,'a',1)
        for i in range(len(xs)):
            f.write('{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\n'.format(float(xs),float(ys),float(zs),float(ws.real),float(gs.real)))

    def write_data_6cols(self, fname, xs, ys, zs, ws, fs, gs):
        f = open(fname,'a',1) 
        for i in range(len(xs)):
            f.write('{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\n'.format(float(xs),float(ys),float(zs),float(ws.real),float(fs.real),float(gs.real)))
                           
###################################################################################
Ts = [1, 0.75, 0.5, 0.44, 0.4, 0.34, 0.3, 0.25, 0.24, 0.225, 0.2, 0.175, 0.17, 0.15, 0.125, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.035, 0.03, 0.025]
Ts = [0.06]#, 0.02]
#channels = ['phcharge','phmag']
#channels = ['phmag']
qs = ['00']#,'pi20','pi0','pipi2','pipi','pi2pi2']
#qs = ['pipi']
Nv = [0,1,2,3,4,5,6,7,8]
Nv = [0]

for T_ind, T in enumerate(Ts):
    #for ch in channels:
    for v in Nv:
        for q in qs:
            #file_tp = './T='+str(Ts[T_ind])+'/dca_tp_'+ch+'_q'+q+'.hdf5'
            #file_tp = './T='+str(Ts[T_ind])+'/dca_tp_mag_q'+str(q)+'.hdf5'
            file_tp = './T='+str(Ts[T_ind])+'/dca_tp.hdf5'
            #file_tp = './T='+str(Ts[T_ind])+'/dca_tp_mag_qpipi'+'.hdf5'
            file_sp = './T='+str(Ts[T_ind])+'/dca_sp.hdf5'
            file_analysis_hdf5 = './T='+str(Ts[T_ind])+'/analysis.hdf5'

            if(os.path.exists(file_tp)):
                print ("\n =================================\n")
                print ("T =", T)
                # model='square','square_txty','bilayer','dsmodel','trilayer','Emery','ddpmodel','sdsmodel','sdsmodel','Eg-bilayer'
                BSE('trilayer',\
                    Ts[T_ind],\
                    file_tp,\
                    file_sp,\
                    file_analysis_hdf5,\
                    draw=False,\
                    useG0=False,\
                    symmetrizeG4=True,\
                    phSymmetry=False,\
                    calcCluster=False,\
                    build_sym_kernel=False,\
                    useGamma_hdf5=False,\
                    nkfine=100,\
                    compare_with_analysishdf5=False,\
                    write_data_file=True)
              

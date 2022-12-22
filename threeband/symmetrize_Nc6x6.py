#! python
import sys
from numpy import *


class symmetrize:

     Kvec =  array([[0.0000000000000000, 0.0000000000000000],
                    [0.0000000000000000, 1.0471975511965976],
                    [0.0000000000000000, 2.0943951023931953],
                    [0.0000000000000000, 3.1415926535897931],
                    [0.0000000000000000, 4.1887902047863905],
                    [0.0000000000000000, 5.235987755982989],
                    [1.0471975511965976, 0.0000000000000000],
                    [1.0471975511965976, 1.0471975511965976],
                    [1.0471975511965976, 2.0943951023931953],
                    [1.0471975511965976, 3.1415926535897931],
                    [1.0471975511965976, 4.1887902047863905],
                    [1.0471975511965976, 5.235987755982989],
                    [2.0943951023931953, 0.0000000000000000],
                    [2.0943951023931953, 1.0471975511965976],
                    [2.0943951023931953, 2.0943951023931953],
                    [2.0943951023931953, 3.1415926535897931],
                    [2.0943951023931953, 4.1887902047863905],
                    [2.0943951023931953, 5.235987755982989],
                    [3.1415926535897931, 0.0000000000000000],
                    [3.1415926535897931, 1.0471975511965976],
                    [3.1415926535897931, 2.0943951023931953],
                    [3.1415926535897931, 3.1415926535897931],
                    [3.1415926535897931, 4.1887902047863905],
                    [3.1415926535897931, 5.235987755982989],
                    [4.1887902047863905, 0.0000000000000000],
                    [4.1887902047863905, 1.0471975511965976],
                    [4.1887902047863905, 2.0943951023931953],
                    [4.1887902047863905, 3.1415926535897931],
                    [4.1887902047863905, 4.1887902047863905],
                    [4.1887902047863905, 5.235987755982989],
                    [5.235987755982989, 0.0000000000000000],
                    [5.235987755982989, 1.0471975511965976],
                    [5.235987755982989, 2.0943951023931953],
                    [5.235987755982989, 3.1415926535897931],
                    [5.235987755982989, 4.1887902047863905],
                    [5.235987755982989, 5.235987755982989]])


     iKWedge      = [0,6,7,12,13,14,18,19,20,21] # K-points in irr. wedge
     #               0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
     iKtoWedge    = [0,6,12,18,12,6,6,7,13,19,13,7,12,13,14,20,14,13,18,19,20,21,20,19,12,13,14,20,14,13,6,7,13,19,13,7] # equivalent K-point in wedge
     symOpToWedge = [0,6,6,6,3,3,0,0,6,6,3,4,0,0,0,6,4,4,0,0,0,0,4,4,5,5,5,1,2,2,5,5,1,1,7,2]


# c  The square 36 site cluster has all 8 point group symmetries:

# c                              rot. by pi
# c  identity   rot. by pi/2     inversion    rot. by -pi/2
# c  x' = x       x' =  y         x' = -x        x' = -y
# c  y' = y       y' = -x         y' = -y        y' =  x
# c  0,0,1,1      1,1,1,-1        2,0,-1,-1      3,1,-1,1
# c
# c  ref. x       ref. y          ref. x=y    ref. x=y inversion
# c  x' =  x      x' = -x         x' =  y         x' = -y
# c  y' = -y      y' =  y         y' =  x         y' = -x
# c  4,0,1,-1     5,0,-1,1        6,1,1,1        7,1,-1,-1
# c   
# c  pgroup(iop,iex,isx,isy) 
# c  iex=0(1) dont (do) exchange x and y
# c  isx      sign of x 
# c  isy      sign of y
# c
# c  x --> isx*[(1-iex)*x + iex*y]
# c  y --> isy*[(1-iex)*y + iex*x]
# c
     iex = zeros((8),dtype='int')
     isx = zeros((8),dtype='int')
     isy = zeros((8),dtype='int')
 
     iex[0]=0; isx[0]= 1; isy[0]= 1
     iex[1]=1; isx[1]= 1; isy[1]=-1
     iex[2]=0; isx[2]=-1; isy[2]=-1
     iex[3]=1; isx[3]=-1; isy[3]= 1
     iex[4]=0; isx[4]= 1; isy[4]=-1
     iex[5]=0; isx[5]=-1; isy[5]= 1
     iex[6]=1; isx[6]= 1; isy[6]= 1
     iex[7]=1; isx[7]=-1; isy[7]=-1


     def __init__(self):
          self.data=[]
          self.Nc=36
          self.NcLin = 6
          self.setup_symm_tables()

     def K_2_iK(self,Kx,Ky):
          small = 1.0e-3
          # iKx = int(2*Kx/pi)
          # iKy = int(2*Ky/pi)
          # NcLin = self.NcLin
             
          # if iKx < 0      : iKx += NcLin
          # if iKy < 0      : iKy += NcLin
          # if iKx >= NcLin : iKx -= NcLin
          # if iKy >= NcLin : iKy -= NcLin

          # return iKx *NcLin + iKy

          # Map to [0 ... 2pi, 0 ... 2pi]:
          # print("Kx,Ky",Kx,Ky)
          if (Kx >= 2.*pi) : Kx -= 2.*pi
          if (Kx < 0.0)    : Kx += 2.*pi
          if (Ky >= 2.*pi) : Ky -= 2.*pi
          if (Ky < 0.0)    : Ky += 2.*pi
          # print("Kx,Ky after mapping",Kx,Ky)

          for iK in range(0,self.Nc):
            if (abs(float(self.Kvec[iK,0])-Kx) < small) & (abs(float(self.Kvec[iK,1])-Ky) < small): return iK


     def symmTrans_of_iK(self,iK,isym):
          Kx = self.Kvec[iK,0]; Ky = self.Kvec[iK,1]
          
          KxTrans = self.isx[isym] * ((1-self.iex[isym])*Kx + self.iex[isym]*Ky)
          KyTrans = self.isy[isym] * ((1-self.iex[isym])*Ky + self.iex[isym]*Kx)

          # print("iK,K_2_iK:",KxTrans,KyTrans,self.K_2_iK(KxTrans,KyTrans))
          return self.K_2_iK(KxTrans,KyTrans)     

     def setup_symm_tables(self):
          self.iKDiff = zeros((self.Nc,self.Nc),dtype='int')
          self.iKSum  = zeros((self.Nc,self.Nc),dtype='int')
          Nc = self.Nc
          NcLin = self.NcLin
          for iK1 in range(0,Nc):
              iKx1 = int(iK1/NcLin); iKy1 = int(iK1 % NcLin)
              for iK2 in range(0,Nc):
                  iKx2 = int(iK2/NcLin); iKy2 = int(iK2 % NcLin)

                  iKxDiff = iKx1-iKx2
                  iKyDiff = iKy1-iKy2

                  iKxSum = iKx1+iKx2
                  iKySum = iKy1+iKy2

                  if iKxDiff < 0      : iKxDiff += NcLin
                  if iKyDiff < 0      : iKyDiff += NcLin
                  if iKxDiff >=  NcLin : iKxDiff -= NcLin
                  if iKyDiff >=  NcLin : iKyDiff -= NcLin

                  if iKxSum < 0      : iKxSum += NcLin
                  if iKySum < 0      : iKySum += NcLin
                  if iKxSum >= NcLin  : iKxSum -= NcLin
                  if iKySum >= NcLin  : iKySum -= NcLin

                  iKD = iKxDiff * NcLin + iKyDiff
                  iKS = iKxSum * NcLin + iKySum

                  self.iKDiff[iK1,iK2] = iKD
                  self.iKSum[iK1,iK2]  = iKS

# Then build the point group symmetry tables needed for generating the 4-point functions

          self.iK2Map = zeros((Nc,Nc),dtype='int')
          for iQ in range(0,Nc):
               isym = self.symOpToWedge[iQ] # index of symm. operation that takes Kvec[iQ] to the irr. wedge
               for iK in range(0,Nc):
                    self.iK2Map[iQ,iK] = self.symmTrans_of_iK(iK,isym)
          print ("Done setting up K-space arrays")

     def apply_point_group_symmetries_Q0(self,G4):
          # for G4[K1,w1,K2,w2]
          # G4(K,K') = G4(Ra(K),Ra(K')) for all frequencies
          Nc = self.Nc
          nwn = G4.shape[1]
          type=dtype(G4[0,0,0,0])
          for iK1 in range(0,Nc):
            for iK2 in range(0,Nc):
              tmp = zeros((nwn,nwn),dtype=type)
              for iSym in range(0,8): # Apply every point-group symmetry operation
                iK1Trans = self.symmTrans_of_iK(iK1,iSym)
                iK2Trans = self.symmTrans_of_iK(iK2,iSym)
                tmp[:,:] += G4[iK1Trans,:,iK2Trans,:]
              for iSym in range(0,8):
                iK1Trans = self.symmTrans_of_iK(iK1,iSym)
                iK2Trans = self.symmTrans_of_iK(iK2,iSym)
                G4[iK1Trans,:,iK2Trans,:] = tmp[:,:]/8.

     def apply_point_group_symmetries_withQ(self,G4):
          #      # for G4[Q,K1,w1,K2,w2]
          #      # G4(Q,K,K') = G4(Ra(Q),Ra(K),Ra(K')) for all frequencies
          #      Nc = self.Nc
          #      nwn = int(G4.shape[0]/Nc)
          Nc = self.Nc
          nwn = G4.shape[3]
          nwm = G4.shape[1]
          type=dtype(G4[0,0,0,0,0,0])
          for iQ in range(0,Nc):
               for iK1 in range(0,Nc):
                    for iK2 in range(0,Nc):
                         tmp = zeros((nwm,nwn,nwn),dtype=type)
                         for iSym in range(0,8): # Apply every point-group symmetry operation
                              iQTrans  = self.symmTrans_of_iK(iQ,iSym)
                              iK1Trans = self.symmTrans_of_iK(iK1,iSym)
                              iK2Trans = self.symmTrans_of_iK(iK2,iSym)
                              tmp[:,:,:] += G4[iQTrans,:,iK1Trans,:,iK2Trans,:]
                         for iSym in range(0,8):
                              iQTrans  = self.symmTrans_of_iK(iQ,iSym)
                              iK1Trans = self.symmTrans_of_iK(iK1,iSym)
                              iK2Trans = self.symmTrans_of_iK(iK2,iSym)
                              G4[iQTrans,:,iK1Trans,:,iK2Trans,:] = tmp[:,:]/8.

     def apply_symmetry_in_wn(self,G4):
          # for G4[K1,w1,K2,w2]
          # apply symmetry G4(K,wn,K',wn') = G4*(K,-wn,K',-wn')
          Nc = self.Nc
          nwn = G4.shape[1]
          iw0=int(nwn/2)
          for iw1 in range(0,nwn):
            for iw2 in range(0,nwn):
              for iK1 in range(0,Nc):
                for iK2 in range(0,Nc):
                  imw1 = nwn-1-iw1
                  imw2 = nwn-1-iw2
                  tmp1 = G4[iK1,iw1,iK2,iw2]
                  tmp2 = G4[iK1,imw1,iK2,imw2]
                  G4[iK1,iw1,iK2,iw2]   = 0.5*(tmp1+conj(tmp2))
                  G4[iK1,imw1,iK2,imw2] = 0.5*(conj(tmp1)+tmp2)

     def apply_transpose_symmetry(self,G4):
          # Now re-apply symmetry GammaPP(K,K') = GammaPP(K',K)
          Nc = self.Nc
          nwn = G4.shape[1]
          GP = G4.reshape(Nc*nwn,Nc*nwn) 
          GP = 0.5*(GP + GP.transpose())
          G4 = GP.reshape(Nc,nwn,Nc,nwn)

# package importing

import cupy as cp
import numpy as np
from time import time
import matplotlib.pyplot as plt
folderName = 'beads_0607'



# fitting parameters

nBlocks = 60
nTimes = 450
finalRounds = [(500,0.8,1e-6,5e-8),\
               (500,0.7,1e-6,2e-8),\
               (500,0.5,5e-7,1e-8),\
               (400,0.3,2e-7,5e-9),\
               (400,0.1,2e-7,1e-9),\
               (300,0.0,1e-7,1e-10)]
nSteps = nBlocks * nTimes
regFreq = 5
alphaList = np.concatenate((2e-3*np.ones(nSteps//200),\
                            4e-4*np.ones(nSteps//50 - nSteps//200),\
                            1e-4*np.ones(nSteps//20 - nSteps//50),\
                            2e-5*np.ones(nSteps//5 - nSteps//20),\
                            1e-5*np.ones(nSteps//3 - nSteps//5),\
                            5e-6*np.ones(nSteps//2 - nSteps//3),\
                            1e-6*np.ones(nSteps - nSteps//2)))
#alphaList = np.zeros(nSteps)
betaList = np.concatenate((1e-3*np.ones(nSteps//200),\
                           1e-4*np.ones(nSteps//100 - nSteps//200),\
                           2e-5*np.ones(nSteps//50 - nSteps//100),\
                           5e-6*np.ones(nSteps//20 - nSteps//50),\
                           2e-6*np.ones(nSteps//10 - nSteps//20),\
                           5e-7*np.ones(nSteps//5 - nSteps//10),\
                           2e-7*np.ones(nSteps//2 - nSteps//5),\
                           5e-8*np.ones(nSteps - nSteps//2)))
#betaList = np.zeros(nSteps)
gamma = 2
etaList = np.concatenate((1e-3*np.ones(nSteps//50),\
                          2e-4*np.ones(nSteps//20 - nSteps//50),\
                          5e-5*np.ones(nSteps//5 - nSteps//20),\
                          1e-5*np.ones(nSteps//3 - nSteps//5),\
                          1e-6*np.ones(nSteps//2 - nSteps//3),\
                          np.zeros(nSteps - nSteps//2)))
#etaList = np.zeros(nSteps)
thetaList = np.concatenate((2e-3*np.ones(nSteps//50),\
                            5e-4*np.ones(nSteps//20 - nSteps//50),\
                            2e-4*np.ones(nSteps//5 - nSteps//20),\
                            5e-5*np.ones(nSteps//3 - nSteps//5),\
                            2e-5*np.ones(nSteps//2 - nSteps//3),\
                            5e-6*np.ones(2*nSteps//3 - nSteps//2),\
                            np.zeros(nSteps - 2*nSteps//3)))
#thetaList = np.zeros(nSteps)

samplingDots = 200
samplingDist = 0.006
decayFraction = 0.12
decayAlpha = 0.602/1
decayGamma = 0.101/1
decayFactor = 1.2
finalDist = samplingDist/200



# layout parameters

distance = cp.float64(13)

heightLength = cp.float64(176)
widthLength = cp.float64(281)
bHeightPixel = 303
bWidthPixel = 483
center = (25,25)
radiusM = 4
radiusB = 0



# data loading

bX = np.load(folderName+'/Bx_0607.npy')
bY = np.load(folderName+'/By_0607.npy')
bZ = np.load(folderName+'/Bz_0607.npy')
bCroppingWidth,bCroppingHeight = bX.shape
bExpNp = np.concatenate((bX[:,:,np.newaxis],bY[:,:,np.newaxis],bZ[:,:,np.newaxis]),axis=2)
bExp = cp.asarray(bExpNp)
del bExpNp


def spinAllocate(f_,r_0,r_):
    x_, y_ = bX.shape
    x_ = round(f_ * x_)
    y_ = round(f_ * y_)
    x0 = round(f_ * r_0[0])
    y0 = round(f_ * r_0[1])
    r_ = round(f_ * r_)
    rg_ = np.zeros((2*r_+1,2*r_+1))
    for i_ in range(-r_,r_+1):
        for j_ in range(-r_,r_+1):
            if (i_*i_ + j_*j_) <= r_*r_:
                rg_[r_+i_,r_+j_] = 1
    RG_ = cp.asarray(rg_)
    del rg_
    return x_, y_, 2*r_+1, 2*r_+1, x0-r_, y0-r_, RG_

def spinAllocateSq(f_,XStart_,XEnd_):
    x_, y_ = bX.shape
    x_ = round(f_ * x_)
    y_ = round(f_ * y_)
    XStart = round(f_ * XStart_[0])
    YStart = round(f_ * XStart_[1])
    XEnd = round(f_ * XEnd_[0])
    YEnd = round(f_ * XEnd_[1])
    return x_, y_, (XEnd-XStart), (YEnd-YStart), XStart, YStart, cp.ones((XEnd-XStart,YEnd-YStart))

def mInitLine(f_,r_0,r_,mx=-1,mz=1):
    x_, y_ = bX.shape
    r__ = r_
    x_ = round(f_ * x_)
    y_ = round(f_ * y_)
    x0 = round(f_ * r_0[0])
    y0 = round(f_ * r_0[1])
    r_ = round(f_ * r_)
    rg_ = np.zeros((2*r_+1,2*r_+1,3))
    for i_ in range(-r_,r_+1):
        for j_ in range(-r_,r_+1):
            rg_[r_+i_,r_+j_,0] = mx*(-i_ * (i_**2 + j_**2)**0.5 *\
                                 (2*r_**2 - i_**2 - j_**2)**0.5 /r_**3)
            rg_[r_+i_,r_+j_,1] = mx*(-j_ * (i_**2 + j_**2)**0.5 *\
                                 (2*r_**2 - i_**2 - j_**2)**0.5 /r_**3)
            rg_[r_+i_,r_+j_,2] = mz*(-1/2 + (i_**2 + j_**2)**0.5 / r_)
    RG_ = cp.asarray(rg_) * spinAllocate(f_,r_0,r__)[-1][:,:,cp.newaxis]
    del rg_
    return RG_

def mInitCirc(f_,r_0,r_,mx=-1,mz=1):
    x_, y_ = bX.shape
    r__ = r_
    x_ = round(f_ * x_)
    y_ = round(f_ * y_)
    x0 = round(f_ * r_0[0])
    y0 = round(f_ * r_0[1])
    r_ = round(f_ * r_)
    rg_ = np.zeros((2*r_+1,2*r_+1,3))
    for i_ in range(-r_,r_+1):
        for j_ in range(-r_,r_+1):
            rg_[r_+i_,r_+j_,0] = mx*(-j_ * (i_**2 + j_**2)**0.5 *\
                                 (2*(r_)**2 - i_**2 - j_**2)**0.5 /r_**3)
            rg_[r_+i_,r_+j_,1] = mx*(i_ * (i_**2 + j_**2)**0.5 *\
                                 (2*(r_)**2 - i_**2 - j_**2)**0.5 /r_**3)
            rg_[r_+i_,r_+j_,2] = mz*(-1/2 + (i_**2 + j_**2)**0.5 / r_)
    RG_ = cp.asarray(rg_) * spinAllocate(f_,r_0,r__)[-1][:,:,cp.newaxis]
    del rg_
    return RG_

'''
mWidthPixel,mHeightPixel,\
mCroppingWidth,mCroppingHeight,\
mCroppingWidthStart,mCroppingHeightStart,\
weightMatrix = spinAllocate(10,center,radiusM)
'''

mWidthPixel,mHeightPixel,\
mCroppingWidth,mCroppingHeight,\
mCroppingWidthStart,mCroppingHeightStart,\
weightMatrix = spinAllocateSq(1,(0,0),(99,109))


''' testing position
circPlot = np.zeros((mWidthPixel,mHeightPixel))
for i_ in range(mCroppingWidth):
    for j_ in range(mCroppingHeight):
        circPlot[mCroppingWidthStart+i_,mCroppingHeightStart+j_] = weightMatrix[i_,j_]
#1/0
#'''


timeStart = time()

rPosition = np.zeros((bCroppingWidth,bCroppingHeight,\
                      mCroppingWidth,mCroppingHeight,3),dtype = cp.float64)
'''
r vector pointing from mMap to bMap.
5 axes: mu, sigma, lam, rho, i
First and second axes are bMap's x and y index
Third and fourth axes are mMap's x and y index
Fifth axis is component,
i = 0 returns x component,
i = 1 returns y component,
i = 2 returns norm.
z component is always -distance
'''
for mu in range(bCroppingWidth):
    for sigma in range(bCroppingHeight):
        for lam in range(mCroppingWidth):
            for rho in range(mCroppingHeight):
                rPosition[mu,sigma,lam,rho,0] = widthLength * bCroppingWidth / bWidthPixel * ((mu)/bCroppingWidth - (lam+mCroppingWidthStart)/mWidthPixel)
                rPosition[mu,sigma,lam,rho,1] = heightLength * bCroppingHeight / bHeightPixel * ((sigma)/bCroppingHeight - (rho+mCroppingHeightStart)/mHeightPixel)
                rPosition[mu,sigma,lam,rho,2] = (\
                    rPosition[mu,sigma,lam,rho,0]**2 +\
                    rPosition[mu,sigma,lam,rho,1]**2 +\
                    distance * distance)**0.5



# transition matrix

matrixANp = np.zeros((bCroppingWidth,bCroppingHeight,\
                      mCroppingWidth,mCroppingHeight,3,3),dtype = cp.float64)
'''
map mMap to bMap
6 axes: mu, sigma, lam, rho, i, j
First and second axes are bMap's x and y index
Third and fourth axes are mMap's x and y index
Fifth and sixth axes are component, i and j,
i = 0 returns x component,
i = 1 returns y component,
i = 2 returns z component
'''
for mu in range(bCroppingWidth):
    for sigma in range(bCroppingHeight):
        for lam in range(mCroppingWidth):
            for rho in range(mCroppingHeight):
                matrixANp[mu,sigma,lam,rho,0,0] = 3*rPosition[mu,sigma,lam,rho,0]*rPosition[mu,sigma,lam,rho,0]/rPosition[mu,sigma,lam,rho,2]**5 -\
                                                1/rPosition[mu,sigma,lam,rho,2]**3
                matrixANp[mu,sigma,lam,rho,0,1] = 3*rPosition[mu,sigma,lam,rho,0]*rPosition[mu,sigma,lam,rho,1]/rPosition[mu,sigma,lam,rho,2]**5
                matrixANp[mu,sigma,lam,rho,0,2] = -3*rPosition[mu,sigma,lam,rho,0]*distance/rPosition[mu,sigma,lam,rho,2]**5
                matrixANp[mu,sigma,lam,rho,1,0] = matrixANp[mu,sigma,lam,rho,0,1]
                matrixANp[mu,sigma,lam,rho,1,1] = 3*rPosition[mu,sigma,lam,rho,1]*rPosition[mu,sigma,lam,rho,1]/rPosition[mu,sigma,lam,rho,2]**5 -\
                                                1/rPosition[mu,sigma,lam,rho,2]**3
                matrixANp[mu,sigma,lam,rho,1,2] = -3*rPosition[mu,sigma,lam,rho,1]*distance/rPosition[mu,sigma,lam,rho,2]**5
                matrixANp[mu,sigma,lam,rho,2,0] = matrixANp[mu,sigma,lam,rho,0,2]
                matrixANp[mu,sigma,lam,rho,2,1] = matrixANp[mu,sigma,lam,rho,1,2]
                matrixANp[mu,sigma,lam,rho,2,2] = 3*distance*distance/rPosition[mu,sigma,lam,rho,2]**5 - 1/rPosition[mu,sigma,lam,rho,2]**3

matrixA = cp.asarray(matrixANp)

del rPosition
del matrixANp

#timeStop = time()
#print(timeStop - timeStart)



# initialization

bRMS = np.sqrt(cp.sum(bExp**2)/3/bCroppingWidth/bCroppingHeight)
numberM = cp.sum(weightMatrix)
#mAVG = (bRMS * distance**3 / 4 / np.sqrt(numberM))
mAVG = (bRMS * distance**3 / numberM) * 15

mPar = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.])
mStep = 0.001
def mSpan(mList):
    mRes_ = mAVG * cp.concatenate((cp.outer(cp.ones(mCroppingWidth),cp.linspace(mList[0]-mList[1],mList[0]+mList[1],mCroppingHeight))[:,:,cp.newaxis],\
                                  cp.outer(cp.linspace(mList[3]-mList[4],mList[3]+mList[4],mCroppingWidth),cp.ones(mCroppingHeight))[:,:,cp.newaxis],\
                                  cp.outer(cp.linspace(mList[6]-mList[7],mList[6]+mList[7],mCroppingWidth),cp.ones(mCroppingHeight))[:,:,cp.newaxis]),axis=2) * weightMatrix[:,:,cp.newaxis] +\
           mAVG * cp.concatenate((cp.outer(cp.linspace(-mList[2],mList[2],mCroppingWidth),cp.ones(mCroppingHeight))[:,:,cp.newaxis],\
                                  cp.outer(cp.ones(mCroppingWidth),cp.linspace(-mList[5],mList[5],mCroppingHeight))[:,:,cp.newaxis],\
                                  cp.outer(cp.ones(mCroppingWidth),cp.linspace(-mList[8],mList[8],mCroppingHeight))[:,:,cp.newaxis]),axis=2) * weightMatrix[:,:,cp.newaxis]
    return mRes_
mRes = mSpan(mPar)
#1/0




timeStop = time()




# loss function

def spinAllocateB(f_,r_0,r_):
    x_, y_ = bX.shape
    x_ = round(f_ * x_)
    y_ = round(f_ * y_)
    x0 = round(f_ * r_0[0])
    y0 = round(f_ * r_0[1])
    r_ = round(f_ * r_)
    rg_ = np.zeros((x_,y_))
    for i_ in range(x_):
        for j_ in range(y_):
            if ((i_-x0)**2 + (j_-y0)**2) <= r_*r_:
                rg_[i_,j_] = 0
            else:
                rg_[i_,j_] = 1
    RG_ = cp.asarray(rg_)
    del rg_
    return RG_

Bweight = spinAllocateB(1,center,0)
print(Bweight[0,0],Bweight[20,20])

def lossF(mMap,alpha=0,beta=0):
    
    bRes = cp.tensordot(matrixA,mMap,axes=([2,3,5],[0,1,2]))
    errTerm = cp.sum(((bRes - bExp)*Bweight[:,:,cp.newaxis]) ** 2)
    del bRes

    if alpha != 0:
        mShiftedX = cp.roll(mMap,1,axis=0) * weightMatrix[:,:,cp.newaxis]
        mShiftedY = cp.roll(mMap,1,axis=1) * weightMatrix[:,:,cp.newaxis]
        gradTermX = cp.sum((mMap - mShiftedX) ** 2)
        gradTermY = cp.sum((mMap - mShiftedY) ** 2)
        gradTerm = gradTermX + gradTermY
        del mShiftedX,mShiftedY
    else:
        gradTerm = 0

    if beta != 0:
        ampTerm = cp.sum(mMap ** gamma)
    else:
        ampTerm = 0

    return errTerm / bRMS**2 / bCroppingWidth / bCroppingHeight / 3 +\
           (alpha * gradTerm / mAVG**2 + beta * ampTerm / mAVG**gamma) / numberM


print('initialization time: %s\n\tLoss = %s'%(timeStop - timeStart,lossF(mRes)))

timeStart = time()

for j in range(1):
    for i in range(200):
        drc = np.random.uniform(size=(9))-0.5
        PP = mPar+mStep*drc
        PM = mPar-mStep*drc
        vP=lossF(mSpan(PP))
        vM=lossF(mSpan(PM))
        v0=lossF(mSpan(mPar))
        F_ = (vP-vM)/2/mStep
        F__ = (vP+vM-2*v0)/mStep/mStep
        if np.abs(F__)>0.00001:
            mPar -= np.float64(0.6 * F_/F__) * drc

mRes = mSpan(mPar)
mResB = cp.copy(mRes)
timeStop = time()
print('adjustment initial condition time: %s\n\tLoss = %s'%(timeStop - timeStart,lossF(mRes)))

# iteration initialization

'''
timeStart = time()

firstDist = samplingDist * mAVG
gradList = []
for _ in range(samplingDots):
    delta_ = firstDist * cp.random.choice([-1,1],size=mRes.shape) * weightMatrix[:,:,cp.newaxis]
    gradList.append(abs(lossF(mRes + delta_) - lossF(mRes - delta_)))
    del delta_
gradAVG = sum(gradList) / samplingDots / firstDist / 2
decayStep = decayFraction / gradAVG

timeStop = time()
print('iteration initialization time (%s gradients): %s\n\tLoss = %s'%\
      (samplingDots,timeStop-timeStart,lossF(mRes)))
#'''



# iteration function

def stepX(mMap,iRound,eta,theta,ifRegulate=False):
    
    decayA = decayStep * mAVG / (decayFactor*iRound/nTimes+1)**decayAlpha
    decayC = samplingDist * mAVG / (decayFactor*iRound/nTimes+1)**decayGamma

    if ifRegulate:
        mMap_ = (1-theta) * mMap
        del mMap
        if eta != 0:
            mMap = mMap_ + (eta * mAVG * cp.random.uniform(-1,1,size=mMap_.shape) *\
                              weightMatrix[:,:,cp.newaxis])
        else:
            mMap = mMap_
        del mMap_

    dirc_ = cp.random.choice([-1,1],size=mMap.shape) * weightMatrix[:,:,cp.newaxis]
    delta_ = decayC * dirc_
    grad_ = lossF(mMap + delta_, alphaList[iRound], betaList[iRound]) -\
            lossF(mMap - delta_, alphaList[iRound], betaList[iRound])

    mMap -= decayA * grad_ / 2 / decayC * dirc_
    del dirc_,delta_

    return mMap

# iteration

'''
for i_b in range(nBlocks):
    timeStart = time()
    for i_t in range(nTimes):
        i_ = i_b*nTimes + i_t
        mRes_ = stepX(mRes,i_,etaList[i_],thetaList[i_],(i_%regFreq==0))
        del mRes
        mRes = mRes_
        del mRes_
    timeStop = time()
    print('block %s (%s iterations) time: %s\n\tLoss = %s'%\
          (i_b+1,nTimes,timeStop-timeStart,lossF(mRes)))
#'''

# final rounds

def stepF(mMap,pos,alpha=0,beta=0):

    decay_ = finalDist * mAVG
    dirc_ = cp.random.choice([-1,1],size=mMap.shape) * weightMatrix[:,:,cp.newaxis]
    delta_ = decay_ * dirc_
    vp = lossF(mMap + delta_, alpha, beta)
    vm = lossF(mMap - delta_, alpha, beta)
    v_ = lossF(mMap, alpha, beta)
    F_ = (vp - vm) / 2 / finalDist
    F__ = (vp + vm - 2*v_) / finalDist**2

    mMap -= pos * F_/F__ * mAVG * dirc_
    return mMap

'''
for i in range(len(finalRounds)):
    i_ = finalRounds[i]
    timeStart = time()
    for j_ in range(i_[0]):
        decay_ = finalDist * mAVG
        dirc_ = cp.random.choice([-1,1],size=mRes.shape) * weightMatrix[:,:,cp.newaxis]
        delta_ = decay_ * dirc_
        vp = lossF(mRes + delta_, i_[2], i_[3])
        vm = lossF(mRes - delta_, i_[2], i_[3])
        v_ = lossF(mRes)
        F_ = (vp - vm) / 2 / finalDist
        F__ = (vp + vm - 2*v_) / finalDist**2
        mRes -= (1-i_[1]) * F_/F__ * mAVG * dirc_
        del dirc_,delta_
    timeStop = time()
    print('final round %s (%s iterations) time: %s\n\tLoss = %s'%\
          (i+1,i_[0],timeStop-timeStart,lossF(mRes)))
#'''

#1/0


'''
for k_ in range(5):
    timeStart = time()

    # multiplicator
    cdL = [(lossF(0.97*mRes),0.97),(lossF(mRes),1),(lossF(1.03*mRes),1.03)]
    cdL.sort(key=lambda x:x[0])
    mRes*=cdL[0][-1]

    # random noise
    mMax = cp.max(mRes)
    mRes += mMax * (0.02+0.06*np.random.random()) * ((cp.random.random(mRes.shape)-0.5) * weightMatrix[:,:,cp.newaxis])

    # iteration
    for j_ in range(4):
        a_=10**(-1.5-2*np.random.random())
        b_=10**(-2.5-3*np.random.random())
        for i_ in range(2000):
            mRes=stepF(mRes,0.5,a_,b_)

    for j_ in range(4):
        a_=10**(-1.5-2*np.random.random())
        b_=10**(-2.5-3*np.random.random())
        for i_ in range(2000):
            mRes=stepF(mRes,0.2,a_,b_)

    cLoss = lossF(mRes)
    pLoss = lossF(mResB)

    if cLoss<pLoss:
        s_ = 'succeeded'
        loss_ = cLoss
        mResB = cp.copy(mRes)
    else:
        s_ = 'failed'
        loss_ = pLoss
        mRes = cp.copy(mResB)

    timeStop = time()
    print('Round %s %s, time: %s\n\tLoss = %s'%(k_+1,s_,timeStop-timeStart,loss_))
#'''    
    

print('finished')

import rundec
import constants as cnst

def mtmt2mtmu(mt, mu): #OK
    crd = rundec.CRunDec()
    asmt = crd.AlphasExact(cnst.asMZ, cnst.MZ, mt, cnst.nflav, cnst.nloops)
    asmu = crd.AlphasExact(cnst.asMZ, cnst.MZ, mu, cnst.nflav, cnst.nloops)
    mtmu = crd.mMS2mMS(mt, asmt, asmu, cnst.nflav, cnst.nloops)
    return mtmu

def mtmu2mtmu (mtmu1, mu1, mu2):
    crd = rundec.CRunDec()
    asmu1 = crd.AlphasExact(cnst.asMZ, cnst.MZ, mu1, cnst.nflav, cnst.nloops)
    asmu2 = crd.AlphasExact(cnst.asMZ, cnst.MZ, mu2, cnst.nflav, cnst.nloops)
    mtmu2 = crd.mMS2mMS(mtmu1, asmu1, asmu2, cnst.nflav, cnst.nloops)
    return mtmu2

def mtmu2mtp(mtmu,mu): #OK
    crd = rundec.CRunDec()
    asmu = crd.AlphasExact(cnst.asMZ, cnst.MZ, mu, cnst.nflav, cnst.nloops)
    mtp = crd.mMS2mOS(mtmu,None,asmu,mu,cnst.nflav,cnst.nloops)
    return mtp


def mtp2mtmt(mtp): # to check
    crd = rundec.CRunDec()
    asmtp = crd.AlphasExact(cnst.asMZ, cnst.MZ, mtp, cnst.nflav, cnst.nloops)
    mtmt = crd.mOS2mSI(mtp,None,asmtp,cnst.nflav,cnst.nloops)
    return mtmt


def mtmu2mtmt(mtmu,mu): #OK
    crd = rundec.CRunDec()
    asmu = crd.AlphasExact(cnst.asMZ, cnst.MZ, mu, cnst.nflav, cnst.nloops)
    mtmt = crd.mMS2mSI(mtmu,asmu,mu,cnst.nflav,cnst.nloops)
    return mtmt


def mtp2mtmu(mtp,mu): #OK
    crd = rundec.CRunDec()
    asmu = crd.AlphasExact(cnst.asMZ, cnst.MZ, mu, cnst.nflav, cnst.nloops)
    mtmu = crd.mOS2mMS(mtp,None,asmu,mu,cnst.nflav,cnst.nloops)
    return mtmu


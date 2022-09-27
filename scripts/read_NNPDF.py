#runs on lxplus

import os
import shutil
import numpy as np

indir = '/afs/cern.ch/work/j/jkiesele/public/Matteo/nnpdf'
od = '../NNLO_dat/NNPDF'

nBins = 4

def writeOut(m_xsec):
    if not os.path.exists(od):
        os.makedirs(od)

    bins = [0,420,550,810,13000]

    f = open('{}/NNPDF_nominal.dat'.format(od),'w')
    for b in range(0,nBins):
        low = bins[b]
        high = bins[b+1]
        for m in m_xsec[b].keys():
            f.write('{}\t{}\tx.xx\tx.xx\tx.xx\t{}\tx.xx\tx.xx\tx.xx\t{}\n'.format(low,high,m_xsec[b][m],m))

    return
                        

def readXsecFromFile(infile):
    f = open(infile,'read')
    lines = f.read().splitlines()
    return float([l for l in lines if 'NNLO' in l][-1].split()[3])

def getMassXsecDict(b,dirlist):
    md = dict()
    for d in dirlist:
        if not d.endswith('bin_{}/'.format(b+1)):
            continue
        mass = d.split('_')[-3]
        xsec = readXsecFromFile(d+d.split('/')[-2]+'/summary/result_summary.dat')
        md[mass] = xsec
    return md        

def readMassPoints(dirname):
    from glob import glob
    dirlist = glob('{}/*/'.format(dirname))
    l = []
    for b in range(0,nBins):
        d = getMassXsecDict(b,dirlist)
        l.append(d)
    return l

def main():
    m_xsec = readMassPoints(indir)
    writeOut(m_xsec)
    
    return

if __name__ == "__main__":
    main()

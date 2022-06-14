#runs on lxplus

import os
import shutil
import numpy as np

indir = '/afs/cern.ch/work/j/jkiesele/public/Matteo'
od = 'NNLO_dat/PDFs_conv'
nPDFs = 30

def readNominal():
    f = open('NNLO_dat/PDFs/pdf_variation_0.dat')
    lines = f.read().splitlines()
    list_d = []
    for line in lines:
        d = dict()
        d['low'] = line.split()[0]
        d['high'] = line.split()[1]
        d['xsec'] = float(line.split()[2])
        d['mass'] = line.split()[-1]
        list_d.append(d)
    return list_d

def convertPDFs(list_d):
    if not os.path.exists(od):
        os.makedirs(od)

    shutil.copyfile('NNLO_dat/PDFs/pdf_variation_0.dat','{}/pdf_variation_0.dat'.format(od))
    for i in range(0,len(list_d)):
        shutil.copyfile('NNLO_dat/PDFs/rel_uncert_bin{}.json'.format(i+1),'{}/rel_uncert_bin{}.json'.format(od,i+1))

    xsec_pdf = []
    for pdf in range(0,nPDFs):
        xsec = np.array([getXsecPole(list_d,mbin,pdf) for mbin in range(0,len(list_d))])
        if pdf>0:
            xsec = xsec/xsec_pdf[0]
        xsec_pdf.append(xsec)
    
    for pdf in range(1,nPDFs):
        f = open('{}/pdf_variation_{}.dat'.format(od,pdf),'w')
        for mbin in range(0,len(list_d)):
            f.write('{}\t{}\t{}\tx.xx\tx.xx\t{}\n'.format(list_d[mbin]['low'], list_d[mbin]['high'],list_d[mbin]['xsec']*xsec_pdf[pdf][mbin],list_d[mbin]['mass']))

    return
                        
def getXsecPole(list_d,mbin,pdf):
    mass = list_d[mbin]['mass']
    if pdf==0:
        infile = '{}/PDF/bin{}/run_NNLO_mt_{}_bin_{}/run_NNLO_mt_{}_bin_{}/summary/result_summary.dat'.format(indir,mbin+1,mass,mbin+1,mass,mbin+1)
    else:
        infile = '{}/PDF/bin{}/run_NNLO_mt_{}_bin_{}_pdf_{}/run_NNLO_mt_{}_bin_{}_pdf_{}/summary/result_summary.dat'.format(indir,mbin+1,mass,mbin+1,pdf,mass,mbin+1,pdf)

    return readXsecFromFile(infile)

def readXsecFromFile(infile):
    f = open(infile,'read')
    lines = f.read().splitlines()
    return float([l for l in lines if 'NNLO' in l][-1].split()[3])


def main():
    list_d = readNominal()
    convertPDFs(list_d)
    
    return

if __name__ == "__main__":
    main()

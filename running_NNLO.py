import os, sys
import numpy as np
import argparse
import pickle

from chi2_running_object import running_object, isGoodFit
from ratio_object import ratio_object

dat_dir = 'NNLO_dat'
PDF_dir = 'PDFs_MSbar'
infile_xsec_mass = '{}/scale_variations/scales_all.dat'.format(dat_dir)
infile_num_unc = '{}/rel_uncert.json'.format(dat_dir)
inpath_PDFs = '{}/{}/pdf_variation_{}.dat'.format(dat_dir,PDF_dir,'{}')
infile_num_unc_PDFs = '{}/{}/rel_uncert_bin{}.json'.format(dat_dir,PDF_dir,'{}')
od = 'fit_results'

def checkFitResults(od):
    with open('{}/minuit_object.pkl'.format(od), 'rb') as inp:
        minuit = pickle.load(inp)
        if not isGoodFit(minuit,printout=True):
            print ('\nERROR: fit did not converge\n')
            sys.exit()
    return
        
def main():

    parser = argparse.ArgumentParser(description='specify options')
    parser.add_argument('--breakdown',action='store_true', help='approximate breakdown of uncertainties')
    parser.add_argument('--PDFsFromNLO',action='store_true', help='use PDF variations estimated using NLO calculation and NNLO PDFs')
    parser.add_argument('--uncorrScales',action='store_true', help='uncorrelated scale variations')
    args = parser.parse_args()

    global od
    
    if args.PDFsFromNLO:
        od += '_nloPDFs'

    main_obj = running_object(infile_xsec_mass,infile_num_unc,inpath_PDFs,infile_num_unc_PDFs,od,args.PDFsFromNLO)
    if not os.path.exists(od):
        print('\ndirectory "{}" not found: re-running fit...\n'.format(od))
        main_obj.doFullFit()
    else:
        print('\nWARNING: directory "{0}" already exisits!\nreading in fit results from "{0}"...\n'.format(od))

    checkFitResults(od)
    
    if args.breakdown:
        print('\nperforming breakdown...\n')
        main_obj.doBreakdown()
        
    r_object = ratio_object(indir=od,scale_vars=main_obj.scale_vars,uncorr_scales=args.uncorrScales)
    
    return

if __name__ == "__main__":
    main()

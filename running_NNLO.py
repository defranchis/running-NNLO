import os
import numpy as np

from chi2_running_object import running_object
from ratio_object import ratio_object

dat_dir = 'NNLO_dat'
infile_xsec_mass = '{}/mass_points.dat'.format(dat_dir)
infile_num_unc = '{}/rel_uncert.json'.format(dat_dir)
inpath_PDFs = '{}/PDFs/pdf_variation_{}.dat'.format(dat_dir,'{}')
infile_num_unc_PDFs = '{}/PDFs/rel_uncert_bin{}.json'.format(dat_dir,'{}')
od = 'fit_results'

def main():

    if not os.path.exists(od):
        print('\ndirectory "{}" not found: re-running fit...\n'.format(od))
        main_obj = running_object(infile_xsec_mass,infile_num_unc,inpath_PDFs,infile_num_unc_PDFs,od)
    else:
        print('\nWARNING: directory "{0}" already exisits!\nreading in fit results from "{0}"...\n'.format(od))

    masses = np.load('{}/mass_results.npy'.format(od))
    covariance = np.load('{}/mass_covariance.npy'.format(od))

    r_object = ratio_object(masses,covariance)
    
    
    return

if __name__ == "__main__":
    main()

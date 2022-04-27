
from running_object import running_object

dat_dir = 'NNLO_dat'
infile_xsec_mass = '{}/mass_points.dat'.format(dat_dir)
infile_num_unc = '{}/rel_uncert.json'.format(dat_dir)
inpath_PDFs = '{}/PDFs/April2022_pdf_{}.dat'.format(dat_dir,'{}')
infile_num_unc_PDFs = '{}/PDFs/rel_uncert.json'.format(dat_dir)

def main():

    main_obj = running_object(infile_xsec_mass,infile_num_unc,inpath_PDFs,infile_num_unc_PDFs)
    
    return

if __name__ == "__main__":
    main()

import os, sys, copy
import numpy as np
import uncertainties as unc
import iminuit
from scipy import stats
from matplotlib import pyplot as plt

import constants as cnst
import mass_convert as conv

# minuit settings
global_tolerance = 1E-6
global_strategy = 2
global_errordef = 1

plotdir = 'plots_running'


class ratio_object():

    def __init__(self,indir,scale_vars,ref_bin=2):
        
        self.ref_bin = ref_bin-1
        self.mass_values = np.load('{}/mass_results.npy'.format(indir))
        self.cov_masses = np.load('{}/mass_covariance.npy'.format(indir))
        self.indir = indir
        self.scale_vars = scale_vars
        self.scales = np.array([cnst.mu_1,cnst.mu_2,cnst.mu_3,cnst.mu_4])
        self.nBins = len(self.scales)
        self.masses_wunc = np.array(unc.correlated_values(self.mass_values,self.cov_masses))
        self.plotMasses()
        self.estimateRatios()
        self.scales_in_chisq = True
        self.estimateBestRunning()
        self.fitDynamicMassGeneration()

    def estimateRatios(self):

        self.ratios_wunc = np.delete(self.masses_wunc / self.masses_wunc[self.ref_bin],self.ref_bin)
        self.ratio_values = np.array([ratio.n for ratio in self.ratios_wunc])
        self.cov_ratios = np.array(unc.covariance_matrix(self.ratios_wunc))

        print ('\nfitted masses:')
        for m in self.masses_wunc:
            print ('{:.2f} GeV'.format(m))

        print ('\ncorrelations:')
        print (np.array(unc.correlation_matrix(self.masses_wunc)).round(2))
        

        scale_up = copy.deepcopy(self.ratio_values)
        scale_down = copy.deepcopy(self.ratio_values)
        self.scale_impacts_ratio = dict()
        for scales in self.scale_vars:
            m_scale = np.load('{}/mass_results_{}.npy'.format(self.indir,scales))
            r_scale = np.delete(m_scale/m_scale[self.ref_bin],self.ref_bin)
            self.scale_impacts_ratio[scales] = r_scale - self.ratio_values
            scale_up = np.maximum(scale_up,r_scale)
            scale_down = np.minimum(scale_down,r_scale)
        scale_up -= self.ratio_values
        scale_down = self.ratio_values-scale_down

        print ('\nfitted ratios:')
        for i,r in enumerate(self.ratios_wunc):
            print ('{:.3f} +/- {:.3f} (exp+PDF+num) +{:.3f} -{:.3f} (scale)'.format(r.n,r.s,scale_up[i],scale_down[i]))

        print ('\ncorrelations:')
        print (np.array(unc.correlation_matrix(self.ratios_wunc)).round(2))

        self.err_ratios_scale_up = scale_up
        self.err_ratios_scale_down = scale_down
        
        return

    def getTheoryRatio(self,scales):
        masses_evolved = np.array([conv.mtmu2mtmu(self.mass_values[self.ref_bin],self.scales[self.ref_bin],scale) for scale in scales])
        return masses_evolved/self.mass_values[self.ref_bin]

    def getRunningX(self,x,scales):
        return x*(self.getTheoryRatio(scales)-1)+1

    def computeChi2(self,x=1):
        if not self.scales_in_chisq:
            tot_cov = self.cov_ratios
        else:
            tot_cov = self.cov_ratios + self.getScaleCovarianceRatios()
        return np.matmul(self.ratio_values-self.getRunningX(x,np.delete(self.scales,self.ref_bin)),
                         np.matmul(np.linalg.inv(tot_cov),self.ratio_values-self.getRunningX(x,np.delete(self.scales,self.ref_bin))))

    def getScaleCovarianceRatios(self):
        for b in range(0,self.nBins-1):
            up = np.array([self.scale_impacts_ratio[scales][b] for scales in self.scale_vars if not 'down' in scales])
            down = np.array([self.scale_impacts_ratio[scales][b] for scales in self.scale_vars if not 'up' in scales])
            up *= up[0]/abs(up[0])
            down *= down[0]/abs(down[0])
            if not (up>0).all() or not (down>0).all():
                print('ERROR: cannot make good estimate of scale covariance: different signs in same-side variation')
                print('please look at the details...\n')
                sys.exit()
            if abs(self.scale_impacts_ratio['muRup_muFup'][b]) < abs(self.scale_impacts_ratio['muRup_muFnom'][b]) \
               or abs(self.scale_impacts_ratio['muRup_muFup'][b]) < abs(self.scale_impacts_ratio['muRnom_muFup'][b]):
                print('ERROR: cannot make good estimate of scale covariance: combined variation smaller than individual variations')
                print('please look at the details...\n')
                sys.exit()
            if abs(self.scale_impacts_ratio['muRdown_muFdown'][b]) < abs(self.scale_impacts_ratio['muRdown_muFnom'][b]) \
               or abs(self.scale_impacts_ratio['muRdown_muFdown'][b]) < abs(self.scale_impacts_ratio['muRnom_muFdown'][b]):
                print('ERROR: cannot make good estimate of scale covariance: combined variation smaller than individual variations')
                print('please look at the details...\n')
                sys.exit()
        signs = self.scale_impacts_ratio['muRup_muFup']/abs(self.scale_impacts_ratio['muRup_muFup'])
        values = np.maximum(abs(self.scale_impacts_ratio['muRup_muFup']),abs(self.scale_impacts_ratio['muRdown_muFdown'])) * signs
        # values = np.array([abs(self.scale_impacts_ratio['muRup_muFup'][b]) if self.ratio_values[b] < self.getTheoryRatio([self.scales[b]])[0] else abs(self.scale_impacts_ratio['muRdown_muFdown'][b]) \
        #                    for b in range(0,self.nBins-1)]) * signs
        return np.matmul(np.diag(values),np.matmul(np.ones((self.nBins-1,self.nBins-1)),np.diag(values)))

    
    def estimateBestRunning(self):

        ndf = self.ratio_values.shape[0]
        self.chi2_QCD = self.computeChi2(1)
        self.chi2_noRunning = self.computeChi2(0)
        self.prob_QCD = stats.chi2.sf(self.chi2_QCD,ndf)
        self.prob_noRunning = stats.chi2.sf(self.chi2_noRunning,ndf)

        print ('\nQCD running (x=1):')
        print('chi2 = {:.2f}, prob = {:.1f}%'.format(self.chi2_QCD,self.prob_QCD*100.))
        print ('\nno running (x=0):')
        print('chi2 = {:.2f}, prob = {:.1f}%'.format(self.chi2_noRunning,self.prob_noRunning*100.))
        print('excluded at {:.1f}% C.L.'.format((1-self.prob_noRunning)*100.))
        print()

        minuit = iminuit.Minuit(self.computeChi2, x=1)
        minuit.errordef=global_errordef
        minuit.strategy=global_strategy
        minuit.tol=global_tolerance
        minuit.migrad()
        
        self.xFit = unc.ufloat(minuit.values['x'], minuit.errors['x'])
        self.chi2_xFit = self.computeChi2(minuit.values['x'])
        self.prob_xFit = stats.chi2.sf(self.chi2_xFit,ndf-1)


        print('\nbest-fit x = {:.2f} +/- {:.2f}'.format(minuit.values['x'], minuit.errors['x']))
        print('chi2 = {:.2f}, prob = {:.1f}%'.format(self.chi2_xFit,self.prob_xFit*100.))
        print()
            
        self.producePlotRatio(scale_variations=False)
        self.producePlotRatio()
            
        return

    def producePlotRatio(self,scale_variations=True):

        err_ratios = np.array([ratio.s for ratio in self.ratios_wunc])

        if scale_variations:
            err_ratios_tot_up = (err_ratios**2+self.err_ratios_scale_up**2)**.5
            err_ratios_tot_down = (err_ratios**2+self.err_ratios_scale_down**2)**.5
            err_ratios_asymm = np.array(list(zip(err_ratios_tot_down, err_ratios_tot_up))).T

            scales = np.delete(self.scales,self.ref_bin)
            ratio_points_tot = plt.errorbar(scales, self.ratio_values, err_ratios_asymm, fmt = '.',ecolor='C0',color='C0')
            ratio_points_tot.set_label('extracted $m_\mathrm{t}(\mu_\mathrm{k}/2) ~ / ~m_\mathrm{t}(\mu_\mathrm{ref})$ at NNLO')

        mu_scan = np.arange(self.scales[0],self.scales[-1],1)
        curve, = plt.plot(mu_scan,self.getTheoryRatio(mu_scan),color='C1')
        curve.set_label('QCD RGE solution at {} loops, {} flavours'.format(cnst.nloops,cnst.nflav))

        # ratio_points = plt.errorbar(self.scales, np.insert(self.ratio_values,self.ref_bin,1), np.insert(err_ratios,self.ref_bin,0), fmt='o',capsize=2,color='C0',ecolor='C0')
        ratio_points = plt.errorbar(np.delete(self.scales,self.ref_bin), self.ratio_values, err_ratios, fmt='o',capsize=2,color='C0',ecolor='C0')
        ref_points = plt.plot(self.scales[self.ref_bin],1,marker='o',color='none',markerfacecolor='none',markeredgecolor='C0',label='$\mu_\mathrm{ref}$'+' = {:.0f} GeV'.format(self.scales[self.ref_bin]))
        
        if not scale_variations:
            ratio_points.set_label('extracted $m_\mathrm{t}(\mu_\mathrm{k}/2) ~ / ~m_\mathrm{t}(\mu_\mathrm{ref})$ at NNLO')

        handles, labels = plt.gca().get_legend_handles_labels()
        order = [0,2,1]
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='lower left') 

        # plt.legend(loc='lower left')
        plt.xlabel('energy scale $\mu_\mathrm{m} = \mu_\mathrm{k}/2$')
        plt.ylabel('$m_\mathrm{t}(\mu_\mathrm{m}) ~ / ~ m_\mathrm{t}(\mu_\mathrm{ref})$')
        plt.title('running of $m_\mathrm{t}$ at NNLO in QCD',loc='right')
        plt.title('Preliminary',loc='left')

        if scale_variations:
            plt.text(200,.87,'data-theory reduced $\chi^2$ = {:.2f}'.format(self.chi2_QCD/(self.nBins-1)))
            plt.text(200,.855,'p-value for QCD RGE = {:.2f}'.format(self.prob_QCD))

        plt.text(385,1.04, '$Matrix$ calculation at NNLO')
        plt.text(385,1.025, 'CMS data at $\sqrt{s} = 13~\mathrm{TeV}$')
        plt.text(385,1.01,'ABMP16_5_nnlo PDF set')
        # plt.text(385,.99,'$\mu_\mathrm{ref}$'+' = {:.0f} GeV'.format(self.scales[self.ref_bin]))
        
        os.makedirs(plotdir,exist_ok = True)

        if scale_variations:
            plt.savefig('{}/running_scale.pdf'.format(plotdir))
            plt.savefig('{}/running_scale.png'.format(plotdir))
        else:
            plt.savefig('{}/running.pdf'.format(plotdir))
            plt.savefig('{}/running.png'.format(plotdir))

        plt.close()
        
        return

    def dynmass(self,scale,mtmt,Lambda):
        Lambda *= 1000. #in TeV
        return mtmt/(1-(mtmt/Lambda)**2)*(1-(scale/Lambda)**2)


    def chi2dynmass(self,mtmt,Lambda):
        return np.matmul(self.mass_values-self.dynmass(self.scales, mtmt, Lambda), np.matmul(np.linalg.inv(self.cov_masses),self.mass_values-self.dynmass(self.scales, mtmt, Lambda)))

    
    def fitDynamicMassGeneration(self):

        ndf = self.mass_values.shape[0]-2

        minuit = iminuit.Minuit(self.chi2dynmass, mtmt=170, Lambda=1)
        minuit.errordef=global_errordef
        minuit.strategy=global_strategy
        minuit.tol=global_tolerance
        minuit.migrad()
        
        mtmt = minuit.values['mtmt']
        err_mtmt = minuit.errors['mtmt']
        Lambda = minuit.values['Lambda']
        err_Lambda = minuit.errors['Lambda']

        self.mtmt_dynmass = unc.ufloat(mtmt,err_mtmt)
        self.Lambda_dynmass = unc.ufloat(Lambda,err_Lambda)
        

        print('\nmt(mt) = {:.2f} +/- {:.2f} GeV'.format(mtmt,err_mtmt))
        print('Lambda = {:.1f} +/- {:.1f} TeV'.format(Lambda,err_Lambda))
        print()
        self.chi2_dynmass = self.chi2dynmass(mtmt,Lambda)
        print('chi2 = {:.2f}, prob = {:.1f}%'.format(self.chi2_dynmass,stats.chi2.sf(self.chi2_dynmass,ndf)*100.))
        print()

        self.producePlotDynamicMass()
            
        return

    def producePlotDynamicMass(self):

        err_mass = np.array([mass.s for mass in self.masses_wunc])
        mass_points = plt.errorbar(self.scales, self.mass_values, err_mass, fmt='o')
        mass_points.set_label('extracted masses $m_\mathrm{t}(\mu_{k})$ at NNLO')

        mu_scan = np.arange(self.scales[0],self.scales[-1],1)
        curve, = plt.plot(mu_scan,self.dynmass(mu_scan,self.mtmt_dynmass.n,self.Lambda_dynmass.n))
        curve.set_label('technicolor: $m_\mathrm{t}(m_\mathrm{t})}$ = '+'{:.1f}'.format(self.mtmt_dynmass.n) +' GeV, $\Lambda$ = '+'{:.1f}'.format(self.Lambda_dynmass.n)+' TeV')

        plt.legend(loc='lower left')
        plt.xlabel('energy scale $\mu = m_\mathrm{t\overline{t}}/2$')
        plt.ylabel('NNLO running mass $m_\mathrm{t}(\mu$)')
        plt.title('dynamic mass generation')

        plt.text(215,142,'best-fit values:')
        plt.text(215,140,'$\Lambda = {:.1f} \pm {:.1f}$ TeV'.format(self.Lambda_dynmass.n,self.Lambda_dynmass.s), fontsize=11)
        plt.text(215,138,'$m_\mathrm{t}(m_\mathrm{t})'+' = {:.1f} \pm {:.1f}$ GeV'.format(self.mtmt_dynmass.n,self.mtmt_dynmass.s), fontsize=11)

        os.makedirs(plotdir,exist_ok = True)

        plt.savefig('{}/dynmass.pdf'.format(plotdir))
        plt.savefig('{}/dynmass.png'.format(plotdir))
        plt.close()
        
        return

    def plotMasses(self):
        err_mass = np.array([mass.s for mass in self.masses_wunc])
        mass_points = plt.errorbar(self.scales, self.mass_values, err_mass, fmt='o')
        mass_points.set_label('extracted $m_\mathrm{t}(\mu_{k})$ at NNLO (from differential)')

        mass_incl = plt.errorbar(cnst.mtmt,cnst.mtmt,cnst.mtmt_err,fmt='o')
        mass_incl.set_label('extracted $m_\mathrm{t}(m_\mathrm{t})$ at NNLO (from inclusive)')

        mu_scan = np.arange(cnst.mtmt,self.scales[-1],1)
        masses_evolved = np.array([conv.mtmt2mtmu(cnst.mtmt,scale) for scale in mu_scan])
        masses_evolved_err = masses_evolved/cnst.mtmt*cnst.mtmt_err

        band = plt.fill_between(mu_scan,masses_evolved-masses_evolved_err,masses_evolved+masses_evolved_err,facecolor='yellow')
        band.set_label('evolved uncertainty: nloops = {}, nflav = {}'.format(cnst.nloops,cnst.nflav))

        plt.text(cnst.mtmt,self.mass_values[-1]+3,'ABMP16_5_nnlo PDF set')
        
        plt.legend(loc='lower left')
        plt.xlabel('energy scale $\mu = m_\mathrm{t\overline{t}}/2$')
        plt.ylabel('NNLO running mass $m_\mathrm{t}(\mu$)')
        plt.title('running $m_\mathrm{t}(\mu_m)$ at NNLO')

        os.makedirs(plotdir,exist_ok = True)
        
        plt.savefig('{}/masses.pdf'.format(plotdir))
        plt.savefig('{}/masses.png'.format(plotdir))
        plt.close()

        return

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

    def __init__(self,masses,covariance,ref_bin=2):
        
        self.ref_bin = ref_bin-1
        self.mass_values = masses
        self.cov_masses = covariance
        self.scales = np.array([cnst.mu_1,cnst.mu_2,cnst.mu_3,cnst.mu_4])
        self.nBins = len(self.scales)
        self.estimateRatios()
        self.estimateBestRunning()
        self.fitDynamicMassGeneration()

    def estimateRatios(self):

        self.masses_wunc = np.array(unc.correlated_values(self.mass_values,self.cov_masses))
        self.ratios_wunc = np.delete(self.masses_wunc / self.masses_wunc[self.ref_bin],self.ref_bin)

        self.ratio_values = np.array([ratio.n for ratio in self.ratios_wunc])
        self.cov_ratios = np.array(unc.covariance_matrix(self.ratios_wunc))

        print ('\nfitted masses:\n{}\n'.format(self.masses_wunc))
        print ('correlations:')
        print (np.array(unc.correlation_matrix(self.masses_wunc)))
        
        print ('\nfitted ratios:\n{}\n'.format(self.ratios_wunc))
        print ('correlations:')
        print (np.array(unc.correlation_matrix(self.ratios_wunc)))
        
        return

    def getTheoryRatio(self,scales):
        masses_evolved = np.array([conv.mtmu2mtmu(self.mass_values[self.ref_bin],self.scales[self.ref_bin],scale) for scale in scales])
        return masses_evolved/self.mass_values[self.ref_bin]

    def getRunningX(self,x,scales):
        return x*(self.getTheoryRatio(scales)-1)+1

    def computeChi2(self,x=1):
        return np.matmul(self.ratio_values-self.getRunningX(x,np.delete(self.scales,self.ref_bin)),
                         np.matmul(np.linalg.inv(self.cov_ratios),self.ratio_values-self.getRunningX(x,np.delete(self.scales,self.ref_bin))))

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
            
        self.producePlotRatio()
            
        return

    def producePlotRatio(self):
        err_ratios = np.array([ratio.s for ratio in self.ratios_wunc])
        ratio_points = plt.errorbar(self.scales, np.insert(self.ratio_values,self.ref_bin,1), np.insert(err_ratios,self.ref_bin,0), fmt='o')
        ratio_points.set_label('exctracted running $m_\mathrm{t}(\mu_{k})/m_\mathrm{t}(\mu_\mathrm{ref})$ at NNLO')

        mu_scan = np.arange(self.scales[0],self.scales[-1],1)
        curve, = plt.plot(mu_scan,self.getTheoryRatio(mu_scan))
        curve.set_label('QCD running: nloops = {}, nflav = {}'.format(cnst.nloops,cnst.nflav))

        plt.legend(loc='lower left')
        plt.xlabel('energy scale $\mu = m_\mathrm{t\overline{t}}/2$')
        plt.ylabel('running $m_\mathrm{t}(\mu) / m_\mathrm{t}(\mu_\mathrm{ref})$')
        plt.title('QCD running at NNLO')

        plt.text(210,.9,'data/theory $\chi^2/ndf$ = {:.1f}'.format(self.chi2_QCD/(self.nBins-1)))
        plt.text(210,.885,'probability = {:.1f}%'.format(self.prob_QCD*100.))
        plt.text(390,1.03,'ABMP16_5_nnlo PDF set')
        plt.text(390,1.015,'$\mu_0 = \mu_\mathrm{ref}$'+' = {:.0f} GeV'.format(self.scales[self.ref_bin]))
        
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)

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
        mass_points.set_label('exctracted masses $m_\mathrm{t}(\mu_{k})$ at NNLO')

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

        if not os.path.exists(plotdir):
            os.makedirs(plotdir)

        plt.savefig('{}/dynmass.pdf'.format(plotdir))
        plt.savefig('{}/dynmass.png'.format(plotdir))
        plt.close()
        
        return

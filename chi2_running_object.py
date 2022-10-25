
import os, sys, json, math, iminuit
import numpy as np
from datetime import datetime
import uncertainties as unc
import variables as var
from matplotlib import pyplot as plt
import pickle
import copy

import constants as cnst
import mass_convert as conv

import matplotlib.ticker as plticker

from ratio_object import setstyles
setstyles()

plotdir = 'plots_xsec'

def save_object(obj, filename):
    if os.path.exists(filename):
        print('\n**********\nWARNING! overwritten file {}\n**********\n'.format(filename))
    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def isGoodFit(minuit,printout=False):

    if printout:
        print('\n-> checking fit')
        print('is valid minumum:',minuit.valid)
        print('has accurate covariance:',minuit.accurate)

    return minuit.valid and minuit.accurate
    
        
class running_object():

    def __init__(self,infile_xsec_mass,infile_num_unc,inpath_PDFs,infile_num_unc_PDFs,output_dir='.',PDFsFromNLO=False,mtmt_only=False,normalised=False):
        
        if normalised and not mtmt_only:
            print('ERROR: normalised fit only for inclusive mt(mt)')
            sys.exit()

        self.od = output_dir
        self.mtmt_only = mtmt_only
        self.normalised = normalised
        self.PDFsFromNLO = PDFsFromNLO
        self.exp_xsec, self.exp_err, self.corr_matrix = self.getExperimentalResults()
        self.exp_cov = np.matmul(np.diag(self.exp_err),np.matmul(self.corr_matrix,np.diag(self.exp_err)))
        self.nBins = self.exp_xsec.shape[0]
        self.extr_cov = self.getExtrapolationImpacts()
        self.d_xsec_vs_mass, self.d_xsec_scales_all = self.readAllXsecAndScaleVariations(infile_xsec_mass)
        self.d_numunc = self.readNumericalUncertJSON(infile_num_unc)
        self.scales = np.array([cnst.mu_1,cnst.mu_2,cnst.mu_3,cnst.mu_4])
        self.nPDFs = 30
        self.d_PDFunc = self.readPDFuncertainties(inpath_PDFs)
        self.d_PDFunc_nlo = self.readPDFuncertaintiesNLO('NNLO_dat/nnloPDFs.json')
        self.d_numunc_PDFs = self.readNumericalUncertPDFsJSON(infile_num_unc_PDFs)
        self.addCentralPDFtoList(onlyNumUnc=True)
        self.defineUsefulVariablesForFit()
        self.drawXsecVsMassNominal()
        
    def getExperimentalResults(self):
        exp_xsec = np.array([var.xsec_1,var.xsec_2,var.xsec_3,var.xsec_4])
        exp_err = np.array([var.err_xsec_1_up/2.+var.err_xsec_1_down/2.,var.err_xsec_2_up/2.+var.err_xsec_2_down/2.,
                            var.err_xsec_3_up/2.+var.err_xsec_3_down/2.,var.err_xsec_4_up/2.+var.err_xsec_4_down/2])*exp_xsec/100.
        corr_matrix = np.array([[1,var.corr_1_2,var.corr_1_3,var.corr_4_1],
                                [var.corr_1_2,1,var.corr_3_2,var.corr_4_2],
                                [var.corr_1_3,var.corr_3_2,1,var.corr_4_3],
                                [var.corr_4_1,var.corr_4_2,var.corr_4_3,1]])
        return exp_xsec, exp_err, corr_matrix

    def getExtrapolationImpacts(self):

        extr_cov = np.zeros((self.nBins,self.nBins))
        
        for i, name in enumerate(var.extr_name):
            up = np.array([var.extr_1_up[i],var.extr_2_up[i],var.extr_3_up[i],var.extr_4_up[i]])*self.exp_xsec/100.
            down = np.array([var.extr_1_down[i],var.extr_2_down[i],var.extr_3_down[i],var.extr_4_down[i]])*self.exp_xsec/100.
            conserv = np.maximum(abs(up),abs(down)) * (up/abs(up))
            cov = np.zeros((self.nBins,self.nBins))
            for i in range(0,self.nBins):
                for j in range(0,self.nBins):
                    cov[i][j] = conserv[i]*conserv[j]
            extr_cov += cov
        return extr_cov
    
    def readAllXsecVsMass(self,filename):
        bin_low = []
        f = open(filename)
        lines = f.read().splitlines()
        for l in lines:
            if l == '': continue
            if l.replace(' ','').startswith('#'): continue
            low = int(l.split()[0])
            if not low in bin_low:
                bin_low.append(low)
        bin_low.sort()
        d = dict()
        d_up = dict()
        d_down = dict()
        for i, low in enumerate(bin_low):
            m_xsec = dict()
            m_xsec_up = dict()
            m_xsec_down = dict()
            for l in lines:
                if l == '': continue
                if l.replace(' ','').startswith('#'): continue
                if int(l.split()[0]) != low: continue
                m_xsec[l.split()[-1]] = float(l.split()[2])
                m_xsec_down[l.split()[-1]] = float(l.split()[3])
                m_xsec_up[l.split()[-1]] = float(l.split()[4])
            d[i]=m_xsec
            d_up[i]=m_xsec_up
            d_down[i]=m_xsec_down
        return d, d_up, d_down

    def readAllXsecAndScaleVariations(self,filename):
        bin_low = []
        f = open(filename)
        lines = f.read().splitlines()
        for l in lines:
            if l == '': continue
            if l.replace(' ','').startswith('#'): continue
            low = int(l.split()[0])
            if not low in bin_low:
                bin_low.append(low)
        bin_low.sort()
        d_nom = dict()
        d_scale = dict()
        for i, low in enumerate(bin_low):
            m_xsec_nom = dict()
            m_xsec_scale = dict()
            for l in lines:
                if l == '': continue
                if l.replace(' ','').startswith('#'): continue
                if int(l.split()[0]) != low: continue
                central = float(l.split()[5])
                m_xsec_nom[l.split()[-1]] = central
                dd = dict()
                dd['muRdown_muFdown'] = float(l.split()[2]) / central
                dd['muRdown_muFnom'] = float(l.split()[3]) / central
                dd['muRup_muFnom'] = float(l.split()[4]) / central
                dd['muRnom_muFup'] = float(l.split()[6]) / central
                dd['muRnom_muFdown'] = float(l.split()[7]) / central
                dd['muRup_muFup'] = float(l.split()[8]) / central
                m_xsec_scale[l.split()[-1]] = dd
                
            d_nom[i]=m_xsec_nom
            d_scale[i]=m_xsec_scale

        self.scale_vars = []
        for muR in ['nom','up','down']:
            for muF in ['nom','up','down']:
                if (muR=='up' and muF=='down') or (muR=='down' and muF=='up') or (muR=='nom' and muF=='nom'):
                    continue
                self.scale_vars.append('muR{}_muF{}'.format(muR,muF))

        return d_nom, d_scale

    def readPDFuncertainties(self, inpah_PDFs):
        d_pdf = dict()
        for pdf in range(0,self.nPDFs):
            d_pdf[pdf] = self.readPDFuncertainty(inpah_PDFs,pdf)
        return d_pdf

    def readPDFuncertaintiesNLO(self, infile):
        d = json.load(open(infile,'r'))
        for k in d.keys():
            d[k] = np.delete(np.array(d[k])/d[k][0],0)
        return d
        
    def readPDFuncertainty(self,inpath_PDFs,pdf):
        bin_low = []
        f = open(inpath_PDFs.format(pdf))
        lines = f.read().splitlines()
        for l in lines:
            if l == '': continue
            if l.replace(' ','').startswith('#'): continue
            low = int(l.split()[0])
            if not low in bin_low:
                bin_low.append(low)
        bin_low.sort()
        d = dict()
        for i, low in enumerate(bin_low):
            m_xsec = dict()
            for l in lines:
                if l == '': continue
                if l.replace(' ','').startswith('#'): continue
                if int(l.split()[0]) != low: continue
                m_xsec[l.split()[-1]] = float(l.split()[2])
            d[i]=m_xsec
        return d

    def addCentralPDFtoList(self,onlyNumUnc=False):
        d = self.d_PDFunc[0]
        for i in d.keys():
            for mass in d[i].keys():
                if not onlyNumUnc:
                    self.d_xsec_vs_mass[i][mass]=d[i][mass]
                self.d_numunc[i][mass]=self.d_numunc_PDFs[0][i][float(mass)]
        return
    
    def readNumericalUncertJSON(self,filename):
        f = open(filename)
        data = json.load(f)
        d = dict()
        for b in range(0,self.nBins):
            dd = dict()
            for i, bb in enumerate(data['bin']):
                if bb-1 != b: continue
                dd[str(data['mass'][i])] = data['relunc'][i]
            d[b]=dd
        return d
    
    def readNumericalUncertPDFsJSON(self,filename):
        d = dict()
        for pdf in range(0,self.nPDFs):
            dd = dict()
            for b in range(0,self.nBins):
                f = open(filename.format(b+1))
                data = json.load(f)
                f.close()
                ddd = dict()
                for i, bb in enumerate(data['bin']):
                    if bb-1 == b and data['pdf'][i] == str(pdf):
                        ddd[data['mass'][i]] = data['relunc'][i]
                        break
                dd[b]=ddd
            d[pdf]=dd
        return d
        
    def fitQuadratic(self,x,a,b,c):
        return a*x*x + b*x + c

    def chi2QuadraticFit(self,a,b,c):
        res = self.xsec_values_for_fit - self.fitQuadratic(self.masses_for_fit,a,b,c)
        return np.matmul(res,np.matmul(np.linalg.inv(self.xsec_cov_for_fit),res))
    

    def getDependencies(self,MCstat_nuisances_bin,MCstat_nuisances_PDFs_bin,nuisances_PDFs):
        a_s = np.empty(self.nBins)
        b_s = np.empty(self.nBins)
        c_s = np.empty(self.nBins)
        for mbin in range(0,self.nBins):
            dep_params = self.getDependencyBin(mbin,MCstat_nuisances_bin[mbin],MCstat_nuisances_PDFs_bin[mbin],nuisances_PDFs)
            a_s[mbin]=dep_params[0]
            b_s[mbin]=dep_params[1]
            c_s[mbin]=dep_params[2]
        return a_s, b_s, c_s

    def getDependencyBin(self,mbin,MCstat_nuisances_bin,MCstat_nuisances_PDFs_bin,nuisances_PDFs):
        xsec = self.xsec_bin[mbin] * (1+MCstat_nuisances_bin*self.rel_err_xsec_bin[mbin]) # MC stat correction
        cov = np.matmul(np.diag(self.rel_err_xsec_bin[mbin]*xsec),np.matmul(np.diag(np.ones(len(xsec))),np.diag(self.rel_err_xsec_bin[mbin]*xsec)))
        xsec_corr_values = np.array(unc.correlated_values(xsec,cov))

        if not self.PDFsFromNLO:
            xsec_PDFs = self.xsec_PDFs_bin[mbin]*(1+MCstat_nuisances_PDFs_bin*self.rel_err_xsec_PDFs_bin[mbin]) # MC stat correction PDF
            xsec_PDFs_wunc = np.array([unc.ufloat(xsec_PDFs[i],xsec_PDFs[i]*self.rel_err_xsec_PDFs_bin[mbin][i]) for i in range(0,len(self.rel_err_xsec_PDFs_bin[mbin]))])
            rel_variations = nuisances_PDFs * (xsec_PDFs_wunc/xsec_corr_values[self.ref_mass_bin[mbin]]-1) +1 # relative PDF variations
        else:
            rel_variations = nuisances_PDFs * (self.d_PDFunc_nlo[str(mbin+1)]-1) + 1
        xsec_corr_values = xsec_corr_values * np.prod(rel_variations)

        self.masses_for_fit = self.masses_bin[mbin]
        self.xsec_cov_for_fit = np.array(unc.covariance_matrix(xsec_corr_values))
        self.xsec_values_for_fit = np.array([xsec.n for xsec in xsec_corr_values])

        self.minuit_dep.migrad()

        del self.masses_for_fit
        del self.xsec_cov_for_fit
        del self.xsec_values_for_fit
        
        return np.array(self.minuit_dep.values)
        
    def globalChi2(self,params):
        masses = params[0:self.n_massParams]
        if self.mtmt_only:
            masses = np.array([conv.mtmt2mtmu(masses[0],scale) for scale in self.scales])

        all_nuisances = params[self.n_massParams:]
        if not self.PDFsFromNLO:
            MCstat_nuisances_bin = [params[self.n_massParams+self.nMassPoints_integrated[i-1]:self.n_massParams+self.nMassPoints_integrated[i]] for i in range(0,self.nBins)]
            MCstat_nuisances_PDFs_bin = [params[self.n_massParams+sum(self.nMassPoints)+(self.nPDFs-1)*i:self.n_massParams+sum(self.nMassPoints)+(self.nPDFs-1)*(i+1)] for i in range(0,self.nBins)]
            nuisances_PDFs = params[self.n_massParams+sum(self.nMassPoints)+(self.nPDFs-1)*self.nBins:]
        else:
            nuisances_PDFs = params[self.n_massParams:self.n_massParams+self.nPDFs-1]
            MCstat_nuisances_bin = [params[self.n_massParams+self.nPDFs-1+self.nMassPoints_integrated[i-1]:self.n_massParams+self.nPDFs-1+self.nMassPoints_integrated[i]] for i in range(0,self.nBins)]
            MCstat_nuisances_PDFs_bin = [[] for _ in range(0,self.nBins)]

        a_s,b_s,c_s = self.getDependencies(MCstat_nuisances_bin,MCstat_nuisances_PDFs_bin,nuisances_PDFs)
        if not self.normalised:
            res_array = self.exp_xsec - self.fitQuadratic(masses,a_s,b_s,c_s)
        else:
            pred = self.fitQuadratic(masses,a_s,b_s,c_s)
            pred_norm = np.delete(self.fitQuadratic(masses,a_s,b_s,c_s)/np.sum(self.fitQuadratic(masses,a_s,b_s,c_s)),-1)
            res_array = self.exp_norm_xsec - pred_norm
        chi2 = np.matmul(res_array,np.matmul(np.linalg.inv(self.cov_tot),res_array))
        return chi2 + sum(all_nuisances**2)

    def defineUsefulVariablesForFit(self):
        
        self.n_massParams = self.nBins if not self.mtmt_only else 1
        self.masses_bin = [np.array([float(m) for m in self.d_xsec_vs_mass[mbin]]) for mbin in range(0,self.nBins)]
        for m in self.masses_bin:
            m.sort()
        self.rel_err_xsec_bin = [np.array([self.d_numunc[mbin][str(m)] for m in self.masses_bin[mbin]]) for mbin in range(0,self.nBins)]
        self.xsec_bin = [np.array([self.d_xsec_vs_mass[mbin][str(m)] for m in self.masses_bin[mbin]]) for mbin in range(0,self.nBins)]

        self.m_ref_bin = [list(self.d_PDFunc[1][mbin].keys())[0] for mbin in range(0,self.nBins)]
        self.xsec_PDFs_bin = [np.array([float(self.d_PDFunc[pdf][mbin][self.m_ref_bin[mbin]]) for pdf in range(1,self.nPDFs)]) for mbin in range(0,self.nBins)]
        self.rel_err_xsec_PDFs_bin = [np.array([float(self.d_numunc_PDFs[pdf][mbin][float(self.m_ref_bin[mbin])]) for pdf in range(1,self.nPDFs)]) for mbin in range(0,self.nBins)]
        self.ref_mass_bin = [list(self.masses_bin[mbin]).index(float(self.m_ref_bin[mbin])) for mbin in range(0,self.nBins)]

        self.minuit_dep = iminuit.Minuit(self.chi2QuadraticFit,a=0,b=-1,c=10)
        self.minuit_dep.errordef=1
        self.minuit_dep.strategy=2
        self.minuit_dep.tol=1E-6

        self.cov_tot = self.exp_cov + self.extr_cov

        if self.normalised:
            exp_xsec_wunc = np.array(unc.correlated_values(self.exp_xsec,self.cov_tot))
            exp_norm_xsec_wunc = np.delete(exp_xsec_wunc/np.sum(exp_xsec_wunc),-1)
            self.cov_tot = np.array(unc.covariance_matrix(exp_norm_xsec_wunc))
            self.exp_norm_xsec = np.array([xsec.n for xsec in exp_norm_xsec_wunc])
            
        return


    def drawXsecVsMassNominal(self):

        if not os.path.exists(plotdir):
            os.makedirs(plotdir)

        for mbin in range(0,self.nBins):
            
            self.masses_for_fit = self.masses_bin[mbin]
            self.xsec_values_for_fit = self.xsec_bin[mbin]
            self.xsec_cov_for_fit = np.matmul(np.diag(self.rel_err_xsec_bin[mbin]*self.xsec_values_for_fit),
                                              np.matmul(np.diag(np.ones(len(self.xsec_values_for_fit))),np.diag(self.rel_err_xsec_bin[mbin]*self.xsec_values_for_fit)))

            self.minuit_dep.migrad()

            plot = plt.errorbar(self.masses_for_fit,self.xsec_values_for_fit,self.rel_err_xsec_bin[mbin]*self.xsec_values_for_fit,linestyle='None',marker='.')
            plot.set_label('NNLO calculation')
            
            m_scan = np.arange(self.masses_for_fit[0],self.masses_for_fit[-1],0.001)
            curve, = plt.plot(m_scan,self.fitQuadratic(m_scan,self.minuit_dep.values[0],self.minuit_dep.values[1],self.minuit_dep.values[2]))
            curve.set_label('Quadratic interpolation')

            exp, = plt.plot(m_scan,np.array([self.exp_xsec[mbin] for _ in m_scan]))
            exp.set_label('Measured cross section')
            
            band = plt.fill_between(m_scan,self.exp_xsec[mbin]-self.exp_err[mbin],self.exp_xsec[mbin]+self.exp_err[mbin],facecolor='yellow')
            band.set_label('Experimental uncertainty')
            
            # plt.title('Preliminary',loc='left')
            plt.title('Measured and calculated $\sigma_\mathrm{t\overline{t}}'+'^{('+str(mbin+1)+')}$',loc='right')
            plt.xlabel('$m_\mathrm{t}'+'(\mu_{}/2)$ [GeV]'.format(mbin+1))
            plt.ylabel('$\sigma_\mathrm{t\overline{t}}'+'^{('+str(mbin+1)+')}$ [pb]')

            if mbin == self.nBins-1: #hack
                plt.ylim(plt.ylim()[0],plt.ylim()[1]*1.05)            
                
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [3,0,1,2]
            plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper right') 

            step =  self.xsec_values_for_fit[0]-self.xsec_values_for_fit[-1]
            step/=15
            offset = self.xsec_values_for_fit[-1] + step/3

            if mbin == 2: #hacks
                offset -= 1
                loc = plticker.MultipleLocator(base=3.0)
                plt.gca().xaxis.set_major_locator(loc)

            plt.text(self.masses_for_fit[0],offset+2*step, 'NNLO calculation: JHEP 08 (2020) 027')
            plt.text(self.masses_for_fit[0],offset+step, 'CMS data at $\sqrt{s} = 13~\mathrm{TeV}$')
            plt.text(self.masses_for_fit[0],offset,'ABMP16_5_nnlo PDF set')
            
            plt.savefig('{}/xsec_mass_bin_{}.pdf'.format(plotdir,mbin+1))
            plt.savefig('{}/xsec_mass_bin_{}.png'.format(plotdir,mbin+1))
            plt.close()

            del self.masses_for_fit
            del self.xsec_cov_for_fit
            del self.xsec_values_for_fit
        
        return

    def doScaleVariations(self):
        print('\n-> performing scale variations...\n')
        for scales in self.scale_vars:
            self.doNominalFitScales(scales)
        print()
        return
        
    def createMinuitObject(self):

        params = np.ones(self.n_massParams)
        params *= 160 # initialise masses
        self.nMassPoints = [len(self.d_numunc[i].keys()) for i in range(0,self.nBins)]
        self.nMassPoints.append(0) # useful in globalChi2

        self.nMassPoints_integrated = copy.deepcopy(self.nMassPoints)
        for i in range(0,self.nBins):
            self.nMassPoints_integrated[i] += self.nMassPoints_integrated[i-1]

        if not self.PDFsFromNLO:
            all_nuisances = np.zeros(sum(self.nMassPoints)+(self.nPDFs-1)*(self.nBins+1)) # MC stat parameters (nominal+PDFs) and PDFs
        else:
            all_nuisances = np.zeros(sum(self.nMassPoints)+(self.nPDFs-1)) # MC stat parameters (nominal+PDFs) and PDFs

        params = np.append(params,all_nuisances)
        
        minuit = iminuit.Minuit(self.globalChi2,params)
        minuit.errordef=1

        return minuit,params

    def doNominalFitScales(self,scales='muRnom_muFnom'):
        print('fit for scale variation {}'.format(scales))
        xsec_bin_orig = copy.deepcopy(self.xsec_bin)
        
        for mbin in range(0,self.nBins):
            scale_var = np.array([self.d_xsec_scales_all[mbin][str(m)][scales] for m in self.masses_bin[mbin]]) if scales != 'muRnom_muFnom' else np.ones(len(self.masses_bin[mbin]))
            self.xsec_bin[mbin] *= scale_var

        minuit, params = self.createMinuitObject()
            
        with open('{}/minuit_object.pkl'.format(self.od), 'rb') as inp:
            minuit_orig = pickle.load(inp)
            minuit.values = copy.deepcopy(minuit_orig.values)

        minuit.strategy = minuit_orig._strategy.strategy
        minuit.tol = minuit_orig.tol

        minuit.fixed = [False if i<self.n_massParams else True for i, _ in enumerate(params)]
        # minuit.migrad()
        self.fitAndCheck(minuit,printout=True)

        par = np.array(minuit.values)
        np.save('{}/mass_results_{}'.format(self.od,scales),par[:self.n_massParams])
        
        self.xsec_bin = copy.deepcopy(xsec_bin_orig)
        return

    def estimatePDFexternalised(self):
        m = np.array([self.doNominalFitPDF(i) for i in range(0,self.nPDFs)])
        return np.sum((m - m[0])**2)**.5
        
    def doNominalFitPDF(self,pdf=0):
        print('fit for PDF variation {}'.format(pdf))
        xsec_bin_orig = copy.deepcopy(self.xsec_bin)

        #m_ref_bin = [list(self.d_PDFunc[1][mbin].keys())[0] for mbin in range(0,self.nBins)]
        #self.xsec_PDFs_bin = [np.array([float(self.d_PDFunc[pdf][mbin][m_ref_bin[mbin]]) for pdf in range(1,self.nPDFs)]) for mbin in range(0,self.nBins)]

        for mbin in range(0,self.nBins):
            if pdf>0:
                self.xsec_bin[mbin] *= self.xsec_PDFs_bin[mbin][pdf-1]/self.xsec_bin[mbin][list(self.masses_bin[mbin]).index(float(self.m_ref_bin[mbin]))]
            
        minuit, params = self.createMinuitObject()
            
        with open('{}/minuit_object.pkl'.format(self.od), 'rb') as inp:
            minuit_orig = pickle.load(inp)
            minuit.values = copy.deepcopy(minuit_orig.values)

        minuit.strategy = minuit_orig._strategy.strategy
        minuit.tol = minuit_orig.tol

        minuit.fixed = [False if i<self.n_massParams else True for i, _ in enumerate(params)]
        # minuit.migrad()
        self.fitAndCheck(minuit)

        self.xsec_bin = copy.deepcopy(xsec_bin_orig)
        return minuit.values[0]

    
    def fitAndCheck(self,minuit,printout=False,hesse=False):
        if hesse:
            minuit.hesse()
        else:
            minuit.simplex()
            minuit.migrad()
        if printout:
            print(minuit.values[0:self.n_massParams])
            print(minuit.errors[0:self.n_massParams])
        return isGoodFit(minuit,printout)
    
    def doFullFit(self):

        minuit, params = self.createMinuitObject()

        minuit.fixed = [False if i<self.n_massParams else True for i, _ in enumerate(params)]
        if not self.mtmt_only:
            minuit.limits = [(-1*math.inf,math.inf) if i<self.n_massParams else (-1,1) for i, _ in enumerate(params)]
        
        pre = datetime.now()
        
        # minuit.migrad() # do first fit (only masses)
        self.fitAndCheck(minuit,printout=True)
        
        print('\nfirst fit took {}\n'.format(datetime.now()-pre))
        
        minuit.fixed = [False for _ in params]
        
        if not self.mtmt_only:
            minuit.limits = [(-1*math.inf,math.inf) if i<self.n_massParams else (-1,1) for i, _ in enumerate(params)]
            minuit.limits = [(minuit.values[i]-.3*minuit.errors[i],minuit.values[i]+.3*minuit.errors[i]) if i<self.n_massParams else (-.3,.3) for i, _ in enumerate(params)]
        else:
            minuit.strategy = 0
        # minuit.migrad() # second fit, with constraints and all nuisances
        self.fitAndCheck(minuit,printout=True)

        print('second fit took {}\n'.format(datetime.now()-pre))

        if not self.mtmt_only:

            pre_RP = datetime.now()

            minuit.strategy=2
            minuit.limits = [(-1*math.inf,math.inf) for _ in params]
            # minuit.migrad() # fit with all parameters
            self.fitAndCheck(minuit,printout=True)
            
            print('full fit (reduced precision) took {}\n'.format(datetime.now()-pre_RP))

            if not self.PDFsFromNLO:
                odRP = self.od+'_RP' # reduced precision
                if not os.path.exists(odRP):
                    os.makedirs(odRP)

                cov = np.array(minuit.covariance)
                par = np.array(minuit.values)
                err = np.array(minuit.errors)

                np.save('{}/mass_results'.format(odRP),par[:self.n_massParams])
                if not self.mtmt_only:
                    np.save('{}/mass_covariance'.format(odRP),cov[:self.nBins,:self.nBins])
                np.save('{}/par_values'.format(odRP),par)
                np.save('{}/par_errors'.format(odRP),err)
                np.save('{}/full_covariance'.format(odRP),cov)

                save_object(minuit,'{}/minuit_object.pkl'.format(odRP))
        
        pre_final = datetime.now()

        minuit.strategy = 2 if not self.mtmt_only else 1
        
        if not self.mtmt_only:
            minuit.fixed = [False for _ in params]
            minuit.limits = [(-1*math.inf,math.inf) for _ in params]
            minuit.tol = 0
            
        # minuit.migrad() # final fit
        success = self.fitAndCheck(minuit,printout=True)

        if self.mtmt_only and not success:
            if not minuit.valid:
                print('ERROR: invalid minimum')
            elif not minuit.accurate:
                print('WARNING: covariance not accurate, re-running hesse')
                success = self.fitAndCheck(minuit,printout=True,hesse=True)
                # print('WARNING: covariance not accurate, improving strategy')
                # minuit.strategy = 2
                success = self.fitAndCheck(minuit,printout=True)
                if not success:
                    print('ERROR: fit failed')
                
        print ('\nlast step took {}'.format(datetime.now()-pre_final))        
        print ('total fit took {}\n'.format(datetime.now()-pre))

        
        cov = np.array(minuit.covariance)
        par = np.array(minuit.values)
        err = np.array(minuit.errors)
        
        if not os.path.exists(self.od):
            os.makedirs(self.od)

        np.save('{}/mass_results'.format(self.od),par[:self.n_massParams])
        if not self.mtmt_only:
            np.save('{}/mass_covariance'.format(self.od),cov[:self.nBins,:self.nBins])
        np.save('{}/par_values'.format(self.od),par)
        np.save('{}/par_errors'.format(self.od),err)
        np.save('{}/full_covariance'.format(self.od),cov)

        save_object(minuit,'{}/minuit_object.pkl'.format(self.od))

        self.doScaleVariations()
        
        return
    
    def doBreakdown(self):

        # ext_PDF = self.estimatePDFexternalised()
        # print('\nexternalised PDF uncertainty = {:.2f} GeV\n'.format(ext_PDF))
        
        with open('{}/minuit_object.pkl'.format(self.od), 'rb') as inp:
            minuit = pickle.load(inp)
            m_err = np.array(minuit.errors)[:self.n_massParams]
            masses = np.array(minuit.values)[:self.n_massParams]
            
            minuit.fixed = [False if i<self.n_massParams else True for i, _ in enumerate(minuit.values)]
            # minuit.migrad() # fit with only exp
            self.fitAndCheck(minuit,printout=True)
            m_err_exp = np.array(minuit.errors)[:self.n_massParams]
            
            m_err_PDF_num = (m_err**2-m_err_exp**2)**.5

        scale_up = copy.deepcopy(masses)
        scale_down = copy.deepcopy(masses)
        for scales in self.scale_vars:
            m_scale = np.load('{}/mass_results_{}.npy'.format(self.od,scales))
            scale_up = np.maximum(scale_up,m_scale)
            scale_down = np.minimum(scale_down,m_scale)

        scale_up-=masses
        scale_down=masses-scale_down

        tot_up = (m_err**2+scale_up**2)**.5
        tot_down = (m_err**2+scale_down**2)**.5
        
        print()
        for i in range(0,self.n_massParams):
            print('m{} = {:.2f} +/- {:.2f} (exp+PDF+num) +{:.2f} -{:.2f} (scale) GeV'.format(i+1 if not self.mtmt_only else '(m)',
                                                                                                           masses[i],m_err[i],scale_up[i],scale_down[i]))
            print ('m{} = {:.2f} +/- {:.2f} (exp) +/- {:.2f} (PDF+num) +{:.2f} -{:.2f} (scale) GeV'.format(i+1 if not self.mtmt_only else '(m)',
                                                                                                           masses[i],m_err_exp[i],m_err_PDF_num[i],scale_up[i],scale_down[i]))
            print ('m{} = {:.2f} +{:.2f} -{:.2f} (tot) GeV'.format(i+1 if not self.mtmt_only else '(m)', masses[i],tot_up[i],tot_down[i]))
            print()
            
            if self.mtmt_only:
                print('corresponding mtp = {:.2f} GeV'.format(conv.mtmu2mtp(masses[i],masses[i])))
                print('corresponding mt({}) = {:.2f} GeV'.format(cnst.mu_1,conv.mtmt2mtmu(masses[i],cnst.mu_1)))
        print()

        return


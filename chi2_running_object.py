import os, sys, copy, json
import numpy as np
import ROOT as rt
import uncertainties as unc
import iminuit
from scipy import stats
from scipy.optimize import fsolve
from matplotlib import pyplot as plt

import variables as var
import constants as cnst
import mass_convert as conv

from datetime import datetime

import math

rt.gROOT.SetBatch(True)

# minuit settings

plotdir = 'plots_running'

class running_object():

    def __init__(self,infile_xsec_mass,infile_num_unc,inpath_PDFs,infile_num_unc_PDFs,ref_bin=2):
        
        self.ref_bin = ref_bin-1
        self.exp_xsec, self.exp_err, self.corr_matrix = self.getExperimentalResults()
        self.exp_cov = np.matmul(np.diag(self.exp_err),np.matmul(self.corr_matrix,np.diag(self.exp_err)))
        self.nBins = self.exp_xsec.shape[0]
        self.scales = np.array([cnst.mu_1,cnst.mu_2,cnst.mu_3,cnst.mu_4])
        self.extr_cov = self.getExtrapolationImpacts()
        self.d_xsec_vs_mass, self.d_xsec_vs_mass_scaleup, self.d_xsec_vs_mass_scaledown = self.readAllXsecVsMass(infile_xsec_mass)
        self.d_numunc = self.readNumericalUncertJSON(infile_num_unc)
        self.nPDFs = 30
        self.d_PDFunc = self.readPDFuncertainties(inpath_PDFs)
        self.d_numunc_PDFs = self.readNumericalUncertPDFsJSON(infile_num_unc_PDFs)
        self.addCentralPDFtoList()
        self.doFit()
        
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

    def readPDFuncertainties(self, inpah_PDFs):
        d_pdf = dict()
        for pdf in range(0,self.nPDFs):
            d_pdf[pdf] = self.readPDFuncertainty(inpah_PDFs,pdf)
        return d_pdf

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

    def addCentralPDFtoList(self):
        d = self.d_PDFunc[0]
        for i in d.keys():
            for mass in d[i].keys():
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

        xsec_PDFs = self.xsec_PDFs_bin[mbin]*(1+MCstat_nuisances_PDFs_bin*self.rel_err_xsec_PDFs_bin[mbin]) # MC stat correction PDF
        xsec_PDFs_wunc = np.array([unc.ufloat(xsec_PDFs[i],xsec_PDFs[i]*self.rel_err_xsec_PDFs_bin[mbin][i]) for i in range(0,len(self.rel_err_xsec_PDFs_bin[mbin]))])

        rel_variations = nuisances_PDFs * (xsec_PDFs_wunc/xsec_corr_values[self.ref_mass_bin[mbin]]-1) +1 # relative PDF variations
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
        masses = params[0:self.nBins]
        all_nuisances = params[self.nBins:]
        MCstat_nuisances_bin = [params[self.nBins+self.nMassPoints[i-1]:self.nBins+self.nMassPoints[i-1]+self.nMassPoints[i]] for i in range(0,self.nBins)]
        MCstat_nuisances_PDFs_bin = [params[self.nBins+sum(self.nMassPoints)+(self.nPDFs-1)*i:self.nBins+sum(self.nMassPoints)+(self.nPDFs-1)*(i+1)] for i in range(0,self.nBins)]
        nuisances_PDFs = params[self.nBins+sum(self.nMassPoints)+(self.nPDFs-1)*self.nBins:]
        a_s,b_s,c_s = self.getDependencies(MCstat_nuisances_bin,MCstat_nuisances_PDFs_bin,nuisances_PDFs)
        res_array = self.exp_xsec - self.fitQuadratic(masses,a_s,b_s,c_s)
        chi2 = np.matmul(res_array,np.matmul(np.linalg.inv(self.cov_tot),res_array))
        return chi2 + sum(all_nuisances**2)

    def defineUsefulVariablesForFit(self):
        self.masses_bin = [np.array([float(m) for m in self.d_xsec_vs_mass[mbin]]) for mbin in range(0,self.nBins)]
        for m in self.masses_bin:
            m.sort()
        self.rel_err_xsec_bin = [np.array([self.d_numunc[mbin][str(m)] for m in self.masses_bin[mbin]]) for mbin in range(0,self.nBins)]
        self.xsec_bin = [np.array([self.d_xsec_vs_mass[mbin][str(m)] for m in self.masses_bin[mbin]]) for mbin in range(0,self.nBins)]

        m_ref_bin = [list(self.d_PDFunc[1][mbin].keys())[0] for mbin in range(0,self.nBins)]
        self.xsec_PDFs_bin = [np.array([float(self.d_PDFunc[pdf][mbin][m_ref_bin[mbin]]) for pdf in range(1,self.nPDFs)]) for mbin in range(0,self.nBins)]
        self.rel_err_xsec_PDFs_bin = [np.array([float(self.d_numunc_PDFs[pdf][mbin][float(m_ref_bin[mbin])]) for pdf in range(1,self.nPDFs)]) for mbin in range(0,self.nBins)]
        self.ref_mass_bin = [list(self.masses_bin[mbin]).index(float(m_ref_bin[mbin])) for mbin in range(0,self.nBins)]

        self.minuit_dep = iminuit.Minuit(self.chi2QuadraticFit,a=0,b=-1,c=10)
        self.minuit_dep.errordef=1
        self.minuit_dep.strategy=2
        self.minuit_dep.tol=1E-6

        self.cov_tot = self.exp_cov + self.extr_cov
        
        return
    
    def doFit(self):
        params = np.ones(self.nBins)
        params *= 160 # initialise masses
        self.nMassPoints = [len(self.d_numunc[i].keys()) for i in range(0,self.nBins)]
        self.nMassPoints.append(0) # useful in globalChi2
        all_nuisances = np.zeros(sum(self.nMassPoints)+(self.nPDFs-1)*5) # MC stat parameters (nominal+PDFs) and PDFs
        params = np.append(params,all_nuisances)

        self.defineUsefulVariablesForFit()
        
        minuit = iminuit.Minuit(self.globalChi2,params)
        minuit.fixed = [False if i<self.nBins else True for i, _ in enumerate(params)]
        minuit.errordef=1
        # minuit.strategy=global_strategy
        # minuit.tol=global_tolerance

        pre = datetime.now()
        
        minuit.limits = [(-1*math.inf,math.inf) if i<self.nBins else (-1,1) for i, _ in enumerate(params)]
        minuit.migrad() # do first fit (only masses)
        
        print('\nfirst fit took {}\n'.format(datetime.now()-pre))
        
        minuit.fixed = [False for _ in params]
        minuit.limits = [(minuit.values[i]-.1*minuit.errors[i],minuit.values[i]+.1*minuit.errors[i]) if i<self.nBins else (-.3,.3) for i, _ in enumerate(params)]
        minuit.migrad() # second fit, with constraints

        print('second fit took {}\n'.format(datetime.now()-pre))

        pre_final = datetime.now()
        minuit.limits = [(-1*math.inf,math.inf) for _ in params]
        minuit.strategy = 2
        minuit.tol = 1E-6
        minuit.migrad() # final fit
        
        print ('last step took {}\n'.format(datetime.now()-pre_final))        
        print ('total fit took {}\n'.format(datetime.now()-pre))

        cov = np.empty((self.nBins,self.nBins))
        for i in range (0,self.nBins):
            for j in range (0,self.nBins):
                cov[i][j] = minuit.covariance[i][j]
        results = [minuit.values[i] for i in range(0,self.nBins)]
        np.save('mass_results',results)
        np.save('mass_covariance',cov)
        
        print()
        print(minuit.values)
        print()
        print(minuit.errors)
        return

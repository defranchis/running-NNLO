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

rt.gROOT.SetBatch(True)

# minuit settings
global_tolerance = 1E-6
global_strategy = 1
global_errordef = 1

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
    

    def getDependencies(self,MCstat_nuisances_bin):
        a_s = np.empty(self.nBins)
        b_s = np.empty(self.nBins)
        c_s = np.empty(self.nBins)
        for mbin in range(0,self.nBins):
            dep_params = self.getDependencyBin(mbin,MCstat_nuisances_bin[mbin])
            a_s[mbin]=dep_params[0]
            b_s[mbin]=dep_params[1]
            c_s[mbin]=dep_params[2]
        return a_s, b_s, c_s

    def getDependencyBin(self,mbin,MCstat_nuisances_bin):
        masses = np.array([float(m) for m in self.d_xsec_vs_mass[mbin]])
        masses.sort()
        rel_err_xsec = np.array([self.d_numunc[mbin][str(m)] for m in masses])
        xsec = np.array([self.d_xsec_vs_mass[mbin][str(m)] for m in masses]) * (1+MCstat_nuisances_bin*rel_err_xsec) # MC stat correction
        cov = np.matmul(np.diag(rel_err_xsec*xsec),np.matmul(np.diag(np.ones(len(xsec))),np.diag(rel_err_xsec*xsec)))

        xsec_corr_values = np.array(unc.correlated_values(xsec,cov))

        # same concept, but to be updated depending on PDF nuisances
        # # here PDF numerical uncertainties and their correlations are propagated
        # if pdf > 0 and mbin in self.d_PDFunc[pdf].keys():  #torm second part
        #     m_ref = list(self.d_PDFunc[pdf][mbin].keys())[0]
        #     xsec_pdf = float(self.d_PDFunc[pdf][mbin][m_ref])
        #     err_xsec_pdf = float(self.d_numunc_PDFs[pdf][mbin][float(m_ref)])
        #     xsec_pdf_wuncert = unc.ufloat(xsec_pdf,err_xsec_pdf*xsec_pdf)
        #     i = list(masses).index(float(m_ref))
        #     xsec_corr_values = xsec_corr_values/xsec_corr_values[i]*xsec_pdf_wuncert

        self.masses_for_fit = masses
        self.xsec_cov_for_fit = np.array(unc.covariance_matrix(xsec_corr_values))
        self.xsec_values_for_fit = np.array([xsec.n for xsec in xsec_corr_values])

        minuit = iminuit.Minuit(self.chi2QuadraticFit,a=0,b=-1,c=10)
        minuit.errordef=global_errordef
        minuit.strategy=global_strategy
        minuit.tol=global_tolerance
        minuit.migrad()

        del self.masses_for_fit
        del self.xsec_cov_for_fit
        del self.xsec_values_for_fit
        
        return np.array(minuit.values)
        
    def globalChi2(self,params):
        masses = params[0:self.nBins]
        MCstat_nuisances = params[self.nBins:]
        MCstat_nuisances_bin = [MCstat_nuisances[self.nMassPoints[i-1]:self.nMassPoints[i-1]+self.nMassPoints[i]] for i in range(0,self.nBins)]
        a_s,b_s,c_s = self.getDependencies(MCstat_nuisances_bin)
        res_array = self.exp_xsec - self.fitQuadratic(masses,a_s,b_s,c_s)
        cov_tot = self.exp_cov + self.extr_cov # add extrapolation
        chi2 = np.matmul(res_array,np.matmul(np.linalg.inv(cov_tot),res_array))
        return chi2 + sum(MCstat_nuisances**2)

    def doFit(self):
        params = np.ones(self.nBins)
        params *= 160 # initialise masses
        self.nMassPoints = [len(self.d_numunc[i].keys()) for i in range(0,self.nBins)]
        self.nMassPoints.append(0) # useful in globalChi2
        MCstat_nuisances = np.zeros(sum(self.nMassPoints)) # MC stat parameters (nominal)
        params = np.append(params,MCstat_nuisances)
        
        minuit = iminuit.Minuit(self.globalChi2,params)
        minuit.errordef=global_errordef
        minuit.strategy=global_strategy
        minuit.tol=global_tolerance

        pre = datetime.now()
        minuit.migrad() # do fit
        post = datetime.now()

        print ('\nfit took', post-pre)
        print()
        print(minuit.values)
        print()
        print(minuit.errors)
        return

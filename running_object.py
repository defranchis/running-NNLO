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

rt.gROOT.SetBatch(True)

# minuit settings
global_tolerance = 1E-6
global_strategy = 2
global_errordef = 1

plotdir = 'plots_running'

class mass_result():
    pass

class running_object():

    def __init__(self,infile_xsec_mass,infile_num_unc,inpath_PDFs,infile_num_unc_PDFs,ref_bin=2):
        
        self.isCloneMass = False
        self.isCloneFull = False
        self.isClone = self.isCloneFull or self.isCloneMass
        self.ref_bin = ref_bin-1
        self.exp_xsec, self.exp_err, self.corr_matrix = self.getExperimentalResults()
        self.nBins = self.exp_xsec.shape[0]
        self.scales = np.array([cnst.mu_1,cnst.mu_2,cnst.mu_3,cnst.mu_4])
        self.extr_up, self.extr_down = self.getExtrapolationImpacts()
        self.d_xsec_vs_mass, self.d_xsec_vs_mass_scaleup, self.d_xsec_vs_mass_scaledown = self.readAllXsecVsMass(infile_xsec_mass)
        self.d_numunc = self.readNumericalUncertJSON(infile_num_unc)
        self.nPDFs = 30
        self.d_PDFunc = self.readPDFuncertainties(inpath_PDFs)
        self.d_numunc_PDFs = self.readNumericalUncertPDFsJSON(infile_num_unc_PDFs)
        self.addCentralPDFtoList()
        self.d_mass_results = self.getAllMasses()
        self.estimateScaleUncertainties()
        self.estimateExtrapolationUncertainties()
        self.estimatePDFuncertaities()
        self.printMassResults()
        self.estimateRatios()
        self.estimateBestRunning()
        self.fitDynamicMassGeneration()
        
    def cloneMass(self):
        tmp = copy.deepcopy(self)
        tmp.isCloneMass = True
        tmp.isCloneFull = False
        tmp.isClone = tmp.isCloneMass or tmp.isCloneFull
        return tmp

    def cloneFull(self):
        if self.isCloneMass:
            raise RuntimeError('cannot make full copy of a partial copy')
        tmp = copy.deepcopy(self)
        tmp.isCloneFull = not tmp.isCloneMass
        tmp.isClone = tmp.isCloneMass or tmp.isCloneFull
        return tmp
        
    def update(self,pdf=0):
        self.d_mass_results = self.getAllMasses(pdf)
        if not self.isCloneMass:
            self.estimateScaleUncertainties()
            self.estimateExtrapolationUncertainties()
            self.estimatePDFuncertaities()
            self.estimateRatios()
            self.estimateBestRunning()
            self.fitDynamicMassGeneration()
        return

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

        extr_up = dict()
        extr_down = dict()

        for i, name in enumerate(var.extr_name):
            up = np.array([var.extr_1_up[i],var.extr_2_up[i],var.extr_3_up[i],var.extr_4_up[i]])*self.exp_xsec/100.
            down = np.array([var.extr_1_down[i],var.extr_2_down[i],var.extr_3_down[i],var.extr_4_down[i]])*self.exp_xsec/100.
            extr_up[name] = up
            extr_down[name] = down
        
        return extr_up, extr_down
    
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

    def estimatePDFuncertaities(self):
        d_pdf = dict()
        for pdf in range(1,self.nPDFs):
            d_pdf[pdf] = self.estimatePDFuncertaity(pdf)
        self.addPDFuncertaintyToResult(d_pdf)
        return

    def addPDFuncertaintyToResult(self,d_pdf):
        for mbin in range(0,self.nBins):
            total = 0
            up = 0
            down = 0
            for pdf in range(1,self.nPDFs):
                uncert = d_pdf[pdf][mbin].value - self.d_mass_results[mbin].value
                setattr(self.d_mass_results[mbin],'PDF_{}'.format(pdf),uncert)
                total += uncert**2
                if uncert > 0:
                    up += uncert**2
                else:
                    down += uncert**2
            setattr(self.d_mass_results[mbin],'PDF',total)
            setattr(self.d_mass_results[mbin],'PDF_up',up)
            setattr(self.d_mass_results[mbin],'PDF_down',down)
        return

    def estimatePDFuncertaity(self,pdf):
        pdf_rel_uncert = np.ones(self.nBins)
        for mbin in range(0,self.nBins):
            if not mbin in self.d_PDFunc[pdf].keys(): #torm
                continue
            if len(self.d_PDFunc[pdf][mbin].keys()) !=1 or len(self.d_PDFunc[0][mbin].keys())!=1:
                raise RuntimeError('more than one mass point for PDF variations or no PDF variation at all')
            if list(self.d_PDFunc[pdf][mbin].keys())[0] != list(self.d_PDFunc[0][mbin].keys())[0]:
                raise ValueError('different mass points used to evaluate PDF variation')
            
        obj_pdf = self.cloneMass()
        obj_pdf.update(pdf)

        return obj_pdf.d_mass_results
    
    def addCentralPDFtoList(self):
        d = self.d_PDFunc[0]
        for i in d.keys():
            for mass in d[i].keys():
                self.d_xsec_vs_mass[i][mass]=d[i][mass]
                if self.d_numunc_PDFs[0][i] != dict(): #torm
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
    
    # def readNumericalUncertPDFsJSON(self,filename):
    #     f = open(filename)
    #     data = json.load(f)

    #     d = dict()
    #     for pdf in range(0,self.nPDFs):
    #         dd = dict()
    #         for b in range(0,self.nBins):
    #             ddd = dict()
    #             for i, bb in enumerate(data['bin']):
    #                 if bb-1 == b and data['pdf'][i] == str(pdf):
    #                     ddd[data['mass'][i]] = data['relunc'][i]
    #                     break
    #             dd[b]=ddd
    #         d[pdf]=dd
    #     return d
        
    def getAllMasses(self,pdf=0):
        d_mass_results = dict()
        for mbin in range(0,self.nBins):
            result = self.getMassInBin(mbin,pdf)
            d_mass_results[mbin] = result
        return d_mass_results


    def fitQuadratic(self,x,a,b,c):
        return a*x*x + b*x + c

    def chi2QuadraticFit(self,a,b,c):
        res = self.xsec_values_for_fit - self.fitQuadratic(self.masses_for_fit,a,b,c)
        return np.matmul(res,np.matmul(np.linalg.inv(self.xsec_cov_for_fit),res))

    def getMassInBin(self,mbin,pdf=0):

        od = 'plots_chi2_err'
        if not os.path.exists(od):
            os.makedirs(od)

        masses = np.array([float(m) for m in self.d_xsec_vs_mass[mbin]])
        masses.sort()
        xsec = np.array([self.d_xsec_vs_mass[mbin][str(m)] for m in masses])
        err_xsec = np.array([self.d_numunc[mbin][str(m)] for m in masses])*xsec
        cov = np.matmul(np.diag(err_xsec),np.matmul(np.diag(np.ones(len(xsec))),np.diag(err_xsec)))

        xsec_corr_values = np.array(unc.correlated_values(xsec,cov))

        # here PDF numerical uncertainties and their correlations are propagated
        if pdf > 0 and mbin in self.d_PDFunc[pdf].keys():  #torm second part
            m_ref = list(self.d_PDFunc[pdf][mbin].keys())[0]
            xsec_pdf = float(self.d_PDFunc[pdf][mbin][m_ref])
            err_xsec_pdf = float(self.d_numunc_PDFs[pdf][mbin][float(m_ref)])
            xsec_pdf_wuncert = unc.ufloat(xsec_pdf,err_xsec_pdf*xsec_pdf)
            i = list(masses).index(float(m_ref))
            xsec_corr_values = xsec_corr_values/xsec_corr_values[i]*xsec_pdf_wuncert

        self.masses_for_fit = masses
        self.xsec_cov_for_fit = np.array(unc.covariance_matrix(xsec_corr_values))
        self.xsec_values_for_fit = np.array([xsec.n for xsec in xsec_corr_values])

        minuit = iminuit.Minuit(self.chi2QuadraticFit,a=0,b=-1,c=10)
        minuit.errordef=global_errordef
        minuit.strategy=global_strategy
        minuit.tol=global_tolerance
        minuit.migrad()
        
        if pdf == 0 and not self.isClone:
            plot = plt.errorbar(self.masses_for_fit,self.xsec_values_for_fit,np.array([xsec.s for xsec in xsec_corr_values]))
            m_scan = np.arange(masses[0],masses[-1],0.001)
            curve, = plt.plot(m_scan,self.fitQuadratic(m_scan,minuit.values[0],minuit.values[1],minuit.values[2]))
            plt.savefig('{}/xsec_bin_{}.png'.format(od,mbin+1))
            plt.close()
            
        del self.masses_for_fit
        del self.xsec_cov_for_fit
        del self.xsec_values_for_fit

        f = lambda x: self.fitQuadratic(x,minuit.values[0],minuit.values[1],minuit.values[2])-self.exp_xsec[mbin]
        f_up = lambda x: self.fitQuadratic(x,minuit.values[0],minuit.values[1],minuit.values[2])-self.exp_xsec[mbin]+self.exp_err[mbin]
        f_down = lambda x: self.fitQuadratic(x,minuit.values[0],minuit.values[1],minuit.values[2])-self.exp_xsec[mbin]-self.exp_err[mbin]
        m = list(set(fsolve(f,[masses[0],masses[-1]])))[0]
        err_up = list(set(fsolve(f_up,[masses[0],masses[-1]])))[0] -m
        err_down = m-list(set(fsolve(f_down,[masses[0],masses[-1]])))[0]

        result = self.fillMassResult(m,err_up,err_down)
        
        return result

    def fillMassResult(self,m,err_up,err_down):
        result = mass_result()
        setattr(result,'value',m)
        setattr(result,'err_up',err_up)
        setattr(result,'err_down',err_down)
        setattr(result,'err',max(err_up,err_down))
        return result

    def estimateScaleUncertainties(self):
        obj_up = self.cloneMass()
        obj_up.d_xsec_vs_mass = self.d_xsec_vs_mass_scaleup
        obj_up.update()
        obj_down = self.cloneMass()
        obj_down.d_xsec_vs_mass = self.d_xsec_vs_mass_scaledown
        obj_down.update()
        self.addScaleUncertaintyToResult(obj_up,obj_down)
        return

    def addScaleUncertaintyToResult(self,obj_up,obj_down):
        for mbin in range(0,self.nBins):
            scale_up = abs(self.d_mass_results[mbin].value-obj_down.d_mass_results[mbin].value)
            scale_down = abs(self.d_mass_results[mbin].value-obj_up.d_mass_results[mbin].value)
            scale_up_sign = obj_up.d_mass_results[mbin].value - self.d_mass_results[mbin].value
            scale_down_sign = obj_down.d_mass_results[mbin].value - self.d_mass_results[mbin].value
            setattr(self.d_mass_results[mbin],'scale_up',scale_up)
            setattr(self.d_mass_results[mbin],'scale_down',scale_down)
            setattr(self.d_mass_results[mbin],'scale_up_sign',scale_up_sign)
            setattr(self.d_mass_results[mbin],'scale_down_sign',scale_down_sign)
            setattr(self.d_mass_results[mbin],'scale',max(scale_up,scale_down))
        return

    def estimateExtrapolationUncertainties(self):
        values = np.array([self.d_mass_results[i].value for i in range(0,self.nBins)])
        extr_uncert_up = np.zeros(self.nBins)
        extr_uncert_down = np.zeros(self.nBins)
        for name in self.extr_up.keys():
            obj_up = self.cloneMass()
            obj_down = self.cloneMass()
            obj_up.exp_xsec += self.extr_up[name]
            obj_down.exp_xsec += self.extr_down[name]
            obj_up.update()
            obj_down.update()
            up_values = np.array([obj_up.d_mass_results[i].value if obj_up.d_mass_results[i].value > self.d_mass_results[i].value else obj_down.d_mass_results[i].value
                                  for i in range(0,self.nBins)])
            down_values = np.array([obj_down.d_mass_results[i].value if obj_down.d_mass_results[i].value < self.d_mass_results[i].value else obj_up.d_mass_results[i].value
                                  for i in range(0,self.nBins)])
            up_values_sign = np.array([obj_up.d_mass_results[i].value - self.d_mass_results[i].value for i in range(0,self.nBins)])
            down_values_sign = np.array([obj_down.d_mass_results[i].value - self.d_mass_results[i].value for i in range(0,self.nBins)])
            for mbin in range(0,self.nBins):
                setattr(self.d_mass_results[mbin],'extr_{}_up'.format(name),up_values_sign[mbin])
                setattr(self.d_mass_results[mbin],'extr_{}_down'.format(name),down_values_sign[mbin])
            extr_uncert_up += (up_values-values)**2
            extr_uncert_down += (down_values-values)**2
        extr_uncert_up = extr_uncert_up**.5
        extr_uncert_down = extr_uncert_down**.5

        self.addExtrapolationUncertaintyToResult(extr_uncert_up,extr_uncert_down)
            
        return
            
    def addExtrapolationUncertaintyToResult(self,extr_uncert_up,extr_uncert_down):
        for mbin in range(0,self.nBins):
            setattr(self.d_mass_results[mbin],'extr_up',extr_uncert_up[mbin])
            setattr(self.d_mass_results[mbin],'extr_down',extr_uncert_down[mbin])
            setattr(self.d_mass_results[mbin],'extr',max(self.d_mass_results[mbin].extr_up,self.d_mass_results[mbin].extr_down))

            
    def printMassResults(self,detailed=True):
        print ('\nmasses:')
        for mbin in range(0,self.nBins):
            if detailed:
                print ('bin {}: mt(mu) = {:.2f} +{:.2f} -{:.2f} (exp) +{:.2f} -{:.2f} (extr) +{:.2f} -{:.2f} (PDF) +{:.2f} -{:.2f} (scale)'
                       .format(mbin+1,self.d_mass_results[mbin].value,
                               self.d_mass_results[mbin].err_up,self.d_mass_results[mbin].err_down,
                               self.d_mass_results[mbin].extr_up,self.d_mass_results[mbin].extr_down,
                               self.d_mass_results[mbin].PDF_up,self.d_mass_results[mbin].PDF_down,
                               self.d_mass_results[mbin].scale_up,self.d_mass_results[mbin].scale_down))
            else:
                print ('bin {}: mt(mu) = {:.2f} +/- {:.2f} (exp) +/- {:.2f} (extr) +/- {:.2f} (PDF) +/- {:.2f} (scale)'
                       .format(mbin+1,self.d_mass_results[mbin].value,
                               self.d_mass_results[mbin].err,
                               self.d_mass_results[mbin].extr,
                               self.d_mass_results[mbin].PDF,
                               self.d_mass_results[mbin].scale))
        return

    def printMassNominalResults(self):
        masses = np.array([self.d_mass_results[i].value for i in range(0,self.nBins)])
        print(masses)
        return

    def getExtrapolationCovarianceMasses(self):
        cov_tot = np.zeros((self.nBins,self.nBins))
        for extr in self.extr_up.keys():
            cov = np.zeros((self.nBins,self.nBins))
            for i in range(0,self.nBins):
                for j in range(0,self.nBins):
                    sign_i = abs(getattr(self.d_mass_results[i],'extr_{}_up'.format(extr)))/getattr(self.d_mass_results[i],'extr_{}_up'.format(extr))
                    sign_j = abs(getattr(self.d_mass_results[j],'extr_{}_up'.format(extr)))/getattr(self.d_mass_results[j],'extr_{}_up'.format(extr))
                    uncert_i = max(abs(getattr(self.d_mass_results[i],'extr_{}_up'.format(extr))),abs(getattr(self.d_mass_results[i],'extr_{}_down'.format(extr))))
                    uncert_j = max(abs(getattr(self.d_mass_results[j],'extr_{}_up'.format(extr))),abs(getattr(self.d_mass_results[j],'extr_{}_down'.format(extr))))
                    cov[i][j] = sign_i*sign_j*uncert_i*uncert_j
            cov_tot += cov
        return cov_tot
                     
    def getPDFCovarianceMasses(self):
        cov_tot = np.zeros((self.nBins,self.nBins))
        for pdf in range(1,self.nPDFs):
            cov = np.zeros((self.nBins,self.nBins))
            for i in range(0,self.nBins):
                for j in range(0,self.nBins):
                    cov[i][j] = getattr(self.d_mass_results[i],'PDF_{}'.format(pdf))*getattr(self.d_mass_results[j],'PDF_{}'.format(pdf))
            cov_tot += cov
        #torm, just for testing:
        for i in range(0,self.nBins):
            cov_tot[i][i] = max(cov_tot[i][i],1E-06)
        return cov_tot

    def getScaleCovarianceMasses(self):
        cov = np.zeros((self.nBins,self.nBins))
        for i in range(0,self.nBins):
            for j in range(0,self.nBins):
                sign_i = abs(getattr(self.d_mass_results[i],'scale_up_sign'))/getattr(self.d_mass_results[i],'scale_up_sign')
                sign_j = abs(getattr(self.d_mass_results[j],'scale_up_sign'))/getattr(self.d_mass_results[j],'scale_up_sign')
                uncert_i = max(abs(getattr(self.d_mass_results[i],'scale_up_sign')),abs(getattr(self.d_mass_results[i],'scale_down_sign')))
                uncert_j = max(abs(getattr(self.d_mass_results[j],'scale_up_sign')),abs(getattr(self.d_mass_results[j],'scale_down_sign')))
                cov[i][j] = sign_i*sign_j*uncert_i*uncert_j
        return cov
        
    def estimateRatios(self):
        uncert_mass = np.array([self.d_mass_results[i].err for i in range(0,self.nBins)])
        values_mass = np.array([self.d_mass_results[i].value for i in range(0,self.nBins)])
        self.mass_values = values_mass
        uncert_mass_diag = np.diag(uncert_mass)
        cov_mass_exp = np.matmul(uncert_mass_diag,np.matmul(self.corr_matrix,uncert_mass_diag))
        self.masses = np.array(unc.correlated_values(values_mass,cov_mass_exp))
        ratios = self.masses / self.masses[self.ref_bin]
        self.ratios = np.delete(ratios,self.ref_bin)
        self.corr_ratio_exp = np.array(unc.correlation_matrix(self.ratios))
        self.cov_ratio_exp = np.array(unc.covariance_matrix(self.ratios))
        if not self.isClone:
            print ('\nratios (exp):')
            for i, ratio in enumerate(self.ratios):
                print ('r_{} = {}'.format(i+1,ratio))
            print ('\ncorrelations (exp):')
            print (self.corr_ratio_exp)
            print()
        cov_mass_extr = self.getExtrapolationCovarianceMasses()
        masses_extr = np.array(unc.correlated_values(values_mass,cov_mass_extr))
        ratios_extr = masses_extr / masses_extr[self.ref_bin]
        self.ratios_extr = np.delete(ratios_extr,self.ref_bin)
        self.corr_ratio_extr = np.array(unc.correlation_matrix(self.ratios_extr))
        self.cov_ratio_extr = np.array(unc.covariance_matrix(self.ratios_extr))
        if not self.isClone:
            print ('\nratios (extr):')
            for i, ratio in enumerate(self.ratios_extr):
                print ('r_{} = {}'.format(i+1,ratio))
            print ('\ncorrelations (extr):')
            print (self.corr_ratio_extr)
            print()
        cov_mass_pdf = self.getPDFCovarianceMasses()
        masses_pdf = np.array(unc.correlated_values(values_mass,cov_mass_pdf))
        ratios_pdf = masses_pdf / masses_pdf[self.ref_bin]
        self.ratios_pdf = np.delete(ratios_pdf,self.ref_bin)
        self.corr_ratio_pdf = np.array(unc.correlation_matrix(self.ratios_pdf))
        self.cov_ratio_pdf = np.array(unc.covariance_matrix(self.ratios_pdf))

        if not self.isClone:
            print ('\nratios (pdf):')
            for i, ratio in enumerate(self.ratios_pdf):
                print ('r_{} = {}'.format(i+1,ratio))
            print ('\ncorrelations (pdf):')
            print (self.corr_ratio_pdf)
            print()
        cov_mass_scale = self.getScaleCovarianceMasses()
        masses_scale = np.array(unc.correlated_values(values_mass,cov_mass_scale))
        ratios_scale = masses_scale / masses_scale[self.ref_bin]
        self.ratios_scale = np.delete(ratios_scale,self.ref_bin)
        self.corr_ratio_scale = np.array(unc.correlation_matrix(self.ratios_scale))
        self.cov_ratio_scale = np.array(unc.covariance_matrix(self.ratios_scale))
        if not self.isClone:
            print ('\nratios (scale):')
            for i, ratio in enumerate(self.ratios_scale):
                print ('r_{} = {}'.format(i+1,ratio))
            print ('\ncorrelations (scale):')
            print (self.corr_ratio_scale)
            print()

        self.cov_ratio_tot_noscale = self.cov_ratio_exp + self.cov_ratio_extr + self.cov_ratio_pdf
        self.cov_ratio_tot = self.cov_ratio_tot_noscale + self.cov_ratio_scale

        self.cov_mass_tot_noscale = cov_mass_exp + cov_mass_extr + cov_mass_pdf
        self.cov_mass_tot = self.cov_mass_tot_noscale + cov_mass_scale
        self.masses_tot_noscale = unc.correlated_values(self.mass_values,self.cov_mass_tot_noscale)
        
        self.ratio_values = np.array([ratio.n for ratio in self.ratios])
        self.ratios_tot_noscale = unc.correlated_values(self.ratio_values,self.cov_ratio_tot_noscale)
        self.corr_ratio_tot_noscale = unc.correlation_matrix(self.ratios_tot_noscale)
        if not self.isClone:
            print ('\nratios (tot, no scale):')
            for i, ratio in enumerate(self.ratios_tot_noscale):
                print ('r_{} = {}'.format(i+1,ratio))
            print ('\ncorrelations (tot, no scale):')
            print (self.corr_ratio_tot_noscale)
            print()
        
        return

    def getTheoryRatio(self,scales):
        masses_evolved = np.array([conv.mtmu2mtmu(self.mass_values[self.ref_bin],self.scales[self.ref_bin],scale) for scale in scales])
        return masses_evolved/self.mass_values[self.ref_bin]

    def getRunningX(self,x,scales):
        return x*(self.getTheoryRatio(scales)-1)+1

    def computeChi2(self,x=1):
        return np.matmul(self.ratio_values-self.getRunningX(x,np.delete(self.scales,self.ref_bin)),
                         np.matmul(np.linalg.inv(self.cov_ratio_tot_noscale),self.ratio_values-self.getRunningX(x,np.delete(self.scales,self.ref_bin))))

    def estimateBestRunning(self):

        ndf = self.ratio_values.shape[0]
        self.chi2_QCD = self.computeChi2(1)
        self.chi2_noRunning = self.computeChi2(0)
        self.prob_QCD = stats.chi2.sf(self.chi2_QCD,ndf)
        self.prob_noRunning = stats.chi2.sf(self.chi2_noRunning,ndf)
        if not self.isClone:
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

        if not self.isClone:
            print('\nbest-fit x = {:.2f} +/- {:.2f}'.format(minuit.values['x'], minuit.errors['x']))
            print('chi2 = {:.2f}, prob = {:.1f}%'.format(self.chi2_xFit,self.prob_xFit*100.))
            print()
            
            self.producePlotRatio()
            
        return

    def producePlotRatio(self):
        err_ratios = np.array([ratio.s for ratio in self.ratios_tot_noscale])
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
        return np.matmul(self.mass_values-self.dynmass(self.scales, mtmt, Lambda), np.matmul(np.linalg.inv(self.cov_mass_tot_noscale),self.mass_values-self.dynmass(self.scales, mtmt, Lambda)))

    
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
        
        if not self.isClone:
            print('\nmt(mt) = {:.2f} +/- {:.2f} GeV'.format(mtmt,err_mtmt))
            print('Lambda = {:.1f} +/- {:.1f} TeV'.format(Lambda,err_Lambda))
            print()
            self.chi2_dynmass = self.chi2dynmass(mtmt,Lambda)
            print('chi2 = {:.2f}, prob = {:.1f}%'.format(self.chi2_dynmass,stats.chi2.sf(self.chi2_dynmass,ndf)*100.))
            print()

            self.producePlotDynamicMass()
            
        return

    def producePlotDynamicMass(self):

        err_mass = np.array([mass.s for mass in self.masses_tot_noscale])
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

import os, sys, copy, json
import numpy as np
import ROOT as rt
import uncertainties

import variables as var
import constants as cnst
import mass_convert as conv

class mass_result():
    pass

class running_object():

    def __init__(self,infile_xsec_mass,infile_num_unc,inpath_PDFs):
        
        self.isClone = False
        self.exp_xsec, self.exp_err, self.corr_matrix = self.getExperimentalResults()
        self.nBins = self.exp_xsec.shape[0]
        self.extr_up, self.extr_down = self.getExtrapolationImpacts()
        self.d_xsec_vs_mass, self.d_xsec_vs_mass_scaleup, self.d_xsec_vs_mass_scaledown = self.readAllXsecVsMass(infile_xsec_mass)
        self.d_numunc = self.readNumericalUncertJSON(infile_num_unc)
        self.nPDFs = 30
        self.d_PDFunc = self.readPDFuncertainties(inpath_PDFs)
        # to add once numerical uncertainties available 
        # self.addCentralPDFtoList()
        self.d_mass_results = self.getAllMasses()
        self.estimateScaleUncertainties()
        self.estimateExtrapolationUncertainties()
        self.estimatePDFuncertaities()
        self.printMassResults()
        self.estimateRatios()
        
    def clone(self):
        tmp = copy.deepcopy(self)
        tmp.isClone = True
        return tmp

    def cloneIdentical(self):
        return copy.deepcopy(self)
    
    def update(self):
        self.d_mass_results = self.getAllMasses()
        if not self.isClone:
            self.estimateScaleUncertainties()
            self.estimateExtrapolationUncertainties()
            self.estimateRatios()
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
            pdf_rel_uncert[mbin] = list(self.d_PDFunc[pdf][mbin].values())[0]/list(self.d_PDFunc[0][mbin].values())[0]
            
        obj_pdf = self.clone()
        for mbin in range(0,self.nBins):
            for mass in obj_pdf.d_xsec_vs_mass[mbin].keys():
                obj_pdf.d_xsec_vs_mass[mbin][mass] *= pdf_rel_uncert[mbin]

        # here should also propagate the stat uncertainty correctly
        # and make sure it's not double-counted for the central PDF (rather replaced)
        # can also consider using only toys

        obj_pdf.update()    

        return obj_pdf.d_mass_results
    
    # to double check
    def addCentralPDFtoList(self):
        d = self.d_PDFunc[0]
        for i in d.keys():
            for mass in d[i].keys():
                self.d_xsec_vs_mass[i][mass]=d[i][mass]
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
    

    def getAllMasses(self):
        d_mass_results = dict()
        for mbin in range(0,self.nBins):
            result = self.getMassInBin(mbin)
            d_mass_results[mbin] = result
        return d_mass_results

    def getMassInBin(self,mbin):

        od = 'plots_chi2_err' if not self.isClone else 'tmp_chi2'
        if not os.path.exists(od):
            os.makedirs(od)

        mass_list = [float(m) for m in self.d_xsec_vs_mass[mbin]]
        mass_list.sort()

        h_xsec = rt.TGraphErrors()
        for i,mass in enumerate(mass_list):
            h_xsec.SetPoint(i,mass,self.d_xsec_vs_mass[mbin][(str(mass))])
            if str(mass) in self.d_numunc[mbin].keys():
                h_xsec.SetPointError(i,0,self.d_xsec_vs_mass[mbin][(str(mass))]*self.d_numunc[mbin][str(mass)])
            else:
                raise RuntimeError('numerical uncertainties not found for bin {} and mass {}'.format(mbin+1,mass))

        f = rt.TF1('f','pol2')
        h_xsec.Fit(f,'Q')
        
        if not self.isClone:

            h_xsec.SetMarkerStyle(8)
            h_xsec.SetTitle('bin {}; mt [GeV]; xsec [pb]'.format(mbin+1))
            l_central = rt.TLine(mass_list[0],self.exp_xsec[mbin],mass_list[-1],self.exp_xsec[mbin])
            l_up = rt.TLine(mass_list[0],self.exp_xsec[mbin]+self.exp_err[mbin],mass_list[-1],self.exp_xsec[mbin]+self.exp_err[mbin])
            l_down = rt.TLine(mass_list[0],self.exp_xsec[mbin]-self.exp_err[mbin],mass_list[-1],self.exp_xsec[mbin]-self.exp_err[mbin])

            l_central.SetLineColor(rt.kBlue)
            l_up.SetLineColor(rt.kGreen)
            l_down.SetLineColor(rt.kGreen)

            c = rt.TCanvas()
            h_xsec.Draw('apl')
            l_central.Draw('same')
            l_up.Draw('same')
            l_down.Draw('same')
            c.SaveAs('{}/xsec_bin_{}.png'.format(od,mbin+1))

        m = f.GetX(self.exp_xsec[mbin],100,200)
        err_up = f.GetX(self.exp_xsec[mbin]-self.exp_err[mbin],100,200) - m
        err_down = m - f.GetX(self.exp_xsec[mbin]+self.exp_err[mbin],100,200)

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
        obj_up = self.clone()
        obj_up.d_xsec_vs_mass = self.d_xsec_vs_mass_scaleup
        obj_up.update()
        obj_down = self.clone()
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
            obj_up = self.clone()
            obj_down = self.clone()
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
        uncert_mass_diag = np.diag(uncert_mass)
        cov_mass_exp = np.matmul(uncert_mass_diag,np.matmul(self.corr_matrix,uncert_mass_diag))
        masses = np.array(uncertainties.correlated_values(values_mass,cov_mass_exp))
        ratios = masses / masses[1]
        self.ratios = np.delete(ratios,1)
        self.corr_ratio_exp = np.array(uncertainties.correlation_matrix(self.ratios))
        self.cov_ratio_exp = np.array(uncertainties.covariance_matrix(self.ratios))
        if not self.isClone:
            print ('\nratios (exp):')
            for i, ratio in enumerate(self.ratios):
                print ('r_{} = {}'.format(i+1,ratio))
            print ('\ncorrelations (exp):')
            print (self.corr_ratio_exp)
            print()
        cov_mass_extr = self.getExtrapolationCovarianceMasses()
        masses_extr = np.array(uncertainties.correlated_values(values_mass,cov_mass_extr))
        ratios_extr = masses_extr / masses_extr[1]
        self.ratios_extr = np.delete(ratios_extr,1)
        self.corr_ratio_extr = np.array(uncertainties.correlation_matrix(self.ratios_extr))
        self.cov_ratio_extr = np.array(uncertainties.covariance_matrix(self.ratios_extr))
        if not self.isClone:
            print ('\nratios (extr):')
            for i, ratio in enumerate(self.ratios_extr):
                print ('r_{} = {}'.format(i+1,ratio))
            print ('\ncorrelations (extr):')
            print (self.corr_ratio_extr)
            print()
        cov_mass_pdf = self.getPDFCovarianceMasses()
        masses_pdf = np.array(uncertainties.correlated_values(values_mass,cov_mass_pdf))
        ratios_pdf = masses_pdf / masses_pdf[1]
        self.ratios_pdf = np.delete(ratios_pdf,1)
        self.corr_ratio_pdf = np.array(uncertainties.correlation_matrix(self.ratios_pdf))
        self.cov_ratio_pdf = np.array(uncertainties.covariance_matrix(self.ratios_pdf))

        if not self.isClone:
            print ('\nratios (pdf):')
            for i, ratio in enumerate(self.ratios_pdf):
                print ('r_{} = {}'.format(i+1,ratio))
            print ('\ncorrelations (pdf):')
            print (self.corr_ratio_pdf)
            print()
        cov_mass_scale = self.getScaleCovarianceMasses()
        masses_scale = np.array(uncertainties.correlated_values(values_mass,cov_mass_scale))
        ratios_scale = masses_scale / masses_scale[1]
        self.ratios_scale = np.delete(ratios_scale,1)
        self.corr_ratio_scale = np.array(uncertainties.correlation_matrix(self.ratios_scale))
        self.cov_ratio_scale = np.array(uncertainties.covariance_matrix(self.ratios_scale))
        if not self.isClone:
            print ('\nratios (scale):')
            for i, ratio in enumerate(self.ratios_scale):
                print ('r_{} = {}'.format(i+1,ratio))
            print ('\ncorrelations (scale):')
            print (self.corr_ratio_scale)
            print()

        self.cov_ratio_tot_noscale = self.cov_ratio_exp + self.cov_ratio_extr + self.cov_ratio_pdf
        self.cov_ratio_tot = self.cov_ratio_tot_noscale + self.cov_ratio_scale

        ratios_values = np.array([ratio.n for ratio in self.ratios])
        self.ratios_tot_noscale = uncertainties.correlated_values(ratios_values,self.cov_ratio_tot_noscale)
        self.corr_ratio_tot_noscale = uncertainties.correlation_matrix(self.ratios_tot_noscale)
        if not self.isClone:
            print ('\nratios (tot, no scale):')
            for i, ratio in enumerate(self.ratios_tot_noscale):
                print ('r_{} = {}'.format(i+1,ratio))
            print ('\ncorrelations (tot, no scale):')
            print (self.corr_ratio_tot_noscale)
            print()
        
        return

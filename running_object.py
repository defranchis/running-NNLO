import os, sys, copy, json
import numpy as np
import ROOT as rt

import variables as var
import constants as cnst
import mass_convert as conv

class mass_result():
    pass

class running_object():

    def __init__(self,infile_xsec_mass,infile_num_unc):
        
        self.isClone = False
        self.exp_xsec = np.array([var.xsec_1,var.xsec_2,var.xsec_3,var.xsec_4])
        self.exp_err = np.array([var.err_xsec_1_up/2.+var.err_xsec_1_down/2.,var.err_xsec_2_up/2.+var.err_xsec_2_down/2.,
                    var.err_xsec_3_up/2.+var.err_xsec_3_down/2.,var.err_xsec_4_up/2.+var.err_xsec_4_down/2])*self.exp_xsec/100.
        self.nBins = self.exp_xsec.shape[0]
        self.d_xsec_vs_mass, self.d_xsec_vs_mass_scaleup, self.d_xsec_vs_mass_scaledown = self.readAllXsecVsMass(infile_xsec_mass)
        self.d_numunc = self.readNumericalUncertJSON(infile_num_unc)
        self.d_mass_results = self.getAllMasses()
        self.estimateScaleUncertainties()
        self.printMassResults()
        
    def clone(self):
        tmp = copy.deepcopy(self)
        tmp.isClone = True
        return tmp

    def update(self):
        self.d_mass_results = self.getAllMasses()
        if not self.isClone:
            self.estimateScaleUncertainties()
        return

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
            scale_down = abs(self.d_mass_results[mbin].value-obj_down.d_mass_results[mbin].value)
            setattr(self.d_mass_results[mbin],'scale_up',scale_up)
            setattr(self.d_mass_results[mbin],'scale_down',scale_down)
            setattr(self.d_mass_results[mbin],'scale',max(scale_up,scale_down))
        return

    def printMassResults(self,detailed=False):
        for mbin in range(0,self.nBins):
            if detailed:
                print ('bin {}: mt(mu) = {:.2f} + {:.2f} - {:.2f} (exp) + {:.2f} - {:.2f} (scale)'
                       .format(mbin+1,self.d_mass_results[mbin].value,
                               self.d_mass_results[mbin].err_up,self.d_mass_results[mbin].err_down,
                               self.d_mass_results[mbin].scale_up,self.d_mass_results[mbin].scale_down))
            else:
                print ('bin {}: mt(mu) = {:.2f} +/- {:.2f} (exp) +/- {:.2f} (scale)'
                       .format(mbin+1,self.d_mass_results[mbin].value,
                               self.d_mass_results[mbin].err,self.d_mass_results[mbin].scale))
        return

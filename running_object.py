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
        self.d_xsec_vs_mass = self.readAllXsecVsMass(infile_xsec_mass)
        self.d_numunc = self.readNumericalUncertJSON(infile_num_unc)
        self.d_mass_results = self.getAllMasses()
        
    def clone(self):
        tmp = copy.deepcopy(self)
        tmp.isClone = True
        return tmp
            
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
        d = {}
        for i, low in enumerate(bin_low):
            m_xsec = dict()
            for l in lines:
                if l == '': continue
                if l.replace(' ','').startswith('#'): continue
                if int(l.split()[0]) != low: continue
                m_xsec[l.split()[-1]] = float(l.split()[2])
            d[i]=m_xsec
        return d

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

        m = f.GetX(self.exp_xsec[mbin],120,180)
        err_up = f.GetX(self.exp_xsec[mbin]-self.exp_err[mbin],120,180) - m
        err_down = m - f.GetX(self.exp_xsec[mbin]+self.exp_err[mbin],120,180)
        print ('fit in bin {}: mt(mu) = {:.2f} + {:.2f} - {:.2f} GeV'.format(mbin,m,err_up,err_down))

        result = self.fillMassResult(m,err_up,err_down)
        
        return result

    def fillMassResult(self,m,err_up,err_down):
        result = mass_result()
        setattr(result,'value',m)
        setattr(result,'err_up',err_up)
        setattr(result,'err_down',err_down)
        setattr(result,'err',max(err_up,err_down))
        return result

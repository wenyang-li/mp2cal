import numpy as np, copy, warnings
from pos import *
from wyl import GPR_interp
c_light=299792458.

def unwrap(arr):
    brr = np.unwrap(arr)
    crr = []
    for ii in range(1,brr.size): crr.append(brr[ii]-brr[ii-1])
    crr = np.unwrap(crr)
    nn = np.round(crr[0]/(2*np.pi))
    crr -= (nn*2.*np.pi)
    drr = np.zeros(brr.shape)+brr[0]
    for ii in range(crr.size): drr[ii+1] += np.sum(crr[:ii+1])
    return drr


class RedGain(object):
    """
    gain object, supporting degeneracy parameters manipulatioin.
    """
    def __init__(self, freqs = None, mask = None):
        self.red = None # Gain from redundant cal
        self.sky = None # Gain from sky cal
        self.mdl = None # Model vis from redundant cal
        self.auto = None # Scaled auto correlations from the data
        self.gbp = None # Global bandpass, needs to be calculated from the input gains
        self.gfit = None # Smoothed gains, needs to be calculated from the input gains
        self.freqs = freqs # Frequency array
        self.mask = mask # Frequency mask boolean array, same shape as frequency array
        if freqs is None:
            warnings.warn("Frequency array is None. Bandpass fitting cannot work without frequency array")
        if mask is None:
            warnings.warn("Frequency mask not provided, using MWA default.")
            nf = self.freqs.size
            self.mask = np.zeros(self.freqs.shape, dtype=bool)
            if nf == 384:
                for ii in range(nf):
                    if ii%16 in [0, 15]: self.mask[ii] = True
            if nf == 768:
                for ii in range(nf):
                    if ii%32 in [0,1,16,30,31]: self.mask[ii] = True

    def get_red(self, g_red):
        """
        Load redundant cal gains.
        """
        self.red = g_red

    def get_sky(self, g_sky):
        """
        Load sky cal gains
        """
        self.sky = g_sky

    def get_mdl(self, v_mdl):
        """
        Load redundant cal model visibilities
        """
        self.mdl = v_mdl
    
    def get_auto(self, uv):
        """
        Get auto correlations from the data. This is for bandpass fitting
        """
        a1 = uv.ant_1_array[:uv.Nbls]
        a2 = uv.ant_2_array[:uv.Nbls]
        self.auto = {'x': {}, 'y': {}}
        ind = np.where(a1 == a2)[0]
        inx = np.where(a1 != a2)[0]
        data = uv.data_array.reshape(uv.Ntimes,uv.Nbls,uv.Nfreqs,uv.Npols)
        wgts = np.logical_not(uv.flag_array.reshape(uv.Ntimes,uv.Nbls,uv.Nfreqs,uv.Npols))
        wgts = (np.sum(wgts[:,inx,:,:], axis=1) >0)
        wgts = np.sum(wgts, axis=1)
        wxmax = np.max(wgts[:,0])
        wymax = np.max(wgts[:,1])
        tindx = np.where(wgts[:,0] == wxmax)[0]
        tindy = np.where(wgts[:,1] == wymax)[0]
        for ii in ind:
            a = a1[ii]
            self.auto['x'][a] = np.sqrt(np.mean(np.abs(data[tindx,ii,:,0]),axis=0))
            self.auto['y'][a] = np.sqrt(np.mean(np.abs(data[tindy,ii,:,1]),axis=0))
        amp_ref_x, amp_ref_y = None, None
        for ii in ind:
            a = a1[ii]
            if np.any(np.isnan(self.sky['x'][a])):
                continue
            else:
                amp_ref_x = np.copy(self.auto['x'][a])
                amp_ref_y = np.copy(self.auto['y'][a])
                break
        for ii in ind:
            a = a1[ii]
            self.auto['x'][a] /= amp_ref_x
            self.auto['y'][a] /= amp_ref_y

    def scale_gains(self, amp_ave=1.):
        """
        scale the gain amplitudes so that they have an average of amp_ave
        """
        assert self.red, "Redundant calibration gains do not exit."
        for p in self.red.keys():
            amp = 0
            n = 0
            for a in self.red[p].keys():
                amp_temp = np.abs(self.red[p][a])
                ind = np.where(amp_temp == 0)
                amp_temp[ind] = 1
                amp += np.log(amp_temp)
                n += 1
            amp /= n
            q = amp - np.log(amp_ave)
            for a in self.red[p].keys(): self.red[p][a] /= np.exp(q)

    def amp_proj(self):
        """
        Calculate the amplitude degeneracy correction according to sky based solutions
        """
        assert self.red, "Redundant calibration gains do not exit."
        assert self.sky, "Sky calibration gains do not exit."
        amppar = {}
        for p in self.red.keys():
            SH = self.red[p][self.red[p].keys()[0]].shape
            s = np.zeros(SH)
            n = np.zeros(SH)
            for a in self.red[p].keys():
                if not a in self.sky[p].keys(): continue
                if np.any(np.isnan(self.sky[p][a])): continue
                if np.any(np.isnan(self.red[p][a])): continue
                num = np.ones(SH)
                amp_in = np.abs(self.red[p][a])
                amp_ta = np.resize(np.abs(self.sky[p][a]),SH)
                ind = np.where(amp_in==0)
                amp_in[ind] = 1.
                amp_ta[ind] = 1.
                num[ind] = 0
                s += np.log(amp_ta/amp_in)
                n += num
            ind = np.where(n==0)
            n[ind] = 1.
            s[ind] = 0.
            amppar[p] = np.exp(s/n)
        return amppar

    def phs_proj(self):
        """
        Calculate the phase degeneracy corrections according to sky based solutions
        """
        assert self.red, "Redundant calibration gains do not exit."
        assert self.sky, "Sky calibration gains do not exit."
        g_input = copy.deepcopy(self.red)
        g_target = copy.deepcopy(self.sky)
        phspar = {}
        ax1,ax2 = [],[]
        for ii in range(EastHex.shape[0]):
            if ii == 3: continue
            ind_east = EastHex[ii]
            ind_south = SouthHex[ii]
            ax1.append(ind_east)
            ax1.append(ind_south)
        for jj in range(EastHex.shape[1]):
            if jj == 3: continue
            ind_east = EastHex[:,jj]
            ind_south = SouthHex[:,jj]
            ax2.append(ind_east)
            ax2.append(ind_south)
        for p in g_input.keys():
            phspar[p] = {}
            a0 = g_input[p].keys()[0]
            nf = g_input[p][a0].size
            SH = g_input[p][a0].shape
            if g_input[p][a0].ndim == 2:
                for a in g_input[p].keys(): g_input[p][a] = g_input[p][a].flatten()
            if g_target[p][a0].ndim == 2:
                for a in g_target[p].keys(): g_target[p][a] = g_target[p][a].flatten()
            for a in g_target[p].keys():
                g_target[p][a] = np.resize(g_target[p][a], g_input[p][a].shape)
            slp1 = []
            slp2 = []
            for ff in range(0,nf):
                #***** East-West direction fit *****#
                slope = []
                for inds in ax1:
                    x,tau = [],[]
                    for ii in inds:
                        if not ii in g_input[p].keys(): continue
                        if not ii in g_target[p].keys(): continue
                        if np.isnan(g_input[p][ii][ff]): continue
                        if np.isnan(g_target[p][ii][ff]): continue
                        x.append(float(np.argwhere(inds==ii)))
                        tau.append(np.angle(g_target[p][ii][ff]*g_input[p][ii][ff].conj()))
                    if len(tau) < 3: continue
                    if np.round(x[-1])-np.round(x[0])+1 != len(x): continue
                    tau = unwrap(tau)
                    try:
                        z = np.polyfit(x,tau,1)
                        slope.append(z[0])
                    except: pass
                if len(slope) == 0:
                    warnings.warn("invalid value of gain in channel " + str(ff))
                    slp1.append(0)
                else:
                    slope = np.unwrap(slope)
                    slp1.append(np.median(slope))
                #***** 60 deg East-South direction fit *****#
                slope = []
                for inds in ax2:
                    x,tau = [],[]
                    for ii in inds:
                        if not ii in g_input[p].keys(): continue
                        if not ii in g_target[p].keys(): continue
                        if np.isnan(g_input[p][ii][ff]): continue
                        if np.isnan(g_target[p][ii][ff]): continue
                        x.append(float(np.argwhere(inds==ii)))
                        tau.append(np.angle(g_target[p][ii][ff]*g_input[p][ii][ff].conj()))
                    if len(tau) < 3: continue
                    if np.round(x[-1])-np.round(x[0])+1 != len(x): continue
                    tau = unwrap(tau)
                    try:
                        z = np.polyfit(x,tau,1)
                        slope.append(z[0])
                    except: pass
                if len(slope) == 0:
                    warnings.warn("invalid value of gain in channel " + str(ff))
                    slp2.append(0)
                else:
                    slope = np.unwrap(slope)
                    slp2.append(np.median(slope))
            phspar[p]['phi1'] = np.array(slp1).reshape(SH)
            phspar[p]['phi2'] = np.array(slp2).reshape(SH)
        return phspar

    def plane_fitting(self, ratio = None):
        """
        If the solutions are supposed to be centered at 1.0, remove any phase gradients or
        phase offsets by fitting a plane in (phase, rx, ry) space.
        """
        assert self.red, "Redundant calibration gains do not exit."
        gc = {}
        if ratio is None: ratio = self.red
        for p in ratio.keys():
            gc[p] = {}
            for a in ratio[p].keys():
                SH = ratio[p][a].shape
                gc[p][a] = ratio[p][a].flatten()
        Nsample = np.product(SH)
        phspar = {}
        for p in gc.keys():
            phspar[p] = {}
            A0 = np.zeros((4,4))
            p0 = np.zeros((4,Nsample))
            Ants = gc[p].keys()
            Ants.sort()
            na = len(gc[p].keys())
            for a in gc[p].keys():
                x = antpos[a]['top_x']
                y = antpos[a]['top_y']
                z = np.angle(gc[p][a])
                ai = Ants.index(a)
                if 56 < a < 93:
                    A0 += np.array([[x*x, x*y, x , 0 ],
                                    [x*y, y*y, y , 0 ],
                                    [ x ,  y , 1 , 0 ],
                                    [ 0 ,  0 , 0 , 0 ]])
                    p0 += np.array([z*x,z*y,z,np.zeros(z.shape)])
                if 92 < a < 128:
                    A0 += np.array([[x*x, x*y, 0 , x ],
                                    [x*y, y*y, 0 , y ],
                                    [ 0 ,  0 , 0 , 0 ],
                                    [ x ,  y , 0 , 1 ]])
                    p0 += np.array([z*x,z*y,np.zeros(z.shape),z])
            C = (np.linalg.inv(A0).dot(p0))
            #Attention: append negative results here
            phspar[p]['phix'] = -C[0].reshape(SH)
            phspar[p]['phiy'] = -C[1].reshape(SH)
            phspar[p]['offset_east'] = -C[2].reshape(SH)
            phspar[p]['offset_south'] = -C[3].reshape(SH)
        return phspar

    def degen_project_to_sky(self, time_average=True):
        """
        Project degeneracy of redundant cal from raw data to sky cal
        """
        assert self.red, "Redundant calibration gains do not exit."
        assert self.sky, "Sky calibration gains do not exit."
        if time_average:
            print("The gains will be averaged in time axis after degeneracy projection by default. \
                    Set time_average=False if you do not want to.")
        for p in self.red.keys():
            a_red = self.red[p].keys()
            a_sky = self.sky[p].keys()
            a_pool = [i for i in a_red if i in a_sky]
            ref1 = min(a_pool)
            ref2 = max(a_pool)
            ref_exp1 = np.exp(1j*np.angle(self.red[p][ref1]*self.sky[p][ref1].conj()))
            ref_exp2 = np.exp(1j*np.angle(self.red[p][ref2]*self.sky[p][ref2].conj()))
            for a in self.red[p].keys():
                if a < 93: self.red[p][a] /= ref_exp1
                else: self.red[p][a] /= ref_exp2
            amppar = self.amp_proj()
            phspar = self.phs_proj()
            for a in self.red[p].keys():
                if a < 93:
                    nx = np.argwhere(EastHex==a)[0][1] - np.argwhere(EastHex==ref1)[0][1]
                    ny = np.argwhere(EastHex==a)[0][0] - np.argwhere(EastHex==ref1)[0][0]
                else:
                    nx = np.argwhere(SouthHex==a)[0][1] - np.argwhere(SouthHex==ref2)[0][1]
                    ny = np.argwhere(SouthHex==a)[0][0] - np.argwhere(SouthHex==ref2)[0][0]
                proj = amppar[p]*np.exp(1j*(nx*phspar[p]['phi1']+ny*phspar[p]['phi2']))
                self.red[p][a] *= proj
            ratio = {p: {}}
            for a in self.red[p].keys():
                r = self.red[p][a]*self.sky[p][a].conj()
                if np.any(np.isnan(r)): continue
                ratio[p][a] = r
            phspar2 = self.plane_fitting(ratio = ratio)
            for a in self.red[p].keys():
                dx = antpos[a]['top_x']
                dy = antpos[a]['top_y']
                proj = np.exp(1j*(dx*phspar2[p]['phix']+dy*phspar2[p]['phiy']))
                if a > 92: proj *= np.exp(1j*phspar2[p]['offset_south'])
                else: proj *= np.exp(1j*phspar2[p]['offset_east'])
                self.red[p][a] *= proj
                if self.red[p][a].ndim == 2 and time_average:
                    self.red[p][a] = np.mean(self.red[p][a], axis=0)
            if self.mdl:
                pp = p + p
                for bl in self.mdl[pp].keys():
                    i,j = bl
                    if i < 93:
                        self.mdl[pp][bl] *= (ref_exp1*np.exp(-1j*phspar2[p]['offset_east']))
                        nxi = np.argwhere(EastHex==i)[0][1] - np.argwhere(EastHex==ref1)[0][1]
                        nyi = np.argwhere(EastHex==i)[0][0] - np.argwhere(EastHex==ref1)[0][0]
                    else:
                        self.mdl[pp][bl] *= (ref_exp2*np.exp(-1j*phspar2[p]['offset_south']))
                        nxi = np.argwhere(SouthHex==i)[0][1] - np.argwhere(SouthHex==ref2)[0][1]
                        nyi = np.argwhere(SouthHex==i)[0][0] - np.argwhere(SouthHex==ref2)[0][0]
                    if j < 93:
                        self.mdl[pp][bl] *= (ref_exp1.conj()*np.exp(1j*phspar2[p]['offset_east']))
                        nxj = np.argwhere(EastHex==j)[0][1] - np.argwhere(EastHex==ref1)[0][1]
                        nyj = np.argwhere(EastHex==j)[0][0] - np.argwhere(EastHex==ref1)[0][0]
                    else:
                        self.mdl[pp][bl] *= (ref_exp2.conj()*np.exp(1j*phspar2[p]['offset_south']))
                        nxj = np.argwhere(SouthHex==j)[0][1] - np.argwhere(SouthHex==ref2)[0][1]
                        nyj = np.argwhere(SouthHex==j)[0][0] - np.argwhere(SouthHex==ref2)[0][0]
                    dx = antpos[i]['top_x']-antpos[j]['top_x']
                    dy = antpos[i]['top_y']-antpos[j]['top_y']
                    proj = amppar[p]*amppar[p]*np.exp(1j*((nxi-nxj)*phspar[p]['phi1']+(nyi-nyj)*phspar[p]['phi2']))*np.exp(1j*(dx*phspar2[p]['phix']+dy*phspar2[p]['phiy']))
                    proj = np.resize(proj, self.mdl[pp][bl].shape)
                    ind = np.where(proj!=0)
                    self.mdl[pp][bl][ind] /= proj[ind]

    def degen_project_to_unit(self, time_average=True):
        """
        Project degeneracy parameters to 1.0
        """
        assert self.red, "Redundant calibration gains does not exit."
        if time_average:
            print("The gains will be averaged in time axis after degeneracy projection by default. \
            Set time_average=False if you do not want to.")
        self.scale_gains()
        phspar = self.plane_fitting()
        for p in self.red.keys():
            for a in self.red[p].keys():
                dx = antpos[a]['top_x']
                dy = antpos[a]['top_y']
                proj = np.exp(1j*(dx*phspar[p]['phix']+dy*phspar[p]['phiy']))
                if a > 92: proj *= np.exp(1j*phspar[p]['offset_south'])
                else: proj *= np.exp(1j*phspar[p]['offset_east'])
                self.red[p][a] *= proj
                if self.red[p][a].ndim == 2 and time_average:
                    self.red[p][a] = np.mean(self.red[p][a], axis=0)
            if self.mdl:
                pp = p + p
                for bl in self.mdl[pp].keys():
                    i,j = bl
                    if i < 93: self.mdl[pp][bl] *= np.exp(-1j*phspar[p]['offset_east'])
                    else: self.mdl[pp][bl] *= np.exp(-1j*phspar[p]['offset_south'])
                    if j < 93: self.mdl[pp][bl] *= np.exp(1j*phspar[p]['offset_east'])
                    else: self.mdl[pp][bl] *= np.exp(1j*phspar[p]['offset_south'])
                    dx = antpos[i]['top_x']-antpos[j]['top_x']
                    dy = antpos[i]['top_y']-antpos[j]['top_y']
                    proj = np.exp(-1j*(dx*phspar[p]['phix']+dy*phspar[p]['phiy']))
                    self.mdl[pp][bl] *= proj

    def decouple_cross_mode(self, res):
        ripple = {}
        dmode = 0.05
        nmode = 50
        nf = self.freqs.size
        band = (self.freqs[-1]-self.freqs[0]) * nf / (nf - 1)
        fq = self.freqs
        for p in res.keys():
            ripple[p] = {}
            for a in res[p].keys():
                resautos = []
                resphase = []
                ind = np.where(res[p][a]!=0)[0]
                flg = np.where(res[p][a]==0)[0]
                for aa in res[p].keys():
                    if tile_info[aa]['cable'] == tile_info[a]['cable']: continue
                    amps = self.auto[p][a] / self.auto[p][aa]
                    amps /= np.mean(amps)
                    resautos.append(amps)
                    resphase.append(res[p][a] - res[p][aa])
                resautos = np.mean(resautos, axis=0)
                resphase = np.mean(resphase, axis=0)
                resautos -= np.mean(resautos)
                reftime = 2.*tile_info[a]['cable'] / (c_light * tile_info[a]['vf'])
                resautos[flg],_ = GPR_interp(fq[ind], resautos[ind], fq[flg], col=1./reftime)
                resphase[flg] = 0.
                resautos[:ind[0]] = 0.
                resautos[ind[-1]+1:] = 0.
                modes = np.linspace(-dmode*nmode, dmode*nmode, 2*nmode+1) + band * reftime
                freq_mat = np.resize(np.arange(nf), (2*nmode+1, nf))
                t1 = np.sum(np.sin(2*np.pi/nf*modes*freq_mat.T).T*resautos, axis=1)
                t2 = np.sum(np.cos(2*np.pi/nf*modes*freq_mat.T).T*resautos, axis=1)
                i = np.argmax(t1**2+t2**2)
                mi = modes[i]
                nu = ind.size
                t1i = np.sum(np.sin(2*np.pi/nf*mi*np.arange(nf))*resphase) / nu
                t2i = np.sum(np.cos(2*np.pi/nf*mi*np.arange(nf))*resphase) / nu
                phase_ripple = 2*t1i*np.sin(2*np.pi*(mi*np.arange(nf)/nf)) + 2*t2i*np.cos(2*np.pi*(mi*np.arange(nf)/nf))
                ripple[p][a] = phase_ripple
        return ripple

    def bandpass_fitting(self, include_red = False):
        """
        Calculate the global bandpass as the amplitude, fit linear in the phase,
        fit 150 m cable reflection
        """
        self.gfit = copy.deepcopy(self.sky)
        for p in self.auto.keys():
            for a in self.auto[p].keys():
                self.gfit[p][a] /= (self.auto[p][a]+1e-10)
        if include_red:
            for p in self.red.keys():
                for a in self.red[p].keys():
                    self.gfit[p][a] *= self.red[p][a]
        nf = self.freqs.size
        flgc = np.where(self.mask)[0]
        self.gbp = {}
        residuals = {}
        for p in self.gfit.keys():
            self.gbp[p] = []
            residuals[p] = {}
            for a in self.gfit[p].keys():
                if np.any(np.isnan(self.gfit[p][a])): continue
                x = np.copy(self.gfit[p][a])
                x[flgc] = 0
                self.gfit[p][a][flgc] = 0
                md = np.ma.masked_array(x, np.zeros(x.shape, dtype=bool))
                ind = np.where(x==0)
                md.mask[ind] = True
                self.gbp[p].append(np.abs(md)/np.mean(np.abs(md)))
                fqc = np.arange(md.size)
                ind = np.logical_not(md.mask)
                x = fqc[ind]
                y = np.unwrap(np.angle(md.data[ind]))
                z = np.polyfit(x,y,1)
                res = self.gfit[p][a] * np.exp(-1j*(z[0]*fqc+z[1]))
                residuals[p][a] = np.zeros(res.shape)
                indp = np.where(md.mask==False)[0]
                residuals[p][a][indp] = np.angle(res[indp])
                self.gfit[p][a] = np.abs(self.gfit[p][a]) * np.exp(1j*(z[0]*fqc+z[1]))
            self.gbp[p] = np.ma.masked_array(self.gbp[p])
            self.gbp[p] = np.mean(self.gbp[p], axis = 0)
            for a in self.gfit[p].keys():
                if np.any(np.isnan(self.gfit[p][a])): continue
                ind = np.where(self.gbp[p].data * self.gfit[p][a] != 0)[0]
                amp = np.mean(np.abs(self.gfit[p][a][ind])) / np.mean(self.gbp[p].data[ind])
                self.gfit[p][a] = (self.auto[p][a]+1e-10) * amp * self.gbp[p].data * np.exp(1j*np.angle(self.gfit[p][a]))
        ripples = self.decouple_cross_mode(residuals)
        for p in ripples.keys():
            for a in ripples[p].keys():
                self.gfit[p][a] *= np.exp(1j*ripples[p][a])

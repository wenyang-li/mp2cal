import numpy as np, copy
from pos import *
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
    def __init__(self, freqs = None):
        self.red = None # Gain from redundant cal
        self.sky = None # Gain from sky cal
        self.mdl = None # Model vis from redundant cal
        self.gbp = None # Global bandpass, needs to be calculated from the input gains
        self.gfit = None # Smoothed gains, needs to be calculated from the input gains
        self.freqs = freqs # Frequency array

    def get_red(self, g_red):
        """
        Load redundant cal gains.
        """
        self.red = g_red
        for p in self.red.keys():
            for a in self.red[p].keys():
                if self.red[p][a].ndim == 2:
                    self.red[p][a] = np.mean(self.red[p][a], axis=0)

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
            if g_input[p][a0].ndim == 2:
                for a in g_input[p].keys(): g_input[p][a] = np.mean(g_input[p][a],axis=0)
            if g_target[p][a0].ndim == 2:
                for a in g_target[p].keys(): g_target[p][a] = np.mean(g_target[p][a],axis=0)
            nf = g_target[p][a0].size
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
                    print "invalid value of gain in channel " + str(ff)
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
                    print "invalid value of gain in channel " + str(ff)
                    slp2.append(0)
                else:
                    slope = np.unwrap(slope)
                    slp2.append(np.median(slope))
            phspar[p]['phi1'] = np.array(slp1)
            phspar[p]['phi2'] = np.array(slp2)
        return phspar

    def plane_fitting(self, ratio = None):
        """
        If the solutions are supposed to be centered at 1.0, remove any phase gradients or
        phase offsets by fitting a plane in (phase, rx, ry) space.
        """
        assert self.red, "Redundant calibration gains do not exit."
        gc = {}
        if ratio: gc = ratio
        else:
            for p in self.red.keys():
                gc[p] = {}
                for a in self.red[p].keys():
                    SH = self.red[p][a].shape
                    gc[p][a] = self.red[p][a].flatten()
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

    def degen_project_OF(self):
        """
        Project degeneracy of redundant cal from raw data to sky cal
        """
        assert self.red, "Redundant calibration gains do not exit."
        assert self.sky, "Sky calibration gains do not exit."
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
                    dx = np.argwhere(EastHex==a)[0][1] - np.argwhere(EastHex==ref1)[0][1]
                    dy = np.argwhere(EastHex==a)[0][0] - np.argwhere(EastHex==ref1)[0][0]
                else:
                    dx = np.argwhere(SouthHex==a)[0][1] - np.argwhere(SouthHex==ref2)[0][1]
                    dy = np.argwhere(SouthHex==a)[0][0] - np.argwhere(SouthHex==ref2)[0][0]
                nx = dx
                ny = dy
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
            if self.mdl:
                pp = p + p
                for bl in self.mdl[pp].keys():
                    i,j = bl
                    if i < 93: self.mdl[pp][bl] *= (ref_exp1*np.exp(-1j*phspar2[p]['offset_east']))
                    else: self.mdl[pp][bl] *= (ref_exp2*np.exp(-1j*phspar2[p]['offset_south']))
                    if i < 93: self.mdl[pp][bl] *= (ref_exp1.conj()*np.exp(1j*phspar2[p]['offset_east']))
                    else: self.mdl[pp][bl] *= (ref_exp2.conj()*np.exp(1j*phspar2[p]['offset_south']))
                    dx = antpos[i]['top_x']-antpos[j]['top_x']
                    dy = antpos[i]['top_y']-antpos[j]['top_y']
                    nx = np.round(dx/0.14-dy/np.sqrt(3)/0.14)
                    ny = np.round(-2*dy/np.sqrt(3)/0.14)
                    proj = amppar[p]*amppar[p]*np.exp(1j*(nx*phspar[p]['phi1']+ny*phspar[p]['phi2']))*np.exp(1j*(dx*phspar2[p]['phix']+dy*phspar2[p]['phiy']))
                    proj = np.resize(proj, self.mdl[pp][bl].shape)
                    ind = np.where(proj!=0)
                    self.mdl[pp][bl][ind] /= proj[ind]

    def degen_project_FO(self):
        """
        Project degeneracy parameters to 1.0
        """
        assert self.red, "Redundant calibration gains does not exit."
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
            if self.mdl:
                pp = p + p
                for bl in self.mdl[pp].keys():
                    i,j = bl
                    if i < 93: self.mdl[pp][bl] *= np.exp(-1j*phspar[p]['offset_east'])
                    else: self.mdl[pp][bl] *= np.exp(-1j*phspar[p]['offset_south'])
                    if i < 93: self.mdl[pp][bl] *= np.exp(1j*phspar[p]['offset_east'])
                    else: self.mdl[pp][bl] *= np.exp(1j*phspar[p]['offset_south'])
                    dx = antpos[i]['top_x']-antpos[j]['top_x']
                    dy = antpos[i]['top_y']-antpos[j]['top_y']
                    proj = np.exp(-1j*(dx*phspar[p]['phix']+dy*phspar[p]['phiy']))
                    self.mdl[pp][bl] *= proj

    def bandpass_fitting(self, include_red = False):
        """
        Calculate the global bandpass as the amplitude, fit linear in the phase,
        fit 150 m cable reflection
        """
        self.gfit = copy.deepcopy(self.sky)
        if include_red:
            for p in self.gfit.keys():
                for a in self.gfit[p].keys():
                    self.gfit[p][a] *= self.red[p][a]
        nf = self.freqs.size
        flgc = []
        if nf == 384:
            for ii in range(nf):
                if ii%16 in [0, 15]: flgc.append(ii)
        if nf == 768:
            for ii in range(nf):
                if ii%32 in [0,1,16,30,31]: flgc.append(ii)
        band = (self.freqs[-1]-self.freqs[0]) * nf / (nf - 1)
        self.gbp = {}
        for p in self.gfit.keys():
            self.gbp[p] = []
            for a in self.gfit[p].keys():
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
                phs = 1.
                if tile_info[a]['cable']==150:
                    reftime = 300. / (c_light * tile_info[a]['vf'])
                    nu = np.sum(np.logical_not(md.mask))
                    res *= np.logical_not(md.mask)
                    res0 = np.zeros(res.shape)
                    indp = np.where(md.mask==False)[0]
                    res0[indp] = np.angle(res[indp])
                    dmode = 0.05
                    nmode = 50
                    modes = np.linspace(-dmode*nmode, dmode*nmode, 2*nmode+1) + band * reftime
                    res = np.resize(res, (2*nmode+1, nf))
                    freq_mat = np.resize(np.arange(nf), (2*nmode+1, nf))
                    t1 = np.sum(np.sin(2*np.pi/nf*modes*freq_mat.T).T*res0, axis=1)
                    t2 = np.sum(np.cos(2*np.pi/nf*modes*freq_mat.T).T*res0, axis=1)
                    i = np.argmax(t1**2+t2**2)
                    mi = modes[i]
                    phase_ripple = 2*t1[i]*np.sin(2*np.pi*(mi*np.arange(nf)/nf))/nu + \
                                   2*t2[i]*np.cos(2*np.pi*(mi*np.arange(nf)/nf))/nu
                    phs = np.exp(1j*phase_ripple)
                self.gfit[p][a] = np.abs(self.gfit[p][a]) * np.exp(1j*(z[0]*fqc+z[1])) * phs
            self.gbp[p] = np.ma.masked_array(self.gbp[p])
            self.gbp[p] = np.mean(self.gbp[p], axis = 0)
            for a in self.gfit[p].keys():
                ind = np.where(self.gbp[p].data * self.gfit[p][a] != 0)[0]
                amp = np.mean(np.abs(self.gfit[p][a][ind])) / np.mean(self.gbp[p].data[ind])
                self.gfit[p][a] = amp * self.gbp[p].data * np.exp(1j*np.angle(self.gfit[p][a]))

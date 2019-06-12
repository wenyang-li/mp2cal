import numpy as np, os, warnings, matplotlib.pyplot as plt
from astropy.io import fits
from gain import RedGain
from pos import *

pol_lookup = {'xx': -5, 'yy': -6, 'xy': -7, 'yx': -8}

def output_mask_array(flag_array):
    wgts = np.logical_not(flag_array)
    wgts = np.sum(wgts, axis=1)
    return wgts == 0

class RedData(object):
    """
    A container that includes all information of the data, as well as the gain object
    containing calibration gains.
    """
    def __init__(self, pol):
        self.pol = pol # Polarization
        self.freqs = None # Frequency
        self.data = {} # Data dictionary for omnical. data={bl:{pol:{data_array}}}
        self.flag = {} # Flag dictionary for omnical, flag={bl:{pol:{flag_array}}}
        self.noise = {}# Noise dictionary
        self.dead = [] # Flagged antenna
        self.mask = None # mask for all baselines, with shape (Ntime, Nfreq) if not tave, else (1, Nfreq)
        self.shape_waterfall = None # shape (Ntime, Nfreq)
        self.gains = None # calibrations
        self.data_backup = {} # raw data backup for chi-square calculation and model vis calculation
        self.flag_backup = {} # raw flag backup for chi-square calculation and model vis calculation
        self.chisq_base = {} # this is for chi-square visualization for different redundant baseline group

    def get_ex_ants(self, ex_ants):
        """
        Get excluded antennas from provided information.
        """
        for a in ex_ants:
            if not a in self.dead:
                self.dead.append(a)

    def get_dead_dipole(self, metafits):
        """
        Get dead dipole information from metafits. Add antennas with dead dipoles to excluded antennas
        """
        if os.path.exists(metafits):
            print("    Finding dead dipoles in metafits")
            hdu = fits.open(metafits)
            inds = np.where(hdu[1].data['Delays']==32)[0]
            for ind in inds:
                a = hdu[1].data['Antenna'][ind]
                if hdu[1].data['Pol'][ind].lower() in self.pol and not a in self.dead:
                    self.dead.append(a)
        else:
            warnings.warn("    Warning: Metafits not found. Cannot get the information of dead dipoles")

    def read_data(self, uv, tave=False):
        """
        Read data from a pyuvdata obsject.
        tave: average data over time.
        """
        self.freqs = uv.freq_array[0]
        a1 = uv.ant_1_array[:uv.Nbls]
        a2 = uv.ant_2_array[:uv.Nbls]
        for a in uv.antenna_numbers:
            if not a in a1 and not a in a2:
                if not a in self.dead: self.dead.append(a)
        pid = np.where(uv.polarization_array == pol_lookup[self.pol])[0][0]
        data = uv.data_array[:,0,:,pid].reshape(uv.Ntimes,uv.Nbls,uv.Nfreqs)
        flag = uv.flag_array[:,0,:,pid].reshape(uv.Ntimes,uv.Nbls,uv.Nfreqs)
        ind = np.where(a1!=a2)[0]
        self.mask = output_mask_array(flag[:,ind])
        self.shape_waterfall = (uv.Ntimes, uv.Nfreqs)
        def creat_dict(ii):
            if a1[ii] < 57 or a2[ii] < 57 or a1[ii] == a2[ii]: return # hard coded for MWA Phase II
            if a1[ii] in self.dead or a2[ii] in self.dead: return
            bl = (a1[ii],a2[ii])
            md = np.ma.masked_array(data[:,ii],flag[:,ii])
            diff = md[1:] - md[:-1]
            self.noise[bl] = np.var(diff,axis=0).data/2
            zerofq = np.where(np.sum(np.logical_not(diff.mask),axis=0) < 3)[0]
            md.mask[:,zerofq] = True
            self.data_backup[bl] = {self.pol: np.complex64(md.data)}
            self.flag_backup[bl] = {self.pol: np.copy(md.mask)}
            if tave:
                md = np.mean(md,axis=0,keepdims=True)
            self.data[bl] = {self.pol: np.complex64(md.data)}
            self.flag[bl] = {self.pol: md.mask}
        map(creat_dict, np.arange(uv.Nbls))
        if tave: self.mask= np.product(self.mask, axis=0, keepdims=True).astype(bool)
        mask = np.copy(self.mask)
        if mask.ndim == 2: mask = np.product(self.mask, axis=0).astype(bool)
        self.gains = RedGain(freqs=self.freqs, mask=mask)
        self.gains.get_auto(uv)

    def apply_fhd(self, gfhd):
        """
        Apply FHD sky based calibration to data.
        """
        for bl in self.data.keys():
            i,j = bl
            p1,p2 = self.pol
            G = gfhd[p1][i]*gfhd[p2][j].conj()
            ind = np.where(G != 0)[0]
            self.data[bl][self.pol][:,ind] /= G[ind]

    def get_gains(self, g_red,  v_mdl = None, g_sky = None):
        """
        Read gain solutions and add to this object
        """
        self.gains.get_red(g_red)
        if g_sky: self.gains.get_sky(g_sky)
        if v_mdl: self.gains.get_mdl(v_mdl)
    
    def recover_model_vis_waterfall(self, info, g = None):
        """
        If tave is set to true, recover model vis using calibration solutions
        """
        reds = info.get_reds()
        p1, p2 = self.pol
        v_mdl = {self.pol: {}}
        if g is None:
            if self.gains.gfit is None:
                self.gains.bandpass_fitting(include_red = True)
            g = self.gains.gfit
        SH = self.shape_waterfall
        if not self.data_backup:
            print("The tave is set to False, no need to recalculate vis model.")
            return
        for r in reds:
            bl0 = r[0]
            num = 0
            den = 0
            for bl in r:
                i,j = bl
                try:
                    di = self.data_backup[bl][self.pol]
                    wi = np.logical_not(self.flag_backup[bl][self.pol])
                except:
                    di = self.data_backup[bl[::-1]][self.pol].conj()
                    wi = np.logical_not(self.flag_backup[bl[::-1]][self.pol])
                num += di * g[p1][i].conj() * g[p2][j] * wi
                den += np.resize(g[p1][i].conj() * g[p1][i] * g[p2][j].conj() * g[p2][j], SH) * wi
            v_mdl[self.pol][bl0] = num / (den+1e-10)
        self.gains.get_mdl(v_mdl)


    def cal_chi_square(self, info, meta, per_bl_chi2=False, g = None):
        """
        Compute omnical chi-square, add them into meta container.
        """
        reds = info.get_reds()
        SH = self.shape_waterfall
        chisq = np.zeros(SH)
        weight = np.zeros(SH)
        p1, p2 = self.pol
        if g is None:
            if self.gains.gfit is None:
                self.gains.bandpass_fitting(include_red = True)
            g = self.gains.gfit
        noise = []
        for bl in self.noise.keys():
            i,j = bl
            try: G = np.abs(g[p1][i]*g[p2][j])
            except: continue
            noise.append(self.noise[bl]/(G*G+1e-10))
        noise = np.ma.masked_array(noise, np.zeros((len(noise), len(noise[0]))))
        noise.mask[np.where(noise==0)] = True
        noise = np.mean(noise, axis=0).data
        if self.gains.mdl is None: self.recover_model_vis_waterfall(info, g = g)
        mdl = self.gains.mdl
        data_arr = None
        flag_arr = None
        if self.data_backup:
            data_arr = self.data_backup
            flag_arr = self.flag_backup
        else:
            data_arr = self.data
            flag_arr = self.flag
        for r in reds:
            if len(r) < 5: continue
            bl0 = None
            yij = None
            for bl in r:
                if mdl[self.pol].has_key(bl):
                    yij = mdl[self.pol][bl]
                    bl0 = bl
                    break
            if bl0 is None: continue
            chis = np.zeros(SH)
            wgts = np.zeros(SH)
            for bl in r:
                try:
                    di = data_arr[bl][self.pol]
                    wi = np.logical_not(flag_arr[bl][self.pol])
                except(KeyError):
                    di = data_arr[bl[::-1]][self.pol].conj()
                    wi = np.logical_not(flag_arr[bl[::-1]][self.pol])
                i,j = bl
                chis += (np.abs(di/(g[p1][i]*g[p2][j].conj()+1e-10)-yij))**2 * wi / (noise + 1e-10)
                wgts += wi
            iuse = np.where(wgts>1)
            self.chisq_base[bl0] = np.mean(chis[iuse]-(wgts[iuse]-1))
            if per_bl_chi2:
                meta['chisq'+'('+str(bl0[0])+','+str(bl0[1])+')'] = chis
                meta['wgts'+'('+str(bl0[0])+','+str(bl0[1])+')'] = wgts
            chisq += chis
            weight += wgts
        meta['chisq'] = chisq * (weight > 1) / (weight - 1 + 1e-10)
        meta['flags'] = weight < 2

    def plot_chisq(self, meta, outdir, obsname):
        fq = self.freqs / 1e6
        m = np.ma.masked_array(meta['chisq'], meta['flags'])
        fig = plt.figure(figsize=(10,5))
        p1 = fig.add_subplot(2,1,1)
        i1 = p1.imshow(m,aspect='auto',cmap='coolwarm',extent=(fq[0],fq[-1],len(m)-1,0))
        plt.colorbar(i1)
        p2 = fig.add_subplot(2,1,2)
        mu=np.mean(m)
        sig=np.std(m)
        b=np.linspace(mu-5*sig,mu+5*sig,100)
        p2.hist(m[np.where(m.mask==False)],bins=b,normed=1,histtype='step')
        p2.plot(b,1/np.sqrt(2*np.pi)/sig*np.exp(-(b-mu)**2/2/sig**2))
        p2.set_yscale('log')
        p2.set_xlabel('$\chi^2$')
        plt.suptitle(obsname+' '+self.pol)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92,hspace=0.25)
        plt.savefig(outdir + obsname + '_chisq_waterfall_'+self.pol + '.png')

    def plot_chisq_per_bl(self, outdir, obsname):
        x, y, c = [], [], []
        for bl in self.chisq_base.keys():
            i,j = bl
            r1 = antpos[i]
            r2 = antpos[j]
            x.append(r2['top_x'] - r1['top_x'])
            y.append(r2['top_y'] - r1['top_y'])
            c.append(self.chisq_base[bl])
        plt.scatter(x, y, c=c, cmap='coolwarm')
        plt.xlabel('East-West Coordinate (m)')
        plt.ylabel('North-South Coordinate (m)')
        plt.colorbar()
        plt.grid(True)
        plt.title(obsname + '_' + self.pol)
        plt.savefig(outdir + obsname + '_bl_chisq_'+self.pol + '.png')


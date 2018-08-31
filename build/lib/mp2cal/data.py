import numpy as np, os
from astropy.io import fits
from gain import RedGain

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
        self.pol = pol
        self.data = {}
        self.flag = {}
        self.noise = {}
        self.dead = []
        self.mask = None
        self.gains = RedGain()

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
            print '    Finding dead dipoles in metafits'
            hdu = fits.open(metafits)
            inds = np.where(hdu[1].data['Delays']==32)[0]
            for ind in inds:
                a = hdu[1].data['Antenna'][ind]
                if hdu[1].data['Pol'][ind].lower() in self.pol and not a in self.dead:
                    self.dead.append(a)
        else:
            print '    Warning: Metafits not found. Cannot get the information of dead dipoles'

    def read_data(self, uv, tave=False):
        """
        Read data from a pyuvdata obsject.
        tave: average data over time.
        """
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
        for ii in range(uv.Nbls):
            if a1[ii] < 57 or a2[ii] < 57 or a1[ii] == a2[ii]: continue # hard coded for MWA Phase II
            if a1[ii] in self.dead or a2[ii] in self.dead: continue
            bl = (a1[ii],a2[ii])
            md = np.ma.masked_array(data[:,ii],flag[:,ii])
            diff = md[1:] - md[:-1]
            self.noise[bl] = np.var(diff,axis=0).data/2
            zerofq = np.where(np.sum(np.logical_not(diff.mask),axis=0) < 3)[0]
            md.mask[:,zerofq] = True
            if tave:
                md = np.mean(md,axis=0,keepdims=True)
                self.noise[bl] /= (uv.Ntimes-np.count_nonzero(np.product(self.mask,axis=1)))
            self.data[bl] = {self.pol: np.complex64(md.data)}
            self.flag[bl] = {self.pol: md.mask}
        if tave: self.mask= np.product(self.mask, axis=0, keepdims=True).astype(bool)

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

    def get_gains(self, g_red,  v_mdl, g_sky = None):
        """
        Read gain solutions and add to this object
        """
        self.gains.get_red(g_red)
        if g_sky: self.gains.get_sky = g_sky
        self.gains.get_mdl(v_mdl)

    def cal_chi_square(self, info, meta):
        """
        Compute omnical chi-square, add them into meta container.
        """
        reds = info.get_reds()
        chisq = 0.
        chisqbls = {}
        g = self.gains.red
        mdl = self.gains.mdl
        p1, p2 = self.pol
        for r in reds:
            bl0 = None
            yij = None
            for bl in r:
                if mdl[self.pol].has_key(bl):
                    yij = np.ma.masked_array(mdl[self.pol][bl], mask=self.mask)
                    bl0 = bl
                    chisqbls[bl0] = 0.
                    break
            for bl in r:
                try: md = np.ma.masked_array(self.data[bl][self.pol], mask=self.mask)
                except(KeyError): np.ma.masked_array(self.data[bl[::-1]][self.pol].conj(), mask=self.mask)
                i,j = bl
                try: chisqterm = (np.abs(md-g[p1][i]*g[p2][j].conj()*yij))**2/self.noise[bl]
                except(KeyError): chisqterm = (np.abs(md-g[p1][i]*g[p2][j].conj()*yij))**2/self.noise[bl[::-1]]
                chisq += chisqterm
                chisqbls[bl0] += chisqterm
            meta['chisq('+str(bl0[0])+','+str(bl0[1])+')'] = chisqbls[bl0].data / (len(r)-1.)
        DOF = (info.nBaseline - info.nAntenna - info.ublcount.size)
        meta['chisq'] = chisq.data / float(DOF)
        meta['flags'] = self.mask

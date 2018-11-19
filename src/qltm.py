import numpy as np, matplotlib.pyplot as plt

class INS(object):
    """
    Using incoherence noise spectrum as data quality metric (Wilensky et al, in prep)
    """
    def __init__(self, uv):
        self.uv = uv
        d = uv.data_array.reshape(uv.Ntimes,uv.Nbls,uv.Nspws,uv.Nfreqs,uv.Npols)
        f = uv.flag_array.reshape(uv.Ntimes,uv.Nbls,uv.Nspws,uv.Nfreqs,uv.Npols)
        md = np.ma.masked_array(d, f)
        md = md[1:] - md[:-1]
        self.ins = np.mean(np.abs(md),axis=1)
        self.mask = np.copy(self.ins.mask)

    def outliers_flagging(self, nsig = 5):
        """
        Assuming frac_ins follows Gaussian distribution, flag (time,freq,pol) samples that deviate
        from the mean by nsig sigmas.
        """
        for niter in range(self.ins.size):
            frac_diff = self.ins / np.mean(self.ins, axis=0) - 1
            m = np.mean(frac_diff)
            s = np.std(frac_diff)
            ind = np.where(abs(frac_diff-m).data*np.logical_not(self.ins.mask) > nsig*s)
            if ind[0].size == 0: break
            else: self.ins.mask[ind] = True

    def smooth_over_freq(self, fc=1):
        """
        Using Gaussian Kernel to smooth the frac ins over frequency for each time step.
        fc = frequency coherence length / coarse bandwidth
        """
        frac_diff = self.ins / np.mean(self.ins, axis=0) - 1
        SH = self.ins.shape
        nc = SH[2] / 24
        cf = int(fc*nc)
        m2 = np.zeros(SH)
        x = np.linspace(-cf,cf,2*cf+1)
        window = np.zeros((SH[0],SH[1],2*cf+1,SH[3]))
        for ii in range(2*cf+1): window[:,:,ii,:]=np.exp(-(x[ii]/float(cf))**2)
        for ff in range(SH[2]):
            min_ind = max(0,ff-cf)
            max_ind = min(SH[2],ff+cf+1)
            dm = np.sum(frac_diff[:,:,min_ind:max_ind,:]*window[:,:,cf-(ff-min_ind):cf+(max_ind-ff),:],axis=2)
            dn = np.sum(np.logical_not(frac_diff.mask[:,:,min_ind:max_ind,:])*window[:,:,cf-(ff-min_ind):cf+(max_ind-ff),:],axis=2)+1e-10
            m2[:,:,ff,:] = dm.data/dn
        return m2

    def time_flagging(self, nthresh=1e-5):
        """
        Flag bad time slices
        """
        maxiter = self.ins.shape[0]
        frac_smt = self.smooth_over_freq(fc=1)
        frac_co = frac_smt[:,:,1:,:]*frac_smt[:,:,:-1,:]
        dt_slice = np.max(np.mean(frac_co,axis=(1,2)),axis=1)
        dt_ind = np.where(np.abs(dt_slice) > nthresh)[0]
        self.ins.mask[dt_ind] = True

    def freq_flagging(self, nsig=6):
        """
        Flag bad frequency channels
        """
        frac_diff = self.ins / np.mean(self.ins, axis=0) - 1
        df_slice = frac_diff[1:]*frac_diff[:-1]
        df_slice = np.mean(df_slice,axis=(0,1,3))
        df_slice -= np.mean(df_slice)
        df_ind = np.where(df_slice>nsig*np.std(df_slice))[0]
        self.ins.mask[:,:,df_ind,:] = True

    def coherence_flagging(self, fc=0.25, nsig=6):
        """
        Flag samples based on smoothed (over frequency) ins. This is to find coherences in noise spectrum
        """
        for niter in range(self.ins.size):
            m2 = self.smooth_over_freq(fc=fc)
            m2 = np.ma.masked_array(m2, self.ins.mask)
            sigma = np.std(m2)
            mean2 = np.mean(m2)
            ind = np.where(abs(m2-mean2).data*np.logical_not(self.ins.mask) > nsig*sigma)
            if ind[0].size == 0: break
            self.ins.mask[ind] = True

    def extend_flagging(self, f_thresh=0.5, t_thresh=0.5):
        """
        Flag the whole time/frequency slice if the number of flagged smaples in that slice is
        above a given threshold
        """
        mt = np.max(np.mean(self.ins.mask, axis=(1,2)),axis=1)
        indt = np.where(mt>t_thresh)[0]
        mf = np.max(np.mean(self.ins.mask, axis=(0,1)),axis=1)
        indf = np.where(mf>f_thresh)[0]
        self.ins.mask[indt,:,:,:] = True
        self.ins.mask[:,:,indf,:] = True

    def apply_flagging(self):
        """
        Apply the flagging to the uv object
        """
        SH = (self.uv.Ntimes,self.uv.Nspws,self.uv.Nfreqs,self.uv.Npols)
        mask1 = np.zeros(SH, dtype=bool)
        mask2 = np.zeros(SH, dtype=bool)
        mask1[0] = True
        mask1[1:] = self.ins.mask
        mask2[-1] = True
        mask2[:-1] = self.ins.mask
        mask_all = np.logical_and(mask1,mask2)
        for ii in range(self.uv.Nbls):
            self.uv.flag_array[ii::self.uv.Nbls] = np.logical_or(self.uv.flag_array[ii::self.uv.Nbls], mask_all)

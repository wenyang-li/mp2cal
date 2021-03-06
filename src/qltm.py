import numpy as np, matplotlib.pyplot as plt, os

class INS(object):
    """
    Using incoherence noise spectrum as data quality metric (Wilensky et al, in prep)
    """
    def __init__(self, uv = None, ins_arr = None):
        """
        uv:
            pyuvdata object, which should have read in interferometric data. The incoherence noise spectrum is
            calculated based on uv.
        ins_arr:
            Directly input ins which has already been calculated.
        If both uv and ins_arr are provided, only use the information of uv object.
        """
        if uv is None and ins_arr is None:
            raise IOError("Object requires either uv object or input INS array")
        elif uv is None and ins_arr is not None:
            print("Reads in INS from saved array. apply_flagging cannot be used because uv is None. ")
            self.ins = ins_arr
            self.mask = np.copy(ins_arr.mask)
        else:
            if ins_arr is not None:
                print("calculate INS from uv object. ins_arr is discarded")
            self.uv = uv
            self.cal_ins()
    
    def cal_ins(self):
        d = self.uv.data_array.reshape(self.uv.Ntimes, self.uv.Nbls, self.uv.Nspws, self.uv.Nfreqs, self.uv.Npols)
        f = self.uv.flag_array.reshape(self.uv.Ntimes, self.uv.Nbls, self.uv.Nspws, self.uv.Nfreqs, self.uv.Npols)
        a1 = self.uv.ant_1_array[:self.uv.Nbls]
        a2 = self.uv.ant_2_array[:self.uv.Nbls]
        ind = np.where(a1!=a2)[0]
        md = np.ma.masked_array(d[:,ind,:,:], f[:,ind,:,:])
        md = md[1:] - md[:-1]
        self.ins = np.mean(np.abs(md),axis=1)
        self.mask = np.copy(self.ins.mask)

    def outliers_flagging(self, nsig = 5):
        """
        Assuming frac_ins follows Gaussian distribution, flag (time,freq,pol) samples that deviate
        from the mean by nsig sigmas.
        """
        for niter in range(self.ins.size):
            frac_diff = self.ins / np.ma.median(self.ins, axis=0) - 1
            m = np.mean(frac_diff)
            s = np.std(frac_diff)
            ind = np.where(abs(frac_diff-m).data*np.logical_not(self.ins.mask) > nsig*s)
            if ind[0].size == 0: break
            else: self.ins.mask[ind] = True

    def outliers_flagging_extreme(self, nsig = 5):
        """
        Assuming frac_ins follows Gaussian distribution, flag time samples where there exists any sample
        that deviate from the mean by nsig sigmas.
        """
        frac_diff = self.ins / np.ma.median(self.ins, axis=0) - 1
        m = np.mean(frac_diff)
        s = np.std(frac_diff)
        ind = np.where(abs(frac_diff-m).data*np.logical_not(self.ins.mask) > nsig*s)
        for ii in range(ind[0].size):
            self.ins.mask[ind[0][ii],:,:,ind[3][ii]] = True

    def smooth_over_freq(self, fc=1):
        """
        Using Gaussian Kernel to smooth the frac ins over frequency for each time step.
        fc = frequency coherence length / coarse bandwidth
        """
        frac_diff = self.ins / np.ma.median(self.ins, axis=0) - 1
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

    def time_flagging(self, nsig=10):
        """
        Flag bad time slices
        """
        frac_diff = self.ins / np.ma.median(self.ins, axis=0) - 1
        s = np.std(frac_diff)
        dt_slice = np.sum(frac_diff,axis=(1,2,3))
        nsample = np.sum(np.logical_not(frac_diff.mask), axis=(1,2,3))
        dt_slice /= np.sqrt(nsample+1e-10)
        ind = np.where(dt_slice > nsig * s)[0]
        self.ins.mask[ind] = True

    def freq_flagging(self, nsig=6, frac_thresh=0.5):
        """
        Flag bad frequency channels
        """
        frac_diff = self.ins / np.ma.median(self.ins, axis=0) - 1
        df_slice = frac_diff[1:]*frac_diff[:-1]
        df_slice = np.mean(df_slice,axis=(0,1,3))
        df_slice -= np.mean(df_slice)
        df_ind = np.where(df_slice>nsig*np.std(df_slice))[0]
        self.ins.mask[:,:,df_ind,:] = True
        wgts = np.mean(self.ins.mask, axis=(0,1,3))
        ind = np.where(wgts > frac_thresh)[0]
        self.ins.mask[:,:,ind,:] = True

    def coherence_flagging(self, fc=0.25, nsig=5):
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

    def merge_flagging(self, ncoarse=24, frac_thresh=0.5):
        """
        Flag isolated pixels that are surrounded by flagged samples. Also check each coarse band, if huge amount
        of samples are flagged in this coarse channel, flag the whole time step.
        """
        m = np.copy(self.ins.mask)
        for ii in range(m.size):
            x = np.zeros(m.shape, dtype=np.int32)
            x[:-1] += np.int32(m[1:])
            x[1:] += np.int32(m[:-1])
            x[:,:,:-1,:]+=np.int32(m[:,:,1:,:])
            x[:,:,1:,:]+=np.int32(m[:,:,:-1,:])
            ind = np.where(x > 2)
            sz = np.where(m[ind]==False)[0].size
            if sz == 0: break
            m[ind]=True
        SH = m.shape
        m = m.reshape(SH[0],SH[1],ncoarse,SH[2]/ncoarse,SH[3])
        y = np.mean(m,axis=3) > frac_thresh
        for ii in range(SH[0]):
            for jj in range(SH[1]):
                for kk in range(SH[3]):
                    if np.any(y[ii,jj,:,kk]):
                        m[ii,:,:,:,:] = True
        m = m.reshape(SH)
        self.ins.mask = m


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

    def saveplots(self, outdir, obsname, clim=(-5,5)):
        """
        Save incoherence noise plots before and after the extra flagging
        """
        fq = self.uv.freq_array[0] / 1e6
        nf = self.uv.Nfreqs
        fi = np.array([0, nf/4-1, nf/2-1, nf/4*3-1, nf-1])
        flabel = np.int32(fq[fi])
        d0 = np.ma.masked_array(self.ins.data, self.mask)
        d0 = d0 / np.ma.median(d0, axis=0) - 1
        s00 = np.std(d0[:,0,:,0])
        s01 = np.std(d0[:,0,:,1])
        d1 = self.ins / np.ma.median(self.ins, axis=0) - 1
        s10 = np.std(d1[:,0,:,0])
        s11 = np.std(d1[:,0,:,1])
        l,r = clim
        fig = plt.figure(figsize=(20,8))
        p1 = fig.add_subplot(2,2,1)
        s0 = np.std(d0[:,0,:,0])
        i1 = p1.imshow(d0[:,0,:,0], aspect='auto', cmap='coolwarm', clim=(l*s00, r*s00))
        p1.set_xlabel('Frequency (MHz)')
        p1.set_ylabel('Time steps')
        p1.set_xticks(fi)
        p1.xaxis.set_ticklabels(flabel)
        p1.set_title('XX masked')
        plt.colorbar(i1)
        p2 = fig.add_subplot(2,2,2)
        i2 = p2.imshow(d0[:,0,:,1], aspect='auto', cmap='coolwarm', clim=(l*s01, r*s01))
        p2.set_xticks(fi)
        p2.xaxis.set_ticklabels(flabel)
        p2.set_xlabel('Frequency (MHz)')
        p2.set_ylabel('Time steps')
        p2.set_title('YY masked')
        plt.colorbar(i2)
        p3 = fig.add_subplot(2,2,3)
        i3 = p3.imshow(d1[:,0,:,0], aspect='auto', cmap='coolwarm', clim=(l*s10, r*s10))
        p3.set_xlabel('Frequency (MHz)')
        p3.set_ylabel('Time steps')
        p3.set_xticks(fi)
        p3.xaxis.set_ticklabels(flabel)
        p3.set_title('XX post flagging')
        plt.colorbar(i3)
        p4 = fig.add_subplot(2,2,4)
        i4 = p4.imshow(d1[:,0,:,1], aspect='auto', cmap='coolwarm', clim=(l*s11, r*s11))
        p4.set_xlabel('Frequency (MHz)')
        p4.set_ylabel('Time steps')
        p4.set_xticks(fi)
        p4.xaxis.set_ticklabels(flabel)
        p4.set_title('YY post flagging')
        plt.colorbar(i4)
        plt.suptitle(obsname)
        plt.subplots_adjust(hspace=0.5)
        figpath = outdir + 'plots/'
        if not os.path.exists(figpath):
            try: os.makedirs(figpath)
            except: pass
        plt.savefig(figpath+obsname+'_INS.png')
        plt.clf()

    def savearrs(self, outdir, obsname):
        """
        Save the original and flagged ins array
        """
        arrpath = outdir + 'arrs/'
        if not os.path.exists(arrpath):
            try: os.makedirs(arrpath)
            except: pass
        d0 = np.ma.masked_array(self.ins.data, self.mask)
        d0.dump(arrpath + obsname + '_Original_INS.npym')
        self.ins.dump(arrpath + obsname + '_Postflag_INS.npym')

class Chisq(object):
    """
        Use omnical chi-square as a quality metric
    """
    def __init__(self, pol, m):
        """
            m: npz class or dictionary containing flags and chisq from omnical
            pol: polarization (xx or yy)
        """
        self.pol = pol
        self.chi = np.ma.masked_array(m['chisq'], m['flags'])
        self.mask = np.copy(m['flags'])
        self.freqs = m['freqs'] #Hz

    def ch7det(self, nsig=5):
        """
            detect any structure from channel 7
        """
        fqr = np.logical_and(self.freqs > 1.811e8, self.freqs < 1.8785e8)
        testset = self.chi[:,np.logical_not(fqr)]
        ch7 = self.chi[:,fqr]
        m0 = np.mean(testset)
        s0 = np.std(testset)
        x = (np.mean(ch7, axis=1) - m0) / s0 * np.sqrt(np.sum(ch7.mask, axis=1)+1e-10)
        ind = np.where(x > nsig)[0]
        for ii in ind: self.chi.mask[ii, fqr] = True

    def outliers_flagging(self, nsig = 6):
        """
            Assuming chisqure follows Gaussian distribution, flag (time,freq,pol) samples that deviate
            from the mean by nsig sigmas.
        """
        for niter in range(self.chi.size):
            m = np.mean(self.chi)
            s = np.std(self.chi)
            ind = np.where(abs(self.chi-m).data*np.logical_not(self.chi.mask) > nsig*s)
            if ind[0].size == 0: break
            else: self.chi.mask[ind] = True

    def freq_flagging(self, nsig=6, frac_thresh=0.5):
        """
            Flag bad frequency channels
        """
        chi = self.chi - np.mean(self.chi)
        df_slice = chi[1:]*chi[:-1]
        df_slice = np.mean(df_slice, axis = 0)
        df_slice -= np.mean(df_slice)
        df_ind = np.where(df_slice>nsig*np.std(df_slice))[0]
        self.chi.mask[:,df_ind] = True
        wgts = np.mean(self.chi.mask, axis= 0)
        ind = np.where(wgts > frac_thresh)[0]
        self.chi.mask[:,ind] = True

    def apply_to_uv(self, uv):
        """
            Apply Chisq flagging to uv object
        """
        for ii in range(uv.Nbls):
            for pid in range(uv.polarization_array.size):
                uv.flag_array[ii::uv.Nbls,0,:,pid] = np.logical_or(uv.flag_array[ii::uv.Nbls,0,:,pid], self.chi.mask)

    def plot_chisq(self, outdir, obsname, clim=(-5,5)):
        """
            Plot Chisq Flagging
        """
        fq = self.freqs / 1e6
        d0 = np.ma.masked_array(self.chi.data, self.mask)
        d1 = self.chi
        m0 = np.ma.median(d0)
        s0 = np.std(d0)
        m1 = np.ma.median(d1)
        s1 = np.std(d1)
        l, r = clim
        b = np.linspace(clim[0], clim[1], 100)
        fig = plt.figure(figsize=(12,6))
        p1 = fig.add_subplot(2,2,1)
        i1 = p1.imshow(d0,aspect='auto',cmap='coolwarm',extent=(fq[0],fq[-1],len(d0)-1,0), clim=(m0+l*s0, m0+r*s0))
        plt.colorbar(i1)
        p1.set_xlabel('Frequency (MHz)')
        p1.set_title('Chi-square')
        p2 = fig.add_subplot(2,2,2)
        p2.hist(d0[np.where(d0.mask==False)], bins=b*s0+m0, density=True, histtype='step')
        p2.plot(b*s0+m0, 1/np.sqrt(2*np.pi*s0*s0)*np.exp(-b*b/2))
        p2.set_yscale('log')
        p2.set_xlabel('$\chi^2$')
        p3 = fig.add_subplot(2,2,3)
        i3 = p3.imshow(d1,aspect='auto',cmap='coolwarm',extent=(fq[0],fq[-1],len(d1)-1,0), clim=(m1+l*s1, m1+r*s1))
        plt.colorbar(i3)
        p3.set_xlabel('Frequency (MHz)')
        p3.set_title('Post Flagging')
        p4 = fig.add_subplot(2,2,4)
        p4.hist(d1[np.where(d1.mask==False)], bins=b*s1+m1, density=True, histtype='step')
        p4.plot(b*s1+m1, 1/np.sqrt(2*np.pi*s1*s1)*np.exp(-b*b/2))
        p4.set_yscale('log')
        p4.set_xlabel('$\chi^2$')
        plt.suptitle(obsname+' '+self.pol)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92,hspace=0.55)
        figpath = outdir + 'plots/'
        if not os.path.exists(figpath):
            try: os.makedirs(figpath)
            except: pass
        plt.savefig(figpath+obsname+'_'+self.pol+'_Chisq.png')
        plt.clf()


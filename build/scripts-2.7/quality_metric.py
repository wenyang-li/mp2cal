from matplotlib import use
use('Agg')
import numpy as np, mp2cal, sys, optparse, subprocess, os
import pyuvdata.uvdata as uvd
from multiprocessing import Pool

o = optparse.OptionParser()
o.set_usage('ssins_flag.py [options] obsid')
o.set_description(__doc__)
o.add_option('-i',dest='inpath',default='./',help='path to input uvfits')
o.add_option('-o',dest='outpath',default='./',help='path to output uvfits')
o.add_option('--xpol', dest='xpol', default=False, action='store_true', help='Toggle: cut out cross polarization')
o.add_option('--fc', dest='fc', default=False, action='store_true', help='Toggle: flag 40kHz channel at coarse band center')
o.add_option('--hex', dest='hex', default=False, action='store_true', help='Toggle: use only hex baselines for SSINS and output')
o.add_option('--ow', dest='ow', default=False, action='store_true', help='Toggle: overwrite flags to original data, use with caution')
opts,args = o.parse_args(sys.argv[1:])
obs = args[0]
filepath = opts.inpath+obs+".uvfits"
print("Reading " + filepath + "...")
antenna_nums = None
if opts.hex:
    print("Check hex subarray for diagnose. No outputs will be generated.")
    from astropy.io import fits
    h = fits.open(opts.inpath+obs+".metafits")
    ants = np.unique(h[1].data['Antenna'][np.where(h[1].data['FLAG']==0)])
    antenna_nums = ants[np.where(ants>56)]
uv = uvd.UVData()
uv.read_uvfits(filepath, antenna_nums=antenna_nums)
if opts.xpol:
    uv.Npols = 2
    uv.flag_array = uv.flag_array[:,:,:,:2]
    uv.data_array = uv.data_array[:,:,:,:2]
    uv.nsample_array = uv.nsample_array[:,:,:,:2]
    uv.polarization_array = uv.polarization_array[:2]
if opts.fc and uv.Nfreqs == 768:
    for ii in range(768):
        if ii%32==16: uv.flag_array[:,:,ii,:] = True
#SSINS
print("Evaluating SSINS...")
ins = mp2cal.qltm.INS(uv)
ins.outliers_flagging()
ins.time_flagging()
ins.coherence_flagging()
ins.outliers_flagging()
ins.merge_flagging()
ins.freq_flagging()
ins.apply_flagging()
ins.saveplots(opts.outpath, obs.split('/')[-1]+'_hex'*opts.hex)
ins.savearrs(opts.outpath, obs.split('/')[-1]+'_hex'*opts.hex)
if np.sum(np.logical_not(ins.ins.mask)) < uv.Nfreqs * uv.Npols:
    raise IOError("All time steps are flagged by SSINS. Skip subsequent steps. Please exclude "+obs)
#Chi-square
print("FirstCal...")
pols = ['xx', 'yy']
data_list = []
for pol in pols:
    RD = mp2cal.data.RedData(pol)
    RD.read_data(uv, tave=True)
    data_list.append(RD)
reds=mp2cal.wyl.cal_reds_from_pos(ex_ants=RD.dead,ex_ubls=[(57,58),(57,59)])
g0 = mp2cal.firstcal.firstcal(uv, reds)
print("OmniCal...")
def omnirun(RD):
    p = RD.pol[0]
    flagged_fqs = np.sum(np.logical_not(RD.mask),axis=0).astype(bool)
    flag_bls = []
    for bl in RD.flag.keys():
        wgt_data = np.logical_not(RD.flag[bl][RD.pol])
        wgt_data = np.sum(wgt_data,axis=0) + np.logical_not(flagged_fqs)
        ind = np.where(wgt_data==0)
        if ind[0].size > 0: flag_bls.append(bl)
    print("exclude baselines in omnical: ", flag_bls)
    for a in g0[p].keys():
        g0[p][a] *= RD.gains.auto[p][a]
    info = mp2cal.wyl.pos_to_info(pols=[p],ex_ubls=[(57,58),(57,59)],ex_bls=flag_bls,ex_ants=RD.dead)
    m2,g2,v2 = mp2cal.wyl.run_omnical(RD.data,info,gains0=g0, maxiter=500, conv=1e-12)
    for a in g2[p[0]].keys():
        g2[p[0]][a] = np.mean(g2[p[0]][a], axis=0)
    RD.get_gains(g2)
    print("Getting chi-square...")
    RD.cal_chi_square(info, m2, per_bl_chi2=True, g=g2)
    m2['freqs'] = uv.freq_array[0]
    cc = mp2cal.qltm.Chisq(RD.pol, m2)
    cc.outliers_flagging()
    cc.ch7det()
    cc.freq_flagging()
    cc.plot_chisq(opts.outpath, obs.split('/')[-1])
    outdir = opts.outpath + 'arrs/'
    if not os.path.exists(outdir):
        try: os.makedirs(outdir)
        except: pass
    mp2cal.io.save_gains_omni(outdir + obs + '.' + RD.pol + '.firstcal.npz', m2, RD.gains.red, RD.gains.mdl)
    return cc
par = Pool(2)
cclist = par.map(omnirun, data_list)
par.close()
if not opts.hex:
    for cc in cclist:
        if np.sum(np.logical_not(cc.chi.mask)) < uv.Nfreqs*2:
            raise IOError("All time steps are flagged Chisq. Skip overwriting. Please exclude "+obs)
        cc.apply_to_uv(uv)
    if opts.ow:
        print("Overwriting to original data.")
        uv.write_uvfits(filepath,write_lst=False)
    else:
        outdir = opts.outpath + '/PostFlag/'
        fout = outdir + obs + '.uvfits'
        print("Write obs to " + fout)
        if not os.path.exists(outdir):
            try: os.makedirs(outdir)
            except: pass
        uv.write_uvfits(fout, write_lst=False)

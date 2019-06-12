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
opts,args = o.parse_args(sys.argv[1:])
obs = args[0]
print "Reading..."
uv = mp2cal.io.read(opts.inpath+obs+'.uvfits')
uv.Npols = 2
uv.flag_array = uv.flag_array[:,:,:,:2]
uv.data_array = uv.data_array[:,:,:,:2]
uv.nsample_array = uv.nsample_array[:,:,:,:2]
uv.polarization_array = uv.polarization_array[:2]
if uv.Nfreqs == 768:
    for ii in range(768):
        if ii%32==16: uv.flag_array[:,:,ii,:] = True
#SSINS
print "Evaluating SSINS..."
ins = mp2cal.qltm.INS(uv)
ins.outliers_flagging()
ins.time_flagging()
ins.coherence_flagging()
ins.outliers_flagging()
ins.merge_flagging()
ins.freq_flagging()
ins.apply_flagging()
ins.saveplots(opts.outpath, obs.split('/')[-1])
ins.savearrs(opts.outpath, obs.split('/')[-1])
#Chi-square
print "FirstCal..."
pols = ['xx', 'yy']
data_list = []
for pol in pols:
    RD = mp2cal.data.RedData(pol)
    RD.read_data(uv, tave=True)
    data_list.append(RD)
reds=mp2cal.wyl.cal_reds_from_pos(ex_ants=RD.dead,ex_ubls=[(57,58),(57,59)])
g0 = mp2cal.firstcal.firstcal(uv, reds)
print "OmniCal..."
def omnirun(RD):
    p = RD.pol[0]
    flagged_fqs = np.sum(np.logical_not(RD.mask),axis=0).astype(bool)
    flag_bls = []
    for bl in RD.flag.keys():
        wgt_data = np.logical_not(RD.flag[bl][RD.pol])
        wgt_data = np.sum(wgt_data,axis=0) + np.logical_not(flagged_fqs)
        ind = np.where(wgt_data==0)
        if ind[0].size > 0: flag_bls.append(bl)
    print 'exclude baselines in omnical: ', flag_bls
    for a in g0[p].keys():
        g0[p][a] *= RD.gains.auto[p][a]
    info = mp2cal.wyl.pos_to_info(pols=[p],ex_ubls=[(57,58),(57,59)],ex_bls=flag_bls)
    m2,g2,v2 = mp2cal.wyl.run_omnical(RD.data,info,gains0=g0, maxiter=500, conv=1e-12)
    for a in g2[p[0]].keys():
        g2[p[0]][a] = np.mean(g2[p[0]][a], axis=0)
    RD.get_gains(g2)
    print 'Getting chi-square...'
    RD.cal_chi_square(info, m2, per_bl_chi2=True, g=g2)
    m2['freqs'] = uv.freq_array[0]
    outdir = opts.outpath + 'arrs/'
    if not os.path.exists(outdir):
        try: os.makedirs(outdir)
        except: pass
    mp2cal.io.save_gains_omni(outdir + obs + '.' + RD.pol + '.firstcal.npz', m2, RD.gains.red, RD.gains.mdl)
    plotdir = opts.outpath + 'plots/'
    RD.plot_chisq(m2, plotdir, obs)
par = Pool(2)
npzlist = par.map(omnirun, data_list)
par.close()
mp2cal.io.write(uv, opts.outpath+obs+'.uvfits')

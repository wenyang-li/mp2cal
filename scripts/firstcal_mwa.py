# edited version of firstcal in heracal
#! /usr/bin/env python
import heracal, mp2cal
import pylab as p, aipy as a
import sys,optparse,glob,os
import numpy as np
from multiprocessing import Pool
import pyuvdata.uvdata as uvd

o = optparse.OptionParser()
o.set_usage('firstcal_mwa.py [options] obsid')
a.scripting.add_standard_options(o,cal=True,pol=True)
o.add_option('--ubls', dest='ubls', default='', help='Unique baselines to use, separated by commas (ex: 1_4,64_49).')
o.add_option('--ex_ubls', dest='ex_ubls', default='', help='Unique baselines to exclude, separated by commas (ex: 1_4,64_49).')
o.add_option('--bls', dest='bls', default='', help='Baselines to use, separated by commas (ex: 1_4,64_49).')
o.add_option('--ex_bls', dest='ex_bls', default='', help='Baselines to exclude, separated by commas (ex: 1_4,64_49).')
o.add_option('--ants', dest='ants', default='', help='Antennas to use, separated by commas (ex: 1,4,64,49).')
o.add_option('--ex_ants', dest='ex_ants', default='', help='Antennas to exclude, separated by commas (ex: 1,4,64,49).')
o.add_option('--outpath', default='/users/wl42/data/wl42/Nov2016EoR0/omni_sol/',help='Output path of solutions.')
o.add_option('--plot', action='store_true', default=False, help='Turn on plotting in firstcal class.')
o.add_option('--verbose', action='store_true', default=False, help='Turn on verbose.')
o.add_option('--ftype', dest='ftype', default='', type='string',
             help='Type of the input file, uvfits or fhd')
opts,args = o.parse_args(sys.argv[1:])

#*****************************************************************************
def bl_parse(bl_args):
    bl_list = []
    for bl in bl_args.split(','):
        i,j = bl.split('_')
        bl_list.append((int(i),int(j)))
    return bl_list

def ant_parse(ant_args):
    ant_list = []
    for a in ant_args.split(','): ant_list.append(int(a))
    return ant_list

#************************************************************************************************
exec('from %s import *'% opts.cal) # Including antpos, realpos, EastHex, SouthHex
pols = opts.pol.split(',')
ubls, ex_ubls, bls, ex_bls, ants, ex_ants = None, [], None, [], None, []
if not opts.ubls == '': ubls = bl_parse(opts.ubls)
if not opts.ex_ubls == '': ex_ubls = bl_parse(opts.ex_ubls)
if not opts.bls == '': bls = bl_parse(opts.bls)
if not opts.ex_bls == '': ex_bls = bl_parse(opts.ex_bls)
if not opts.ants == '': ants = ant_parse(opts.ants)
if not opts.ex_ants == '': ex_ants = ant_parse(opts.ex_ants)

#********************************** load and wrap data ******************************************
if not len(args) == 1: raise IOError('Do not support multiple files.')
obsid = args[0]
uv = uvd.UVData()
if opts.ftype == 'uvfits':
    uv.read_uvfits(obsid+'.uvfits',run_check=False,run_check_acceptability=False)
elif opts.ftype == 'fhd':
    uv.read_fhd(glob.glob(opts.fhdpath+'/vis_data/'+obsid+'*')+glob.glob(opts.fhdpath+'/metadata/'+obsid+'*'),use_model=False,run_check=False,run_check_acceptability=False)
else: IOError('invalid filetype, it should be uvfits or fhd')
ex_ants_find = mp2cal.wyl.find_ex_ant(uv)
for a in ex_ants_find:
    if not a in ex_ants: ex_ants.append(a)
reds = mp2cal.wyl.cal_reds_from_pos(antpos)
redbls = [bl for red in reds for bl in red]
print 'Number of redundant baselines:',len(redbls)
wrap_list = mp2cal.wyl.uv_wrap_fc(uv,redbls,pols=pols)

#************************************************************************************************
def firstcal(data_wrap):
    pp = data_wrap['pol']
    p = pp[0]
    outname = opts.outpath + obsid + '.' + pp + '.first.calfits'
    if os.path.exists(outname): raise IOError("File {0} already exists".format(outname))
    info = mp2cal.wyl.pos_to_info(antpos,pols=[p],fcal=True,ubls=ubls,ex_ubls=ex_ubls,bls=bls,ex_bls=ex_bls,ants=ants,ex_ants=ex_ants)
    fqs = uv.freq_array[0]/1e9
    fc = heracal.FirstCal(data_wrap['data'],data_wrap['flag'],fqs,info)
    print "     running firstcal"
    sols = fc.run(finetune=True,verbose=False,plot=False,noclean=False,offset=False,average=True,window='none')
    meta = {}
    meta['lsts'] = uv.lst_array[::uv.Nbls]
    meta['times'] = uv.time_array[::uv.Nbls]
    meta['freqs'] = uv.freq_array[0]
    meta['inttime'] = uv.integration_time
    meta['chwidth'] = uv.channel_width
    delays = {p:{}}
    antflags = {p:{}}
    for a in sols.keys():
        delays[a.pol()][a.val] = sols[a].T
        antflags[a.pol()][a.val] = np.zeros((uv.Ntimes,uv.Nfreqs))
        meta['chisq{0}'.format(str(a))] = np.ones((uv.Ntimes,1))
    meta['chisq'] = np.ones_like(sols[a].T)
    hc = heracal.omni.HERACal(meta, delays, flags=antflags, ex_ants=ex_ants, DELAY=True, appendhist=' '.join(sys.argv), optional={})
    print('     Saving {0}'.format(outname))
    hc.write_calfits(outname)

#*****************************************************************************
par = Pool(2)
npzlist = par.map(firstcal, wrap_list)
par.close()

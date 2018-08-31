import numpy as np, sys, optparse, mp2cal

o = optparse.OptionParser()
o.add_option('--fhd', dest='fhd', default='/users/wl42/data/wl42/FHD_out/fhd_Calibration_PhaseII/',
             help='fhd output path, please include /.')
opts,args = o.parse_args(sys.argv[1:])

fx = args[0] # fx and fy are good obs ids for pol xx and pol yy, respectively
fy = args[1] # They should only contain obs ids from a single pointing. x first, y second.

fhdpath = opts.fhd + 'calibration/'
days = {}
fn = open(fx,'rb')
for line in fn:
    obs = line.strip()
    pointing = mp2cal.io.getpointing(obs)
    day = int(obs)/86164
    if not days.has_key(day): days[day] = []
    bp = mp2cal.io.load_bp_txt(fhdpath+obs+'_bandpass.txt', 'x')
    days[day].append(bp)
for day in days.keys():
    days[day] = np.array(days[day])
    SH = days[day].shape
    md = np.ma.masked_array(days[day], np.zeros(SH,dtype=bool))
    md.mask[np.where(days[day]==0)] = True
    md = np.mean(md,axis=0)
    np.save(fhdpath+str(day)+'_p'+pointing+'_xx.npy', md.data)
del days
days = {}
fn = open(fy,'rb')
for line in fn:
    obs = line.strip()
    pointing = mp2cal.io.getpointing(obs)
    day = int(obs)/86164
    if not days.has_key(day): days[day] = []
    bp = mp2cal.io.load_bp_txt(fhdpath+obs+'_bandpass.txt', 'y')
    days[day].append(bp)
for day in days.keys():
    days[day] = np.array(days[day])
    SH = days[day].shape
    md = np.ma.masked_array(days[day], np.zeros(SH,dtype=bool))
    md.mask[np.where(days[day]==0)] = True
    md = np.mean(md,axis=0)
    np.save(fhdpath+str(day)+'_p'+pointing+'_yy.npy', md.data)

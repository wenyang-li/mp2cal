from matplotlib import use
use('Agg')
import glob, numpy as np, optparse, sys, matplotlib.pyplot as plt
import matplotlib.mlab as mlab

o = optparse.OptionParser()
o.set_usage('chisq.py [options] obs')
o.set_description(__doc__)
o.add_option('-o',dest='omnipath',type='string',default='chi_sol/',help='path to omnical solutions')
o.add_option('-s',dest='savepath',type='string',default='plot_chi/',help='path to save plots')
#o.add_option('-p',dest='pol',type='string',default='xx',help='polarization')
opts,args = o.parse_args(sys.argv[1:])
#exec('from plot_vis import *')
obs = args[0]
#pol = opts.pol
fn = glob.glob(opts.omnipath+obs+'*xx.omni.npz')
obslist = []
for f in fn: obslist.append(f.split('/')[-1].split('.')[0])
obslist.sort()
n = 0
#m = np.zeros((56,384),dtype=bool)
#m[0]=True;m[-1]=True;m[-2]=True
#for ii in range(384):
#    if ii%16 in [0,15]: m[:,ii]=True
for o in obslist:
    dx = np.load(opts.omnipath+o+'.xx.omni.npz')
    dy = np.load(opts.omnipath+o+'.yy.omni.npz')
    freq = dx['freqs']/1e6
    if n == 0:
        ddx = np.ma.masked_array(dx['chisq'],dx['flags'])
        ddy = np.ma.masked_array(dy['chisq'],dy['flags'])
    else:
        ddx = np.concatenate((ddx,np.ma.masked_array(dx['chisq'],dx['flags'])))
        ddy = np.concatenate((ddy,np.ma.masked_array(dy['chisq'],dy['flags'])))
    n += 1

#def Gaussian(x,mean,sigma):
#    return 1/np.sqrt(2*np.pi*sigma)*np.exp(-(x-mean)**2/sigma**2)
def Gaussianfit(n,bins):
    b=(bins[:-1]+bins[1:])/2
    ind=np.where(n>5)
    x = b[ind].flatten()
    y = np.log(n[ind]).flatten()
    z = np.polyfit(x,y,2)
    return np.exp(z[0]*bins**2+z[1]*bins+z[2])

meanx = np.mean(ddx)
stdx = np.std(ddx)
meany = np.mean(ddy)
stdy = np.std(ddy)
fig = plt.figure()
fig.suptitle(obs)
p1 = fig.add_subplot(2,1,1)
i1 = p1.imshow(ddx,aspect='auto',cmap='coolwarm',extent=(freq[0],freq[-1],ddx.shape[0]-1,0),clim=(meanx-5*stdx,meanx+5*stdx))
#p1.set_xlabel('Frequency (MHz)')
p1.set_ylabel('Time step')
p1.set_title('Chisq/DoF XX')
fig.colorbar(i1)
p2 = fig.add_subplot(2,1,2)
i2 = p2.imshow(ddy,aspect='auto',cmap='coolwarm',extent=(freq[0],freq[-1],ddy.shape[0]-1,0),clim=(meany-5*stdy,meany+5*stdy))
p2.set_xlabel('Frequency (MHz)')
p2.set_ylabel('Time step')
p2.set_title('Chisq/DoF YY')
fig.colorbar(i2)
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig(opts.savepath+obs+'_waterfall.png')
#plt.show()

fig = plt.figure()
fig.suptitle(obs)
p1 = fig.add_subplot(2,1,1)
#meanx = np.mean(ddx)
#stdx = np.std(ddx)
n, bins, patches=p1.hist(ddx.data[np.logical_not(ddx.mask)].flatten(), bins=np.linspace(meanx-5*stdx,meanx+5*stdx,500), facecolor='blue', alpha=0.5, histtype='step')
p1.plot(bins,Gaussianfit(n,bins),c='r')
p1.set_ylabel('normed counts')
p1.set_yscale('log')
p1.set_title('XX')
p1.set_xlim(meanx-5*stdx,meanx+5*stdx)
p2 = fig.add_subplot(2,1,2)
#meany = np.mean(ddy)
#stdy = np.std(ddy)
n, bins, patches=p2.hist(ddy.data[np.logical_not(ddy.mask)].flatten(), bins=np.linspace(meany-5*stdy,meany+5*stdy,500), facecolor='blue', alpha=0.5, histtype='step')
p2.plot(bins,Gaussianfit(n,bins),c='r')
p2.set_ylabel('normed counts')
p2.set_yscale('log')
p2.set_xlabel('Chisq/DoF')
p2.set_title('YY')
p2.set_xlim(meany-5*stdy,meany+5*stdy)
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig(opts.savepath+obs+'_hist.png')
#plt.show()
#







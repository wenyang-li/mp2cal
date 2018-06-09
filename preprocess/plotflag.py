from matplotlib import use
use('Agg')
from cnn import *
import sys
fpath = './mwilensky/Unflagged/all_spw/arrs/'
obs = sys.argv[1]
ins = np.load(fpath+obs+'_Unflagged_Amp_INS_mask.npym')
smooth_plotter(ins)
plt.savefig("./furtherflg/"+obs+"_smt.png")
plt.gcf().clear()
frac_diff_plotter(ins)
#plt.show()
plt.savefig("./furtherflg/"+obs+"_flg.png")
plt.gcf().clear()

# mp2cal
MWA Phase II calibration

Dependencies:

    numpy >= 1.10 
    scipy
    matplotlib
    astropy >= 1.2
    omnical (https://github.com/wenyang-li/omnical)
    pyuvdata (https://github.com/RadioAstronomySoftwareGroup/pyuvdata)
    
Install as admin:

    sudo python setup.py install
    
Install as user:

    python setup.py install --prefix=<your python path>
    
Main Scripts (listed in pipeline order):

1. scripts/quality_metric.py : use SSINS and redundant calibration chi-square for further RFI detection and flagging. This script can output SSINS array, chi-square array, and new uvfits data which has new flaggings applied.
2. scripts/omni_run_mwa.py: Please run FHD calibration first, then run this script. This does redundant calibration correction to the hexagonal cores in addition to FHD calibration. This script outputs per frequency omnical solutions (the output ends in omni.pol.npz). It does bandpass fitting for both FHD sky calibration only and hybrid calibration which has redundant calibration correction included. This script also outputs the difference between the fitted calibration in these two cases, i.e., fitted hybrid calibration divided by fitted FHD sky calibration only (the output ends in difffit.pol.npz).
3. scripts/cal_apply_average.py: Apply calibration solutions to the data. This script takes in the per frequency calibration, either FHD sky calibration only or hybrid calibration which has redundant calibration correction included, then automatically do bandpass fitting, and finally apply the calirbation to the data. This script outputs the calibrated data. It also does averaging in frequency if the frequency resolution is 40 kHz. After the frequency averaging, we double check SSINS, and output SSINS arrays as well.

Examples:

1. test/orbcomm.ipynb: This notebook shows how omnical is working using orbcomm observation as an example. This also demostrates how degeneracy projection is working (see src/gain.py)
2. test/bandpass_fitting.ipynb: This notebook demostrates how bandpass fitting using auto-correlations works. 


# MAIN FUNCTION TO CALL THE ISM MODULE

from ism.src.ism import ism

# Directory - this is the common directory for the execution of the E2E, all modules
auxdir = '/home/daniel/EODP/auxiliary'
indir = '/home/daniel/EODP/data/EODP-TS-ISM/input/gradient_alt100_act150'
outdir = '/home/daniel/EODP/data/EODP-TS-ISM/output'

# Initialise the ISM
myIsm = ism(auxdir, indir, outdir)
myIsm.processModule()

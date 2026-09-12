
# MAIN FUNCTION TO CALL THE L1C MODULE

from l1c.src.l1c import l1c

# Directory - this is the common directory for the execution of the E2E, all modules
auxdir = '/home/daniel/EODP/auxiliary'
# GM dir + L1B dir
indir = '/home/daniel/EODP/data/EODP-TS-L1C/input/gm_alt100_act_150,/home/daniel/EODP/data/EODP-TS-L1B/output'
outdir = '/home/daniel/EODP/data/EODP-TS-L1C/output'

# Initialise the ISM
myL1c = l1c(auxdir, indir, outdir)
myL1c.processModule()

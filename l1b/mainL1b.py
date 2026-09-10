
# MAIN FUNCTION TO CALL THE L1B MODULE

from l1b.src.l1b import l1b

# Directory - this is the common directory for the execution of the E2E, all modules
auxdir = '/home/daniel/EODP/auxiliary'
indir = '/home/daniel/EODP/data/EODP-TS-L1B/input'
outdir = '/home/daniel/EODP/data/EODP-TS-L1B/output'

# Initialise the ISM
myL1b = l1b(auxdir, indir, outdir)
myL1b.processModule()

from backup.load_data import load_propcontrol0
from backup.abersim_io import VARISNULL, FILEISNULL


def load_propcontrol(fnstr, prop, logstr):
    ret = load_propcontrol0(fnstr, prop)

    if ret == VARISNULL:
        print("Propcontrol was not loaded")
    elif ret == FILEISNULL:
        print("Invalid filename. No data loaded")

    return ret

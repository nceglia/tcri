from . import metrics as tl
from . import preprocessing as pp
from . import plotting as pl
from . import utils as ut

import sys

sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['tl', 'pp', 'pl', 'ut']})
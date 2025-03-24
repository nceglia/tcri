from . import metrics as tl
from . import preprocessing as pp
from . import plotting as pl
from . import utils as ut
from . import model as ml

import sys

sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['tl', 'pp', 'pl', 'ut', 'ml']})
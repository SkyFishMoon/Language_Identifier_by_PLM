from model.mBert import mBERT
from model.XLM import XLM
from model.XLMR import XLMR


encoder = {'mbert': mBERT, 'xlm': XLM, 'xlmr': XLMR}

__all__ = [
    'mBERT', 'XLM', 'XLMR'
]
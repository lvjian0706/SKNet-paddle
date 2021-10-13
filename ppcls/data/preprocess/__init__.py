from ppcls.data.preprocess.ops.operators import DecodeImage
from ppcls.data.preprocess.ops.operators import ResizeImage
from ppcls.data.preprocess.ops.operators import CropImage
from ppcls.data.preprocess.ops.operators import RandCropImage
from ppcls.data.preprocess.ops.operators import RandFlipImage
from ppcls.data.preprocess.ops.operators import NormalizeImage
from ppcls.data.preprocess.ops.operators import ToCHWImage

import numpy as np
from PIL import Image


def transform(data, ops=[]):
    """ transform """
    for op in ops:
        data = op(data)
    return data

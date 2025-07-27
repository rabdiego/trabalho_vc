import numpy as np
from visao.BaseProcessor import BaseProcessor
from visao.Histogram import HistogramContrast
from visao.Laplace import LaplaceVariance
from visao.Fourier import FrequencyRatio

class ImageVectorizer(BaseProcessor):
    def __call__(self, x):
        hist = HistogramContrast()
        laplace = LaplaceVariance()
        ratio = FrequencyRatio()
        return np.array([hist(x), laplace(x), ratio(x)])
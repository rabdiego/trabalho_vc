import cv2
import numpy as np
from visao.BaseProcessor import BaseProcessor

class Histogram(BaseProcessor):
    def __call__(self, x):
        return cv2.calcHist([x], [0], None, [256], [0, 256])
        

class HistogramContrast(BaseProcessor):
    def __call__(self, x):
        hist = Histogram()
        return np.std(hist(x)) / np.mean(hist(x))


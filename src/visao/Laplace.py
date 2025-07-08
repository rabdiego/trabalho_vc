import cv2
import numpy as np
from visao.BaseProcessor import BaseProcessor

class Laplace(BaseProcessor):
    def __call__(self, x):
        return np.abs(cv2.Laplacian(x, ddepth=cv2.CV_64F))


class LaplaceVariance(BaseProcessor):
    def __call__(self, x):
        laplace = Laplace()
        return np.std(laplace(x)) / np.mean(laplace(x))


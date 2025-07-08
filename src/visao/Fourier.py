import cv2
import numpy as np
from visao.BaseProcessor import BaseProcessor

class FourierTransform(BaseProcessor):
    def __call__(self, x):
        f = np.fft.fft2(x)
        return np.fft.fftshift(f)


class FrequencyGraph(BaseProcessor):
    def __call__(self, x):
        return 20 * np.log(np.abs(x) + 1)


class FrequencyRatio(BaseProcessor):
    def __call__(self, x, cutoff_ratio: float = 0.05):
        rows, cols = x.shape
        fourier = FourierTransform()
        grapher = FrequencyGraph()
        x_transformed = fourier(x)

        total_energy = grapher(x_transformed)

        crow, ccol = rows // 2, cols // 2
        radius = int(cutoff_ratio * min(rows, cols))

        mask = np.zeros((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), radius, 1, thickness=-1)

        x_filtered = x_transformed * mask

        low_energy = grapher(x_filtered)

        return 1 - low_energy.sum() / total_energy.sum()


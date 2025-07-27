from visao.Histogram import HistogramContrast
from visao.Laplace import LaplaceVariance
from visao.Fourier import FourierTransform, FrequencyRatio, FrequencyGraph
from visao.ImageLoader import ImageLoader
from visao.ImageVectorizer import ImageVectorizer

img = ImageLoader.load('../data/good/0001x4.png')
low = ImageLoader.load('../data/low_contrast/a0063-IMG_4185_N1.5.JPG')
vec = ImageVectorizer()
hist = HistogramContrast()
laplace = LaplaceVariance()
fourier = FourierTransform()
grapher = FrequencyGraph()
ratio = FrequencyRatio()

print(f'G: {vec(img)}')
print(f'L: {vec(low)}')

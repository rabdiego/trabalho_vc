from visao.Histogram import HistogramContrast
from visao.Laplace import LaplaceVariance
from visao.Fourier import FourierTransform, FrequencyRatio, FrequencyGraph
from visao.ImageLoader import ImageLoader

img = ImageLoader.load('../data/emperor.jpg')
low = ImageLoader.load('../data/low.jpg')
hist = HistogramContrast()
laplace = LaplaceVariance()
fourier = FourierTransform()
grapher = FrequencyGraph()
ratio = FrequencyRatio()

print(f'G: {hist(img)},{laplace(img)},{ratio(img)}')
print(f'L: {hist(low)},{laplace(low)},{ratio(low)}')

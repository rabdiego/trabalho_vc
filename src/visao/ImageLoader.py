import cv2

class ImageLoader:
    @staticmethod
    def load(path: str, bw: bool = True) -> None:
        img = cv2.imread(path)
        transform = cv2.COLOR_BGR2GRAY if bw else cv2.COLOR_BGR2RGB
        return cv2.cvtColor(img, transform)


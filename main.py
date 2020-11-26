import SearchEngine
import cv2 as cv
import numpy as np


class SearchEngine:

    def __init__(self, min_ratio=4, max_ratio=5, debug=True):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.debug = debug

    def debug_display(self, name, image):
        if self.debug:
            cv.imshow(name, image)
            cv.waitKey()
            cv.destroyAllWindows()

    def objectDetection(self, image, keep=5):
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (13, 5))  # Kernel Creation
        morpho = cv.morphologyEx(image, cv.MORPH_BLACKHAT, kernel)  # Performing morphological transformation
        self.debug_display('Morpho', morpho)

        # Find the light regions
        square_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        light = cv.morphologyEx(image, cv.MORPH_CLOSE, square_kernel)
        light = cv.threshold(light, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        self.debug_display('Light', image)  # porencjalne błedy w tym miejscu

        gradX = cv.Sobel(morpho, ddepth=cv.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
        gradX = gradX.astype("uint8")
        self.debug_display("Scharr", gradX)

        gradX = cv.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv.morphologyEx(gradX, cv.MORPH_CLOSE, kernel)
        thresh = cv.threshold(gradX, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
        self.debug_display("Grad Thresh", thresh)


def main():
    img = cv.imread('licenseplates/001.jpg', 0)

    search = SearchEngine()
    search.objectDetection(img)


if __name__ == "__main__":
    main()

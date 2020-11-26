import cv2 as cv
import numpy as np


class SearchEngine:
    def __init__(self, min_ratio=4, max_ratio=5, debug=False):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.debug = debug

    def debug_display(self, name, image):
        if self.debug:
            cv.imshow(name, image)
            cv.waitKey()
            cv.destroyAllWindows()

    def objectDetection(self, gray, keep=5):
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (13, 5))  # Kernel Creation
        morpho = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)  # Performing morphological transformation
        self.debug_display('Morpho', morpho)

import pytesseract
import imutils
import cv2 as cv
import numpy as np
import xlsxwriter
from skimage.segmentation import clear_border
import tkinter as tk
from tkinter import filedialog, Text
from PIL import Image
from PIL import ImageTk
from tkinter import *


def cleanup_text(text):
    """Delete any watermarks on the image so one can draw easier.
                                Parameters
                                ----------
                                text : Any
                                    Text to be cleared
                                """
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()


class GUI:
    def __init__(self, root):
        self.root = tk.Tk()


class ExtractPlateFromPhoto:
    def __init__(self, min_ratio=4, max_ratio=5, debug=False):
        """Store min and max ratio. Note: debug is used only in case program doesn't work
                            Parameters
                            ----------
                            min_ratio : int
                               Minimal ratio
                            max_ratio : int
                               Maximal ratio
                            debug : bool
                                Only for easier debugging
                            """
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.debug = debug

    # TODO Write proper documentation of method
    def debug_display(self, name, image, waitKey=False):

        if self.debug:
            cv.imshow(name, image)

            if waitKey:
                cv.waitKey(0)

    # TODO Write proper documentation of method
    def objectDetection(self, image_bw):
        rect_kern = cv.getStructuringElement(cv.MORPH_RECT, (13, 5))
        morpho = cv.morphologyEx(image_bw, cv.MORPH_BLACKHAT, rect_kern)
        self.debug_display("Morpho", morpho)

        square_kern = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        light = cv.morphologyEx(image_bw, cv.MORPH_CLOSE, square_kern)
        light = cv.threshold(light, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
        self.debug_display("Light Regions", light)

        grad_x = cv.Sobel(morpho, ddepth=cv.CV_32F, dx=1, dy=0, ksize=-1)
        grad_x = np.absolute(grad_x)
        (min_val, max_val) = (np.min(grad_x), np.max(grad_x))
        grad_x = 255 * ((grad_x - min_val) / (max_val - min_val))
        grad_x = grad_x.astype("uint8")
        self.debug_display("Edge detection", grad_x)

        grad_x = cv.GaussianBlur(grad_x, (5, 5), 0)
        grad_x = cv.morphologyEx(grad_x, cv.MORPH_CLOSE, rect_kern)
        thresh = cv.threshold(grad_x, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
        self.debug_display("Thresholding", thresh)

        thresh = cv.erode(thresh, None, iterations=2)
        thresh = cv.dilate(thresh, None, iterations=2)
        self.debug_display("Noise elimination", thresh)

        thresh = cv.bitwise_and(thresh, thresh, mask=light)
        thresh = cv.dilate(thresh, None, iterations=2)
        thresh = cv.erode(thresh, None, iterations=1)
        self.debug_display("Final image", thresh, waitKey=True)

        contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv.contourArea, reverse=True)[:5]

        return contours

    # TODO Write proper documentation of method
    def locatePlate(self, image_BW, candidate, clearBorder=False):

        lpCnt = None
        roi = None

        for c in candidate:
            (x, y, w, h) = cv.boundingRect(c)
            ar = w / float(h)

            if self.min_ratio <= ar <= self.max_ratio:

                lpCnt = c
                licensePlate = image_BW[y:y + h, x:x + w]
                roi = cv.threshold(licensePlate, 0, 255,
                                   cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]

                if clearBorder:
                    roi = clear_border(roi)

                self.debug_display("License Plate", licensePlate)
                self.debug_display("ROI", roi, waitKey=True)
                break

        # return a 2-tuple of the license plate ROI and the contour
        # associated with it
        return (roi, lpCnt)

    # TODO Write proper documentation of method
    @staticmethod
    def tesseractInitialization(psm=7):

        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = "-c tessedit_char_whitelist={}".format(alphanumeric)
        options += " --psm {}".format(psm)

        return options

    # TODO Write proper documentation of method
    def OCR(self, image_col, psm=7, clearBorder=False):

        lpText = None
        image_no_colour = cv.cvtColor(image_col, cv.COLOR_BGR2GRAY)
        candidates = self.objectDetection(image_no_colour)
        (lp, lpCnt) = self.locatePlate(image_no_colour, candidates, clearBorder=clearBorder)

        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        if lp is not None:
            options = self.tesseractInitialization(psm=psm)
            lpText = pytesseract.image_to_string(lp, config=options)
            self.debug_display("License Plate", lp)

        return lpText, lpCnt


class GUI:
    def __init__(self, image, ocr, license_plate, license_counter, frame):
        self.image = image
        self.ocr = ocr
        self.license_plate = license_plate
        self.license_counter = license_counter
        self.frame = frame

    def generate_ui(self):
        root = tk.Tk()

        # specify window properties
        canvas = tk.Canvas(root, height=700, width=700, bg="#444746")
        canvas.pack()

        self.frame = tk.Frame(root, bg="#0cc769")
        self.frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)

        openFile = tk.Button(root, text="Load Licence Plate", padx=50, pady=5, fg="white", bg="#444746",
                             command=self.load_file)
        openFile.pack(side="left")

        export = tk.Button(root, text="Export to Excel", padx=50, pady=5, fg="white", bg="#444746",
                           command=self.create_csv)
        export.pack(side="right")

        recognize = tk.Button(root, text="Recognize", padx=100, pady=5, fg="white", bg="#444746",
                              command=self.recognize)
        recognize.pack(side="top")

        root.mainloop()

    def load_file(self):
        path = filedialog.askopenfilename()
        if len(path) > 0:
            self.image = cv.imread(path)
            self.image = imutils.resize(self.image, width=600)

    def get_image(self):
        self.image
        return self.image

    def recognize(self):
        counter = 0
        (self.license_plate, self.license_counter) = self.ocr.OCR(self.image, 7, True)

        if self.license_plate is not None and self.license_counter is not None:
            box = cv.boxPoints(cv.minAreaRect(self.license_counter))
            box = box.astype("int")
            cv.drawContours(self.image, [box], -1, (0, 255, 0), 2)
            (x, y, w, h) = cv.boundingRect(self.license_counter)
            cv.putText(self.image, cleanup_text(self.license_plate), (x, y - 15), cv.FONT_HERSHEY_SIMPLEX, 0.75,
                       (255, 0, 255), 2)

            print("[INFO] {}".format(self.license_plate))
            cv.imshow("Output ANPR", self.image)
            cv.waitKey(0)
            cv.destroyAllWindows()

            label = tk.Label(self.frame, text="License plate: " + self.license_plate, bg="gray")
            label.pack()

    def create_csv(self):

        workbook = xlsxwriter.Workbook('Plates.xlsx')
        worksheet = workbook.add_worksheet()
        row = 1
        col = 1

        worksheet.write(row, col, "License plate")
        worksheet.write(row, col + 1, self.license_plate)
        row += 1

        workbook.close()


# TODO Describe main method
def main():
    plate = ExtractPlateFromPhoto(0)  # debug 0
    gui = GUI(None, plate, None, None, None)

    gui.generate_ui()


if __name__ == "__main__":
    main()

# TODO Implement User interface

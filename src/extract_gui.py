# import the necessary packages
import time
import tkinter as tk
from tkinter import filedialog

import cv2
import joblib
import numpy as np
from PIL import Image, ImageTk

from core import extract_signature


def resize(image, size):
    w, h = image.size
    if w == 0 or h == 0:
        return Image.fromarray(np.ones(size) * 255.0)

    _w, _h = size
    if w > h:
        h = int(h * float(_w) / w)
        w = int(_w)
    else:
        w = int(w * float(_h) / h)
        h = int(_h)
    image = image.resize((w, h), Image.ANTIALIAS)

    max_w, max_h = size
    img_w, img_h = image.size

    img = np.array(image)
    canvas = np.ones(shape=(max_h, max_w, 3), dtype=img.dtype) * 255
    x = int((max_w - img_w) / 2)
    y = int((max_h - img_h) / 2)

    canvas[y:y + img_h, x:x + img_w, :] = img[0:img_h, 0:img_w, :]
    return Image.fromarray(canvas)


def detect_signature():
    global app

    if len(app.current_file) > 0:
        clf = app.model

        app.status("Extracting signature...")
        start_time = time.time()

        im = cv2.imread(app.current_file, 0)
        mask = extract_signature(im, clf, preprocess=True)

        im = cv2.imread(app.current_file)
        im[np.where(mask == 255)] = (0, 0, 255)

        # Draw bounding box on image
        points = np.argwhere(mask == 255)  # find where the black pixels are
        points = np.fliplr(points)  # store them in x,y coordinates instead of row,col indices
        x, y, w, h = cv2.boundingRect(points)  # create a rectangle around those points
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        app.show(im, app.input_view)
        app.status("Done in %.2fs." % (time.time() - start_time))


def open_image():
    global app

    # open a file chooser dialog and allow the user to select an input image
    current_file = filedialog.askopenfilename()

    # ensure a file path was selected
    if len(current_file) > 0:
        app.status("Opening " + current_file.split("/")[-1] + "...")
        app.current_file = current_file

        # Open and display selected image
        src = cv2.imread(app.current_file)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        app.show(src, app.input_view)
        app.status("Step 2: Detect Signature")


class SignatureExtractor:

    def __init__(self):
        self.__root = tk.Tk()
        self.__root.configure(background="white")
        self.__root.title("Signature Extractor")
        self.__root.resizable(width=False, height=False)
        self.__root.geometry('{}x{}'.format(960, 720))
        tk.Grid.rowconfigure(self.__root, 0, weight=1)
        tk.Grid.columnconfigure(self.__root, 0, weight=1)
        self.__center()

        # Add a grid
        mainframe = tk.Frame(self.__root)
        mainframe.grid(rowspan=12, columnspan=4, sticky=(tk.N, tk.W, tk.E, tk.S))
        tk.Grid.rowconfigure(mainframe, 0, weight=1)
        tk.Grid.columnconfigure(mainframe, 0, weight=1)

        # Create a Tkinter variable
        self.model = joblib.load("models/decision-tree.pkl")

        tk.Button(mainframe, text="Open an Image", command=open_image).grid(row=0, column=0, sticky=tk.E)
        tk.Button(mainframe, text="Detect Signature", command=detect_signature).grid(row=0, column=1, sticky=tk.E)

        # Create canvas where source image will be displayed
        self.input_view = tk.Label(mainframe)
        self.input_view.grid(row=1, column=0, columnspan=2)
        self.show(np.ones((100, 100)) * 255, self.input_view)

        self.__status = tk.Label(mainframe, text="Step 1: Open an Image")
        self.__status.grid(row=2, column=0, sticky=tk.W)

        self.current_file = ""

    def __center(self):
        self.__root.update_idletasks()
        w = self.__root.winfo_screenwidth()
        h = self.__root.winfo_screenheight()
        size = tuple(int(_) for _ in self.__root.geometry().split('+')[0].split('x'))
        x = w / 2 - size[0] / 2
        y = h / 2 - size[1] / 2
        self.__root.geometry("%dx%d+%d+%d" % (size + (x, y)))

    def show(self, im, target):
        try:
            im = Image.fromarray(im).convert("RGB")
            im = resize(im, (960, 640))
        except Exception as ex:
            im = Image.fromarray(np.ones((960, 640)) * 255.0)

        im = ImageTk.PhotoImage(im)
        target.configure(image=im)
        target.image = im

    def status(self, text):
        self.__status['text'] = text

    def start(self):
        self.__root.mainloop()


if __name__ == '__main__':
    app = SignatureExtractor()
    app.start()

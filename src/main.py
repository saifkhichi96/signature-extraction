# import the necessary packages
import Tkinter as tk
import time
import tkFileDialog

import cv2
import numpy as np
from PIL import Image
from PIL import ImageTk
from sklearn.externals import joblib

from classify import prepare


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
    x = (max_w - img_w) / 2
    y = (max_h - img_h) / 2

    canvas[y:y + img_h, x:x + img_w, :] = img[0:img_h, 0:img_w, :]
    return Image.fromarray(canvas)


def detect_signature():
    global app

    if len(app.current_file) > 0:
        app.status("loading model...")
        clf = app.get_classifier()

        app.status("segmenting...")
        start_time = time.time()
        im, mask = prepare(app.current_file, clf)

        app.status("done (%.2fs)" % (time.time() - start_time))
        app.show(mask, app.input_view)


def open_image():
    global app

    # open a file chooser dialog and allow the user to select an input image
    current_file = tkFileDialog.askopenfilename()

    # ensure a file path was selected
    if len(current_file) > 0:
        app.current_file = current_file

        # Open and display selected image
        src = cv2.imread(app.current_file)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        app.show(src, app.input_view)

        app.status("file selected")


class SignatureExtractor:

    def __init__(self):
        self.__root = tk.Tk()
        self.__root.configure(background="white")
        self.__root.title("Signature Extractor")
        self.__root.resizable(width=False, height=False)
        self.__root.geometry('{}x{}'.format(800, 600))
        self.__center()

        # Add a grid
        mainframe = tk.Frame(self.__root)
        mainframe.grid(column=0, row=0, columnspan=5, rowspan=11, sticky=(tk.N, tk.W, tk.E, tk.S))
        mainframe.pack(pady=10, padx=10)

        # Create a Tkinter variable
        self.__selected_model = tk.StringVar(self.__root)

        # Dictionary with options
        choices = {'Decision Tree', 'KNN', 'MLP', 'SGD', 'SVC-Linear', 'SVC-RBF'}
        self.__selected_model.set('Decision Tree')  # set the default option

        tk.Label(mainframe, text="Select model").grid(row=0, column=0)

        popup_menu = tk.OptionMenu(mainframe, self.__selected_model, *choices)
        popup_menu.grid(row=0, column=1)

        tk.Label(mainframe, text="Select image").grid(row=0, column=3)
        tk.Button(mainframe, text="Open", command=open_image).grid(row=0, column=4)

        # Create canvas where source image will be displayed
        self.input_view = tk.Label(mainframe)
        self.input_view.grid(row=1, rowspan=8, column=1, columnspan=3)
        self.show(np.ones((100, 100)) * 255, self.input_view)

        self.__status = tk.Label(mainframe, text="")
        self.__status.grid(row=10, column=0)

        tk.Button(mainframe, text="Detect", command=detect_signature).grid(row=10, column=4)

        self.current_file = ""

    def __center(self):
        self.__root.update_idletasks()
        w = self.__root.winfo_screenwidth()
        h = self.__root.winfo_screenheight()
        size = tuple(int(_) for _ in self.__root.geometry().split('+')[0].split('x'))
        x = w / 2 - size[0] / 2
        y = h / 2 - size[1] / 2
        self.__root.geometry("%dx%d+%d+%d" % (size + (x, y)))

    def get_classifier(self):
        return joblib.load("../out/models/" + self.__selected_model.get().replace(" ", "-").lower() + ".pkl")

    def show(self, im, target):
        im = Image.fromarray(im).convert("RGB")
        try:
            im = resize(im, (480, 480))
        except:
            im = Image.fromarray(np.ones((480, 480)) * 255.0)

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

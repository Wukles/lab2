import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        self.root.geometry("1500x700")  # Увеличен для размещения трех изображений

        self.original_image = None
        self.processed_image = None
        self.new_image = None
        self.watercolor_image = None

        # Frame for the original and processed images
        self.image_frame = Frame(self.root)
        self.image_frame.pack(side=TOP, pady=10)

        # Original image panel
        self.original_panel = Label(self.image_frame)
        self.original_panel.pack(side=LEFT, padx=10)

        # Processed image panel
        self.processed_panel = Label(self.image_frame)
        self.processed_panel.pack(side=LEFT, padx=10)

        # New image panel
        self.new_image_panel = Label(self.image_frame)
        self.new_image_panel.pack(side=LEFT, padx=10)

        # Frame for the buttons
        self.button_frame = Frame(self.root)
        self.button_frame.pack(side=TOP, pady=10)

        # Frame for the sliders
        self.slider_frame = Frame(self.root)
        self.slider_frame.pack(side=TOP, pady=5)

        # Buttons for loading and processing images
        self.load_button = Button(self.button_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=LEFT, padx=10)

        self.gray_button = Button(self.button_frame, text="Gray Scale", command=self.convert_to_gray)
        self.gray_button.pack(side=LEFT, padx=10)

        self.sepia_button = Button(self.button_frame, text="Sepia", command=self.convert_to_sepia)
        self.sepia_button.pack(side=LEFT, padx=10)

        self.brightness_contrast_button = Button(self.button_frame, text="Brightness & Contrast", command=self.show_brightness_contrast_sliders)
        self.brightness_contrast_button.pack(side=LEFT, padx=10)

        self.hsv_button = Button(self.button_frame, text="Convert to HSV", command=self.show_hsv_slider)
        self.hsv_button.pack(side=LEFT, padx=10)

        self.median_blur_button = Button(self.button_frame, text="Median Blur", command=self.median_blur)
        self.median_blur_button.pack(side=LEFT, padx=10)

        self.cartoon_button = Button(self.button_frame, text="Cartoon Filter", command=self.show_cartoon_filter_slider)
        self.cartoon_button.pack(side=LEFT, padx=10)

        self.window_filter_button = Button(self.button_frame, text="Window Filter", command=self.show_window_filter_fields)
        self.window_filter_button.pack(side=LEFT, padx=10)

        self.watercolor_button = Button(self.button_frame, text="Watercolor Filter", command=self.show_watercolor_options)
        self.watercolor_button.pack(side=LEFT, padx=10)

        # Sliders for cartoon filter threshold, brightness and contrast (initially hidden)
        self.threshold_slider = Scale(self.slider_frame, from_=1, to=50, orient=HORIZONTAL, label="Cartoon Threshold")
        self.threshold_slider.set(100)

        self.brightness_slider = Scale(self.slider_frame, from_=-100, to=100, orient=HORIZONTAL, label="Contrast")
        self.brightness_slider.set(0)

        self.contrast_slider = Scale(self.slider_frame, from_=1, to=3, orient=HORIZONTAL, label="Brightness", resolution=0.1)
        self.contrast_slider.set(1)

        # Slider for HSV hue adjustment (initially hidden)
        self.hue_slider = Scale(self.slider_frame, from_=0, to=180, orient=HORIZONTAL, label="Hue")
        self.hue_slider.set(0)

        # Slider for Watercolor mix
        self.watercolor_slider = Scale(self.slider_frame, from_=0, to=100, orient=HORIZONTAL, label="Watercolor Mix")
        self.watercolor_slider.set(50)

        self.filter_matrix_entry = []
        self.filter_matrix_values = []
        self.apply_filter_button = None

        self.add_image_button = None

        self.channel_var = StringVar(self.root)
        self.channel_var.set("Red")  # Set default value
        self.channel_menu = OptionMenu(self.button_frame, self.channel_var, "Red", "Green", "Blue", command=self.extract_channel)
        self.channel_menu.pack(side=LEFT, padx=10)

    def load_image(self):
        self.hide_sliders()
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.processed_image = self.original_image.copy()
            self.watercolor_image = self.original_image.copy()
            self.display_images()

    def load_new_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            new_image = cv2.imread(file_path)
            # Resize the new image to match the original image size
            self.new_image = cv2.resize(new_image, (self.original_image.shape[1], self.original_image.shape[0]))
            self.watercolor_filter()

    def display_images(self):
        if self.original_image is not None:
            original_img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            processed_img = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
            new_img = cv2.cvtColor(self.new_image, cv2.COLOR_BGR2RGB) if self.new_image is not None else None

            original_img = Image.fromarray(original_img)
            processed_img = Image.fromarray(processed_img)
            new_img = Image.fromarray(new_img) if new_img is not None else None

            original_img = ImageTk.PhotoImage(original_img)
            processed_img = ImageTk.PhotoImage(processed_img)
            new_img = ImageTk.PhotoImage(new_img) if new_img is not None else None

            self.original_panel.config(image=original_img)
            self.original_panel.image = original_img

            self.processed_panel.config(image=processed_img)
            self.processed_panel.image = processed_img

            if new_img is not None:
                self.new_image_panel.config(image=new_img)
                self.new_image_panel.image = new_img

    def convert_to_gray(self):
        self.hide_sliders()
        if self.original_image is not None:
            self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR)
            self.display_images()

    def convert_to_sepia(self):
        self.hide_sliders()
        if self.original_image is not None:
            kernel = np.array([[0.272, 0.534, 0.131],
                               [0.349, 0.686, 0.168],
                               [0.393, 0.769, 0.189]])
            self.processed_image = cv2.transform(self.original_image, kernel)
            self.processed_image = np.clip(self.processed_image, 0, 255)
            self.display_images()

    def convert_to_hsv(self):
        if self.original_image is not None:
            hsv_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
            hue_shift = self.hue_slider.get()
            h, s, v = cv2.split(hsv_image)
            h = (h + hue_shift) % 180  # Ensure hue values wrap around correctly
            hsv_image = cv2.merge([h, s, v])
            self.processed_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
            self.display_images()

    def update_hsv(self, event):
        self.convert_to_hsv()

    def show_hsv_slider(self):
        self.hide_sliders()
        self.hue_slider.pack(side=LEFT, padx=5)
        self.hue_slider.bind("<Motion>", self.update_hsv)
        self.convert_to_hsv()

    def median_blur(self):
        self.hide_sliders()
        if self.original_image is not None:
            self.processed_image = cv2.medianBlur(self.original_image, 5)
            self.display_images()

    def cartoon_filter(self):
        if self.original_image is not None:
            threshold = self.threshold_slider.get()
            img_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.medianBlur(img_gray, 7)
            img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY, 9, threshold)
            img_color = cv2.bilateralFilter(self.original_image, 9, 300, 300)
            img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)
            self.processed_image = cv2.bitwise_and(img_color, img_edge)
            self.display_images()

    def update_cartoon_filter(self, event):
        self.cartoon_filter()

    def show_cartoon_filter_slider(self):
        self.hide_sliders()
        self.threshold_slider.pack(side=LEFT, padx=5)
        self.threshold_slider.bind("<Motion>", self.update_cartoon_filter)
        self.cartoon_filter()

    def adjust_brightness_contrast(self):
        if self.original_image is not None:
            brightness = self.brightness_slider.get()
            contrast = self.contrast_slider.get()
            self.processed_image = cv2.convertScaleAbs(self.original_image, alpha=contrast, beta=brightness)
            self.display_images()

    def update_brightness_contrast(self, event):
        self.adjust_brightness_contrast()

    def show_brightness_contrast_sliders(self):
        self.hide_sliders()
        self.brightness_slider.pack(side=LEFT, padx=5)
        self.contrast_slider.pack(side=LEFT, padx=5)
        self.brightness_slider.bind("<Motion>", self.update_brightness_contrast)
        self.contrast_slider.bind("<Motion>", self.update_brightness_contrast)
        self.adjust_brightness_contrast()

    def show_window_filter_fields(self):
        self.hide_sliders()
        self.filter_matrix_entry = []
        for i in range(3):
            row = Frame(self.slider_frame)
            row.pack(side=TOP, padx=5, pady=2)
            row_entry = []
            for j in range(3):
                entry = Entry(row, width=5)
                entry.pack(side=LEFT)
                entry.insert(END, '0')
                row_entry.append(entry)
            self.filter_matrix_entry.append(row_entry)
        if self.apply_filter_button:
            self.apply_filter_button.pack_forget()
        self.apply_filter_button = Button(self.slider_frame, text="Apply Filter", command=self.apply_filter)
        self.apply_filter_button.pack(side=TOP, pady=5)

    def apply_filter(self):
        if self.original_image is not None:
            filter_matrix = np.zeros((3, 3), dtype=np.float32)
            for i in range(3):
                for j in range(3):
                    filter_matrix[i, j] = float(self.filter_matrix_entry[i][j].get())
            self.processed_image = cv2.filter2D(self.original_image, -1, filter_matrix)
            self.display_images()

    def extract_channel(self, channel):
        self.hide_sliders()
        if self.original_image is not None:
            b, g, r = cv2.split(self.original_image)
            if channel == "Red":
                self.processed_image = r
            elif channel == "Green":
                self.processed_image = g
            elif channel == "Blue":
                self.processed_image = b
            self.processed_image = cv2.merge([self.processed_image] * 3)
            self.display_images()

    def watercolor_filter(self):
        if self.original_image is not None and self.new_image is not None:
            mix_ratio = self.watercolor_slider.get() / 100.0
            self.watercolor_image = cv2.addWeighted(self.original_image, 1 - mix_ratio, self.new_image, mix_ratio, 0)
            self.processed_image = self.watercolor_image
            self.display_images()

    def update_watercolor_filter(self, event):
        self.watercolor_filter()

    def show_watercolor_options(self):
        self.hide_sliders()
        self.watercolor_slider.pack(side=LEFT, padx=5)
        self.watercolor_slider.bind("<Motion>", self.update_watercolor_filter)

        if self.add_image_button:
            self.add_image_button.pack_forget()

        self.add_image_button = Button(self.slider_frame, text="Add New Image", command=self.load_new_image)
        self.add_image_button.pack(side=TOP, pady=5)

        self.watercolor_filter()

    def hide_sliders(self):
        for widget in self.slider_frame.winfo_children():
            widget.pack_forget()

        # Hide and remove filter matrix entries and apply button
        for row in self.filter_matrix_entry:
            for entry in row:
                entry.pack_forget()
        self.filter_matrix_entry = []

        if self.apply_filter_button:
            self.apply_filter_button.pack_forget()
            self.apply_filter_button = None

        if self.add_image_button:
            self.add_image_button.pack_forget()
            self.add_image_button = None


if __name__ == "__main__":
    root = Tk()
    app = ImageProcessor(root)
    root.mainloop()

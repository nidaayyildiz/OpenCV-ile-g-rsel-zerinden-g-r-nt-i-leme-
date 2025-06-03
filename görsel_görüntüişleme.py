
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None
    print("Uyarı: sklearn yüklü değil. K-Means fonksiyonu devre dışı bırakılacak.")

try:
    from scipy import ndimage
except ImportError:
    ndimage = None
    print("Uyarı: scipy yüklü değil. watershed gibi bazı işlemler devre dışı bırakılacak.")

def apply_sobel(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.uint8(np.clip(sobel, 0, 255))
    return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)

def apply_median_filter(img):
    return cv2.medianBlur(img, 5)

def apply_gaussian_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def apply_kmeans_segmentation(img):
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    k = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((img.shape))
    return segmented_image

def apply_watershed(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    return img

def apply_rotation(img):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 45, 1.0)
    return cv2.warpAffine(img, M, (w, h))

def apply_zoom(img):
    return img  # Zoom işlemi artık scale üzerinden yapılacak

def apply_stretching(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    stretched = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)

def apply_hough_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img

def apply_shearing(img):
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0.5, 0], [0, 1, 0]])
    return cv2.warpAffine(img, M, (cols, rows))

def apply_erode(img):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(img, kernel, iterations=1)

def apply_dilate(img):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)

class ImageProcessor:
    def __init__(self, root):
        self.zoom_scale = 1.0
        self.polygon_sides = 5
        self.root = root
        self.root.title("Görüntü İşleme Arayüzü")
        self.image = None
        self.history = []
        self.mask_center = None
        self.create_widgets()

    def create_widgets(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Resim Aç", command=self.load_image)
        file_menu.add_command(label="Resmi Kaydet", command=self.save_image)
        # file_menu.add_command(label="Geri Al", command=self.undo)  # Menüden kaldırıldı
        menubar.add_cascade(label="Dosya", menu=file_menu)
        self.root.config(menu=menubar)

        self.placeholder_frame = tk.Frame(self.root, width=800, height=600, bg='lightgray')
        self.placeholder_frame.pack_propagate(False)
        self.placeholder_frame.pack(pady=10)

        self.load_button = tk.Button(self.placeholder_frame, text="Görsel Aç", font=("Arial", 14), command=self.load_image)
        self.load_button.pack(expand=True)

        self.image_label = tk.Label(self.placeholder_frame)
        self.image_label.bind("<Button-1>", self.set_mask_center)

        self.image_label = tk.Label(self.placeholder_frame)

        # Sağ alt köşeye 'Geri Al' butonu
        undo_frame = tk.Frame(self.root)
        undo_frame.pack(anchor="se", padx=10, pady=10)
        tk.Button(undo_frame, text="Geri Al", command=self.undo).pack(side=tk.RIGHT, padx=5)
        tk.Button(undo_frame, text="Resmi Kaydet", command=self.save_image).pack(side=tk.RIGHT, padx=5)
        tk.Button(undo_frame, text="Tümünü Sıfırla", command=self.reset_image).pack(side=tk.RIGHT)

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="x")

        categories = {
            "Temel İşlemler": [
                ("Mavi Kanal", lambda: self.split_channel(0)),
                ("Yeşil Kanal", lambda: self.split_channel(1)),
                ("Kırmızı Kanal", lambda: self.split_channel(2)),
                ("Arka Plandan Nesne Ayır", self.subtract_background),
                ("Mantıksal Operatör", self.logical_operation),
                ("Toplama/Çıkarma", self.blend_images),
                ("Parlaklık +", lambda: self.adjust_brightness(30)),
                ("Parlaklık -", lambda: self.adjust_brightness(-30)),
                ("Kontrast +", lambda: self.adjust_contrast(1.5)),
                ("Kontrast -", lambda: self.adjust_contrast(0.5)),
                ("Negatif", self.negative),
                ("Gri", self.to_gray),
                ("Histogram", self.show_histogram),
                ("Eşikleme", self.threshold_image)
            ],
            "Filtreler": [
                ("Bulanıklaştır (Median/Gaussian)", self.blur_image),
                ("Netleştir (Keskinleştirme)", self.sharpen_image),
                ("Sobel", lambda: self.apply_custom_filter(apply_sobel)),
                ("Median Blur", lambda: self.apply_custom_filter(apply_median_filter)),
                ("Gaussian Blur", lambda: self.apply_custom_filter(apply_gaussian_blur))
            ],
            "Segmentasyon": [] if KMeans is None or ndimage is None else [
                ("Watershed", lambda: self.apply_custom_filter(apply_watershed)),
                ("K-Means", lambda: self.apply_custom_filter(apply_kmeans_segmentation))
            ],
            "Geometrik": [
                ("Zoom +", self.zoom_in),
                ("Zoom -", self.zoom_out),
                ("Taşıma", self.translate_image),
                ("Aynalama", self.mirror_image),
                ("Rotate", lambda: self.apply_custom_filter(apply_rotation)),
                                ("Stretching", lambda: self.apply_custom_filter(apply_stretching)),
                ("Shearing", lambda: self.apply_custom_filter(apply_shearing)),
                ("Dikdörtgen Maskele", lambda: self.apply_custom_filter(lambda img: self.mask_shape(img, 'rectangle'))),
                ("Çokgen Maskele", self.polygon_mask),
                ("Daire Maskele", lambda: self.apply_custom_filter(lambda img: self.mask_shape(img, 'circle'))),
                ("Elips Maskele", lambda: self.apply_custom_filter(lambda img: self.mask_shape(img, 'ellipse'))),
                            ],
            "Kenar & Morfoloji": [
                ("Kapalı Alanları Renklendir", self.fill_closed_regions),
                ("Kenar Bul (Sobel/Canny)", self.edge_detection),
                ("Hough", lambda: self.apply_custom_filter(apply_hough_lines)),
                ("Erode", lambda: self.apply_custom_filter(apply_erode)),
                ("Dilate", lambda: self.apply_custom_filter(apply_dilate))
            ]
        }

        for tab_name, button_list in categories.items():
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=tab_name)
            for label, cmd in button_list:
                tk.Button(frame, text=label, command=cmd).pack(side=tk.LEFT, padx=5, pady=5)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.image = cv2.imread(path)
            self.history.clear()
            self.display_image(self.image)

    def save_image(self):
        if self.image is not None:
            path = filedialog.asksaveasfilename(defaultextension=".png")
            if path:
                cv2.imwrite(path, self.image)
                messagebox.showinfo("Başarılı", "Resim kaydedildi")

    def undo(self):
        if self.history:
            self.image = self.history.pop()
            self.display_image(self.image)

    def display_image(self, img):
        if hasattr(self, 'load_button') and self.load_button.winfo_exists():
            self.load_button.destroy()
        if hasattr(self, 'image_label'):
            self.image_label.pack()

        max_width, max_height = 800, 600
        h, w = img.shape[:2]
        h = int(h * self.zoom_scale)
        w = int(w * self.zoom_scale)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        scale = min(max_width / w, max_height / h, 1.0)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk

    def adjust_brightness(self, value):
        if self.image is not None:
            self.history.append(self.image.copy())
            self.image = cv2.convertScaleAbs(self.image, beta=value)
            self.display_image(self.image)

    def adjust_contrast(self, alpha):
        if self.image is not None:
            self.history.append(self.image.copy())
            self.image = cv2.convertScaleAbs(self.image, alpha=alpha)
            self.display_image(self.image)

    def negative(self):
        if self.image is not None:
            self.history.append(self.image.copy())
            self.image = cv2.bitwise_not(self.image)
            self.display_image(self.image)

    def to_gray(self):
        if self.image is not None:
            self.history.append(self.image.copy())
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            self.display_image(self.image)

    def show_histogram(self):
        if self.image is not None:
            color = ('b', 'g', 'r')
            for i, col in enumerate(color):
                histr = cv2.calcHist([self.image], [i], None, [256], [0, 256])
                plt.plot(histr, color=col)
                plt.xlim([0, 256])
            plt.title("Histogram")
            plt.show()

    def threshold_image(self):
        if self.image is not None:
            self.history.append(self.image.copy())
            self.show_histogram()

            response = messagebox.askyesno("Eşikleme", "Çift eşik değeri kullanmak ister misiniz?")
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            if response:
                import tkinter.simpledialog as sd
                lower = sd.askinteger("Alt Eşik Değeri", "Alt eşik değerini girin:", minvalue=0, maxvalue=255)
                upper = sd.askinteger("Üst Eşik Değeri", "Üst eşik değerini girin:", minvalue=0, maxvalue=255)
                if lower is not None and upper is not None:
                    thresh = cv2.inRange(gray, lower, upper)
            else:
                value = tk.simpledialog.askinteger("Eşik Değeri", "Eşik değerini girin:", minvalue=0, maxvalue=255)
                if value is not None:
                    _, thresh = cv2.threshold(gray, value, 255, cv2.THRESH_BINARY)
                else:
                    return

            self.image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            self.display_image(self.image)

    def mask_shape(self, img, shape):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        h, w = img.shape[:2]
        center = self.mask_center if self.mask_center else (w // 2, h // 2)

        if shape == 'rectangle':
            cv2.rectangle(mask, (int(center[0] - w*0.25), int(center[1] - h*0.25)),
                          (int(center[0] + w*0.25), int(center[1] + h*0.25)), 255, -1)

        elif shape == 'circle':
            cv2.circle(mask, center, min(h, w) // 4, 255, -1)

        elif shape == 'ellipse':
            cv2.ellipse(mask, center, (w//4, h//6), 0, 0, 360, 255, -1)

        
        result = cv2.bitwise_and(img, img, mask=mask)
        background = np.full(img.shape, 255, dtype=np.uint8)
        inv_mask = cv2.bitwise_not(mask)
        outside = cv2.bitwise_and(background, background, mask=inv_mask)
        return cv2.add(result, outside)

    def set_mask_center(self, event):
        if self.image_label.image and self.image is not None:
            label_width = self.image_label.winfo_width()
            label_height = self.image_label.winfo_height()
            img_height, img_width = self.image.shape[:2]

            scale_x = img_width / label_width
            scale_y = img_height / label_height

            real_x = int(event.x * scale_x)
            real_y = int(event.y * scale_y)
            self.mask_center = (real_x, real_y)
            print(f"Maskeleme merkezi: {self.mask_center}")
            print(f"Maskeleme merkezi: {self.mask_center}")

    def translate_image(self):
        if self.image is not None:
            dx = tk.simpledialog.askinteger("X Ekseni", "Sağa/sola kaç piksel taşıyayım? (+sağ, -sol):", initialvalue=50)
            dy = tk.simpledialog.askinteger("Y Ekseni", "Yukarı/aşağı kaç piksel taşıyayım? (+aşağı, -yukarı):", initialvalue=50)
            if dx is not None and dy is not None:
                self.history.append(self.image.copy())
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                self.image = cv2.warpAffine(self.image, M, (self.image.shape[1], self.image.shape[0]))
                self.display_image(self.image)

    def mirror_image(self):
        if self.image is not None:
            option = tk.simpledialog.askstring("Aynalama", "Hangi eksende aynalama yapayım? (x/y/herikisi):", initialvalue="x")
            if option:
                self.history.append(self.image.copy())
                if option.lower() == 'x':
                    self.image = cv2.flip(self.image, 0)
                elif option.lower() == 'y':
                    self.image = cv2.flip(self.image, 1)
                elif option.lower() == 'herikisi':
                    self.image = cv2.flip(self.image, -1)
                self.display_image(self.image)

    def blur_image(self):
        if self.image is not None:
            choice = tk.simpledialog.askstring("Bulanıklaştırma", "Algoritma seç (median / gaussian):", initialvalue="median")
            if choice:
                self.history.append(self.image.copy())
                if choice.lower() == "median":
                    self.image = cv2.medianBlur(self.image, 5)
                elif choice.lower() == "gaussian":
                    self.image = cv2.GaussianBlur(self.image, (5, 5), 0)
                self.display_image(self.image)

    def sharpen_image(self):
        if self.image is not None:
            choice = tk.simpledialog.askstring("Netleştirme", "Algoritma seç (unsharp / laplacian):", initialvalue="unsharp")
            if choice:
                self.history.append(self.image.copy())
                if choice.lower() == "unsharp":
                    gaussian = cv2.GaussianBlur(self.image, (9, 9), 10.0)
                    self.image = cv2.addWeighted(self.image, 1.5, gaussian, -0.5, 0)
                elif choice.lower() == "laplacian":
                    lap = cv2.Laplacian(self.image, cv2.CV_64F)
                    self.image = cv2.convertScaleAbs(self.image + lap)
                self.display_image(self.image)

    def edge_detection(self):
        if self.image is not None:
            method = tk.simpledialog.askstring("Kenar Algılama", "Yöntem seçin: sobel / canny", initialvalue="sobel")
            if method:
                self.history.append(self.image.copy())
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                if method.lower() == 'sobel':
                    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                    edges = cv2.magnitude(sobelx, sobely)
                    edges = np.uint8(np.clip(edges, 0, 255))
                elif method.lower() == 'canny':
                    t1 = tk.simpledialog.askinteger("Canny Alt Eşik", "Alt eşik değerini girin:", initialvalue=100)
                    t2 = tk.simpledialog.askinteger("Canny Üst Eşik", "Üst eşik değerini girin:", initialvalue=200)
                    if t1 is None or t2 is None:
                        return
                    edges = cv2.Canny(gray, t1, t2)
                else:
                    return
                self.image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                self.display_image(self.image)

    def fill_closed_regions(self):
        if self.image is not None:
            self.history.append(self.image.copy())
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Alanları etiketle
            num_labels, labels_im = cv2.connectedComponents(thresh)
            colored = np.zeros_like(self.image)
            for label in range(1, num_labels):
                mask = labels_im == label
                color = np.random.randint(0, 255, size=3, dtype=np.uint8)
                colored[mask] = color

            # Aşındırma ve genişletme işlemi uygula
            operation = tk.simpledialog.askstring("Morfolojik İşlem", "Hangi işlem? (erode/dilate/none):", initialvalue="none")
            kernel = np.ones((3, 3), np.uint8)
            if operation == 'erode':
                colored = cv2.erode(colored, kernel, iterations=1)
            elif operation == 'dilate':
                colored = cv2.dilate(colored, kernel, iterations=1)

            self.image = colored
            self.display_image(self.image)

    def blend_images(self):
        if self.image is not None:
            path = filedialog.askopenfilename(title="İkinci resmi seçin")
            if not path:
                return
            img2 = cv2.imread(path)
            img1 = cv2.resize(self.image, (img2.shape[1], img2.shape[0]))

            method = tk.simpledialog.askstring("İşlem Türü", "Toplama mı çıkarma mı? (add / subtract):", initialvalue="add")
            strategy = tk.simpledialog.askstring("Sınır Stratejisi", "Sınırı aşarsa ne yapılsın? (clip / normalize):", initialvalue="clip")
            alpha = tk.simpledialog.askfloat("Ağırlık", "1. resim için ağırlık değeri (0.0-1.0):", minvalue=0.0, maxvalue=1.0, initialvalue=0.5)

            if method and strategy and alpha is not None:
                self.history.append(self.image.copy())
                if method == 'add':
                    result = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
                elif method == 'subtract':
                    result = cv2.subtract(img1, img2)

                if strategy == 'normalize':
                    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
                else:
                    result = np.clip(result, 0, 255).astype(np.uint8)

                self.image = result
                self.display_image(self.image)

    def logical_operation(self):
        if self.image is not None:
            path = filedialog.askopenfilename(title="İkinci resmi seçin")
            if not path:
                return
            img2 = cv2.imread(path)
            img1 = cv2.resize(self.image, (img2.shape[1], img2.shape[0]))

            method = tk.simpledialog.askstring("Mantıksal Operatör", "Operatörü seçin (and/or/xor/nand/nor):", initialvalue="and")
            strategy = tk.simpledialog.askstring("Sınır Stratejisi", "Sonuç nasıl işlenmeli? (clip / normalize):", initialvalue="clip")

            if method and strategy:
                self.history.append(self.image.copy())

                if method == 'and':
                    result = cv2.bitwise_and(img1, img2)
                elif method == 'or':
                    result = cv2.bitwise_or(img1, img2)
                elif method == 'xor':
                    result = cv2.bitwise_xor(img1, img2)
                elif method == 'nand':
                    result = cv2.bitwise_not(cv2.bitwise_and(img1, img2))
                elif method == 'nor':
                    result = cv2.bitwise_not(cv2.bitwise_or(img1, img2))
                else:
                    return

                if strategy == 'normalize':
                    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
                else:
                    result = np.clip(result, 0, 255).astype(np.uint8)

                self.image = result
                self.display_image(self.image)

    def subtract_background(self):
        if self.image is not None:
            path = filedialog.askopenfilename(title="Arka plan resmini seçin")
            if not path:
                return
            bg = cv2.imread(path)
            if bg is None:
                messagebox.showerror("Hata", "Arka plan resmi yüklenemedi.")
                return

            fg = self.image
            if fg.shape != bg.shape:
                bg = cv2.resize(bg, (fg.shape[1], fg.shape[0]))

            mode = tk.simpledialog.askstring("Mod Seçimi", "Çıkarılacak görüntü tipi: renkli / gri", initialvalue="renkli")
            self.history.append(self.image.copy())

            diff = cv2.absdiff(fg, bg)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

            if mode == "gri":
                obj = cv2.bitwise_and(gray, gray, mask=mask)
                self.image = cv2.cvtColor(obj, cv2.COLOR_GRAY2BGR)
            else:
                self.image = cv2.bitwise_and(fg, fg, mask=mask)

            self.display_image(self.image)

    def split_channel(self, index):
        if self.image is not None:
            self.history.append(self.image.copy())
            channels = cv2.split(self.image)
            zeros = np.zeros_like(channels[0])
            merged = [zeros, zeros, zeros]
            merged[index] = channels[index]
            self.image = cv2.merge(merged)
            self.display_image(self.image)

    def reset_image(self):
        if self.history:
            self.image = self.history[0]
            self.history = []
            self.display_image(self.image)

    def zoom_in(self):
        if self.image is not None:
            self.zoom_scale *= 1.2
            self.display_image(self.image)

    def zoom_out(self):
        if self.image is not None:
            self.zoom_scale = max(0.2, self.zoom_scale / 1.2)
            self.display_image(self.image)

    def polygon_mask(self):
        if self.image is not None:
            self.history.append(self.image.copy())
            img = self.image.copy()
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            h, w = img.shape[:2]
            center = self.mask_center if self.mask_center else (w // 2, h // 2)
            sides = 6
            radius = min(h, w) // 4
            angle_step = 2 * np.pi / sides
            pts = [
                [int(center[0] + radius * np.cos(i * angle_step)),
                 int(center[1] + radius * np.sin(i * angle_step))] for i in range(sides)
            ]
            pts = np.array(pts, np.int32)
            cv2.fillPoly(mask, [pts], 255)
            result = cv2.bitwise_and(img, img, mask=mask)
            background = np.full(img.shape, 255, dtype=np.uint8)
            inv_mask = cv2.bitwise_not(mask)
            outside = cv2.bitwise_and(background, background, mask=inv_mask)
            self.image = cv2.add(result, outside)
            self.display_image(self.image)

    def apply_custom_filter(self, func):
        if self.image is not None:
            self.history.append(self.image.copy())
            self.image = func(self.image)
            self.display_image(self.image)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessor(root)
    root.mainloop()
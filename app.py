import sys
import os
import json
import shutil
import torch
import numpy as np
import cv2  
import tempfile 
from datetime import datetime
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QMarginsF, QSize, QEvent
from PyQt6.QtGui import QPixmap, QTextDocument, QPageLayout, QIcon, QImage, QIntValidator    
from PyQt6.QtPrintSupport import QPrinter 
from PyQt6.QtWidgets import (QApplication, QMainWindow, QFileDialog, QVBoxLayout,
                             QHBoxLayout, QWidget, QProgressBar, QLabel, QPushButton, 
                             QStackedWidget, QMessageBox, QScrollArea, QLineEdit, 
                             QComboBox, QGroupBox, QListWidget, QFormLayout, 
                             QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView, QDialog,
                             QFrame, QListWidgetItem, QAbstractItemView)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pydicom

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from mri_analyzer import analyze_mri_unet

# [CẬP NHẬT] Đã loại bỏ ResNet và EEG, chuyển sang UNet
from unet_model import UNet  # Hãy đổi tên UNetMRI thành tên class UNet thực tế của bạn nếu cần
def normalize_mri_for_app(image):
    """
    Đồng bộ preprocessing với lúc train U-Net mới.
    Output luôn nằm trong [0, 1].
    """
    image = image.astype(np.float32)
    brain = image > 0

    if np.sum(brain) > 10:
        pixels = image[brain]
        lo = np.percentile(pixels, 1)
        hi = np.percentile(pixels, 99)

        if hi > lo:
            image = (image - lo) / (hi - lo)
        else:
            image = image / 255.0
    else:
        image = image / 255.0

    image = np.clip(image, 0.0, 1.0)
    return image


def load_unet_checkpoint(model_path, device):
    """
    Hỗ trợ cả 2 kiểu:
    1. state_dict thuần
    2. checkpoint dict có model_state_dict
    """
    model = UNet(in_channels=1, out_channels=1).to(device)

    checkpoint = torch.load(model_path, map_location=device)

    image_size = (256, 256)
    threshold = 0.35

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        image_size = tuple(checkpoint.get("image_size", (256, 256)))
        threshold = float(checkpoint.get("threshold", 0.35))
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, image_size, threshold
# ====================== DATABASE ======================
HISTORY_DIR = "patient_database"
os.makedirs(HISTORY_DIR, exist_ok=True)

APP_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #f4f7fb;
    color: #17324d;
    font-family: "Segoe UI";
}
QLabel#PageTitle {
    font-size: 32px;
    font-weight: 700;
    color: #11263c;
}
QLabel#PageSubtitle {
    font-size: 14px;
    color: #58708a;
}
QFrame#HeroCard, QFrame#MetricCard, QFrame#SurfaceCard, QFrame#PatientBar {
    background-color: #ffffff;
    border: 1px solid #dbe4ee;
    border-radius: 18px;
}
QFrame#HeroCard {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                      stop:0 #103a5c, stop:0.55 #195177, stop:1 #2b6d77);
    border: none;
}
QLabel#HeroTitle {
    color: white;
    font-size: 30px;
    font-weight: 700;
}
QLabel#HeroSubtitle {
    color: rgba(255, 255, 255, 0.84);
    font-size: 14px;
}
QLabel#MetricValue {
    font-size: 24px;
    font-weight: 700;
    color: #11263c;
}
QLabel#MetricLabel {
    font-size: 12px;
    color: #688098;
    text-transform: uppercase;
    font-weight: 600;
}
QLabel#MetricHint {
    font-size: 12px;
    color: #6f8599;
}
QGroupBox {
    font-size: 15px;
    font-weight: 700;
    color: #16314c;
    border: 1px solid #d6e1eb;
    border-radius: 18px;
    margin-top: 14px;
    background-color: white;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 16px;
    padding: 0 8px;
    color: #1d6d78;
}
QLineEdit, QComboBox, QTextEdit {
    background-color: #f8fbfd;
    border: 1px solid #ccd9e5;
    border-radius: 12px;
    padding: 10px 12px;
    font-size: 14px;
}
QLineEdit:focus, QComboBox:focus, QTextEdit:focus {
    border: 1px solid #1f8a8a;
}
QPushButton {
    border-radius: 12px;
    padding: 10px 18px;
    font-size: 14px;
    font-weight: 700;
}
QPushButton#PrimaryButton {
    background-color: #167c80;
    color: white;
    border: none;
}
QPushButton#PrimaryButton:hover {
    background-color: #11686b;
}
QPushButton#SecondaryButton {
    background-color: #eef6fa;
    color: #16587e;
    border: 1px solid #c6d8e6;
}
QPushButton#SecondaryButton:hover {
    background-color: #deedf6;
}
QPushButton#DangerButton {
    background-color: #fff1f0;
    color: #b34237;
    border: 1px solid #f0cbc7;
}
QPushButton#DangerButton:hover {
    background-color: #ffdeda;
}
QPushButton#GhostButton {
    background-color: transparent;
    color: white;
    border: 1px solid rgba(255, 255, 255, 0.35);
}
QPushButton#GhostButton:hover {
    background-color: rgba(255, 255, 255, 0.12);
}
QProgressBar {
    border: none;
    border-radius: 10px;
    background-color: #e8eef4;
    text-align: center;
    min-height: 16px;
}
QProgressBar::chunk {
    background-color: #17928b;
    border-radius: 10px;
}
QScrollArea {
    border: none;
    background: transparent;
}
QTableWidget {
    font-size: 14px;
    background-color: white;
    alternate-background-color: #f8fbfd;
    border: 1px solid #d6e1eb;
    border-radius: 14px;
    gridline-color: transparent;
    outline: none;
}
QTableWidget::item {
    padding: 6px;
    border-bottom: 1px solid #edf2f6;
}
QTableWidget::item:selected {
    background-color: #167c80;
    color: white;
}
QHeaderView::section {
    background-color: #18344f;
    color: white;
    font-weight: 700;
    padding: 10px;
    border: none;
}
"""

# ====================== DIALOG: SỬA THÔNG TIN BỆNH NHÂN ======================
class PatientEditDialog(QDialog):
    def __init__(self, patient_info, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Patient Information")
        self.setFixedSize(400, 250)
        self.patient_info = patient_info

        layout = QFormLayout(self)
        self.txt_id = QLineEdit(patient_info.get("id", ""))
        self.txt_id.setReadOnly(True)
        self.txt_id.setStyleSheet("background-color: #ecf0f1; border-radius: 5px; padding: 8px; font-size: 14px; color: #7f8c8d;")
        
        self.txt_name = QLineEdit(patient_info.get("name", ""))
        self.txt_name.setStyleSheet("border: 1px solid #bdc3c7; border-radius: 5px; padding: 8px; font-size: 14px;")
        
        self.txt_age = QLineEdit(str(patient_info.get("age", "")))
        self.txt_age.setStyleSheet("border: 1px solid #bdc3c7; border-radius: 5px; padding: 8px; font-size: 14px;")
        
        self.cb_gender = QComboBox()
        self.cb_gender.addItems(["-- Choose --", "Male", "Female", "Other"])
        self.cb_gender.setCurrentText(patient_info.get("gender", "-- Choose --"))
        self.cb_gender.setStyleSheet("border: 1px solid #bdc3c7; border-radius: 5px; padding: 8px; font-size: 14px;")

        layout.addRow("Full Name:", self.txt_name)
        layout.addRow("Age:", self.txt_age)
        layout.addRow("Gender:", self.cb_gender)

        btn_layout = QHBoxLayout()
        btn_save = QPushButton("Save Changes")
        btn_save.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 8px; border-radius: 5px;")
        btn_save.clicked.connect(self.accept)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_save)
        layout.addRow(btn_layout)

    def get_updated_info(self):
        self.patient_info["name"] = self.txt_name.text().strip()
        self.patient_info["age"] = self.txt_age.text().strip()
        self.patient_info["gender"] = self.cb_gender.currentText()
        return self.patient_info

# ====================== DIALOG: SỬA RAW AI DATA ======================
class EditAIResultDialog(QDialog):
    def __init__(self, raw_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit AI Diagnostic Findings")
        self.setMinimumSize(600, 400)
        layout = QVBoxLayout(self)
        info_label = QLabel("<b>Edit the raw AI output below.</b><br>You can safely delete incorrect finding rows. The formal HTML table will auto-regenerate.")
        info_label.setStyleSheet("color: #2980b9; margin-bottom: 10px;")
        layout.addWidget(info_label)

        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(raw_text)
        self.text_edit.setStyleSheet("font-family: 'Courier New', monospace; font-size: 13px; padding: 10px; border: 1px solid #bdc3c7;")
        layout.addWidget(self.text_edit)

        btn_layout = QHBoxLayout()
        btn_save = QPushButton("Save & Update Table")
        btn_save.setStyleSheet("background-color: #e67e22; color: white; font-weight: bold; padding: 10px; border-radius: 5px;")
        btn_save.clicked.connect(self.accept)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_save)
        layout.addLayout(btn_layout)

    def get_updated_text(self): return self.text_edit.toPlainText().strip()

# ====================== DIALOG: GHI CHÚ CỦA BÁC SĨ ======================
class EditNotesDialog(QDialog):
    def __init__(self, current_notes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Clinical Notes")
        self.setMinimumSize(500, 300)
        layout = QVBoxLayout(self)
        info_label = QLabel("<b>Add your clinical observations below:</b><br>If left blank, this section will NOT appear in the final PDF report.")
        info_label.setStyleSheet("color: #8e44ad; margin-bottom: 10px;")
        layout.addWidget(info_label)

        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Type clinical notes, recommendations, or correlation here...")
        self.text_edit.setPlainText(current_notes)
        self.text_edit.setStyleSheet("font-size: 14px; padding: 10px; border: 1px solid #bdc3c7;")
        layout.addWidget(self.text_edit)

        btn_layout = QHBoxLayout()
        btn_save = QPushButton("Save Notes")
        btn_save.setStyleSheet("background-color: #8e44ad; color: white; font-weight: bold; padding: 10px; border-radius: 5px;")
        btn_save.clicked.connect(self.accept)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_save)
        layout.addLayout(btn_layout)

    def get_notes(self): return self.text_edit.toPlainText().strip()

# ====================== CÁC COMPONENT GIAO DIỆN ======================
class FileDropList(QListWidget):
    file_count_changed = pyqtSignal(int) # Tín hiệu để báo ra ngoài cập nhật số lượng

    def __init__(self):
        super().__init__()
        # Cài đặt chế độ Grid (Lưới) và Ảnh thu nhỏ
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setIconSize(QSize(90, 90))
        self.setGridSize(QSize(110, 130)) # Kích thước ô đủ chứa 3 ảnh 1 hàng
        self.setSpacing(10)
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setMovement(QListWidget.Movement.Snap) # Hỗ trợ kéo thả đổi thứ tự
        self.setWordWrap(True) # Tự xuống dòng nếu tên file dài
        
        # Bật tính năng Drag & Drop cả từ ngoài vào và kéo thả nội bộ
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)

        self._default_style = (
            "QListWidget { border: 2px dashed #9bc7d8; border-radius: 16px; "
            "background-color: #f8fbfd; padding: 10px; font-size: 12px; color: #17324d; }"
            "QListWidget::item { border-radius: 8px; padding: 5px; }"
            "QListWidget::item:hover { background-color: #e8f7f6; }"
            "QListWidget::item:selected { background-color: #167c80; color: white; }"
        )
        self._active_style = self._default_style.replace("#9bc7d8", "#167c80").replace("#f8fbfd", "#e8f7f6")
        self.setStyleSheet(self._default_style)

    # Hàm tạo ảnh thu nhỏ (kể cả DICOM)
    def get_thumbnail(self, path):
        try:
            ext = path.lower().split('.')[-1]
            if ext in ['dcm', 'ima']:
                ds = pydicom.dcmread(path)
                img = ds.pixel_array.astype(np.float32)
                if img.ndim == 3: img = img[img.shape[0] // 2, :, :]
                img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
                img_rgb = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                h, w, ch = img_rgb.shape
                qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                return QIcon(QPixmap.fromImage(qimg))
            else:
                return QIcon(path)
        except Exception:
            pix = QPixmap(90, 90)
            pix.fill(Qt.GlobalColor.lightGray)
            return QIcon(pix)

    def add_file(self, file_path):
        existing_paths = [self.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.count())]
        if file_path not in existing_paths:
            icon = self.get_thumbnail(file_path)
            item = QListWidgetItem(icon, os.path.basename(file_path)) # Chỉ lấy tên file, bỏ đường dẫn
            item.setToolTip(file_path) # Hover chuột để xem full đường dẫn
            item.setData(Qt.ItemDataRole.UserRole, file_path) # Giấu đường dẫn thật ở dưới nền
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.addItem(item)
            self.file_count_changed.emit(self.count())

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(self._active_style)
        else: super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls(): event.acceptProposedAction()
        else: super().dragMoveEvent(event)

    def dragLeaveEvent(self, event):
        self.setStyleSheet(self._default_style)
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        self.setStyleSheet(self._default_style)
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if os.path.isfile(file_path) and file_path.lower().endswith(('.ima', '.dcm', '.jpg', '.jpeg', '.png')):
                    self.add_file(file_path)
                elif os.path.isdir(file_path):
                    for root, dirs, files in os.walk(file_path):
                        for file in files:
                            if file.lower().endswith(('.ima', '.dcm', '.jpg', '.jpeg', '.png')):
                                self.add_file(os.path.join(root, file))
        else:
            super().dropEvent(event) # Xử lý việc kéo thả đảo vị trí nội bộ
            self.file_count_changed.emit(self.count())

class MRICanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 5), facecolor='white')
        self.ax_orig = self.fig.add_subplot(121)
        self.ax_heat = self.fig.add_subplot(122)
        self.fig.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95, wspace=0.2)
        self.ax_orig.axis('off')
        self.ax_heat.axis('off')
        super().__init__(self.fig)

    def plot_image(self, img_with_boxes, heatmap, title=""):
        self.ax_orig.clear()
        self.ax_heat.clear()
        self.ax_orig.axis('on')
        self.ax_heat.axis('on')
        self.ax_orig.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
        self.ax_orig.set_title(f"MRI Scan\n[File: {title}]", fontsize=10, fontweight='bold')
        self.ax_orig.tick_params(axis='both', labelsize=8)
        self.ax_heat.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
        self.ax_heat.set_title(f"AI Activation Map\n[File: {title}]", fontsize=10, fontweight='bold')
        self.ax_heat.tick_params(axis='both', labelsize=8)
        self.draw()

    def wheelEvent(self, event): event.ignore()

# ====================== Worker nạp Model ngầm ======================
class ModelLoaderWorker(QThread):
    finished = pyqtSignal(object, object, object, float)

    def run(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, image_size, threshold = load_unet_checkpoint("models/unet_best.pth", device)
            self.finished.emit(model, device, image_size, threshold)
        except Exception as e:
            print("Lỗi load model ngầm:", e)
            
# ====================== Worker xử lý MRI (Đã Tối Ưu Siêu Tốc) ======================
class MRIPredictWorker(QThread):
    progress = pyqtSignal(int)
    done = pyqtSignal(str, np.ndarray, np.ndarray, str)

    def __init__(self, file_path, model, device, image_size=(256, 256), threshold=0.35):
        super().__init__()
        self.file_path = file_path
        self.model = model
        self.device = device
        self.image_size = tuple(image_size)
        self.threshold = float(threshold)

    def run(self):
        try:
            ext = self.file_path.lower()

            if ext.endswith(('.ima', '.dcm')):
                ds = pydicom.dcmread(self.file_path)
                image = ds.pixel_array.astype(np.float32)
            else:
                image = cv2.imread(self.file_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise Exception("Unsupported image format.")
                image = image.astype(np.float32)

            if image.ndim == 3:
                image = image[image.shape[0] // 2, :, :]

            # Resize theo đúng checkpoint đã train
            image_resized = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)

            # Đồng bộ preprocessing với train
            image_normalized = normalize_mri_for_app(image_resized)

            # Tensor shape: [1, 1, H, W]
            input_tensor = torch.from_numpy(image_normalized).unsqueeze(0).unsqueeze(0).float().to(self.device)

            self.progress.emit(50)

            # Gọi pipeline phân tích của bạn
            img_with_boxes, heatmap, prob_overall, has_anom, suggestions = analyze_mri_unet(
                input_tensor,
                self.model,
                image_normalized
            )

            self.progress.emit(100)

            img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

            if has_anom:
                details = "\n".join(suggestions)
                result_text = (
                    f"AI DETECTED ABNORMALITY (Confidence: {prob_overall:.1f}%)\n"
                    f"----------------------------------------------------\n"
                    f"{details}\n"
                    f"----------------------------------------------------\n"
                    f"Threshold used: {self.threshold:.2f}\n"
                    f"Heatmap generated directly from AI's deep neural layers."
                )
            else:
                result_text = (
                    f"NO ABNORMALITY DETECTED\n"
                    f"AI Confidence for normal structure: {(100 - prob_overall):.1f}%\n"
                    f"Threshold used: {self.threshold:.2f}\n"
                    f"The deep learning model found no pathological features."
                )

            self.done.emit(
                result_text,
                img_with_boxes,
                heatmap,
                os.path.basename(self.file_path)
            )

        except Exception as e:
            h, w = self.image_size
            self.done.emit(
                f"Processing Error: {str(e)}",
                np.zeros((h, w, 3), dtype=np.uint8),
                np.zeros((h, w, 3), dtype=np.uint8),
                "Error"
            )
class SortItem(QTableWidgetItem):
    def __init__(self, display_text, sort_value, tie_breaker_id):
        super().__init__(display_text)
        self.sort_value = sort_value          
        self.tie_breaker_id = tie_breaker_id  

    def __lt__(self, other):
        if self.sort_value == other.sort_value:
            return self.tie_breaker_id < other.tie_breaker_id
        return self.sort_value < other.sort_value
    
# ====================== CỬA SỔ CHÍNH ======================
class NeuroVisionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroVision - MRI Review Workspace")
        self.resize(1360, 900)
        self.setStyleSheet(APP_STYLESHEET)
        self.setAcceptDrops(True)
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        
        # ==========================================================
        # [CẬP NHẬT] LOAD UNET MODEL 1 LẦN DUY NHẤT LÚC MỞ APP
        # ==========================================================
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ai_model, self.model_image_size, self.model_threshold = load_unet_checkpoint(
            "models/unet_best.pth",
            self.device
        )
        # ==========================================================
        
        self.patient_info = {}
        self.files_to_process = []
        self.current_processing_index = 0
        self.history = []
        self.current_index = -1

        btn_style = "QPushButton { border: 2px solid #7f8c8d; border-radius: 10px; padding: 12px 30px; font-size: 16px; font-weight: bold; background-color: white; color: black; } QPushButton:hover { background-color: #ecf0f1; }"

        # ================= TRANG 1: HOME =================
        self.home_page = QWidget()
        home_layout = QVBoxLayout(self.home_page)
        home_layout.setContentsMargins(28, 24, 28, 24)
        home_layout.setSpacing(18)

        header_layout = QHBoxLayout()
        title = QLabel("NeuroVision MRI Analysis")
        title.setObjectName("PageTitle")
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        btn_open_history = QPushButton("PATIENT DATABASE")
        btn_open_history.setStyleSheet("QPushButton { border: none; background-color: #8e44ad; border-radius: 8px; padding: 10px 20px; font-weight: bold; color: white; font-size: 16px; } QPushButton:hover { background-color: #9b59b6; }")
        btn_open_history.clicked.connect(self.open_patient_browser)
        header_layout.addWidget(btn_open_history)
        home_layout.addLayout(header_layout)

        body_layout = QHBoxLayout()
        
        patient_group = QGroupBox("1. Patient Information")
        patient_group.setStyleSheet("QGroupBox { font-size: 16px; font-weight: 700; border: 1px solid #d6e1eb; border-radius: 18px; margin-top: 12px; background-color: white; } QGroupBox::title { subcontrol-origin: margin; left: 16px; padding: 0 8px; color: #1d6d78; }")
        form_layout = QFormLayout(patient_group)
        form_layout.setContentsMargins(20, 30, 20, 20)
        
        self.txt_id = QLineEdit(f"PID-{datetime.now().strftime('%Y%m%d%H%M')}")
        self.txt_id.setReadOnly(True)
        self.txt_id.setStyleSheet("background-color: #eef3f7; border-radius: 12px; padding: 10px; font-size: 14px; color: #6c8196; font-weight: 600;")
        
        self.txt_name = QLineEdit()
        self.txt_name.setPlaceholderText("e.g. Nguyen Van A")
        self.txt_name.setStyleSheet("border: 1px solid #ccd9e5; border-radius: 12px; padding: 10px; font-size: 14px; background-color: #f8fbfd;")
        self.lbl_err_name = QLabel("* Missing Patient Name")
        self.lbl_err_name.setStyleSheet("color: #c0392b; font-size: 12px; font-weight: bold; margin-bottom: 5px;")
        self.lbl_err_name.setVisible(False)
        
        self.txt_age = QLineEdit()
        self.txt_age.setPlaceholderText("0 - 200")
        self.txt_age.setValidator(QIntValidator(0, 200)) # KHÓA CHỈ CHO NHẬP SỐ TỪ 0 ĐẾN 200
        self.txt_age.setStyleSheet("border: 1px solid #ccd9e5; border-radius: 12px; padding: 10px; font-size: 14px; background-color: #f8fbfd;")
        self.lbl_err_age = QLabel("* Invalid Age (Must be 0-200)")
        self.lbl_err_age.setStyleSheet("color: #c0392b; font-size: 12px; font-weight: bold; margin-bottom: 5px;")
        self.lbl_err_age.setVisible(False)
        
        self.cb_gender = QComboBox()
        self.cb_gender.addItems(["-- Choose --", "Male", "Female", "Other"])
        self.cb_gender.setStyleSheet("border: 1px solid #ccd9e5; border-radius: 12px; padding: 10px; font-size: 14px; background-color: #f8fbfd;")
        self.lbl_err_gender = QLabel("* Please select Gender")
        self.lbl_err_gender.setStyleSheet("color: #c0392b; font-size: 12px; font-weight: bold; margin-bottom: 5px;")
        self.lbl_err_gender.setVisible(False)

        self.txt_name.textChanged.connect(self.clear_name_error)
        self.txt_age.textChanged.connect(self.clear_age_error)
        self.cb_gender.currentIndexChanged.connect(self.clear_gender_error)

        self.txt_name.installEventFilter(self)
        self.txt_age.installEventFilter(self)

        form_layout.addRow(QLabel("Patient ID:"), self.txt_id)
        form_layout.addRow(QLabel("Full Name:"), self.txt_name)
        form_layout.addRow("", self.lbl_err_name)
        form_layout.addRow(QLabel("Age:"), self.txt_age)
        form_layout.addRow("", self.lbl_err_age)
        form_layout.addRow(QLabel("Gender:"), self.cb_gender)
        form_layout.addRow("", self.lbl_err_gender)
        
        file_group = QGroupBox("2. MRI Scans Queue (Drag & Drop Here)")
        file_group.setStyleSheet("QGroupBox { font-size: 16px; font-weight: 700; border: 1px solid #d6e1eb; border-radius: 18px; margin-top: 12px; background-color: white; } QGroupBox::title { subcontrol-origin: margin; left: 16px; padding: 0 8px; color: #1d6d78; }")
        file_layout = QVBoxLayout(file_group)
        
        self.lbl_file_count = QLabel("Total files in queue: 0")
        self.lbl_file_count.setStyleSheet("font-size: 14px; font-weight: bold; color: #167c80; margin-left: 5px;")
        file_layout.addWidget(self.lbl_file_count)

        self.file_list = FileDropList()
        self.file_list.file_count_changed.connect(lambda count: self.lbl_file_count.setText(f"Total files in queue: {count}"))
        file_layout.addWidget(self.file_list)

        queue_btns = QHBoxLayout()
        btn_browse = QPushButton("Browse")
        btn_clear = QPushButton("Clear")
        btn_browse.setStyleSheet("QPushButton { border: 1px solid #c6d8e6; border-radius: 12px; padding: 10px 16px; color: #16587e; font-weight: 700; background-color: #eef6fa; } QPushButton:hover { background-color: #deedf6; }")
        btn_clear.setStyleSheet("QPushButton { border: 1px solid #f0cbc7; border-radius: 12px; padding: 10px 16px; color: #b34237; font-weight: 700; background-color: #fff1f0; } QPushButton:hover { background-color: #ffdeda; }")
        btn_browse.clicked.connect(self.browse_files)
        btn_clear.clicked.connect(self.clear_file_queue)
        queue_btns.addWidget(btn_browse)
        queue_btns.addWidget(btn_clear)
        file_layout.addLayout(queue_btns)

        body_layout.addWidget(patient_group, 4)
        body_layout.addWidget(file_group, 6)
        home_layout.addLayout(body_layout)

        self.btn_start = QPushButton("START MRI ANALYSIS")
        self.btn_start.setFixedHeight(60)
        self.btn_start.setStyleSheet("QPushButton { border: none; border-radius: 18px; font-size: 20px; font-weight: 700; background-color: #167c80; color: white; margin-top: 10px; } QPushButton:hover { background-color: #11686b; }")
        self.btn_start.clicked.connect(self.start_batch_analysis)
        home_layout.addWidget(self.btn_start)
        self.stack.addWidget(self.home_page)

        # ================= TRANG 2: RESULTS =================
        self.mri_page = QWidget()
        mri_main_layout = QVBoxLayout(self.mri_page)

        self.patient_bar = QWidget()
        self.patient_bar.setStyleSheet("background-color: white; border: 1px solid #d6e1eb; border-radius: 16px; padding: 8px;")
        bar_layout = QHBoxLayout(self.patient_bar)
        self.lbl_patient_info = QLabel("Patient: ---")
        self.lbl_patient_info.setStyleSheet("font-size: 16px; font-weight: 700; color: #16314c;")
        
        btn_edit_patient = QPushButton("Edit Info")
        btn_edit_patient.setStyleSheet("background-color: #eef6fa; color: #16587e; font-weight: 700; border: 1px solid #c6d8e6; border-radius: 12px; padding: 10px 16px;")
        btn_edit_patient.clicked.connect(self.edit_patient_info)
        
        bar_layout.addWidget(self.lbl_patient_info)
        bar_layout.addStretch()
        bar_layout.addWidget(btn_edit_patient)
        mri_main_layout.addWidget(self.patient_bar)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; }")
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        
        self.display_stack = QStackedWidget()
        self.mri_canvas = MRICanvas(self)
        self.mri_canvas.setMinimumHeight(450)
        self.display_stack.addWidget(self.mri_canvas) 
        
        self.loading_label = QLabel("Analyzing... Please wait")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.setStyleSheet("font-size: 32px; font-weight: 700; color: #547089; background-color: #f8fbfd; border: 2px dashed #b6ccda; border-radius: 20px;")
        self.display_stack.addWidget(self.loading_label) 
        self.scroll_layout.addWidget(self.display_stack)

        self.mri_progress = QProgressBar()
        self.mri_progress.setStyleSheet("QProgressBar { border: none; border-radius: 10px; text-align: center; height: 18px; background-color: #e8eef4; } QProgressBar::chunk { background-color: #17928b; border-radius: 10px; }")
        self.mri_progress.setVisible(False)
        self.scroll_layout.addWidget(self.mri_progress)

        result_header = QHBoxLayout()
        findings_title = QLabel("Clinical Notes & Findings")
        findings_title.setStyleSheet("font-size: 18px; font-weight: 700; color: #16314c;")
        result_header.addWidget(findings_title)
        result_header.addStretch()
        
        btn_edit_ai = QPushButton("Edit AI Findings")
        btn_edit_ai.setStyleSheet("background-color: #eef6fa; color: #16587e; font-weight: 700; border: 1px solid #c6d8e6; border-radius: 12px; padding: 10px 16px;")
        btn_edit_ai.clicked.connect(self.edit_ai_raw_data)
        
        btn_edit_notes = QPushButton("Add Clinical Notes")
        btn_edit_notes.setStyleSheet("background-color: #167c80; color: white; font-weight: 700; border-radius: 12px; padding: 10px 16px;")
        btn_edit_notes.clicked.connect(self.edit_dr_notes)
        
        result_header.addWidget(btn_edit_ai)
        result_header.addWidget(btn_edit_notes)
        self.scroll_layout.addLayout(result_header)

        self.mri_result = QTextEdit()
        self.mri_result.setReadOnly(True)
        self.mri_result.setMinimumHeight(350)
        self.mri_result.setStyleSheet("QTextEdit { background-color: #f8fbfd; border: 1px solid #d6e1eb; border-radius: 18px; padding: 18px; font-family: 'Segoe UI'; }")
        self.scroll_layout.addWidget(self.mri_result)

        self.scroll_area.setWidget(self.scroll_content)
        mri_main_layout.addWidget(self.scroll_area)

        nav_mri = QHBoxLayout()
        self.btn_prev = QPushButton("Prev Scan")
        self.btn_next = QPushButton("Next Scan")
        self.btn_delete_scan = QPushButton("Delete Scan")
        self.btn_add_scan = QPushButton("Add Scans")
        
        self.btn_prev.setStyleSheet("QPushButton { border: 1px solid #c6d8e6; border-radius: 12px; padding: 12px 18px; font-size: 15px; font-weight: 700; background-color: #eef6fa; color: #16587e; } QPushButton:hover { background-color: #deedf6; }")
        self.btn_next.setStyleSheet("QPushButton { border: 1px solid #c6d8e6; border-radius: 12px; padding: 12px 18px; font-size: 15px; font-weight: 700; background-color: #eef6fa; color: #16587e; } QPushButton:hover { background-color: #deedf6; }")
        self.btn_delete_scan.setStyleSheet("QPushButton { border: 1px solid #f0cbc7; border-radius: 12px; padding: 12px 18px; font-size: 15px; font-weight: 700; background-color: #fff1f0; color: #b34237; } QPushButton:hover { background-color: #ffdeda; }")
        self.btn_add_scan.setStyleSheet("QPushButton { border: 1px solid #c6d8e6; border-radius: 12px; padding: 12px 18px; font-size: 15px; font-weight: 700; background-color: #eef6fa; color: #16587e; } QPushButton:hover { background-color: #deedf6; }")

        for btn in [self.btn_prev, self.btn_next, self.btn_delete_scan]:
            sp = btn.sizePolicy()
            sp.setRetainSizeWhenHidden(True)
            btn.setSizePolicy(sp)

        self.btn_prev.clicked.connect(self.show_prev)
        self.btn_next.clicked.connect(self.show_next)
        self.btn_delete_scan.clicked.connect(self.delete_current_scan)
        self.btn_add_scan.clicked.connect(self.add_more_scans_browse)

        self.btn_export_img = QPushButton("Export Image")
        self.btn_export_pdf = QPushButton("Export Full Report")
        
        self.btn_export_img.setStyleSheet("QPushButton { border: 1px solid #c6d8e6; border-radius: 12px; padding: 12px 18px; font-size: 15px; font-weight: 700; background-color: #eef6fa; color: #16587e; } QPushButton:hover { background-color: #deedf6; }")
        self.btn_export_pdf.setStyleSheet("QPushButton { border: none; border-radius: 12px; padding: 12px 18px; font-size: 15px; font-weight: 700; background-color: #167c80; color: white; } QPushButton:hover { background-color: #11686b; }")
        
        self.btn_export_img.clicked.connect(self.export_image)
        self.btn_export_pdf.clicked.connect(lambda: self.export_pdf_report(show_msg=True))

        btn_new_patient = QPushButton("Main Menu")
        btn_new_patient.setStyleSheet("QPushButton { border: 1px solid #c6d8e6; border-radius: 12px; padding: 12px 18px; font-size: 15px; font-weight: 700; background-color: #ffffff; color: #16587e; } QPushButton:hover { background-color: #eef6fa; }")
        btn_new_patient.clicked.connect(self.reset_home)
        
        nav_mri.addWidget(self.btn_prev)
        nav_mri.addWidget(self.btn_next)
        nav_mri.addWidget(self.btn_delete_scan)
        nav_mri.addWidget(self.btn_add_scan)
        nav_mri.addStretch()
        nav_mri.addWidget(self.btn_export_img)
        nav_mri.addWidget(self.btn_export_pdf)
        nav_mri.addWidget(btn_new_patient) 
        mri_main_layout.addLayout(nav_mri)
        self.stack.addWidget(self.mri_page)

        # ================= TRANG 3: BROWSER LỊCH SỬ BỆNH NHÂN =================
        self.browser_page = QWidget()
        browser_layout = QVBoxLayout(self.browser_page)
        
        browser_header = QHBoxLayout()
        b_title = QLabel("PATIENT DATABASE")
        b_title.setStyleSheet("font-size: 28px; font-weight: 700; color: #16314c;")
        btn_b_back = QPushButton("Back to Home")
        btn_b_back.setStyleSheet("QPushButton { border: 1px solid #c6d8e6; border-radius: 12px; padding: 10px 16px; font-weight: 700; color: #16587e; background: #eef6fa;} QPushButton:hover { background-color: #deedf6; }")
        btn_b_back.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        browser_header.addWidget(b_title)
        browser_header.addStretch()
        browser_header.addWidget(btn_b_back)
        browser_layout.addLayout(browser_header)

        search_layout = QHBoxLayout()
        self.txt_search = QLineEdit()
        self.txt_search.setPlaceholderText("Search by ID, Name, or Date...")
        self.txt_search.setStyleSheet("padding: 12px; font-size: 14px; border: 1px solid #ccd9e5; border-radius: 12px; background-color: #f8fbfd;")
        self.txt_search.textChanged.connect(self.filter_patients)
        search_layout.addWidget(self.txt_search)
        browser_layout.addLayout(search_layout)

        # [CẬP NHẬT] Bảng Lịch sử chuyên nghiệp
        self.table_patients = QTableWidget()
        self.table_patients.setColumnCount(6)
        self.table_patients.setHorizontalHeaderLabels(["Date", "Patient ID", "Name", "Age/Gender", "Scans", "Status"])
        self.table_patients.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_patients.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_patients.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_patients.setAlternatingRowColors(True)
        self.table_patients.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        
        self.table_patients.verticalHeader().setVisible(False)
        self.table_patients.setSortingEnabled(True)

        self.table_patients.setStyleSheet("""
            QTableWidget { font-size: 14px; background-color: white; alternate-background-color: #f8fbfd; border: 1px solid #d6e1eb; border-radius: 14px; gridline-color: transparent; }
            QTableWidget::item { padding: 6px; border-bottom: 1px solid #edf2f6; }
            QTableWidget::item:hover { background-color: #e6f3f5; }
            QTableWidget::item:selected { background-color: #167c80; color: white; }
            QHeaderView::section { background-color: #18344f; color: white; font-weight: 700; padding: 10px; border: none; }
        """)
        
        self.table_patients.cellDoubleClicked.connect(self.load_selected_patient_from_cell)
        browser_layout.addWidget(self.table_patients)

        b_footer = QLabel("<i>* Double-click on any patient row to open their full medical record. Click headers to sort.</i>")
        b_footer.setStyleSheet("color: #71879b; font-size: 14px;")
        browser_layout.addWidget(b_footer)
        self.stack.addWidget(self.browser_page)

    def clear_file_queue(self):
        self.file_list.clear()
        self.lbl_file_count.setText("Total files in queue: 0")

    def filter_patients(self, text):
        search_text = text.lower()
        for row in range(self.table_patients.rowCount()):
            match = False
            for col in range(self.table_patients.columnCount()):
                item = self.table_patients.item(row, col)
                if item and search_text in item.text().lower():
                    match = True
                    break
            self.table_patients.setRowHidden(row, not match)

    def browse_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select MRI Files", "", "Images (*.png *.jpg *.jpeg *.dcm *.ima)")
        for f in files:
            self.file_list.add_file(f)

    def reset_home(self):
        self.txt_name.clear()
        self.txt_age.clear()
        self.cb_gender.setCurrentIndex(0)
        self.file_list.clear()
        self.txt_id.setText(f"PID-{datetime.now().strftime('%Y%m%d%H%M')}")
        self.stack.setCurrentIndex(0)

    def update_patient_bar(self):
        self.lbl_patient_info.setText(f"Patient: {self.patient_info.get('name', 'N/A')} | Age: {self.patient_info.get('age', '-')} | Gender: {self.patient_info.get('gender', '-')} | ID: {self.patient_info.get('id', '-')}")

    def edit_patient_info(self):
        dialog = PatientEditDialog(self.patient_info, self)
        if dialog.exec():
            self.patient_info = dialog.get_updated_info()
            self.update_patient_bar()
            self.save_patient_to_db()

    def edit_ai_raw_data(self):
        if self.current_index < 0 or self.current_index >= len(self.history): return
        record = self.history[self.current_index]
        dialog = EditAIResultDialog(record["raw_text"], self)
        if dialog.exec():
            record["raw_text"] = dialog.get_updated_text()
            self.render_record()
            self.save_patient_to_db()
            QMessageBox.information(self, "Success", "AI Data updated.")

    def edit_dr_notes(self):
        if self.current_index < 0 or self.current_index >= len(self.history): return
        record = self.history[self.current_index]
        dialog = EditNotesDialog(record.get("dr_notes", ""), self)
        if dialog.exec():
            record["dr_notes"] = dialog.get_notes()
            self.render_record()
            self.save_patient_to_db()

    def delete_current_scan(self):
        if self.current_index < 0 or self.current_index >= len(self.history): return
        reply = QMessageBox.question(self, 'Delete Scan', "Are you sure you want to remove this scan?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            del self.history[self.current_index]
            self.save_patient_to_db()
            if len(self.history) == 0:
                QMessageBox.information(self, "Empty", "All scans deleted. Returning to main menu.")
                self.reset_home()
            else:
                if self.current_index >= len(self.history): self.current_index -= 1
                self.render_record()

    def start_batch_analysis(self):
        if not self.txt_name.text().strip():
            QMessageBox.warning(self, "Missing Info", "Please enter the Patient's Name.")
            return
        if not self.txt_age.text().strip():
            QMessageBox.warning(self, "Missing Info", "Please enter the Patient's Age.")
            return
        if self.cb_gender.currentText() == "-- Choose --":
            QMessageBox.warning(self, "Missing Info", "Please select the Patient's Gender.")
            return
        if self.file_list.count() == 0:
            QMessageBox.warning(self, "Missing Files", "Please add at least one MRI file to the queue.")
            return

        self.patient_info = {
            "id": self.txt_id.text(),
            "name": self.txt_name.text().strip(),
            "age": self.txt_age.text().strip(),
            "gender": self.cb_gender.currentText(),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.patient_folder = os.path.join(HISTORY_DIR, self.patient_info["id"])
        os.makedirs(self.patient_folder, exist_ok=True)

        self.files_to_process = [self.file_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.file_list.count())]
        self.history = []
        self.current_processing_index = 0
        
        self.update_patient_bar()
        self.stack.setCurrentIndex(1)
        self.process_next_file()

    def process_next_file(self):
        if self.current_processing_index < len(self.files_to_process):
            file_path = self.files_to_process[self.current_processing_index]
            self.display_stack.setCurrentIndex(1)
            self.mri_progress.setValue(0)
            self.mri_progress.setVisible(True)
            self.mri_result.clear()
            self.loading_label.setText(f"Scanning File {self.current_processing_index + 1} of {len(self.files_to_process)}...\n{os.path.basename(file_path)}")
            
            self.btn_export_pdf.setEnabled(False)
            self.btn_export_img.setEnabled(False)
            self.btn_prev.setVisible(False)
            self.btn_next.setVisible(False)
            self.btn_delete_scan.setVisible(False)

            self.mri_worker = MRIPredictWorker(
                file_path,
                self.ai_model,
                self.device,
                image_size=self.model_image_size,
                threshold=self.model_threshold
            )
            self.mri_worker.progress.connect(self.update_progress)
            self.mri_worker.done.connect(self.handle_mri_result)
            self.mri_worker.start()
        else:
            self.save_patient_to_db()
            self.current_index = 0
            self.render_record()
            QMessageBox.information(self, "Batch Complete", f"Successfully analyzed {len(self.files_to_process)} files.\nRecord saved.")

    def update_progress(self, val):
        self.mri_progress.setValue(val)
        if val == 100: self.mri_progress.setVisible(False)

    def generate_base_html(self, raw_text):
        is_abnormal = "NO ABNORMALITY" not in raw_text
        theme_color = "#c0392b" if is_abnormal else "#27ae60"
        lines = raw_text.split('\n')
        overall_status = ""
        table_rows = ""
        notes = "" # Thêm biến này để hứng text của UNet

        for line in lines:
            line = line.strip()
            if not line or "--------" in line: continue
            if "Threshold used" in line: continue
            if "DETECTED" in line or "NORMAL" in line:
                overall_status = line.replace("⚠️", "").replace("✅", "").strip()
            elif "| AI" in line:
                try:
                    line = line.lstrip("•").strip()
                    parts = line.split('|')
                    p0 = parts[0].strip()
                    box_info, pathology = p0.split(']', 1)
                    box_info = box_info.replace('[', '').replace('•', '').strip()
                    box_parts = box_info.split('-')
                    box_id = box_parts[0].strip() if len(box_parts) > 0 else "-"
                    box_color = box_parts[1].strip() if len(box_parts) > 1 else "-"
                    color_css = "color: #27ae60;" if "Green" in box_color else "color: #2980b9;" if "Blue" in box_color else "color: #34495e;"
                    box_color_html = f"<b style='{color_css}'>{box_color}</b>"
                    table_rows += f"<tr><td align='center'><b>{box_id}</b></td><td align='center'>{box_color_html}</td><td><b>{pathology.strip()}</b></td><td align='center' style='color: #c0392b;'><b>{parts[1].replace('AI Focus:', '').strip()}</b></td><td align='center' style='font-family: Courier New; font-size: 9pt;'>{parts[2].replace('Pos:', '').strip()}</td></tr>"
                except Exception: pass
            else:
                # Hứng các dòng text kết quả từ UNet
                cleaned_line = line.replace('👉', '').replace('•', '').strip()
                if cleaned_line and "Heatmap generated" not in cleaned_line:
                    notes += f"<li>{cleaned_line}</li>"

        findings_html = f"<p style='font-size: 14pt; font-weight: bold; color: {theme_color}; margin-bottom: 5px;'>{overall_status}</p>"
        if table_rows:
            findings_html += f"<table width='100%' border='1' cellspacing='0' cellpadding='8' style='border-collapse: collapse; border: 1px solid #bdc3c7;'><thead><tr style='background-color: #18344f; color: white; font-weight: bold;'><th width='8%'>ID</th><th width='15%'>Marker</th><th width='40%'>Pathological Finding</th><th width='15%'>AI Confidence</th><th width='22%'>Coordinates</th></tr></thead><tbody>{table_rows}</tbody></table>"
        if notes:
            findings_html += f"<ul style='font-size: 12pt; color: #34495e; margin-top: 15px;'>{notes}</ul>"
            
        return findings_html

    def handle_mri_result(self, result_text, img_with_boxes, heatmap, file_name):
        box_img_path = os.path.join(self.patient_folder, f"scan_{self.current_processing_index}_box.jpg")
        heat_img_path = os.path.join(self.patient_folder, f"scan_{self.current_processing_index}_heat.jpg")
        cv2.imwrite(box_img_path, img_with_boxes)
        cv2.imwrite(heat_img_path, heatmap)

        record = {
            "raw_text": result_text,
            "dr_notes": "",  
            "file": file_name,
            "box_img": box_img_path,
            "heat_img": heat_img_path,
            "img": img_with_boxes, 
            "heat": heatmap
        }
        self.history.append(record)
        self.current_processing_index += 1
        self.process_next_file() 

    def save_patient_to_db(self):
        json_path = os.path.join(self.patient_folder, "data.json")
        data_to_save = {
            "patient_info": self.patient_info,
            "scans": [{"file": r["file"], "raw_text": r.get("raw_text", ""), "dr_notes": r.get("dr_notes", ""), "box_img": r["box_img"], "heat_img": r["heat_img"]} for r in self.history]
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=4)

    def open_patient_browser(self):
        self.table_patients.setSortingEnabled(False) 
        self.table_patients.setRowCount(0)
        # [CẬP NHẬT] Đổi lại tiêu đề 7 cột
        self.table_patients.setColumnCount(7)
        self.table_patients.setHorizontalHeaderLabels(["Date", "Patient ID", "Name", "Age", "Gender", "Scans", "Status"])
        self.txt_search.clear() 
        folders = [f.path for f in os.scandir(HISTORY_DIR) if f.is_dir()]
        
        for folder in folders:
            json_path = os.path.join(folder, "data.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    p_info = data["patient_info"]
                    scans = data.get("scans", [])
                    
                    pid = p_info.get("id", "")
                    date_str = p_info.get("date", "N/A")
                    name = p_info.get("name", "")
                    age_str = p_info.get("age", "0")
                    age_int = int(age_str) if age_str.isdigit() else 0 
                    gender = p_info.get("gender", "")
                    scans_count = len(scans)

                    has_abnormal = any("NO ABNORMALITY" not in s.get("raw_text", "") for s in scans)
                    status_text = "⚠️ Abnormal" if has_abnormal else "✅ Normal"

                    row = self.table_patients.rowCount()
                    self.table_patients.insertRow(row)
                    
                    item_date = SortItem(date_str, date_str, pid)
                    item_id = SortItem(pid, pid, pid)
                    item_name = SortItem(name, name.lower(), pid)
                    item_age = SortItem(age_str, age_int, pid) # Tách riêng Tuổi
                    item_gender = SortItem(gender, gender, pid) # Tách riêng Giới tính
                    
                    item_scans = SortItem(f"{scans_count} scans", scans_count, pid)
                    item_scans.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    
                    item_status = SortItem(status_text, status_text, pid)
                    item_status.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    
                    if has_abnormal:
                        item_status.setForeground(Qt.GlobalColor.red)
                    else:
                        item_status.setForeground(Qt.GlobalColor.darkGreen)
                    font = item_status.font()
                    font.setBold(True)
                    item_status.setFont(font)

                    self.table_patients.setItem(row, 0, item_date)
                    self.table_patients.setItem(row, 1, item_id)
                    self.table_patients.setItem(row, 2, item_name)
                    self.table_patients.setItem(row, 3, item_age)
                    self.table_patients.setItem(row, 4, item_gender)
                    self.table_patients.setItem(row, 5, item_scans)
                    self.table_patients.setItem(row, 6, item_status)

                    self.table_patients.item(row, 0).setData(Qt.ItemDataRole.UserRole, folder) 
                except Exception as e: print(f"Error loading {json_path}: {e}")
        
        self.table_patients.sortItems(1, Qt.SortOrder.DescendingOrder)
        self.table_patients.setSortingEnabled(True) 
        self.stack.setCurrentIndex(2)

    def load_selected_patient_from_cell(self, row, column):
        folder = self.table_patients.item(row, 0).data(Qt.ItemDataRole.UserRole)
        json_path = os.path.join(folder, "data.json")
        try:
            with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
            self.patient_info = data["patient_info"]
            self.patient_folder = folder
            self.history = []
            
            for s in data["scans"]:
                box_img = cv2.imread(s["box_img"])
                heat_img = cv2.imread(s["heat_img"])
                if box_img is None or heat_img is None: continue
                self.history.append({
                    "file": s["file"],
                    "raw_text": s.get("raw_text", ""),
                    "dr_notes": s.get("dr_notes", ""),
                    "box_img": s["box_img"],
                    "heat_img": s["heat_img"],
                    "img": box_img,
                    "heat": heat_img
                })

            if len(self.history) > 0:
                self.update_patient_bar()
                self.current_index = 0
                self.stack.setCurrentIndex(1)
                self.render_record()
            else:
                QMessageBox.warning(self, "Error", "No image data found in this record.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load patient data: {e}")

    def render_record(self):
        record = self.history[self.current_index]
        self.display_stack.setCurrentIndex(0)
        self.mri_canvas.plot_image(record['img'], record['heat'], title=record['file'])

        base_html = self.generate_base_html(record['raw_text'])
        
        notes_html = ""
        if record.get('dr_notes'):
            notes_html = f"<br><p style='font-size: 12pt; font-weight: bold; color: #8e44ad; margin-bottom: 5px;'>Dr. Notes:</p><p style='font-size: 12pt; color: #34495e;'>{record['dr_notes'].replace(chr(10), '<br>')}</p>"

        file_counter = f"<div style='text-align: right; color: #7f8c8d; font-weight: bold;'>Reviewing Scan {self.current_index + 1} of {len(self.history)}</div><hr>"
        self.mri_result.setHtml(file_counter + base_html + notes_html)
        
        self.btn_export_pdf.setEnabled(True)
        self.btn_export_img.setEnabled(True)
        self.btn_delete_scan.setVisible(True) 

        self.btn_prev.setVisible(self.current_index > 0)
        self.btn_next.setVisible(self.current_index < len(self.history) - 1)

    def show_prev(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.render_record()

    def show_next(self):
        if self.current_index < len(self.history) - 1:
            self.current_index += 1
            self.render_record()

    def export_image(self):
        if self.current_index < 0: return
        file_name = self.history[self.current_index]['file']
        base_name = os.path.splitext(file_name)[0]
        default_img_name = f"{self.patient_info['id']}_{base_name}.png"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Analysis Image", default_img_name, "PNG Files (*.png);;JPEG Files (*.jpg)")
        if file_path:
            self.mri_canvas.fig.savefig(file_path, bbox_inches='tight', dpi=300)
            QMessageBox.information(self, "Success", f"Analysis image saved successfully.")

    def export_pdf_report(self, show_msg=True):
        import os
        from datetime import datetime
        default_pdf_name = f"{self.patient_info['id']}_{self.patient_info['name'].replace(' ', '_')}_Report.pdf"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Multi-Scan PDF Report", default_pdf_name, "PDF Files (*.pdf)")
            
        if file_path:
            current_time = datetime.now().strftime("%B %d, %Y - %H:%M")
            report_id = f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.save_patient_to_db()

            all_scans_html = ""
            temp_images = [] 
            for i, record in enumerate(self.history):
                self.mri_canvas.plot_image(record['img'], record['heat'], title=record['file'])
                temp_img = tempfile.mktemp(suffix=".png")
                self.mri_canvas.fig.savefig(temp_img, bbox_inches='tight', pad_inches=0.05, dpi=200)
                img_path = temp_img.replace('\\', '/')
                temp_images.append(temp_img)

                clean_doctor_html = self.generate_base_html(record['raw_text'])
                
                if record.get('dr_notes'):
                    clean_doctor_html += f"<br><p style='font-size: 11pt; font-weight: bold; color: #8e44ad; margin-bottom: 5px;'>Dr. Notes:</p><p style='font-size: 10pt; color: #34495e;'>{record['dr_notes'].replace(chr(10), '<br>')}</p>"

                all_scans_html += f"""
                <div style="margin-bottom: 40px; page-break-inside: avoid;">
                    <div class="result-box">{clean_doctor_html}</div>
                    <div align="center" style="margin-top: 15px; margin-bottom: 20px;">
                        <img src="{img_path}" width="650">
                    </div>
                </div>
                """

            html_content = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #2c3e50; line-height: 1.5; }}
                    .header-table {{ border-bottom: 3px solid #34495e; padding-bottom: 10px; margin-bottom: 15px; }}
                    .section-title {{ color: #2980b9; font-size: 14pt; text-transform: uppercase; margin-bottom: 5px; border-bottom: 2px solid #bdc3c7; padding-bottom: 3px; font-weight: bold; }}
                    .info-table {{ width: 100%; border-collapse: collapse; font-size: 11pt; margin-bottom: 20px; }}
                    .info-table td {{ border: 1px solid #bdc3c7; padding: 8px; }}
                    .info-table .bg-gray {{ background-color: #ecf0f1; font-weight: bold; width: 15%; color: #2c3e50; }}
                    .result-box {{ border: 1px solid #bdc3c7; padding: 15px; margin-bottom: 10px; background-color: #fdfefe; }}
                    .footer-text {{ text-align: justify; font-size: 9pt; color: #7f8c8d; margin-top: 20px; font-style: italic; border-top: 1px solid #ecf0f1; padding-top: 10px; }}
                </style>
            </head>
            <body>
                <table width="100%" class="header-table">
                    <tr>
                        <td width="60%">
                            <h1 style="color: #2c3e50; margin: 0; font-size: 22pt; letter-spacing: 1px;">NEUROVISION AI</h1>
                            <span style="color: #7f8c8d; font-size: 11pt; font-weight: bold;">AUTOMATED MULTI-SCAN RADIOLOGY REPORT</span>
                        </td>
                        <td width="40%" align="right" style="font-size: 10pt; color: #34495e;">
                            <b>Report ID:</b> {report_id}<br>
                            <b>Date Generated:</b> {current_time}<br>
                            <b>Total Scans Analyzed:</b> {len(self.history)}
                        </td>
                    </tr>
                </table>
                <div class="section-title">I. PATIENT DEMOGRAPHICS</div>
                <table class="info-table">
                    <tr>
                        <td class="bg-gray">Patient ID:</td>
                        <td><b style="color:#c0392b; font-size: 12pt;">{self.patient_info.get('id', 'N/A')}</b></td>
                        <td class="bg-gray">Full Name:</td>
                        <td><b style="font-size: 12pt;">{self.patient_info.get('name', 'N/A').upper()}</b></td>
                    </tr>
                    <tr>
                        <td class="bg-gray">Age / Gender:</td>
                        <td>{self.patient_info.get('age', 'N/A')} Y / {self.patient_info.get('gender', 'N/A')}</td>
                        <td class="bg-gray">Referring Dept:</td>
                        <td>Neurology Imaging Dept.</td>
                    </tr>
                </table>
                <div class="section-title">II. CLINICAL FINDINGS & AI IMAGING</div>
                {all_scans_html}
                <table width="100%" style="margin-top: 20px; page-break-inside: avoid;">
                    <tr>
                        <td width="50%"></td>
                        <td width="50%" align="center">
                            <p style="margin: 0; font-size: 11pt; color: #34495e;"><i>Electronically Reviewed & Verified By</i></p>
                            <h2 style="margin: 15px 0; color: #2980b9; font-family: 'Times New Roman', serif; font-style: italic;">NeuroVision</h2>
                            <hr style="width: 60%; border: 0.5px solid #bdc3c7;">
                            <p style="margin: 0; font-size: 10pt; font-weight: bold; color: #2c3e50;">Signature / Stamp</p>
                        </td>
                    </tr>
                </table>
                <div class="footer-text"><b>DISCLAIMER:</b> This report was generated with AI assistance. Final diagnostic decisions must be verified by certified professionals.</div>
            </body>
            </html>
            """
            doc = QTextDocument()
            doc.setHtml(html_content)
            printer = QPrinter(QPrinter.PrinterMode.HighResolution)
            printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
            printer.setOutputFileName(file_path)
            printer.setPageMargins(QMarginsF(12, 12, 12, 12), QPageLayout.Unit.Millimeter)
            doc.print(printer)

            for t_img in temp_images:
                if os.path.exists(t_img): os.remove(t_img)
            self.render_record()
            
            if show_msg: QMessageBox.information(self, "Success", f"Full Patient Report saved at:\n{file_path}")

# ==================== TÍNH NĂNG ADD MORE SCANS ====================
    def add_more_scans_browse(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Additional MRI Files", "", "Images (*.png *.jpg *.jpeg *.dcm *.ima)")
        if files:
            self.process_additional_scans(files)

    def process_additional_scans(self, files):
        if not files: return
        self.files_to_process = files
        self.current_processing_index = 0
        self.process_next_additional_file()

    def process_next_additional_file(self):
        if self.current_processing_index < len(self.files_to_process):
            file_path = self.files_to_process[self.current_processing_index]
            self.display_stack.setCurrentIndex(1) 
            self.mri_progress.setValue(0)
            self.mri_progress.setVisible(True)
            self.mri_result.clear()
            self.loading_label.setText(f"Scanning Extra File {self.current_processing_index + 1} of {len(self.files_to_process)}...\n{os.path.basename(file_path)}")
            
            self.btn_export_pdf.setEnabled(False)
            self.btn_export_img.setEnabled(False)
            self.btn_prev.setVisible(False)
            self.btn_next.setVisible(False)
            self.btn_delete_scan.setVisible(False)
            self.btn_add_scan.setVisible(False)

            self.mri_worker = MRIPredictWorker(file_path, self.ai_model, self.device)
            self.mri_worker.progress.connect(self.update_progress)
            self.mri_worker.done.connect(self.handle_additional_mri_result)
            self.mri_worker.start()
        else:
            self.save_patient_to_db()
            self.current_index = len(self.history) - 1 # Chuyển ngay đến scan vừa thêm
            self.render_record()
            self.btn_add_scan.setVisible(True)
            QMessageBox.information(self, "Added Scans", f"Successfully added {len(self.files_to_process)} new scans to patient record.")

    def handle_additional_mri_result(self, result_text, img_with_boxes, heatmap, file_name):
        # Đặt tên file cache an toàn để không ghi đè các scan cũ
        time_stamp = datetime.now().strftime('%H%M%S')
        box_img_path = os.path.join(self.patient_folder, f"scan_extra_{time_stamp}_{self.current_processing_index}_box.jpg")
        heat_img_path = os.path.join(self.patient_folder, f"scan_extra_{time_stamp}_{self.current_processing_index}_heat.jpg")
        cv2.imwrite(box_img_path, img_with_boxes)
        cv2.imwrite(heat_img_path, heatmap)

        record = {
            "raw_text": result_text,
            "dr_notes": "",  
            "file": file_name,
            "box_img": box_img_path,
            "heat_img": heat_img_path,
            "img": img_with_boxes, 
            "heat": heatmap
        }
        self.history.append(record)
        self.current_processing_index += 1
        self.process_next_additional_file()

    # ==================== KÉO THẢ (DRAG & DROP) CHO TOÀN ỨNG DỤNG ====================
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            files = []
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if os.path.isfile(file_path) and file_path.lower().endswith(('.ima', '.dcm', '.jpg', '.jpeg', '.png')):
                    files.append(file_path)
                elif os.path.isdir(file_path):
                    for root, dirs, f in os.walk(file_path):
                        for file in f:
                            if file.lower().endswith(('.ima', '.dcm', '.jpg', '.jpeg', '.png')):
                                files.append(os.path.join(root, file))
            
            # Nếu đang đứng ở màn hình Kết quả (Index 1), kéo thả sẽ tự động thêm scan mới!
            if files and self.stack.currentIndex() == 1:
                self.process_additional_scans(files)

# ==================== BỘ LỌC SỰ KIỆN (VALIDATION LOGIC) ====================
    def eventFilter(self, source, event):
        # Nếu bác sĩ click chuột ra ngoài ô nhập liệu (FocusOut)
        if event.type() == QEvent.Type.FocusOut:
            if source == self.txt_name:
                self.validate_name()
            elif source == self.txt_age:
                self.validate_age()
        return super().eventFilter(source, event)

    # Các hàm XÓA LỖI NGAY LẬP TỨC khi người dùng bắt đầu gõ hoặc chọn
    def clear_name_error(self):
        self.txt_name.setStyleSheet("border: 1px solid #1f8a8a; border-radius: 12px; padding: 10px; font-size: 14px; background-color: #f8fbfd;")
        self.lbl_err_name.setVisible(False)

    def clear_age_error(self):
        self.txt_age.setStyleSheet("border: 1px solid #1f8a8a; border-radius: 12px; padding: 10px; font-size: 14px; background-color: #f8fbfd;")
        self.lbl_err_age.setVisible(False)

    def clear_gender_error(self):
        if self.cb_gender.currentText() != "-- Choose --":
            self.cb_gender.setStyleSheet("border: 1px solid #1f8a8a; border-radius: 12px; padding: 10px; font-size: 14px; background-color: #f8fbfd;")
            self.lbl_err_gender.setVisible(False)

    # Các hàm BÁO LỖI (Bôi đỏ)
    def validate_name(self):
        if not self.txt_name.text().strip():
            self.txt_name.setStyleSheet("border: 2px solid #e74c3c; border-radius: 12px; padding: 10px; font-size: 14px; background-color: #fce8e6;")
            self.lbl_err_name.setVisible(True)
            return False
        return True

    def validate_age(self):
        text = self.txt_age.text().strip()
        if not text or not (0 <= int(text) <= 200):
            self.txt_age.setStyleSheet("border: 2px solid #e74c3c; border-radius: 12px; padding: 10px; font-size: 14px; background-color: #fce8e6;")
            self.lbl_err_age.setVisible(True)
            return False
        return True

    def validate_gender(self):
        if self.cb_gender.currentText() == "-- Choose --":
            self.cb_gender.setStyleSheet("border: 2px solid #e74c3c; border-radius: 12px; padding: 10px; font-size: 14px; background-color: #fce8e6;")
            self.lbl_err_gender.setVisible(True)
            return False
        return True

    # Cập nhật lại nút START để dùng hàm check lỗi mới
    def start_batch_analysis(self):
        v_name = self.validate_name()
        v_age = self.validate_age()
        v_gender = self.validate_gender()
        
        if not (v_name and v_age and v_gender):
            QMessageBox.critical(self, "Validation Error", "Please fill in all required patient information correctly before starting.")
            return
            
        if self.file_list.count() == 0:
            QMessageBox.warning(self, "Missing Files", "Please add at least one MRI file to the queue.")
            return

        self.patient_info = {
            "id": self.txt_id.text(),
            "name": self.txt_name.text().strip(),
            "age": self.txt_age.text().strip(),
            "gender": self.cb_gender.currentText(),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.patient_folder = os.path.join(HISTORY_DIR, self.patient_info["id"])
        os.makedirs(self.patient_folder, exist_ok=True)

        self.files_to_process = [self.file_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.file_list.count())]
        self.history = []
        self.current_processing_index = 0
        
        self.update_patient_bar()
        self.stack.setCurrentIndex(1)
        self.process_next_file()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NeuroVisionApp()
    window.show()
    sys.exit(app.exec())
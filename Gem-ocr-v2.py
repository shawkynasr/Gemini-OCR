import sys
import os
import requests
import json
import base64
import io
import time
import textwrap
import cv2
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
import PIL.Image
import PIL.ImageOps
import google.generativeai as genai

from PySide6.QtCore import (
    Qt, QObject, QThread, Signal, Slot, QDateTime
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QPushButton, QFileDialog,
    QStatusBar, QFrame, QSizePolicy, QComboBox,
    QDialog, QCheckBox, QSpinBox, QDoubleSpinBox, QMessageBox,
    QSplitter, QScrollArea, QGroupBox
)
from PySide6.QtGui import QPixmap, QFont, QIcon

# ==========================================
# --- 1. Image Preview Dialog ---
# ==========================================
class ImagePreviewDialog(QDialog):
    def __init__(self, image_paths, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Preview")
        self.setMinimumSize(700, 600)
        self.setModal(True)
        self.setLayoutDirection(Qt.LeftToRight)
        
        self.image_paths = image_paths
        self.current_index = 0
        
        layout = QVBoxLayout(self)
        
        self.index_label = QLabel()
        self.index_label.setAlignment(Qt.AlignCenter)
        self.index_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.index_label)
        
        self.image_display = QLabel("Loading image...")
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_display.setStyleSheet("border: 1px solid rgba(128, 128, 128, 0.4); border-radius: 8px;")
        layout.addWidget(self.image_display)
        
        button_layout = QHBoxLayout()
        self.prev_button = QPushButton("◀ Previous")
        self.next_button = QPushButton("Next ▶")
        self.close_button = QPushButton("✖ Close")
        
        for btn in [self.prev_button, self.next_button, self.close_button]:
            btn.setCursor(Qt.PointingHandCursor)
            
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)
        
        self.prev_button.clicked.connect(self.show_previous_image)
        self.next_button.clicked.connect(self.show_next_image)
        self.close_button.clicked.connect(self.accept)
        
        self.show_image()

    def show_image(self):
        if not self.image_paths:
            self.image_display.setText("No images found.")
            return
        path = self.image_paths[self.current_index]
        pixmap = QPixmap(path)
        scaled_pixmap = pixmap.scaled(self.image_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_display.setPixmap(scaled_pixmap)
        
        self.index_label.setText(f"Image {self.current_index + 1} of {len(self.image_paths)}")
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < len(self.image_paths) - 1)

    def show_previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()

    def show_next_image(self):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.show_image()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.show_image()

# ==========================================
# --- 2. RequestWorker (Handles OCR Tasks) ---
# ==========================================
class RequestWorker(QObject):
    finished = Signal(str)
    error = Signal(str)
    status_update = Signal(str)
    completed_all = Signal()
    issues_report = Signal(list)

    def __init__(self, engine_type, api_url, api_key, model_name, prompt_text, image_paths, file_paths,
                 pdf_dpi=300, image_batch_size=10, delay_seconds=60, 
                 process_images_binarization=False, process_images_denoising=False,
                 process_images_resize_factor=1.0, 
                 all_pages=True, start_page=1, end_page=1,
                 num_columns=1): 
        super().__init__()
        self.engine_type = engine_type
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.prompt_text = prompt_text
        self.image_paths = image_paths if image_paths else []
        self.file_paths = file_paths if file_paths else []
        
        self.PDF_DPI = pdf_dpi
        self.IMAGE_BATCH_SIZE = image_batch_size
        self.FREE_TIER_DELAY = delay_seconds
        
        self.PROCESS_BINARIZATION = process_images_binarization
        self.PROCESS_DENOISING = process_images_denoising
        self.RESIZE_FACTOR = process_images_resize_factor
        
        self.ALL_PAGES = all_pages
        self.START_PAGE = start_page
        self.END_PAGE = end_page
        self.NUM_COLUMNS = num_columns 

        self.MAX_RETRIES = 2
        self.RETRY_DELAY = 20

    def split_image_into_columns(self, pil_img):
        if self.NUM_COLUMNS <= 1:
            return [pil_img]
            
        self.status_update.emit(f"... ✂️ Auto-slicing image into {self.NUM_COLUMNS} columns...")
        img_np = np.array(pil_img)
        
        if img_np.ndim == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
            
        h, w = gray.shape[:2]
        max_dim = max(h, w)
        scale = 1.0 if max_dim <= 1000 else 1000.0 / max_dim
        small_gray = cv2.resize(gray, (int(w * scale), int(h * scale)))
        
        _, thresh_deskew = cv2.threshold(small_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        coords = cv2.findNonZero(thresh_deskew)
        angle = 0.0
        
        if coords is not None and len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
        if -5.0 < angle < 5.0 and abs(angle) > 0.1:
            self.status_update.emit(f"... 📐 Auto-deskewing: fixing {angle:.2f} degree tilt...")
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_np = cv2.warpAffine(img_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        (h, w) = gray.shape[:2]
        start_y = int(h * 0.15)
        end_y = int(h * 0.85)
        middle_section = gray[start_y:end_y, :]
        
        _, thresh = cv2.threshold(middle_section, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
        kernel = np.ones((1, 5), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        
        proj = np.sum(dilated, axis=0)
        proj_smoothed = np.convolve(proj, np.ones(30)/30, mode='same') 
        
        expected_width = w // self.NUM_COLUMNS
        split_x_coords = []
        
        for i in range(1, self.NUM_COLUMNS):
            expected_split = i * expected_width
            window = int(w * 0.12)
            start_x = max(0, expected_split - window)
            end_x = min(w, expected_split + window)
            
            if start_x < end_x:
                local_min_idx = np.argmin(proj_smoothed[start_x:end_x])
                best_x = start_x + local_min_idx
                split_x_coords.append(best_x)
            else:
                split_x_coords.append(expected_split)
                
        if img_np.ndim == 2:
            rotated_pil = PIL.Image.fromarray(img_np, mode='L') 
        else:
            rotated_pil = PIL.Image.fromarray(img_np)
            
        slices = []
        last_x = 0
        split_x_coords.append(w)
        
        for x in split_x_coords:
            box = (last_x, 0, x, h)
            sliced_img = rotated_pil.crop(box)
            slices.append(sliced_img)
            last_x = x
            
        return slices

    def process_single_image(self, img):
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
            
        img_processed = img.copy()

        if self.RESIZE_FACTOR < 1.0:
            new_size = (int(img_processed.width * self.RESIZE_FACTOR), 
                        int(img_processed.height * self.RESIZE_FACTOR))
            img_processed = img_processed.resize(new_size, PIL.Image.Resampling.LANCZOS)

        img_np = np.array(img_processed)
        
        if img_np.ndim == 3:
             img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
             img_gray = img_np

        if self.PROCESS_BINARIZATION:
            _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img_np = img_thresh

        if self.PROCESS_DENOISING and img_np.ndim == 2:
            kernel = np.ones((2, 2), np.uint8) 
            img_np = cv2.morphologyEx(img_np, cv2.MORPH_OPEN, kernel, iterations=1)
            img_np = cv2.morphologyEx(img_np, cv2.MORPH_CLOSE, kernel, iterations=1)

        if img_np.ndim == 2:
            return PIL.Image.fromarray(img_np, mode='L') 
        else:
            return PIL.Image.fromarray(img_np)
        
    def convert_pdf_to_images(self, pdf_path):
        self.status_update.emit(f"... ⏳ Converting PDF to images at {self.PDF_DPI} DPI...")
        images = []
        doc = fitz.open(pdf_path)
        
        num_pages = len(doc)
        start_index = 0
        end_index = num_pages - 1

        if not self.ALL_PAGES:
            start_index = max(0, self.START_PAGE - 1) 
            end_index = min(num_pages - 1, self.END_PAGE - 1) 
            if start_index > end_index:
                 self.status_update.emit("Warning: Invalid page range. Ignoring.")
                 return [] 
        
        for i in range(start_index, end_index + 1): 
            page = doc[i] 
            pix = page.get_pixmap(dpi=self.PDF_DPI)
            img = PIL.Image.frombytes("RGB", [pix.width, pix.height], pix.samples) 
            
            img_processed = self.process_single_image(img) 
            slices = self.split_image_into_columns(img_processed)
            images.extend(slices)
            
        doc.close()
        return images

    def _call_gemini_api(self, model, batch, batch_idx, total_batches, prefix):
        self.status_update.emit(f"... 🚀 [{prefix}] Sending batch {batch_idx+1} of {total_batches} to Google Gemini...")
        text_response, issues = "", []
        
        custom_safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
        
        payload = [self.prompt_text] 
        payload.extend(batch) 
        
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                custom_generation_config = genai.types.GenerationConfig(max_output_tokens=8192, temperature=0.1)
                response = model.generate_content(
                    payload, 
                    safety_settings=custom_safety_settings,
                    generation_config=custom_generation_config
                )
                
                if not response.candidates:
                    text_response = f"\n[⚠️ WARNING: Batch {batch_idx+1} Failed. API Blocked.]\n"
                    issues.append(f"{prefix} Batch {batch_idx+1}: Blocked by API")
                    break
                else:
                    finish_reason = response.candidates[0].finish_reason
                    if finish_reason in [3, 8]:
                        text_response = f"\n[⚠️ WARNING: Batch {batch_idx+1} Failed. Safety filters.]\n"
                        issues.append(f"{prefix} Batch {batch_idx+1}: Safety filters")
                        break 
                    elif finish_reason == 2:
                        text_response = f"\n[⚠️ WARNING: Batch {batch_idx+1} Failed. MAX_TOKENS limit.]\n"
                        issues.append(f"{prefix} Batch {batch_idx+1}: Stopped due to MAX_TOKENS")
                        break 
                    else:
                        text_response = response.text
                        break 
                        
            except Exception as api_error:
                if attempt < self.MAX_RETRIES:
                    self.status_update.emit(f"⚠️ API Error on Batch {batch_idx+1}. Retrying...")
                    time.sleep(self.RETRY_DELAY)
                else:
                    text_response = f"\n[⚠️ WARNING: Batch {batch_idx+1} Exception - {str(api_error)}]\n"
                    issues.append(f"{prefix} Batch {batch_idx+1}: API Exception")
                    
        return text_response, issues

    def _call_paddle_api(self, batch, batch_idx, total_batches, prefix):
        self.status_update.emit(f"... 🚀 [{prefix}] Sending batch {batch_idx+1} of {total_batches} to Baidu PaddleOCR...")
        text_response, issues = "", []
        
        url = self.api_url
        headers = {
            "Authorization": f"token {self.api_key}",
            "Content-Type": "application/json"
        }
        
        proxies = {}
        if os.environ.get("http_proxy"):
            proxies = {"http": os.environ.get("http_proxy"), "https": os.environ.get("https_proxy")}

        for img_idx, img in enumerate(batch):
            self.status_update.emit(f"    -> Uploading Image {img_idx+1}/{len(batch)} to Baidu API...")
            for attempt in range(self.MAX_RETRIES + 1):
                try:
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG", quality=90)
                    img_str = base64.b64encode(buffered.getvalue()).decode("ascii")
                    
                    payload = {
                        "file": img_str,
                        "fileType": 1
                    }
                    
                    response = requests.post(url, json=payload, headers=headers, proxies=proxies, timeout=120)
                    
                    if response.status_code == 200:
                        res_json = response.json()
                        page_text = ""
                        
                        if "result" in res_json and "layoutParsingResults" in res_json["result"]:
                            for res in res_json["result"]["layoutParsingResults"]:
                                if "markdown" in res and "text" in res["markdown"]:
                                    page_text += res["markdown"]["text"] + "\n\n"
                        else:
                            page_text = json.dumps(res_json, ensure_ascii=False)
                            
                        text_response += page_text
                        break 
                    else:
                        if attempt < self.MAX_RETRIES:
                            self.status_update.emit(f"⚠️ Paddle API Error {response.status_code}. Retrying...")
                            time.sleep(self.RETRY_DELAY)
                        else:
                            text_response += f"\n[⚠️ PADDLE ERROR: HTTP {response.status_code}]\n"
                            issues.append(f"{prefix} Batch {batch_idx+1} (Img {img_idx+1}): HTTP {response.status_code}")
                            
                except Exception as api_error:
                    if attempt < self.MAX_RETRIES:
                        self.status_update.emit(f"⚠️ Paddle Network Error. Retrying...")
                        time.sleep(self.RETRY_DELAY)
                    else:
                        text_response += f"\n[⚠️ EXCEPTION: {str(api_error)}]\n"
                        issues.append(f"{prefix} Batch {batch_idx+1} (Img {img_idx+1}): Network Exception")
                        
        return text_response, issues

    @Slot()
    def run(self):
        try:
            gemini_model = None
            if self.engine_type == 0:
                genai.configure(api_key=self.api_key)
                gemini_model = genai.GenerativeModel(self.model_name)
            
            all_issues = []

            if self.image_paths: 
                self.status_update.emit(f"=== 🖼️ Processing Standalone Images ===")
                images_to_process = []
                for p in self.image_paths:
                    img = PIL.Image.open(p)
                    if img.mode not in ('RGB', 'L'): img = img.convert('RGB')
                    images_to_process.extend(self.split_image_into_columns(self.process_single_image(img)))
                
                if images_to_process:
                    image_batches = [images_to_process[i:i+self.IMAGE_BATCH_SIZE] 
                                     for i in range(0, len(images_to_process), self.IMAGE_BATCH_SIZE)]
                    total_jobs = len(image_batches)

                    for i, batch in enumerate(image_batches):
                        if self.engine_type == 0:
                            text_resp, issues = self._call_gemini_api(gemini_model, batch, i, total_jobs, "Images")
                        else:
                            text_resp, issues = self._call_paddle_api(batch, i, total_jobs, "Images")
                            
                        all_issues.extend(issues)
                        self.finished.emit(f"--- OCR Result (Images Batch {i+1}/{total_jobs}) ---\n{text_resp}")
                        
                        if i < total_jobs - 1:
                            time.sleep(self.FREE_TIER_DELAY)

            if self.file_paths:
                total_pdfs = len(self.file_paths)
                for pdf_idx, pdf_path in enumerate(self.file_paths):
                    pdf_name = os.path.basename(pdf_path)
                    self.status_update.emit(f"\n{'='*40}\n=== 📄 Starting PDF {pdf_idx+1}/{total_pdfs}: {pdf_name} ===\n{'='*40}")
                    
                    pdf_images = self.convert_pdf_to_images(pdf_path)
                    
                    if not pdf_images:
                        self.status_update.emit(f"⚠️ Warning: File {pdf_name} yielded no images. Skipping.")
                        continue
                    
                    image_batches = [pdf_images[i:i+self.IMAGE_BATCH_SIZE] 
                                     for i in range(0, len(pdf_images), self.IMAGE_BATCH_SIZE)]
                    total_jobs = len(image_batches)
                    pdf_full_text = "" 
                    
                    for i, batch in enumerate(image_batches):
                        if self.engine_type == 0:
                            text_resp, issues = self._call_gemini_api(gemini_model, batch, i, total_jobs, pdf_name)
                        else:
                            text_resp, issues = self._call_paddle_api(batch, i, total_jobs, pdf_name)
                            
                        all_issues.extend(issues)
                        pdf_full_text += text_resp + "\n\n"
                        self.finished.emit(f"--- OCR Result ({pdf_name} - Batch {i+1}/{total_jobs}) ---\n{text_resp}")
                        
                        if i < total_jobs - 1 or pdf_idx < total_pdfs - 1:
                            time.sleep(self.FREE_TIER_DELAY)

                    out_md_path = os.path.splitext(pdf_path)[0] + "_OCR.md"
                    try:
                        with open(out_md_path, "w", encoding="utf-8") as f:
                            f.write(pdf_full_text)
                        self.status_update.emit(f"✅ 💾 Auto-saved Markdown to: {out_md_path}")
                    except Exception as e:
                        self.status_update.emit(f"❌ Failed to auto-save {pdf_name}: {str(e)}")
                        all_issues.append(f"{pdf_name}: Auto-save Failed ({str(e)})")

            self.status_update.emit("✅ All OCR tasks completed.")
            self.issues_report.emit(all_issues)  
            
        except Exception as critical_error: 
            self.error.emit(f"Critical System Error: {critical_error}")
        finally: 
            self.completed_all.emit()


# ==========================================
# --- 3. Main Window UI (Adaptive Styling) ---
# ==========================================
class GeminiApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gemini & Paddle OCR Hub By (Shawky Nasr) shawkynasr@126.com")
        self.setGeometry(100, 100, 1200, 800)
        self.setLayoutDirection(Qt.LeftToRight) 
        self.apply_modern_styling()

        self.GEMINI_MODELS_LIST = [
            "models/gemini-2.5-pro", 
            "models/gemini-2.5-flash",
            "models/gemini-pro-latest"
        ]
        self.PADDLE_MODELS_LIST = ["PaddleOCR-VL-1.5"]
        self.DEFAULT_OCR_PROMPT = "Extract text exactly as it appears. Preserve paragraphs and layout. Convert tables to Markdown."
        
        self.current_image_paths = [] 
        self.current_file_paths = [] 

        # --- QSplitter Central Layout ---
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(4)
        self.setCentralWidget(self.main_splitter)

        # --- Left Panel (Settings) ---
        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setMinimumWidth(400)
        
        left_widget = QWidget()
        left_column_layout = QVBoxLayout(left_widget)
        left_column_layout.setContentsMargins(15, 15, 15, 15)
        left_column_layout.setSpacing(15)
        self.left_scroll.setWidget(left_widget)

        # --- Right Panel (Outputs) ---
        right_widget = QWidget()
        right_column_layout = QVBoxLayout(right_widget)
        right_column_layout.setContentsMargins(15, 15, 15, 15)
        
        self.main_splitter.addWidget(self.left_scroll)
        self.main_splitter.addWidget(right_widget)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 2)

        # --- 1. Engine & API Group ---
        api_group = QGroupBox("⚙️ API Configuration")
        api_layout = QVBoxLayout(api_group)

        engine_row = QHBoxLayout()
        engine_row.addWidget(QLabel("OCR Engine:"))
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["Google Gemini (VLM)", "Baidu PaddleOCR (VL)"])
        engine_row.addWidget(self.engine_combo)
        api_layout.addLayout(engine_row)

        self.api_url_label = QLabel("API URL Endpoint:")
        self.api_url_input = QLineEdit()
        self.api_url_input.setPlaceholderText("https://paddleocr.aistudio-app.com/api/v2/ocr/jobs")
        api_layout.addWidget(self.api_url_label)
        api_layout.addWidget(self.api_url_input)

        token_row = QHBoxLayout()
        self.api_label = QLabel("API Key/Token:")
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password) 
        self.test_conn_button = QPushButton("Test")
        self.test_conn_button.setObjectName("btn_test")
        token_row.addWidget(self.api_label)
        token_row.addWidget(self.api_key_input)
        token_row.addWidget(self.test_conn_button)
        api_layout.addLayout(token_row)

        proxy_row = QHBoxLayout()
        proxy_row.addWidget(QLabel("Local Proxy:"))
        self.proxy_input = QLineEdit()
        self.proxy_input.setPlaceholderText("e.g. 127.0.0.1:8888")
        proxy_row.addWidget(self.proxy_input)
        api_layout.addLayout(proxy_row)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Select Model:"))
        self.model_combo = QComboBox()
        model_row.addWidget(self.model_combo)
        api_layout.addLayout(model_row)

        left_column_layout.addWidget(api_group)

        # --- 2. Files Input Group ---
        files_group = QGroupBox("📂 Input Sources")
        files_layout = QVBoxLayout(files_group)

        img_row = QHBoxLayout()
        self.select_image_button = QPushButton("🖼️ Load Images Folder")
        self.image_status_label = QLabel("No images")
        self.view_images_button = QPushButton("👁️ View")
        self.view_images_button.setEnabled(False)
        img_row.addWidget(self.select_image_button)
        img_row.addWidget(self.image_status_label, 1)
        img_row.addWidget(self.view_images_button)
        files_layout.addLayout(img_row)
        
        self.select_file_button = QPushButton("📄 Select Multiple PDFs (Batch Mode)") 
        files_layout.addWidget(self.select_file_button)
        self.file_preview_label = QLabel("No file selected.")
        self.file_preview_label.setAlignment(Qt.AlignCenter)
        self.file_preview_label.setObjectName("file_preview")
        files_layout.addWidget(self.file_preview_label)

        left_column_layout.addWidget(files_group)

        # --- 3. Processing Options Group ---
        options_group = QGroupBox("🛠️ Processing Options")
        options_layout = QVBoxLayout(options_group)
        
        self.all_pages_checkbox = QCheckBox("✅ Process All Pages in PDF")
        self.all_pages_checkbox.setChecked(True)
        options_layout.addWidget(self.all_pages_checkbox)
        
        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("From:"))
        self.start_page_spin = QSpinBox(); self.start_page_spin.setRange(1, 9999); self.start_page_spin.setEnabled(False) 
        range_row.addWidget(self.start_page_spin)
        range_row.addWidget(QLabel("To:"))
        self.end_page_spin = QSpinBox(); self.end_page_spin.setRange(1, 9999); self.end_page_spin.setValue(10); self.end_page_spin.setEnabled(False) 
        range_row.addWidget(self.end_page_spin)
        options_layout.addLayout(range_row)

        self.binarization_checkbox = QCheckBox("✨ Advanced Binarization (Otsu)")
        self.binarization_checkbox.setChecked(True)
        options_layout.addWidget(self.binarization_checkbox)
        
        self.denoising_checkbox = QCheckBox("🧽 Morphological Noise Reduction")
        self.denoising_checkbox.setChecked(True)
        options_layout.addWidget(self.denoising_checkbox)
        
        dpi_row = QHBoxLayout()
        dpi_row.addWidget(QLabel("PDF DPI:"))
        self.dpi_spin = QSpinBox(); self.dpi_spin.setRange(150, 600); self.dpi_spin.setSingleStep(50); self.dpi_spin.setValue(300)
        dpi_row.addWidget(self.dpi_spin)
        dpi_row.addWidget(QLabel("Resize Factor:"))
        self.resize_factor_spin = QDoubleSpinBox(); self.resize_factor_spin.setRange(0.25, 1.0); self.resize_factor_spin.setSingleStep(0.05); self.resize_factor_spin.setValue(1.0)
        dpi_row.addWidget(self.resize_factor_spin)
        options_layout.addLayout(dpi_row)

        col_row = QHBoxLayout()
        col_row.addWidget(QLabel("✂️ Columns:"))
        self.columns_spin = QSpinBox(); self.columns_spin.setRange(1, 5); self.columns_spin.setValue(1)
        col_row.addWidget(self.columns_spin)
        col_row.addWidget(QLabel("Batch Size:"))
        self.image_batch_spin = QSpinBox(); self.image_batch_spin.setRange(1, 50); self.image_batch_spin.setValue(10)
        col_row.addWidget(self.image_batch_spin)
        options_layout.addLayout(col_row)

        delay_row = QHBoxLayout()
        delay_row.addWidget(QLabel("Wait Delay (Sec):"))
        self.delay_spin = QSpinBox(); self.delay_spin.setRange(0, 600); self.delay_spin.setValue(60)
        delay_row.addWidget(self.delay_spin)
        options_layout.addLayout(delay_row)

        left_column_layout.addWidget(options_group)
        
        # --- 4. Prompt & Action ---
        left_column_layout.addWidget(QLabel("💬 OCR Extraction Prompt:"))
        self.prompt_input = QTextEdit()
        self.prompt_input.setText(self.DEFAULT_OCR_PROMPT)
        self.prompt_input.setMaximumHeight(80)
        left_column_layout.addWidget(self.prompt_input)

        self.send_button = QPushButton("🚀 START OCR EXTRACTION")
        self.send_button.setEnabled(False) 
        self.send_button.setObjectName("btn_start")
        left_column_layout.addWidget(self.send_button)

        io_layout = QHBoxLayout()
        self.import_settings_button = QPushButton("📥 Import Settings")
        self.export_settings_button = QPushButton("📤 Export Settings")
        io_layout.addWidget(self.import_settings_button)
        io_layout.addWidget(self.export_settings_button)
        left_column_layout.addLayout(io_layout)
        left_column_layout.addStretch()

        # --- Right Column (Outputs) ---
        right_column_layout.addWidget(QLabel("📝 Markdown Result (Auto-saved to PDF folder):"))
        self.response_output = QTextEdit()
        self.response_output.setReadOnly(True) 
        right_column_layout.addWidget(self.response_output, 4)
        
        tools_layout = QHBoxLayout()
        self.save_button = QPushButton("💾 Manual Save As...")
        self.save_button.setEnabled(False)
        self.clear_button = QPushButton("🧹 Clear Workspace")
        tools_layout.addWidget(self.save_button)
        tools_layout.addWidget(self.clear_button)
        right_column_layout.addLayout(tools_layout)
        
        right_column_layout.addWidget(QLabel("🖥️ System Event Log:"))
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setObjectName("log_output")
        right_column_layout.addWidget(self.log_output, 1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # --- Connections ---
        self.engine_combo.currentIndexChanged.connect(self.on_engine_changed) 
        self.test_conn_button.clicked.connect(self.test_connection)
        self.select_image_button.clicked.connect(self.open_image_dialog)
        self.view_images_button.clicked.connect(self.show_image_previewer) 
        self.select_file_button.clicked.connect(self.open_file_dialog)
        self.api_key_input.textChanged.connect(self.check_inputs)
        self.prompt_input.textChanged.connect(self.check_inputs)
        self.response_output.textChanged.connect(self.check_save_button_status)
        self.send_button.clicked.connect(self.start_request)
        self.save_button.clicked.connect(self.save_results_to_file)
        self.clear_button.clicked.connect(self.clear_all_inputs)
        self.import_settings_button.clicked.connect(self.import_settings_from_file)
        self.export_settings_button.clicked.connect(self.export_settings_to_file)
        self.all_pages_checkbox.stateChanged.connect(self.toggle_page_spins) 
        
        self.on_engine_changed()
        self.check_inputs()
        self.append_to_log("OCR App Started. Flexible Adaptive UI initialized.")

    def apply_modern_styling(self):
        # Using transparent borders and avoiding hardcoded background/text colors
        # This allows the OS native Dark/Light mode to handle the palette perfectly!
        self.setStyleSheet("""
            QGroupBox { 
                font-weight: bold; 
                border: 1px solid rgba(128, 128, 128, 0.4); 
                border-radius: 6px; 
                margin-top: 12px; 
                padding-top: 15px; 
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                subcontrol-position: top left; 
                left: 10px;
                padding: 0 5px; 
            }
            QPushButton { 
                border-radius: 5px; 
                padding: 6px 12px; 
                border: 1px solid rgba(128, 128, 128, 0.4);
            }
            QPushButton:hover { 
                background-color: rgba(128, 128, 128, 0.15); 
            }
            QPushButton#btn_start { 
                background-color: #28a745; 
                color: white; 
                font-weight: bold; 
                font-size: 14px; 
                padding: 12px; 
                border: none;
                border-radius: 6px; 
            }
            QPushButton#btn_start:hover { background-color: #218838; }
            QPushButton#btn_start:disabled { background-color: rgba(128, 128, 128, 0.4); color: rgba(255, 255, 255, 0.5); }
            
            QPushButton#btn_test { 
                background-color: #17a2b8; 
                color: white; 
                border: none;
                border-radius: 4px; 
            }
            QPushButton#btn_test:hover { background-color: #138496; }
            
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { 
                border-radius: 4px; 
                padding: 5px; 
                border: 1px solid rgba(128, 128, 128, 0.4); 
            }
            QTextEdit { 
                border-radius: 4px; 
                border: 1px solid rgba(128, 128, 128, 0.4); 
            }
            QTextEdit#log_output { 
                background-color: #1e1e1e; 
                color: #0bd10b; 
                font-family: Consolas, Monaco, monospace; 
            }
            QLabel#file_preview { 
                border: 1px dashed rgba(128, 128, 128, 0.5); 
                border-radius: 6px; 
                padding: 10px; 
            }
            QSplitter::handle { 
                background-color: rgba(128, 128, 128, 0.2); 
            }
        """)

    @Slot()
    def on_engine_changed(self):
        engine_idx = self.engine_combo.currentIndex()
        self.model_combo.clear()
        if engine_idx == 0:
            self.api_url_label.hide()
            self.api_url_input.hide()
            self.api_label.setText("Google AI Key:")
            self.model_combo.addItems(self.GEMINI_MODELS_LIST)
        else:
            self.api_url_label.show()
            self.api_url_input.show()
            if not self.api_url_input.text():
                self.api_url_input.setText("https://paddleocr.aistudio-app.com/api/v2/ocr/jobs")
            self.api_label.setText("Baidu Token:")
            self.model_combo.addItems(self.PADDLE_MODELS_LIST)
            
    @Slot()
    def export_settings_to_file(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Settings", "", "JSON Files (*.json)")
        if not file_path: return
        settings = {
            "engine_idx": self.engine_combo.currentIndex(),
            "api_url": self.api_url_input.text().strip(),
            "api_key": self.api_key_input.text().strip(),
            "proxy": self.proxy_input.text().strip(),
            "model_index": self.model_combo.currentIndex(),
            "prompt": self.prompt_input.toPlainText(),
            "binarization": self.binarization_checkbox.isChecked(),
            "denoising": self.denoising_checkbox.isChecked(),
            "dpi": self.dpi_spin.value(),
            "resize_factor": self.resize_factor_spin.value(),
            "columns": self.columns_spin.value(),
            "batch_size": self.image_batch_spin.value(),
            "delay": self.delay_spin.value(),
            "all_pages": self.all_pages_checkbox.isChecked(),
            "start_page": self.start_page_spin.value(),
            "end_page": self.end_page_spin.value()
        }
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=4, ensure_ascii=False)
            self.append_to_log(f"✅ Settings exported to: {file_path}")
            self.status_bar.showMessage("✅ Settings Exported", 5000)
        except Exception as e:
            self.append_to_log(f"❌ Failed to export settings: {e}")

    @Slot()
    def import_settings_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Import Settings", "", "JSON Files (*.json)")
        if not file_path: return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "engine_idx" in data: self.engine_combo.setCurrentIndex(data["engine_idx"])
            if "api_url" in data: self.api_url_input.setText(data["api_url"])
            if "api_key" in data: self.api_key_input.setText(data["api_key"])
            if "proxy" in data: self.proxy_input.setText(data["proxy"])
            if "model_index" in data: self.model_combo.setCurrentIndex(data["model_index"])
            if "prompt" in data: self.prompt_input.setText(data["prompt"])
            if "binarization" in data: self.binarization_checkbox.setChecked(data["binarization"])
            if "denoising" in data: self.denoising_checkbox.setChecked(data["denoising"])
            if "dpi" in data: self.dpi_spin.setValue(data["dpi"])
            if "resize_factor" in data: self.resize_factor_spin.setValue(data["resize_factor"])
            if "columns" in data: self.columns_spin.setValue(data["columns"])
            if "batch_size" in data: self.image_batch_spin.setValue(data["batch_size"])
            if "delay" in data: self.delay_spin.setValue(data["delay"])
            if "all_pages" in data: 
                self.all_pages_checkbox.setChecked(data["all_pages"])
                self.toggle_page_spins(Qt.Checked if data["all_pages"] else Qt.Unchecked)
            if "start_page" in data: self.start_page_spin.setValue(data["start_page"])
            if "end_page" in data: self.end_page_spin.setValue(data["end_page"])
            
            self.append_to_log(f"✅ Settings imported from: {file_path}")
            self.status_bar.showMessage("✅ Settings Imported", 5000)
        except Exception as e:
            self.append_to_log(f"❌ Failed to import settings: {e}")

    @Slot(int)
    def toggle_page_spins(self, state):
        enabled = state != Qt.Checked 
        self.start_page_spin.setEnabled(enabled)
        self.end_page_spin.setEnabled(enabled)

    @Slot(str)
    def append_to_log(self, message):
        now = QDateTime.currentDateTime().toString("hh:mm:ss")
        self.log_output.append(f"[{now}] {message}")
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())

    @Slot()
    def save_results_to_file(self):
        text_to_save = self.response_output.toPlainText()
        if not text_to_save: return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Result As...", "", "Markdown (*.md);;Text Files (*.txt);;All (*)")
        if not file_path: return
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text_to_save)
            self.append_to_log(f"Result saved to: {file_path}")
            self.status_bar.showMessage(f"✅ Saved in: {file_path}", 5000)
        except Exception as e: 
            self.append_to_log(f"ERROR: Save failed - {e}")

    @Slot()
    def clear_all_inputs(self):
        self.api_key_input.clear()
        self.prompt_input.setText(self.DEFAULT_OCR_PROMPT)
        self.response_output.clear()
        self.current_image_paths = []
        self.current_file_paths = [] 
        self.image_status_label.setText("No images loaded")
        self.view_images_button.setEnabled(False)
        self.file_preview_label.setText("No file selected.")
        self.engine_combo.setCurrentIndex(0)
        self.binarization_checkbox.setChecked(True)
        self.denoising_checkbox.setChecked(True)
        self.dpi_spin.setValue(300)
        self.resize_factor_spin.setValue(1.0)
        self.columns_spin.setValue(1)
        self.image_batch_spin.setValue(10)
        self.delay_spin.setValue(60)
        self.proxy_input.clear() 
        self.all_pages_checkbox.setChecked(True)
        self.toggle_page_spins(Qt.Checked) 
        self.status_bar.showMessage("🧹 All inputs cleared.")
        self.log_output.clear()
        self.check_inputs()

    @Slot()
    def check_inputs(self):
        api_key_ok = bool(self.api_key_input.text().strip())
        image_ok = bool(self.current_image_paths)
        file_ok = bool(self.current_file_paths) 
        self.send_button.setEnabled(api_key_ok and (image_ok or file_ok))
        self.view_images_button.setEnabled(image_ok)

    @Slot()
    def check_save_button_status(self):
        self.save_button.setEnabled(bool(self.response_output.toPlainText().strip()))

    @Slot()
    def open_image_dialog(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.ExistingFiles) 
        dialog.setNameFilter("Image Files (*.png *.jpg *.jpeg *.webp)")
        if dialog.exec() == QDialog.Accepted:
            file_paths = list(dialog.selectedFiles())
            if file_paths:
                self.current_image_paths = file_paths
                self.image_status_label.setText(f"{len(file_paths)} images")
                self.append_to_log(f"Loaded {len(file_paths)} images.")
            else:
                self.current_image_paths = []
                self.image_status_label.setText("No images")
        self.check_inputs()

    @Slot()
    def show_image_previewer(self):
        if not self.current_image_paths: return
        dialog = ImagePreviewDialog(self.current_image_paths, self)
        dialog.exec()

    @Slot()
    def open_file_dialog(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select PDF File(s)", "", "PDF Files (*.pdf)")
        if file_paths:
            self.current_file_paths = file_paths
            if len(file_paths) == 1:
                self.file_preview_label.setText(f"File: {os.path.basename(file_paths[0])}")
            else:
                self.file_preview_label.setText(f"Selected {len(file_paths)} PDF Files.")
            self.append_to_log(f"Loaded {len(file_paths)} PDF File(s).")
            
            if len(file_paths) == 1:
                try:
                    doc = fitz.open(file_paths[0])
                    num_pages = len(doc)
                    self.start_page_spin.setRange(1, num_pages)
                    self.end_page_spin.setRange(1, num_pages)
                    self.end_page_spin.setValue(num_pages) 
                    doc.close()
                except:
                    pass
        else:
            self.current_file_paths = []
            self.file_preview_label.setText("No file selected.")
        self.check_inputs()

    @Slot()
    def test_connection(self):
        api_key = self.api_key_input.text().strip()
        if not api_key:
            self.append_to_log("❌ Error: Enter Key/Token first.")
            return

        self.append_to_log("🔍 Testing connection...")
        self.test_conn_button.setEnabled(False)
        
        proxies = {}
        if self.proxy_input.text().strip():
            p = f"http://{self.proxy_input.text().strip()}"
            proxies = {"http": p, "https": p}

        try:
            if self.engine_combo.currentIndex() == 0:
                test_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
                response = requests.get(test_url, proxies=proxies, timeout=10)
            else:
                test_url = self.api_url_input.text().strip()
                headers = {"Authorization": f"token {api_key}", "Content-Type": "application/json"}
                response = requests.post(test_url, json={"fileType": 1}, headers=headers, proxies=proxies, timeout=10)

            if response.status_code in [200, 400, 422, 500]: 
                self.append_to_log("✅ Connection Successful! Server reached.")
                self.status_bar.showMessage("✅ Success", 5000)
            elif response.status_code == 401:
                self.append_to_log("❌ Invalid Token (401 Unauthorized).")
            else:
                self.append_to_log(f"⚠️ Server returned error {response.status_code}.")
                
        except requests.exceptions.Timeout:
            self.append_to_log("❌ Connection Timeout. Check Proxy.")
        except Exception as e:
            self.append_to_log(f"❌ Connection Failed: {str(e)}")
        finally:
            self.test_conn_button.setEnabled(True)

    @Slot()
    def start_request(self):
        self.append_to_log("Starting OCR request process...")
        self.send_button.setEnabled(False)
        self.response_output.clear() 
        
        if self.proxy_input.text().strip():
            p = f"http://{self.proxy_input.text().strip()}"
            os.environ["http_proxy"] = p; os.environ["https_proxy"] = p
        else:
            os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)
        
        engine_type = self.engine_combo.currentIndex()
        api_url = self.api_url_input.text().strip()
        api_key = self.api_key_input.text()
        model = self.model_combo.currentText()
        prompt = self.prompt_input.toPlainText()
        
        if not model: 
            self.handle_error("Please select a model from the list first.")
            self.send_button.setEnabled(True)
            return

        self.thread = QThread()
        self.worker = RequestWorker(
            engine_type, api_url, api_key, model, prompt, 
            self.current_image_paths, self.current_file_paths, 
            self.dpi_spin.value(), self.image_batch_spin.value(), self.delay_spin.value(),
            self.binarization_checkbox.isChecked(), self.denoising_checkbox.isChecked(), 
            self.resize_factor_spin.value(),
            self.all_pages_checkbox.isChecked(), self.start_page_spin.value(), self.end_page_spin.value(), 
            self.columns_spin.value()
        ) 
        
        self.worker.moveToThread(self.thread)
        self.worker.status_update.connect(self.append_to_log)
        self.worker.issues_report.connect(self.show_issues_report)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.handle_partial_response)
        self.worker.error.connect(self.handle_error)
        self.worker.completed_all.connect(self.handle_all_completed)
        self.worker.completed_all.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.worker.completed_all.connect(self.worker.deleteLater)
        self.worker.error.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.thread.start()

    @Slot(list)
    def show_issues_report(self, issues):
        if not issues:
            self.status_bar.showMessage("✅ Task finished perfectly.", 5000)
        else:
            issue_text = "\n".join([f"• {issue}" for issue in issues])
            self.response_output.append(f"\n\n{'='*40}\n⚠️ TASK COMPLETED WITH WARNINGS:\n{issue_text}\n{'='*40}\n")
            QMessageBox.warning(self, "Task Completed with Issues", f"{len(issues)} issue(s) were found.\nCheck the log.")

    @Slot(str)
    def handle_partial_response(self, response_text):
        self.response_output.append(response_text + "\n" + ("-"*40) + "\n")
        self.status_bar.showMessage("✅ Partial result received...", 3000) 

    @Slot()
    def handle_all_completed(self):
        self.status_bar.showMessage("✅ All tasks completed.", 5000) 
        self.check_inputs()

    @Slot(str)
    def handle_error(self, error_message):
        self.response_output.append(f"❌ Error:\n{error_message}\n" + ("-"*40) + "\n")
        self.status_bar.showMessage(f"❌ Error", 10000) 
        self.append_to_log(f"ERROR: {error_message}")
        self.check_inputs()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GeminiApp()
    window.show()
    sys.exit(app.exec())

import sys
import os
# --- Fix for Users in China ---
# This forces the app to use the local VPN proxy tunnel
# 7890 is the default for Clash; 10809 is the default for v2ray
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
# ------------------------------
import google.generativeai as genai
import PIL.Image
import PIL.ImageOps
import pandas as pd
import io
import fitz  # (pymupdf)
import time
import textwrap
import cv2        # <== OpenCV library for image processing
import numpy as np  # <== numpy library for handling cv2 arrays

from PySide6.QtCore import (
    Qt, QObject, QThread, Signal, Slot, QDateTime
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QPushButton, QFileDialog,
    QStatusBar, QFrame, QSizePolicy, QComboBox,
    QDialog, QDialogButtonBox, QCheckBox, QSpinBox, QDoubleSpinBox
)
from PySide6.QtGui import QPixmap

# --- (1. Image Preview Dialog) ---
class ImagePreviewDialog(QDialog):
    def __init__(self, image_paths, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Preview"); self.setMinimumSize(600, 500); self.setModal(True)
        self.setLayoutDirection(Qt.LeftToRight) # Ensure LTR
        self.image_paths = image_paths; self.current_index = 0
        layout = QVBoxLayout(self); self.index_label = QLabel()
        self.index_label.setAlignment(Qt.AlignCenter); layout.addWidget(self.index_label)
        self.image_display = QLabel("Loading image..."); self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding); layout.addWidget(self.image_display)
        button_layout = QHBoxLayout(); self.prev_button = QPushButton("Previous"); self.next_button = QPushButton("Next")
        button_layout.addWidget(self.prev_button); button_layout.addWidget(self.next_button)
        self.close_button = QPushButton("Close"); button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout); self.prev_button.clicked.connect(self.show_previous_image)
        self.next_button.clicked.connect(self.show_next_image); self.close_button.clicked.connect(self.accept)
        self.show_image()
    def show_image(self):
        if not self.image_paths: self.image_display.setText("No images found."); return
        path = self.image_paths[self.current_index]; pixmap = QPixmap(path)
        scaled_pixmap = pixmap.scaled(self.image_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_display.setPixmap(scaled_pixmap)
        self.index_label.setText(f"Image {self.current_index + 1} of {len(self.image_paths)}")
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < len(self.image_paths) - 1)
    def show_previous_image(self):
        if self.current_index > 0: self.current_index -= 1; self.show_image()
    def show_next_image(self):
        if self.current_index < len(self.image_paths) - 1: self.current_index += 1; self.show_image()
    def resizeEvent(self, event): super().resizeEvent(event); self.show_image()

# --- (2. RequestWorker - Focused on OCR) ---
class RequestWorker(QObject):
    finished = Signal(str)
    error = Signal(str)
    status_update = Signal(str)
    completed_all = Signal()

    def __init__(self, api_key, model_name, prompt_text, image_paths, file_path,
                 pdf_dpi=300, image_batch_size=10, delay_seconds=60, 
                 process_images_binarization=False, process_images_denoising=False,
                 process_images_resize_factor=1.0, 
                 all_pages=True, start_page=1, end_page=1): # <== New additions
        super().__init__()
        self.api_key = api_key; self.model_name = model_name; self.prompt_text = prompt_text
        self.image_paths = image_paths if image_paths else []; self.file_path = file_path
        self.PDF_DPI = pdf_dpi; self.IMAGE_BATCH_SIZE = image_batch_size
        self.FREE_TIER_DELAY = delay_seconds
        self.PROCESS_BINARIZATION = process_images_binarization
        self.PROCESS_DENOISING = process_images_denoising
        self.RESIZE_FACTOR = process_images_resize_factor
        
        # --- New Page Range Variables ---
        self.ALL_PAGES = all_pages
        self.START_PAGE = start_page
        self.END_PAGE = end_page
        
    def process_single_image(self, img):
        """Apply pre-processing filters (Resize, Otsu Binarization, Denoising)."""
        img_processed = img.copy()

        # 1. Resize first (to speed up processing and control size)
        if self.RESIZE_FACTOR < 1.0:
            new_size = (int(img_processed.width * self.RESIZE_FACTOR), 
                        int(img_processed.height * self.RESIZE_FACTOR))
            img_processed = img_processed.resize(new_size, PIL.Image.Resampling.LANCZOS)
            self.status_update.emit(f"... Image resized by factor: {self.RESIZE_FACTOR * 100:.0f}%")

        # 2. Convert to OpenCV (numpy array)
        img_np = np.array(img_processed)
        
        # 3. Convert to Grayscale (if not already)
        if img_np.ndim == 3:
             img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
             img_gray = img_np

        # 4. Advanced Binary Thresholding (Otsu's Binarization)
        if self.PROCESS_BINARIZATION:
            _, img_thresh = cv2.threshold(img_gray, 0, 255, 
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img_np = img_thresh
            self.status_update.emit("... Applying advanced binary thresholding (Otsu) ...")
        else:
            img_np = img_gray 

        # 5. Morphological Denoising (Salt & Pepper)
        if self.PROCESS_DENOISING and img_np.ndim == 2:
            kernel = np.ones((2, 2), np.uint8) 
            
            # Opening: Removes white dots (salt noise)
            img_np = cv2.morphologyEx(img_np, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Closing: Removes black dots/fills small holes (pepper noise)
            img_np = cv2.morphologyEx(img_np, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            self.status_update.emit("... Applying morphological noise reduction ...")

        # 6. Convert back to PIL Image
        if img_np.ndim == 2:
            return PIL.Image.fromarray(img_np, mode='L') 
        else:
            return PIL.Image.fromarray(img_np)
        
    def convert_pdf_to_images(self):
        self.status_update.emit(f"... ⏳ Converting PDF to images at {self.PDF_DPI} DPI...")
        images = []; doc = fitz.open(self.file_path)
        
        # --- Apply modified page range ---
        num_pages = len(doc)
        start_index = 0  # Page 1 in PDF is index 0
        end_index = num_pages - 1

        if not self.ALL_PAGES:
            # Pages start from 1 for user, but fitz starts from 0
            start_index = max(0, self.START_PAGE - 1) 
            # (End page is inclusive, so no need to add 1 later for range limit)
            end_index = min(num_pages - 1, self.END_PAGE - 1) 
            
            if start_index > end_index:
                 self.status_update.emit(f"Warning: Invalid page range ({self.START_PAGE}-{self.END_PAGE}). PDF range will be ignored.")
                 return [] 
                 
            self.status_update.emit(f"... Processing PDF page range: from {start_index + 1} to {end_index + 1}...")
        else:
             self.status_update.emit(f"... Processing all PDF pages: {num_pages} pages.")
        
        # Iterate only over the specified range (End is inclusive, so +1)
        for i in range(start_index, end_index + 1): 
            page = doc[i] 
            pix = page.get_pixmap(dpi=self.PDF_DPI)
            img = PIL.Image.frombytes("RGB", [pix.width, pix.height], pix.samples) 
            
            img_processed = self.process_single_image(img) 
            images.append(img_processed)
            
        doc.close(); self.status_update.emit(f"... PDF converted to {len(images)} images."); return images

    @Slot()
    def run(self):
        try:
            genai.configure(api_key=self.api_key); model = genai.GenerativeModel(self.model_name)
            prompt = self.prompt_text; images_to_process = []
            
            # 1. Process Attached Images (Single or Folder)
            if self.image_paths: 
                for p in self.image_paths:
                    img = PIL.Image.open(p)
                    images_to_process.append(self.process_single_image(img)) 
            
            # 2. Process File (PDF)
            if self.file_path:
                file_ext = os.path.splitext(self.file_path)[1].lower()
                if file_ext == '.pdf':
                    pdf_images = self.convert_pdf_to_images()
                    images_to_process.extend(pdf_images)
                else:
                    self.status_update.emit(f"Warning: File type {file_ext} not supported for OCR (Must be PDF or Images). Ignoring.")

            if not images_to_process: raise Exception("No images or PDF found for OCR processing.")
            
            # 3. Create Jobs (Batching images)
            jobs = []
            image_batches = [images_to_process[i:i+self.IMAGE_BATCH_SIZE] 
                             for i in range(0, len(images_to_process), self.IMAGE_BATCH_SIZE)]
            for batch in image_batches: 
                jobs.append( {"type": "image_batch", "content": batch} )

            total_jobs = len(jobs)
            for i, job in enumerate(jobs):
                job_content = job["content"]
                self.status_update.emit(f"... 🚀 Sending image batch {i+1} of {total_jobs}...")
                
                payload = [prompt] 
                payload.extend(job_content) 
                
                response = model.generate_content(payload)
                
                if response.candidates and response.candidates[0].finish_reason == 2: raise Exception(f"Job {i+1} stopped due to Safety filters. Input or output blocked.")
                try: text_response = response.text
                except Exception as e: 
                    if response.prompt_feedback.block_reason:
                        raise Exception(f"Job {i+1} failed due to prompt blocking: {response.prompt_feedback.block_reason}.")
                    raise Exception(f"Job {i+1} returned empty response. Error: {e}")
                
                self.finished.emit(f"--- OCR Result (Batch {i+1}/{total_jobs}) ---\n{text_response}")
                
                if i < total_jobs - 1:
                    self.status_update.emit(f"... ⏳ Waiting {self.FREE_TIER_DELAY} seconds (Free Tier Limit)...")
                    time.sleep(self.FREE_TIER_DELAY)
                    
            self.status_update.emit(f"✅ All OCR tasks completed ({total_jobs}).")
        except Exception as e: self.error.emit(f"OCR Error: {e}")
        finally: self.completed_all.emit()

# --- (3. Main Window - Focused on OCR) ---
class GeminiApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gemini-OCR By (Thecataloger) manuscriptscataloger@gmail.com")
        self.setGeometry(100, 100, 1200, 800)
        self.setLayoutDirection(Qt.LeftToRight) # Set Layout to LTR

        self.VLM_MODELS_LIST = [
            "models/gemini-2.5-pro", 
            "models/gemini-2.5-flash",
            "models/gemini-pro-latest", "models/gemini-flash-latest",
        ]
        
        # Professional and General OCR Prompt
        self.DEFAULT_OCR_PROMPT = """You are an advanced Optical Character Recognition (OCR) engine. Your task is to extract the *full and accurate* text content from the attached images (book pages/documents).

**Key Instructions:**
1.  **Accuracy:** Extract the text exactly as it appears in the image.
2.  **Formatting:** Preserve paragraphs, lines, and indentation as much as possible.
3.  **Language Detection:** Detect the language automatically and extract text accordingly.
4.  **Structured Data:** If you find tables, convert them to **Markdown** table format.
5.  **Sequence:** Process multiple images (batches) sequentially without mixing content.
6.  **Clean Output:** Return **ONLY the extracted text** without any explanations, introductions, or markdown code blocks tags."""
        
        self.current_image_paths = [] 
        self.current_file_path = None
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        left_column_widget = QWidget()
        left_column_layout = QVBoxLayout(left_column_widget)
        right_column_widget = QWidget()
        right_column_layout = QVBoxLayout(right_column_widget)
        main_layout.addWidget(left_column_widget, 1) 
        main_layout.addWidget(right_column_widget, 2)

        # --- Fill Left Column (Inputs) ---
        
        api_layout = QHBoxLayout()
        api_label = QLabel("🔑 Google AI Key:")
        self.api_key_input = QLineEdit(); self.api_key_input.setPlaceholderText("Enter your key..."); self.api_key_input.setEchoMode(QLineEdit.Password) 
        self.import_key_button = QPushButton("Import..."); self.import_key_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        api_layout.addWidget(api_label); api_layout.addWidget(self.api_key_input); api_layout.addWidget(self.import_key_button) 
        left_column_layout.addLayout(api_layout)

        model_layout = QHBoxLayout()
        model_label = QLabel("🤖 Select Model:")
        self.model_combo = QComboBox(); self.model_combo.addItems(self.VLM_MODELS_LIST)
        model_layout.addWidget(model_label); model_layout.addWidget(self.model_combo)
        left_column_layout.addLayout(model_layout)

        # 1. Load Images / Folder
        image_layout = QHBoxLayout()
        self.select_image_button = QPushButton("🖼️ 1. Load Images / Image Folder")
        self.image_status_label = QLabel("No images loaded."); self.image_status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.view_images_button = QPushButton("👁️ View"); self.view_images_button.setEnabled(False)
        image_layout.addWidget(self.select_image_button); image_layout.addWidget(self.image_status_label); image_layout.addWidget(self.view_images_button)
        left_column_layout.addLayout(image_layout)
        
        # 2. Load PDF
        file_layout = QVBoxLayout()
        self.select_file_button = QPushButton("📄 2. Select Scanned PDF File")
        file_layout.addWidget(self.select_file_button)
        self.file_preview_label = QLabel("No file selected."); self.file_preview_label.setAlignment(Qt.AlignCenter)
        self.file_preview_label.setStyleSheet("QLabel { background-color: #F0F0F0; border: 1px dashed #CCC; padding: 5px; }")
        file_layout.addWidget(self.file_preview_label)
        left_column_layout.addLayout(file_layout)

        # --- PDF Page Range Options ---
        pdf_page_range_frame = QFrame(); pdf_page_range_frame.setFrameShape(QFrame.StyledPanel)
        pdf_page_range_layout = QVBoxLayout(pdf_page_range_frame)
        pdf_page_range_label = QLabel("PDF Page Range Selection:"); pdf_page_range_layout.addWidget(pdf_page_range_label)
        
        self.all_pages_checkbox = QCheckBox("✅ Process **All Pages** (Ignore Start/Stop)"); 
        self.all_pages_checkbox.setChecked(True) 
        pdf_page_range_layout.addWidget(self.all_pages_checkbox)
        
        range_layout = QHBoxLayout()
        start_page_label = QLabel("From Page:"); range_layout.addWidget(start_page_label)
        self.start_page_spin = QSpinBox(); self.start_page_spin.setRange(1, 9999); self.start_page_spin.setValue(1);
        self.start_page_spin.setEnabled(False) 
        range_layout.addWidget(self.start_page_spin)
        
        end_page_label = QLabel("To Page:"); range_layout.addWidget(end_page_label)
        self.end_page_spin = QSpinBox(); self.end_page_spin.setRange(1, 9999); self.end_page_spin.setValue(10);
        self.end_page_spin.setEnabled(False) 
        range_layout.addWidget(self.end_page_spin)
        
        pdf_page_range_layout.addLayout(range_layout)
        left_column_layout.addWidget(pdf_page_range_frame) 
        # --- End PDF Options ---

        # --- OCR Specific Options & Pre-processing ---
        options_frame = QFrame(); options_frame.setFrameShape(QFrame.StyledPanel)
        options_layout = QVBoxLayout(options_frame)
        options_label = QLabel("Image Pre-processing Options:"); options_layout.addWidget(options_label)
        
        self.binarization_checkbox = QCheckBox("✨ Advanced Binarization (Otsu) - Recommended for books"); 
        self.binarization_checkbox.setChecked(True) 
        options_layout.addWidget(self.binarization_checkbox)
        
        self.denoising_checkbox = QCheckBox("🧽 Morphological Noise Reduction (Salt & Pepper)");
        self.denoising_checkbox.setChecked(True) 
        options_layout.addWidget(self.denoising_checkbox)
        
        # DPI
        dpi_layout = QHBoxLayout()
        dpi_label = QLabel("PDF Conversion DPI:"); dpi_layout.addWidget(dpi_label)
        self.dpi_spin = QSpinBox(); self.dpi_spin.setRange(150, 600); self.dpi_spin.setValue(300); self.dpi_spin.setSingleStep(50)
        dpi_layout.addWidget(self.dpi_spin)
        options_layout.addLayout(dpi_layout)
        
        # Resize Factor
        resize_layout = QHBoxLayout()
        resize_label = QLabel("Resize Factor (Save Memory):")
        self.resize_factor_spin = QDoubleSpinBox()
        self.resize_factor_spin.setRange(0.25, 1.0)
        self.resize_factor_spin.setSingleStep(0.05)
        self.resize_factor_spin.setValue(1.0) 
        resize_layout.addWidget(resize_label); resize_layout.addWidget(self.resize_factor_spin)
        options_layout.addLayout(resize_layout)
        
        left_column_layout.addWidget(options_frame)
        
        batch_frame = QFrame(); batch_frame.setFrameShape(QFrame.StyledPanel)
        batch_layout = QVBoxLayout(batch_frame)
        batch_label = QLabel("Batch Sending Settings:"); batch_layout.addWidget(batch_label)
        
        image_batch_layout = QHBoxLayout()
        image_batch_label = QLabel("Image Batch Size (Pages):")
        self.image_batch_spin = QSpinBox(); self.image_batch_spin.setRange(1, 50); self.image_batch_spin.setValue(10) 
        image_batch_layout.addWidget(image_batch_label); image_batch_layout.addWidget(self.image_batch_spin)
        batch_layout.addLayout(image_batch_layout)
        
        delay_layout = QHBoxLayout()
        delay_label = QLabel("Wait Delay (Seconds):")
        self.delay_spin = QSpinBox(); self.delay_spin.setRange(5, 600); self.delay_spin.setValue(60)
        delay_layout.addWidget(delay_label); delay_layout.addWidget(self.delay_spin)
        batch_layout.addLayout(delay_layout)
        
        left_column_layout.addWidget(batch_frame)
        
        prompt_label = QLabel("💬 3. OCR Text Extraction Prompt:"); left_column_layout.addWidget(prompt_label)
        self.prompt_input = QTextEdit(); self.prompt_input.setText(self.DEFAULT_OCR_PROMPT) 
        self.prompt_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding); left_column_layout.addWidget(self.prompt_input)

        self.send_button = QPushButton("🚀 Send OCR Request"); self.send_button.setEnabled(False) 
        self.send_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #4CAF50; color: white;"); 
        left_column_layout.addWidget(self.send_button)

        # --- Fill Right Column (Outputs) ---
        response_label = QLabel("🤖 Model Response (Partial results appear here):"); right_column_layout.addWidget(response_label)
        self.response_output = QTextEdit(); self.response_output.setReadOnly(True) 
        self.response_output.setObjectName("responseOutputBox"); 
        right_column_layout.addWidget(self.response_output, 5)
        
        tools_layout = QHBoxLayout()
        self.save_button = QPushButton("💾 Save Result..."); self.save_button.setEnabled(False)
        tools_layout.addWidget(self.save_button)
        self.clear_button = QPushButton("🧹 Clear All"); tools_layout.addWidget(self.clear_button)
        right_column_layout.addLayout(tools_layout)
        
        log_label = QLabel("🖥️ Backend Event Log:"); right_column_layout.addWidget(log_label)
        self.log_output = QTextEdit(); self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("""
            QTextEdit {
                background-color: #000033; color: #E0E0E0;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px; border-radius: 4px;
            }
        """)
        right_column_layout.addWidget(self.log_output, 1)

        # --- 4. Status Bar and Connections ---
        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar)
        
        self.select_image_button.clicked.connect(self.open_image_dialog)
        self.view_images_button.clicked.connect(self.show_image_previewer) 
        self.select_file_button.clicked.connect(self.open_file_dialog)
        self.import_key_button.clicked.connect(self.open_key_file_dialog)
        self.api_key_input.textChanged.connect(self.check_inputs)
        self.prompt_input.textChanged.connect(self.check_inputs)
        self.response_output.textChanged.connect(self.check_save_button_status)
        self.send_button.clicked.connect(self.start_request)
        self.save_button.clicked.connect(self.save_results_to_file)
        self.clear_button.clicked.connect(self.clear_all_inputs)
        
        # --- Connect new page range options ---
        self.all_pages_checkbox.stateChanged.connect(self.toggle_page_spins) 
        
        self.check_inputs(); self.check_save_button_status()
        self.append_to_log("Gemini OCR App started.")

    @Slot(int)
    def toggle_page_spins(self, state):
        """Enable/Disable Start/Stop spins based on 'All Pages' selection."""
        enabled = state != Qt.Checked 
        self.start_page_spin.setEnabled(enabled)
        self.end_page_spin.setEnabled(enabled)

    @Slot(str)
    def append_to_log(self, message):
        now = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        self.log_output.append(f"[{now}] {message}")
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())

    @Slot()
    def save_results_to_file(self):
        text_to_save = self.response_output.toPlainText()
        if not text_to_save: return
        file_path, selected_filter = QFileDialog.getSaveFileName(self, "Save Result As...", "", "CSV (*.csv);;Text Files (*.txt);;Markdown (*.md);;All (*)")
        if not file_path: return
        try:
            if file_path.endswith('.csv'): self.save_as_csv(text_to_save, file_path)
            else: self.save_as_text(text_to_save, file_path)
            self.append_to_log(f"Result saved to: {file_path}")
        except Exception as e: self.append_to_log(f"ERROR: Save failed - {e}")

    def save_as_text(self, text, path):
        with open(path, 'w', encoding='utf-8') as f: f.write(text)
        self.status_bar.showMessage(f"✅ Saved as text in: {path}", 5000)

    def save_as_csv(self, markdown_text, path):
        try:
            lines = markdown_text.strip().split('\n'); table_lines = [line.strip() for line in lines if line.strip().startswith('|') and line.strip().endswith('|')]
            if not table_lines: raise Exception("No Markdown table found")
            data = [[c.strip() for c in line.strip('|').split('|')] for line in table_lines]
            if len(data) > 1 and all(c.replace('-', '').strip() == '' for c in data[1]): header, rows = data[0], data[2:]
            else: header, rows = data[0], data[1:]
            df = pd.DataFrame(rows, columns=header)
            df.to_csv(path, index=False, encoding='utf-8-sig'); self.status_bar.showMessage(f"✅ Saved as Table (CSV) in: {path}", 5000)
        except Exception as e:
            error_msg = f"❌ Table conversion failed: {e}. Saving as plain text."; self.status_bar.showMessage(error_msg, 7000)
            text_path = os.path.splitext(path)[0] + ".txt"; self.save_as_text(markdown_text, text_path)

    @Slot()
    def clear_all_inputs(self):
        self.api_key_input.clear(); self.prompt_input.setText(self.DEFAULT_OCR_PROMPT); self.response_output.clear()
        self.current_image_paths = []; self.image_status_label.setText("No images loaded.")
        self.view_images_button.setEnabled(False)
        self.current_file_path = None; self.file_preview_label.setText("No file selected.")
        self.model_combo.setCurrentIndex(0)
        self.binarization_checkbox.setChecked(True); self.denoising_checkbox.setChecked(True)
        self.dpi_spin.setValue(300); self.resize_factor_spin.setValue(1.0)
        self.image_batch_spin.setValue(10); self.delay_spin.setValue(60)
        
        # --- Reset Page Range Options ---
        self.all_pages_checkbox.setChecked(True)
        self.start_page_spin.setValue(1)
        self.end_page_spin.setValue(10)
        self.toggle_page_spins(Qt.Checked) 
        
        self.status_bar.showMessage("🧹 All inputs and results cleared.")
        self.append_to_log("Interface cleared."); self.log_output.clear()
        self.check_inputs()

    @Slot()
    def check_inputs(self):
        api_key_ok = bool(self.api_key_input.text().strip()); 
        image_ok = bool(self.current_image_paths); file_ok = bool(self.current_file_path)
        self.send_button.setEnabled(api_key_ok and (image_ok or file_ok))
        self.view_images_button.setEnabled(image_ok)

    @Slot()
    def check_save_button_status(self):
        response_ok = bool(self.response_output.toPlainText().strip()); self.save_button.setEnabled(response_ok)

    @Slot()
    def open_key_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select API Key File", "", "Text Files (*.txt)")
        if file_path:
            try:
                with open(file_path, "r") as f: key = f.read().strip()
                if key: self.api_key_input.setText(key); self.append_to_log("API Key loaded from file.")
            except Exception as e: self.append_to_log(f"ERROR: Failed reading key - {e}")

    @Slot()
    def open_image_dialog(self):
        # Allow selecting folder or multiple files
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.ExistingFiles) 
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        
        dialog.setNameFilter("Image Files (*.png *.jpg *.jpeg *.webp)")
        
        open_folder_button = QPushButton("Select Folder")
        h_layout = QHBoxLayout()
        h_layout.addWidget(open_folder_button)
        try:
             dialog.layout().addWidget(open_folder_button) # Try adding as widget
        except Exception:
             pass 

        folder_selected = []
        @Slot()
        def on_folder_button_clicked():
            folder_path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
            if folder_path:
                for filename in os.listdir(folder_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        folder_selected.append(os.path.join(folder_path, filename))
            dialog.done(QDialog.Accepted)
            
        open_folder_button.clicked.connect(on_folder_button_clicked)
        
        if dialog.exec() == QDialog.Accepted:
            file_paths = list(dialog.selectedFiles())
            
            if folder_selected: 
                file_paths = folder_selected
            
            if file_paths:
                self.current_image_paths = file_paths
                self.image_status_label.setText(f"Loaded {len(file_paths)} images (Single/Folder)")
                self.append_to_log(f"Loaded {len(file_paths)} images for OCR processing.")
            else:
                self.current_image_paths = []
                self.image_status_label.setText("No images loaded.")
        
        self.check_inputs()

    @Slot()
    def show_image_previewer(self):
        if not self.current_image_paths: return
        dialog = ImagePreviewDialog(self.current_image_paths, self); dialog.exec()

    @Slot()
    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Scanned PDF File", "", "PDF Files (*.pdf)")
        if file_path:
            self.current_file_path = file_path
            self.file_preview_label.setText(f"Selected File: {os.path.basename(file_path)}")
            self.append_to_log(f"PDF File Loaded: {os.path.basename(file_path)}")
        else:
            self.current_file_path = None
            self.file_preview_label.setText("No file selected.")
            
        # Try reading PDF pages to set limits
        if file_path:
            try:
                doc = fitz.open(file_path)
                num_pages = len(doc)
                self.start_page_spin.setRange(1, num_pages)
                self.end_page_spin.setRange(1, num_pages)
                self.end_page_spin.setValue(num_pages) # Set stop to max pages
                self.append_to_log(f"Detected {num_pages} pages in PDF.")
                doc.close()
            except Exception as e:
                self.append_to_log(f"Warning: Failed reading PDF pages - {e}")
                self.start_page_spin.setRange(1, 9999)
                self.end_page_spin.setRange(1, 9999)

        self.check_inputs()

    @Slot()
    def start_request(self):
        self.append_to_log("Starting OCR request process...")
        self.send_button.setEnabled(False)
        self.response_output.clear() 
        api_key = self.api_key_input.text()
        model = self.model_combo.currentText()
        prompt = self.prompt_input.toPlainText()
        image_paths = self.current_image_paths
        file_path = self.current_file_path
        
        process_images_binarization = self.binarization_checkbox.isChecked()
        process_images_denoising = self.denoising_checkbox.isChecked()
        process_images_resize_factor = self.resize_factor_spin.value()
        
        pdf_dpi = self.dpi_spin.value()
        image_batch_size = self.image_batch_spin.value()
        delay_seconds = self.delay_spin.value() 
        
        # --- Collect new PDF Page Range settings ---
        all_pages = self.all_pages_checkbox.isChecked()
        start_page = self.start_page_spin.value()
        end_page = self.end_page_spin.value()
        
        if not model: self.handle_error("Please select a model from the list first."); return
        
        # Logic check for page range
        if file_path and not all_pages and start_page > end_page:
            self.handle_error("Logic Error: Start page must be less than or equal to End page.");
            self.send_button.setEnabled(True); return

        self.thread = QThread()
        self.worker = RequestWorker(api_key, model, prompt, image_paths, file_path, 
                                    pdf_dpi, image_batch_size, delay_seconds,
                                    process_images_binarization, 
                                    process_images_denoising, 
                                    process_images_resize_factor,
                                    all_pages, start_page, end_page) 
        self.worker.moveToThread(self.thread)
        self.worker.status_update.connect(self.append_to_log)
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

    @Slot(str)
    def handle_partial_response(self, response_text):
        self.response_output.append(response_text + "\n" + ("-"*40) + "\n")
        self.status_bar.showMessage("✅ Partial result received...", 3000) 
        self.append_to_log("SUCCESS: Partial result received.")

    @Slot()
    def handle_all_completed(self):
        self.status_bar.showMessage("✅ All tasks completed successfully!", 5000) 
        self.append_to_log("SUCCESS: All tasks completed.")
        self.check_inputs()

    @Slot(str)
    def handle_error(self, error_message):
        self.response_output.append(f"❌ Error:\n\n{error_message}\n" + ("-"*40) + "\n")
        self.status_bar.showMessage(f"❌ Error: {error_message}", 10000) 
        self.append_to_log(f"ERROR: {error_message}")
        self.check_inputs()

# --- 4. Run Application ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GeminiApp()
    window.show()
    sys.exit(app.exec())

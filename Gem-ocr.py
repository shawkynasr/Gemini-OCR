import sys
import os
import google.generativeai as genai
import PIL.Image
import PIL.ImageOps
import pandas as pd
import io
import fitz  # (pymupdf)
import time
import textwrap
import cv2        # <== مكتبة OpenCV لمعالجة الصور
import numpy as np  # <== مكتبة numpy للتعامل مع مصفوفات cv2

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

# --- (1. نافذة عرض الصور المنبثقة) ---
class ImagePreviewDialog(QDialog):
    def __init__(self, image_paths, parent=None):
        super().__init__(parent)
        self.setWindowTitle("مُعرض الصور"); self.setMinimumSize(600, 500); self.setModal(True)
        self.image_paths = image_paths; self.current_index = 0
        layout = QVBoxLayout(self); self.index_label = QLabel()
        self.index_label.setAlignment(Qt.AlignCenter); layout.addWidget(self.index_label)
        self.image_display = QLabel("جاري تحميل الصورة..."); self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding); layout.addWidget(self.image_display)
        button_layout = QHBoxLayout(); self.prev_button = QPushButton("السابق"); self.next_button = QPushButton("التالي")
        button_layout.addWidget(self.prev_button); button_layout.addWidget(self.next_button)
        self.close_button = QPushButton("إغلاق"); button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout); self.prev_button.clicked.connect(self.show_previous_image)
        self.next_button.clicked.connect(self.show_next_image); self.close_button.clicked.connect(self.accept)
        self.show_image()
    def show_image(self):
        if not self.image_paths: self.image_display.setText("لا توجد صور."); return
        path = self.image_paths[self.current_index]; pixmap = QPixmap(path)
        scaled_pixmap = pixmap.scaled(self.image_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_display.setPixmap(scaled_pixmap)
        self.index_label.setText(f"صورة {self.current_index + 1} من {len(self.image_paths)}")
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < len(self.image_paths) - 1)
    def show_previous_image(self):
        if self.current_index > 0: self.current_index -= 1; self.show_image()
    def show_next_image(self):
        if self.current_index < len(self.image_paths) - 1: self.current_index += 1; self.show_image()
    def resizeEvent(self, event): super().resizeEvent(event); self.show_image()

# --- (2. العامل RequestWorker - مركز على OCR) ---
class RequestWorker(QObject):
    finished = Signal(str)
    error = Signal(str)
    status_update = Signal(str)
    completed_all = Signal()

    def __init__(self, api_key, model_name, prompt_text, image_paths, file_path,
                 pdf_dpi=300, image_batch_size=10, delay_seconds=60, 
                 process_images_binarization=False, process_images_denoising=False,
                 process_images_resize_factor=1.0, 
                 all_pages=True, start_page=1, end_page=1): # <== الإضافات الجديدة
        super().__init__()
        self.api_key = api_key; self.model_name = model_name; self.prompt_text = prompt_text
        self.image_paths = image_paths if image_paths else []; self.file_path = file_path
        self.PDF_DPI = pdf_dpi; self.IMAGE_BATCH_SIZE = image_batch_size
        self.FREE_TIER_DELAY = delay_seconds
        self.PROCESS_BINARIZATION = process_images_binarization
        self.PROCESS_DENOISING = process_images_denoising
        self.RESIZE_FACTOR = process_images_resize_factor
        
        # --- متغيرات نطاق الصفحات الجديدة ---
        self.ALL_PAGES = all_pages
        self.START_PAGE = start_page
        self.END_PAGE = end_page
        
    def process_single_image(self, img):
        """تطبيق فلاتر التنقية المسبقة (Resize, Otsu Binarization, Denoising)."""
        img_processed = img.copy()

        # 1. إعادة التحجيم أولاً (لتسريع المعالجة والتحكم بالحجم)
        if self.RESIZE_FACTOR < 1.0:
            new_size = (int(img_processed.width * self.RESIZE_FACTOR), 
                        int(img_processed.height * self.RESIZE_FACTOR))
            img_processed = img_processed.resize(new_size, PIL.Image.Resampling.LANCZOS)
            self.status_update.emit(f"... تم إعادة تحجيم الصورة بنسبة: {self.RESIZE_FACTOR * 100:.0f}%")

        # 2. التحويل إلى OpenCV (numpy array)
        img_np = np.array(img_processed)
        
        # 3. التحويل إلى الرمادي (إذا لم يكن كذلك)
        if img_np.ndim == 3:
             img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
             img_gray = img_np

        # 4. التحويل الثنائي المتقدم (Otsu's Binarization)
        if self.PROCESS_BINARIZATION:
            _, img_thresh = cv2.threshold(img_gray, 0, 255, 
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img_np = img_thresh
            self.status_update.emit("... تطبيق التحويل الثنائي المتقدم (Otsu Binarization) ...")
        else:
            img_np = img_gray 

        # 5. تنقية الملح والفلفل (Morphological Operations)
        if self.PROCESS_DENOISING and img_np.ndim == 2:
            kernel = np.ones((2, 2), np.uint8) 
            
            # فتح (Opening): لإزالة النقاط البيضاء (ضوضاء الملح)
            img_np = cv2.morphologyEx(img_np, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # إغلاق (Closing): لإزالة النقاط السوداء/ملء الفجوات الصغيرة (ضوضاء الفلفل)
            img_np = cv2.morphologyEx(img_np, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            self.status_update.emit("... تطبيق تنقية الضوضاء المورفولوجية ...")

        # 6. التحويل مرة أخرى إلى PIL Image
        if img_np.ndim == 2:
            return PIL.Image.fromarray(img_np, mode='L') 
        else:
            return PIL.Image.fromarray(img_np)
        
    def convert_pdf_to_images(self):
        self.status_update.emit(f"... ⏳ جاري تحويل PDF إلى صور بدقة {self.PDF_DPI} DPI...")
        images = []; doc = fitz.open(self.file_path)
        
        # --- تطبيق نطاق الصفحات المُعدل ---
        num_pages = len(doc)
        start_index = 0  # صفحة 1 في PDF هي index 0
        end_index = num_pages - 1

        if not self.ALL_PAGES:
            # الصفحات تبدأ من 1 للمستخدم، لكن fitz يبدأ من 0
            start_index = max(0, self.START_PAGE - 1) 
            # (الصفحة الأخيرة شاملة، لذا لا حاجة لإضافة 1 لاحقاً)
            end_index = min(num_pages - 1, self.END_PAGE - 1) 
            
            if start_index > end_index:
                 self.status_update.emit(f"تحذير: نطاق الصفحات غير صالح ({self.START_PAGE}-{self.END_PAGE}). سيتم تجاهل نطاق PDF.")
                 return [] 
                 
            self.status_update.emit(f"... معالجة نطاق صفحات PDF: من {start_index + 1} إلى {end_index + 1}...")
        else:
             self.status_update.emit(f"... معالجة كل صفحات PDF: {num_pages} صفحة.")
        
        # التكرار فقط على النطاق المحدد (النهاية شاملة لذلك + 1)
        for i in range(start_index, end_index + 1): 
            page = doc[i] 
            pix = page.get_pixmap(dpi=self.PDF_DPI)
            img = PIL.Image.frombytes("RGB", [pix.width, pix.height], pix.samples) 
            
            img_processed = self.process_single_image(img) 
            images.append(img_processed)
            
        doc.close(); self.status_update.emit(f"... تم تحويل PDF إلى {len(images)} صورة."); return images

    @Slot()
    def run(self):
        try:
            genai.configure(api_key=self.api_key); model = genai.GenerativeModel(self.model_name)
            prompt = self.prompt_text; images_to_process = []
            
            # 1. معالجة الصور المرفقة (فردية أو مجلد)
            if self.image_paths: 
                for p in self.image_paths:
                    img = PIL.Image.open(p)
                    images_to_process.append(self.process_single_image(img)) 
            
            # 2. معالجة الملف (PDF)
            if self.file_path:
                file_ext = os.path.splitext(self.file_path)[1].lower()
                if file_ext == '.pdf':
                    pdf_images = self.convert_pdf_to_images()
                    images_to_process.extend(pdf_images)
                else:
                    self.status_update.emit(f"تحذير: نوع الملف {file_ext} غير مدعوم لـ OCR (يجب أن يكون PDF أو صور). سيتم تجاهله.")

            if not images_to_process: raise Exception("لا يوجد أي صور أو ملف PDF للإرسال لعملية OCR.")
            
            # 3. إنشاء المهام (تقسيم الصور إلى دفعات)
            jobs = []
            image_batches = [images_to_process[i:i+self.IMAGE_BATCH_SIZE] 
                             for i in range(0, len(images_to_process), self.IMAGE_BATCH_SIZE)]
            for batch in image_batches: 
                jobs.append( {"type": "image_batch", "content": batch} )

            total_jobs = len(jobs)
            for i, job in enumerate(jobs):
                job_content = job["content"]
                self.status_update.emit(f"... 🚀 جاري إرسال دفعة الصور {i+1} من {total_jobs}...")
                
                payload = [prompt] 
                payload.extend(job_content) 
                
                response = model.generate_content(payload)
                
                if response.candidates and response.candidates[0].finish_reason == 2: raise Exception(f"المهمة {i+1} توقفت بسبب الأمان (Safety). المدخلات أو المخرجات تم حجبها.")
                try: text_response = response.text
                except Exception as e: 
                    if response.prompt_feedback.block_reason:
                        raise Exception(f"المهمة {i+1} فشلت بسبب حظر المطالبة: {response.prompt_feedback.block_reason}.")
                    raise Exception(f"المهمة {i+1} أعادت رداً فارغاً. خطأ: {e}")
                
                self.finished.emit(f"--- نتيجة OCR (الدفعة {i+1}/{total_jobs}) ---\n{text_response}")
                
                if i < total_jobs - 1:
                    self.status_update.emit(f"... ⏳ انتظار {self.FREE_TIER_DELAY} ثانية (حد الطبقة المجانية)...")
                    time.sleep(self.FREE_TIER_DELAY)
                    
            self.status_update.emit(f"✅ اكتملت جميع مهام OCR ({total_jobs}).")
        except Exception as e: self.error.emit(f"خطأ OCR: {e}")
        finally: self.completed_all.emit()

# --- (3. الواجهة الرئيسية (MainWindow) - مركزة على OCR) ---
class GeminiApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gemini-OCR By (Thecataloger) manuscriptscataloger@gmail.com")
        self.setGeometry(100, 100, 1200, 800) 

        self.VLM_MODELS_LIST = [
            "models/gemini-2.5-pro", 
            "models/gemini-2.5-flash",
            "models/gemini-pro-latest", "models/gemini-flash-latest",
        ]
        
        # المطالبة الاحترافية والمركزة على اللغة العربية
        self.DEFAULT_OCR_PROMPT = """أنت محرك استخراج نصوص ضوئي (OCR) متقدم ومختص باللغة العربية. مهمتك هي استخراج المحتوى النصي *الكامل والدقيق* من الصور المرفقة (صفحات كتاب/وثيقة) حصراً باللغة العربية.

**التعليمات الأساسية (التي يجب الالتزام بها):**
1.  **التركيز اللغوي:** يجب أن يكون الناتج باللغة **العربية الفصحى**، مع دقة متناهية في نقل النص.
2.  **حفظ التنسيق:** حافظ على **ترتيب الفقرات والأسطر والمسافات البادئة** (Indentation) كما هي في المصدر الأصلي قدر الإمكان.
3.  **تحدي الخط:** انتبه جيداً للتشكيل (الحركات)، والهمزات، والشدات، وتفريق الياء المقصورة (ى) عن الألف المقصورة (آ)، لضمان أعلى مستويات الدقة اللغوية.
4.  **معالجة الجداول:** إذا عثرت على أي بيانات مُنظَّمة (جداول)، فقم بتحويلها إلى جدول بصيغة **Markdown** لسهولة استيرادها.
5.  **التسلسل:** تعامل مع الصور المتعددة (الدفعات) بالتسلسل، ولا تخلط الصفحات.
6.  **الإخراج النظيف:** قم بإرجاع **النص المستخرج فقط** دون أي شروحات، أو تحليل، أو مقدمات، أو إضافات غير موجودة في الصورة."""
        
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

        # --- ملء العمود الأيسر (المدخلات) ---
        
        api_layout = QHBoxLayout()
        api_label = QLabel("🔑 Google AI Key:")
        self.api_key_input = QLineEdit(); self.api_key_input.setPlaceholderText("أدخل مفتاحك..."); self.api_key_input.setEchoMode(QLineEdit.Password) 
        self.import_key_button = QPushButton("استيراد..."); self.import_key_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        api_layout.addWidget(api_label); api_layout.addWidget(self.api_key_input); api_layout.addWidget(self.import_key_button) 
        left_column_layout.addLayout(api_layout)

        model_layout = QHBoxLayout()
        model_label = QLabel("🤖 اختر النموذج:")
        self.model_combo = QComboBox(); self.model_combo.addItems(self.VLM_MODELS_LIST)
        model_layout.addWidget(model_label); model_layout.addWidget(self.model_combo)
        left_column_layout.addLayout(model_layout)

        # 1. تحميل الصور / المجلد
        image_layout = QHBoxLayout()
        self.select_image_button = QPushButton("🖼️ 1. تحميل صور / مجلد صور")
        self.image_status_label = QLabel("لم يتم تحميل صور."); self.image_status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.view_images_button = QPushButton("👁️ عرض"); self.view_images_button.setEnabled(False)
        image_layout.addWidget(self.select_image_button); image_layout.addWidget(self.image_status_label); image_layout.addWidget(self.view_images_button)
        left_column_layout.addLayout(image_layout)
        
        # 2. تحميل PDF
        file_layout = QVBoxLayout()
        self.select_file_button = QPushButton("📄 2. اختر ملف كتاب ممسوح (PDF)")
        file_layout.addWidget(self.select_file_button)
        self.file_preview_label = QLabel("لم يتم اختيار ملف."); self.file_preview_label.setAlignment(Qt.AlignCenter)
        self.file_preview_label.setStyleSheet("QLabel { background-color: #F0F0F0; border: 1px dashed #CCC; padding: 5px; }")
        file_layout.addWidget(self.file_preview_label)
        left_column_layout.addLayout(file_layout)

        # --- خيارات نطاق صفحات PDF الجديدة ---
        pdf_page_range_frame = QFrame(); pdf_page_range_frame.setFrameShape(QFrame.StyledPanel)
        pdf_page_range_layout = QVBoxLayout(pdf_page_range_frame)
        pdf_page_range_label = QLabel("اختيار نطاق صفحات PDF:"); pdf_page_range_layout.addWidget(pdf_page_range_label)
        
        self.all_pages_checkbox = QCheckBox("✅ معالجة **كل الصفحات** (تجاهل البدء/التوقف)"); 
        self.all_pages_checkbox.setChecked(True) 
        pdf_page_range_layout.addWidget(self.all_pages_checkbox)
        
        range_layout = QHBoxLayout()
        start_page_label = QLabel("من صفحة:"); range_layout.addWidget(start_page_label)
        self.start_page_spin = QSpinBox(); self.start_page_spin.setRange(1, 9999); self.start_page_spin.setValue(1);
        self.start_page_spin.setEnabled(False) 
        range_layout.addWidget(self.start_page_spin)
        
        end_page_label = QLabel("إلى صفحة:"); range_layout.addWidget(end_page_label)
        self.end_page_spin = QSpinBox(); self.end_page_spin.setRange(1, 9999); self.end_page_spin.setValue(10);
        self.end_page_spin.setEnabled(False) 
        range_layout.addWidget(self.end_page_spin)
        
        pdf_page_range_layout.addLayout(range_layout)
        left_column_layout.addWidget(pdf_page_range_frame) # أضف الإطار إلى العمود الأيسر
        # --- نهاية خيارات نطاق صفحات PDF ---

        # --- خيارات OCR المحددة والتنقية ---
        options_frame = QFrame(); options_frame.setFrameShape(QFrame.StyledPanel)
        options_layout = QVBoxLayout(options_frame)
        options_label = QLabel("خيارات التنقية والمعالجة المسبقة (OCR Pre-processing):"); options_layout.addWidget(options_label)
        
        self.binarization_checkbox = QCheckBox("✨ تحويل ثنائي متقدم (Otsu) - موصى به للكتب"); 
        self.binarization_checkbox.setChecked(True) 
        options_layout.addWidget(self.binarization_checkbox)
        
        self.denoising_checkbox = QCheckBox("🧽 تنقية الضوضاء (الملح والفلفل) مورفولوجياً");
        self.denoising_checkbox.setChecked(True) 
        options_layout.addWidget(self.denoising_checkbox)
        
        # DPI
        dpi_layout = QHBoxLayout()
        dpi_label = QLabel("دقة تحويل PDF (DPI):"); dpi_layout.addWidget(dpi_label)
        self.dpi_spin = QSpinBox(); self.dpi_spin.setRange(150, 600); self.dpi_spin.setValue(300); self.dpi_spin.setSingleStep(50)
        dpi_layout.addWidget(self.dpi_spin)
        options_layout.addLayout(dpi_layout)
        
        # Resize Factor
        resize_layout = QHBoxLayout()
        resize_label = QLabel("نسبة إعادة التحجيم (لتوفير الذاكرة):")
        self.resize_factor_spin = QDoubleSpinBox()
        self.resize_factor_spin.setRange(0.25, 1.0)
        self.resize_factor_spin.setSingleStep(0.05)
        self.resize_factor_spin.setValue(1.0) 
        resize_layout.addWidget(resize_label); resize_layout.addWidget(self.resize_factor_spin)
        options_layout.addLayout(resize_layout)
        
        left_column_layout.addWidget(options_frame)
        
        batch_frame = QFrame(); batch_frame.setFrameShape(QFrame.StyledPanel)
        batch_layout = QVBoxLayout(batch_frame)
        batch_label = QLabel("إعدادات الإرسال:"); batch_layout.addWidget(batch_label)
        
        image_batch_layout = QHBoxLayout()
        image_batch_label = QLabel("حجم دفعة الصور (صفحات):")
        self.image_batch_spin = QSpinBox(); self.image_batch_spin.setRange(1, 50); self.image_batch_spin.setValue(10) 
        image_batch_layout.addWidget(image_batch_label); image_batch_layout.addWidget(self.image_batch_spin)
        batch_layout.addLayout(image_batch_layout)
        
        delay_layout = QHBoxLayout()
        delay_label = QLabel("مدة الانتظار (ثانية):")
        self.delay_spin = QSpinBox(); self.delay_spin.setRange(5, 600); self.delay_spin.setValue(60)
        delay_layout.addWidget(delay_label); delay_layout.addWidget(self.delay_spin)
        batch_layout.addLayout(delay_layout)
        
        left_column_layout.addWidget(batch_frame)
        
        prompt_label = QLabel("💬 3. مطالبة استخراج النصوص (OCR Prompt):"); left_column_layout.addWidget(prompt_label)
        self.prompt_input = QTextEdit(); self.prompt_input.setText(self.DEFAULT_OCR_PROMPT) 
        self.prompt_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding); left_column_layout.addWidget(self.prompt_input)

        self.send_button = QPushButton("🚀 إرسال طلب OCR"); self.send_button.setEnabled(False) 
        self.send_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #4CAF50; color: white;"); 
        left_column_layout.addWidget(self.send_button)

        # --- ملء العمود الأيمن (المخرجات) ---
        response_label = QLabel("🤖 رد النموذج (النتائج الجزئية تظهر هنا):"); right_column_layout.addWidget(response_label)
        self.response_output = QTextEdit(); self.response_output.setReadOnly(True) 
        self.response_output.setObjectName("responseOutputBox"); 
        right_column_layout.addWidget(self.response_output, 5)
        
        tools_layout = QHBoxLayout()
        self.save_button = QPushButton("💾 حفظ النتيجة..."); self.save_button.setEnabled(False)
        tools_layout.addWidget(self.save_button)
        self.clear_button = QPushButton("🧹 تفريغ الكل"); tools_layout.addWidget(self.clear_button)
        right_column_layout.addLayout(tools_layout)
        
        log_label = QLabel("🖥️ سجل الأحداث والأخطاء (Backend Log):"); right_column_layout.addWidget(log_label)
        self.log_output = QTextEdit(); self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("""
            QTextEdit {
                background-color: #000033; color: #E0E0E0;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px; border-radius: 4px;
            }
        """)
        right_column_layout.addWidget(self.log_output, 1)

        # --- 4. إعداد شريط الحالة والربط ---
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
        
        # --- ربط الإضافة الجديدة لنطاق الصفحات ---
        self.all_pages_checkbox.stateChanged.connect(self.toggle_page_spins) 
        
        self.check_inputs(); self.check_save_button_status()
        self.append_to_log("تم تشغيل تطبيق Gemini OCR المُحسن.")

    @Slot(int)
    def toggle_page_spins(self, state):
        """تمكين/تعطيل خانات البدء والتوقف بناءً على اختيار 'كل الصفحات'."""
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
        file_path, selected_filter = QFileDialog.getSaveFileName(self, "حفظ النتيجة كـ...", "", "CSV (*.csv);;Text Files (*.txt);;Markdown (*.md);;All (*)")
        if not file_path: return
        try:
            if file_path.endswith('.csv'): self.save_as_csv(text_to_save, file_path)
            else: self.save_as_text(text_to_save, file_path)
            self.append_to_log(f"تم حفظ النتيجة في: {file_path}")
        except Exception as e: self.append_to_log(f"ERROR: فشل الحفظ - {e}")

    def save_as_text(self, text, path):
        with open(path, 'w', encoding='utf-8') as f: f.write(text)
        self.status_bar.showMessage(f"✅ تم الحفظ كنص في: {path}", 5000)

    def save_as_csv(self, markdown_text, path):
        try:
            lines = markdown_text.strip().split('\n'); table_lines = [line.strip() for line in lines if line.strip().startswith('|') and line.strip().endswith('|')]
            if not table_lines: raise Exception("لا يوجد جدول Markdown")
            data = [[c.strip() for c in line.strip('|').split('|')] for line in table_lines]
            if len(data) > 1 and all(c.replace('-', '').strip() == '' for c in data[1]): header, rows = data[0], data[2:]
            else: header, rows = data[0], data[1:]
            df = pd.DataFrame(rows, columns=header)
            df.to_csv(path, index=False, encoding='utf-8-sig'); self.status_bar.showMessage(f"✅ تم الحفظ كجدول (CSV) في: {path}", 5000)
        except Exception as e:
            error_msg = f"❌ فشل تحويل الجدول: {e}. سيتم الحفظ كنص عادي."; self.status_bar.showMessage(error_msg, 7000)
            text_path = os.path.splitext(path)[0] + ".txt"; self.save_as_text(markdown_text, text_path)

    @Slot()
    def clear_all_inputs(self):
        self.api_key_input.clear(); self.prompt_input.setText(self.DEFAULT_OCR_PROMPT); self.response_output.clear()
        self.current_image_paths = []; self.image_status_label.setText("لم يتم تحميل صور.")
        self.view_images_button.setEnabled(False)
        self.current_file_path = None; self.file_preview_label.setText("لم يتم اختيار ملف.")
        self.model_combo.setCurrentIndex(0)
        self.binarization_checkbox.setChecked(True); self.denoising_checkbox.setChecked(True)
        self.dpi_spin.setValue(300); self.resize_factor_spin.setValue(1.0)
        self.image_batch_spin.setValue(10); self.delay_spin.setValue(60)
        
        # --- تهيئة إعدادات نطاق الصفحات الجديدة ---
        self.all_pages_checkbox.setChecked(True)
        self.start_page_spin.setValue(1)
        self.end_page_spin.setValue(10)
        self.toggle_page_spins(Qt.Checked) # تحديث حالة تمكين/تعطيل الخانات
        
        self.status_bar.showMessage("🧹 تم تفريغ جميع المدخلات والنتائج.")
        self.append_to_log("تم تفريغ الواجهة."); self.log_output.clear()
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
        file_path, _ = QFileDialog.getOpenFileName(self, "اختر ملف مفتاح API", "", "Text Files (*.txt)")
        if file_path:
            try:
                with open(file_path, "r") as f: key = f.read().strip()
                if key: self.api_key_input.setText(key); self.append_to_log("تم تحميل مفتاح API من ملف.")
            except Exception as e: self.append_to_log(f"ERROR: خطأ قراءة المفتاح - {e}")

    @Slot()
    def open_image_dialog(self):
        # السماح باختيار مجلد أو عدة ملفات فردية
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.ExistingFiles) 
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        
        dialog.setNameFilter("Image Files (*.png *.jpg *.jpeg *.webp)")
        
        open_folder_button = QPushButton("اختيار مجلد")
        h_layout = QHBoxLayout()
        h_layout.addWidget(open_folder_button)
        # هذا الجزء قد يختلف في طريقة إضافة الزر حسب إصدار PySide6/Qt
        try:
             dialog.layout().addWidget(open_folder_button) # محاولة إضافته كـ widget
        except Exception:
             pass # تجاهل إذا لم يتمكن من الإضافة مباشرة

        folder_selected = []
        @Slot()
        def on_folder_button_clicked():
            folder_path = QFileDialog.getExistingDirectory(self, "اختر مجلد الصور")
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
                self.image_status_label.setText(f"تم تحميل {len(file_paths)} صورة (فردية/مجلد)")
                self.append_to_log(f"تم تحميل {len(file_paths)} صورة لمعالجة OCR.")
            else:
                self.current_image_paths = []
                self.image_status_label.setText("لم يتم تحميل صور.")
        
        self.check_inputs()

    @Slot()
    def show_image_previewer(self):
        if not self.current_image_paths: return
        dialog = ImagePreviewDialog(self.current_image_paths, self); dialog.exec()

    @Slot()
    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "اختر ملف PDF للكتاب الممسوح", "", "PDF Files (*.pdf)")
        if file_path:
            self.current_file_path = file_path
            self.file_preview_label.setText(f"الملف المختار: {os.path.basename(file_path)}")
            self.append_to_log(f"تم تحميل الملف PDF: {os.path.basename(file_path)}")
        else:
            self.current_file_path = None
            self.file_preview_label.setText("لم يتم اختيار ملف.")
            
        # محاولة قراءة عدد صفحات PDF لتعيين الحد الأقصى لخانات البدء والتوقف
        if file_path:
            try:
                doc = fitz.open(file_path)
                num_pages = len(doc)
                self.start_page_spin.setRange(1, num_pages)
                self.end_page_spin.setRange(1, num_pages)
                self.end_page_spin.setValue(num_pages) # تعيين التوقف كأقصى عدد للصفحات
                self.append_to_log(f"تم اكتشاف {num_pages} صفحة في ملف PDF.")
                doc.close()
            except Exception as e:
                self.append_to_log(f"تحذير: فشل قراءة صفحات PDF - {e}")
                self.start_page_spin.setRange(1, 9999)
                self.end_page_spin.setRange(1, 9999)

        self.check_inputs()

    @Slot()
    def start_request(self):
        self.append_to_log("بدء عملية إرسال OCR...")
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
        
        # --- جمع إعدادات نطاق صفحات PDF الجديدة ---
        all_pages = self.all_pages_checkbox.isChecked()
        start_page = self.start_page_spin.value()
        end_page = self.end_page_spin.value()
        
        if not model: self.handle_error("الرجاء اختيار نموذج من القائمة أولاً."); return
        
        # فحص إضافي بسيط لنطاق الصفحات إذا لم يتم اختيار "كل الصفحات"
        if file_path and not all_pages and start_page > end_page:
            self.handle_error("خطأ منطقي: يجب أن تكون صفحة البدء أصغر من أو تساوي صفحة التوقف.");
            self.send_button.setEnabled(True); return

        self.thread = QThread()
        self.worker = RequestWorker(api_key, model, prompt, image_paths, file_path, 
                                    pdf_dpi, image_batch_size, delay_seconds,
                                    process_images_binarization, 
                                    process_images_denoising, 
                                    process_images_resize_factor,
                                    all_pages, start_page, end_page) # <== تمرير المتغيرات الجديدة
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
        self.status_bar.showMessage("✅ تم استلام نتيجة جزئية...", 3000) 
        self.append_to_log("SUCCESS: تم استلام نتيجة جزئية.")

    @Slot()
    def handle_all_completed(self):
        self.status_bar.showMessage("✅ اكتملت جميع المهام بنجاح!", 5000) 
        self.append_to_log("SUCCESS: اكتملت جميع المهام.")
        self.check_inputs()

    @Slot(str)
    def handle_error(self, error_message):
        self.response_output.append(f"❌ حدث خطأ:\n\n{error_message}\n" + ("-"*40) + "\n")
        self.status_bar.showMessage(f"❌ خطأ: {error_message}", 10000) 
        self.append_to_log(f"ERROR: {error_message}")
        self.check_inputs()

# --- 4. تشغيل التطبيق ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GeminiApp()
    window.show()
    sys.exit(app.exec())

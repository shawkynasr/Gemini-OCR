# 📄 GemOCR - Arabic OCR Tool (Powered by Gemini)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Gemini API](https://img.shields.io/badge/AI-Gemini%201.5-orange)

**GemOCR** is a robust desktop application designed to extract Arabic text from images and PDF files with exceptional accuracy. It leverages Google Gemini AI to preserve structural layouts, paragraphs, and tables during the conversion process.

## ✨ Features

* **High Accuracy:** Specialized support for complex Arabic scripts, including diacritics (Tashkeel).
* **Document Processing:** Seamlessly handles entire PDF books or specific page ranges.
* **Table Extraction:** Automatically converts tables within images into Markdown format (easily exportable to CSV/Excel).
* **Image Pre-processing:** Built-in **Otsu Binarization** algorithms to enhance the legibility of old or low-quality scans before processing.
* **Cost-Efficient:** Completely free to use via your personal Google Gemini API key.
* **Multilingual Interface:** Native support for Arabic, English, and Chinese interfaces with dynamic layout direction (RTL/LTR).

---

## 🚀 Getting Started (User Guide)

1.  **Download:** Get the latest executable (`.exe` for Windows or `.app` for Mac) from the **[Releases](https://github.com/the-cataloger/Gemini-OCR-Arabic/releases)** section.
2.  **API Key:** Obtain a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
3.  **Setup:** Launch the application and paste your API key into the designated field.
4.  **Process:** Select your images or PDF files, then click **"Run OCR Process"** to begin.

---

## 💻 Developer Installation

If you prefer to run the source code directly or contribute to the project:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/the-cataloger/Gemini-OCR-Arabic.git](https://github.com/the-cataloger/Gemini-OCR-Arabic.git)
    cd Gemini-OCR-Arabic
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application:**
    ```bash
    python Gem-ocr_v2.py
    ```

---

## ⚙️ How it Works (Technical Overview)

1.  **Image Enhancement:** The tool uses **OpenCV** to apply filters (like Binarization and Denoising) to increase the contrast between text and background.
2.  **PDF Conversion:** **PyMuPDF (fitz)** converts PDF pages into high-resolution images (DPI adjustable) for analysis.
3.  **AI Analysis:** The processed images are sent to the **Gemini 1.5** model, which performs advanced visual analysis to interpret Arabic characters and formatting.
4.  **Output:** The text is returned in real-time, maintaining the logical flow and structure of the original document.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
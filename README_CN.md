# 📄 GemOCR - 阿拉伯语 OCR 工具 (基于 Gemini AI)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Gemini API](https://img.shields.io/badge/AI-Gemini%201.5-orange)

**GemOCR** 是一款功能强大的桌面应用程序，旨在以极高的准确率从图像和 PDF 文件中提取阿拉伯语文本。它利用 Google Gemini AI 技术，在转换过程中能够完美保留结构化布局、段落以及表格。

## ✨ 功能特性

* **高准确率：** 专门针对复杂的阿拉伯语脚本（包括变音符号/Tashkeel）进行了优化支持。
* **文档处理：** 支持无缝处理整本 PDF 书籍或指定的页面范围。
* **表格提取：** 自动识别图像中的表格并将其转换为 Markdown 格式（可轻松导出为 CSV/Excel）。
* **图像预处理：** 内置 **Otsu 二值化算法**，可在处理前显著提高旧扫描件或低质量图像的清晰度。
* **成本效益高：** 通过您个人的 Google Gemini API 密钥即可免费使用。
* **多语言界面：** 原生支持中文、英文和阿拉伯语界面，并具备动态布局方向切换（RTL/LTR）功能。

---

## 🚀 快速上手 (用户指南)

1.  **下载：** 从 **[Releases](https://github.com/the-cataloger/Gemini-OCR-Arabic/releases)** 栏目下载最新的可执行文件（Windows 为 `.exe`，Mac 为 `.app`）。
2.  **API 密钥：** 访问 [Google AI Studio](https://aistudio.google.com/app/apikey) 获取免费的 API 密钥。
3.  **设置：** 启动应用程序，并将您的 API 密钥粘贴到指定输入框中。
4.  **开始处理：** 选择您的图像或 PDF 文件，然后点击 **“开始 OCR 处理”** 按钮。

---

## 💻 开发者安装

如果您希望直接运行源代码或为项目做出贡献：

1.  **克隆仓库：**
    ```bash
    git clone [https://github.com/the-cataloger/Gemini-OCR-Arabic.git](https://github.com/the-cataloger/Gemini-OCR-Arabic.git)
    cd Gemini-OCR-Arabic
    ```

2.  **安装依赖项：**
    ```bash
    pip install -r requirements.txt
    ```

3.  **运行程序：**
    ```bash
    python Gem-ocr_v2.py
    ```

---

## ⚙️ 技术原理 (核心架构)

1.  **图像增强：** 工具使用 **OpenCV** 应用滤镜（如二值化和降噪），以增加文本与背景的对比度。
2.  **PDF 转换：** 使用 **PyMuPDF (fitz)** 将 PDF 页面转换为高分辨率图像（DPI 可调）以供分析。
3.  **AI 分析：** 处理后的图像被发送至 **Gemini 1.5** 模型，该模型通过先进的视觉分析技术解读阿拉伯语字符及其格式。
4.  **结果输出：** 文本实时返回，保持原文档的逻辑流和结构。

---

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE) 开源。
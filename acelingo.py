import sys
import os
import re
import json
import numpy as np
from PySide6.QtWidgets import (QApplication, QLabel, QWidget,
    QSystemTrayIcon, QMenu, QVBoxLayout, QHBoxLayout, QPushButton,
    QMainWindow, QTextEdit, QFrame, QSizeGrip, QFileDialog,
    QStackedWidget, QListWidget, QListWidgetItem, QSplitter,
    QMenuBar, QAbstractItemView, QMessageBox, QLineEdit)
from PySide6.QtGui import (QPixmap, QPainter, QColor, QFont, QAction,
    QIcon, QPen, QCursor, QImage)
from PySide6.QtCore import (Qt, QRect, Signal, QObject, QThread, QSize,
    QStandardPaths, QTimer)

from rapidocr_onnxruntime import RapidOCR
import ctranslate2
import transformers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "en-zh-int8")
MAX_TOKENS = 512
HISTORY_ITEM_HEIGHT = 70


# ================= 1. 模型 =================
class LocalTranslator:
    def __init__(self):
        self.translator = ctranslate2.Translator(MODEL_DIR, device="cpu", compute_type="int8")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

    def _count_tokens(self, text):
        ids = self.tokenizer.encode(text, truncation=False)
        return len(ids)

    def _merge_to_segments(self, chunks):
        """将小块合并为不超 MAX_TOKENS 的段落"""
        segments = []
        current = ""
        for chunk in chunks:
            if not current:
                current = chunk
                continue
            test = current + " " + chunk
            if self._count_tokens(test) > MAX_TOKENS:
                segments.append(current)
                current = chunk
            else:
                current = test
        if current:
            segments.append(current)
        return segments

    def _force_split(self, text):
        """按 token 硬切分超长文本"""
        words = text.split()
        segments = []
        current = ""
        for word in words:
            test = (current + " " + word).strip()
            if self._count_tokens(test) > MAX_TOKENS and current:
                segments.append(current)
                current = word
            else:
                current = test
        if current:
            segments.append(current)
        return segments

    def split_segments(self, text):
        """将文本切分为不超 MAX_TOKENS 的段落"""
        text = text.strip()
        if not text:
            return []
        if self._count_tokens(text) <= MAX_TOKENS:
            return [text]

        # 第一层：按换行符切分
        lines = [l.strip() for l in text.split('\n') if l.strip()]

        # 第二层：按句子标点切分
        chunks = []
        for line in lines:
            parts = re.split(r'(?<=[.!?。！？;；,，:：])\s*', line)
            for p in parts:
                p = p.strip()
                if p:
                    chunks.append(p)
        if not chunks:
            chunks = lines

        # 第三层：合并到段落
        segments = self._merge_to_segments(chunks)

        # 第四层：兜底硬切
        final = []
        for seg in segments:
            if self._count_tokens(seg) <= MAX_TOKENS:
                final.append(seg)
            else:
                final.extend(self._force_split(seg))
        return final

    def translate(self, text):
        segments = self.split_segments(text)
        results = [self._translate_segment(seg) for seg in segments]
        return " ".join(results)

    def _translate_segment(self, text):
        ids = self.tokenizer.encode(text, truncation=False)
        # 安全截断
        if len(ids) > MAX_TOKENS:
            ids = ids[:MAX_TOKENS]
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        results = self.translator.translate_batch([tokens], beam_size=1)
        return self.tokenizer.decode(
            self.tokenizer.convert_tokens_to_ids(results[0].hypotheses[0]),
            skip_special_tokens=True)


class LocalOCR:
    def __init__(self):
        self.ocr = RapidOCR()

    def extract(self, img_array):
        result, _ = self.ocr(img_array)
        return "\n".join([line[1] for line in result]) if result else ""


# ================= 2. 后台线程 =================
class ModelLoaderThread(QThread):
    finished = Signal(object, object)
    error = Signal(str)

    def run(self):
        try:
            translator = LocalTranslator()
            ocr = LocalOCR()
            self.finished.emit(translator, ocr)
        except Exception as e:
            self.error.emit(str(e))


class TranslateThread(QThread):
    finished = Signal(str, str)
    segment_signal = Signal(str, str, bool)

    def __init__(self, translator, ocr):
        super().__init__()
        self.translator = translator
        self.ocr = ocr
        self.text = None
        self.img_array = None
        self._stopped = False

    def set_text(self, text):
        self.text = text
        self.img_array = None

    def set_image(self, img_array):
        self.img_array = img_array
        self.text = None

    def stop(self):
        self._stopped = True

    def run(self):
        self._stopped = False
        try:
            if self.img_array is not None:
                text = self.ocr.extract(self.img_array)
                if not text.strip():
                    self.finished.emit("", "未识别到文字")
                    return
                original = text
            elif self.text:
                original = self.text
            else:
                return

            segments = self.translator.split_segments(original)
            for i, seg in enumerate(segments):
                if self._stopped:
                    self.finished.emit("", "翻译已停止")
                    return
                result = self.translator._translate_segment(seg)
                is_last = (i == len(segments) - 1)
                self.segment_signal.emit(seg, result, is_last)

        except Exception as e:
            self.finished.emit("", f"翻译出错: {e}")


# ================= 3. 翻译记录数据 =================
class TranslateRecord:
    """单条翻译记录"""
    def __init__(self, mode, original_text, translated_text, image_path=None):
        self.mode = mode            # "text" or "image"
        self.original_text = original_text
        self.translated_text = translated_text
        self.image_path = image_path

    def to_dict(self):
        return {
            "mode": self.mode,
            "original_text": self.original_text,
            "translated_text": self.translated_text,
            "image_path": self.image_path or "",
        }

    @staticmethod
    def from_dict(d):
        return TranslateRecord(
            mode=d.get("mode", "text"),
            original_text=d.get("original_text", ""),
            translated_text=d.get("translated_text", ""),
            image_path=d.get("image_path") or None,
        )


# ================= 4. 历史列表项 Widget（懒渲染） =================
class HistoryItemWidget(QWidget):
    """单条历史记录的显示 Widget，仅在可见时创建"""

    clicked = Signal(int)
    delete_requested = Signal(int)

    def __init__(self, record, index, parent=None):
        super().__init__(parent)
        self.index = index
        self.setFixedHeight(HISTORY_ITEM_HEIGHT)
        self.setCursor(QCursor(Qt.PointingHandCursor))

        outer = QHBoxLayout(self)
        outer.setContentsMargins(4, 2, 4, 2)

        item = QFrame()
        item.setStyleSheet("background:#f0f0f0; border-radius:6px;")
        layout = QHBoxLayout(item)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # 图片记录显示缩略图
        if record.mode == "image" and record.image_path:
            thumb = QLabel()
            pix = QPixmap(record.image_path).scaled(
                56, 56, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            thumb.setPixmap(pix)
            thumb.setFixedSize(56, 56)
            thumb.setStyleSheet("border-radius:4px; background:white;")
            layout.addWidget(thumb)

        # 文本预览
        text_col = QVBoxLayout()
        text_col.setSpacing(4)

        orig_preview = record.original_text[:80].replace('\n', ' ')
        lbl_orig = QLabel(orig_preview)
        lbl_orig.setFont(QFont("Microsoft YaHei", 9))
        lbl_orig.setStyleSheet("color:#888;")
        lbl_orig.setMaximumHeight(20)
        text_col.addWidget(lbl_orig)

        trans_preview = record.translated_text[:80].replace('\n', ' ')
        lbl_trans = QLabel(trans_preview)
        lbl_trans.setFont(QFont("Microsoft YaHei", 10))
        lbl_trans.setStyleSheet("color:#333;")
        lbl_trans.setMaximumHeight(24)
        text_col.addWidget(lbl_trans)

        layout.addLayout(text_col, 1)
        outer.addWidget(item)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.index)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        act_delete = menu.addAction("删除此记录")
        action = menu.exec(event.globalPos())
        if action == act_delete:
            self.delete_requested.emit(self.index)


# ================= 5. 主窗口 =================
class MainWindow(QMainWindow):
    translate_requested = Signal(str, object, object)
    translate_stop_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AceLingo")
        self.setMinimumSize(760, 520)
        self.resize(900, 600)

        self.records = []  # TranslateRecord 列表
        self.current_record_index = -1

        # 持久化目录
        self.data_dir = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        os.makedirs(self.data_dir, exist_ok=True)
        self.history_file = os.path.join(self.data_dir, "history.json")

        self._build_menu_bar()
        self._build_ui()
        self._build_status_bar()

    # -- 菜单栏 --
    def _build_menu_bar(self):
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar { background: #f8f9fa; border-bottom: 1px solid #e0e0e0; }
            QMenuBar::item:selected { background: #e0e0e0; }
        """)

        menu_file = menubar.addMenu("文件")
        act_open = QAction("打开图片", self)
        act_open.triggered.connect(self.on_select_image)
        menu_file.addAction(act_open)
        menu_file.addSeparator()
        act_quit = QAction("退出", self)
        act_quit.triggered.connect(QApplication.instance().quit)
        menu_file.addAction(act_quit)

        menu_edit = menubar.addMenu("编辑")
        act_clear = QAction("清空历史", self)
        act_clear.triggered.connect(self.clear_history)
        menu_edit.addAction(act_clear)

        menu_help = menubar.addMenu("帮助")
        act_about = QAction("关于", self)
        act_about.triggered.connect(self.show_about)
        menu_help.addAction(act_about)
        act_qq = QAction("QQ 群", self)
        act_qq.triggered.connect(self.show_qq_group)
        menu_help.addAction(act_qq)

    # -- 主体 UI --
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)

        # ---- 左侧：历史列表 ----
        left_panel = QFrame()
        left_panel.setMinimumWidth(160)
        left_panel.setStyleSheet("QFrame{background:#fafafa; border-right:1px solid #e0e0e0;}")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        lbl_history = QLabel("  翻译记录")
        lbl_history.setFixedHeight(32)
        lbl_history.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        lbl_history.setStyleSheet("color:#555; background:#f0f0f0; border-bottom:1px solid #e0e0e0;")
        left_layout.addWidget(lbl_history)

        self.history_list = QListWidget()
        self.history_list.setStyleSheet("""
            QListWidget { background:#fafafa; border:none; outline:none; }
            QListWidget::item { padding:0; }
            QListWidget::item:selected { background:#e3f2fd; }
        """)
        self.history_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.history_list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.history_list.currentRowChanged.connect(self.on_history_select)
        left_layout.addWidget(self.history_list)

        splitter.addWidget(left_panel)

        # ---- 右侧：翻译区域 ----
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(10)

        # Tab 按钮
        tab_bar = QHBoxLayout()
        tab_bar.setSpacing(0)
        self.btn_text_tab = QPushButton("文本翻译")
        self.btn_img_tab = QPushButton("图片翻译")
        for btn in (self.btn_text_tab, self.btn_img_tab):
            btn.setFixedHeight(32)
            btn.setCursor(QCursor(Qt.PointingHandCursor))
            btn.setFont(QFont("Microsoft YaHei", 9))
        self.btn_text_tab.clicked.connect(lambda: self.switch_tab(0))
        self.btn_img_tab.clicked.connect(lambda: self.switch_tab(1))
        tab_bar.addWidget(self.btn_text_tab)
        tab_bar.addWidget(self.btn_img_tab)
        right_layout.addLayout(tab_bar)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_text_page())
        self.stack.addWidget(self._build_image_page())
        right_layout.addWidget(self.stack)

        # 译文区
        self.txt_result = QTextEdit()
        self.txt_result.setReadOnly(True)
        self.txt_result.setFont(QFont("Microsoft YaHei", 11))
        self.txt_result.setStyleSheet(
            "QTextEdit{background:white; border:1px solid #ddd; border-radius:8px; "
            "padding:6px 10px; color:#333; line-height:140%;}")
        self.txt_result.setPlaceholderText("翻译结果将显示在这里...")
        self.txt_result.setMinimumHeight(60)
        right_layout.addWidget(self.txt_result)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([240, 660])
        main_layout.addWidget(splitter)

        self.switch_tab(0)
        self.load_history()

    def load_history(self):
        """从磁盘加载历史记录"""
        if not os.path.exists(self.history_file):
            return
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for d in data:
                rec = TranslateRecord.from_dict(d)
                self.records.append(rec)
                item = QListWidgetItem()
                item.setSizeHint(QSize(0, HISTORY_ITEM_HEIGHT))
                self.history_list.addItem(item)
            # 懒渲染可见项
            self._render_visible_items()
        except Exception as e:
            print(f"加载历史失败: {e}")

    def save_history(self):
        """将历史记录持久化到磁盘"""
        try:
            data = [r.to_dict() for r in self.records]
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存历史失败: {e}")

    def _build_status_bar(self):
        self.lbl_status = QLabel("就绪")
        self.lbl_status.setFont(QFont("Microsoft YaHei", 8))
        self.lbl_status.setStyleSheet("color:#999; padding:2px 8px;")
        self.statusBar().addWidget(self.lbl_status)

    # -- 文本翻译页 --
    def _build_text_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        lbl = QLabel("原文")
        lbl.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        lbl.setStyleSheet("color:#555")
        layout.addWidget(lbl)

        input_wrap = QWidget()
        input_layout = QVBoxLayout(input_wrap)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(0)

        self.txt_input = QTextEdit()
        self.txt_input.setFont(QFont("Microsoft YaHei", 10))
        self.txt_input.setStyleSheet(
            "QTextEdit{background:white; border:1px solid #ddd; border-radius:8px; "
            "padding:6px 10px; color:#333;}")
        self.txt_input.setPlaceholderText("输入要翻译的英文文本...")
        input_layout.addWidget(self.txt_input)

        self.lbl_char_count = QLabel("0 字符")
        self.lbl_char_count.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_char_count.setFont(QFont("Microsoft YaHei", 8))
        self.lbl_char_count.setStyleSheet("color:#aaa; padding:2px 6px;")
        self.txt_input.textChanged.connect(self._update_char_count)
        input_layout.addWidget(self.lbl_char_count)

        layout.addWidget(input_wrap)

        btn_row_text = QHBoxLayout()

        self.lbl_loading = QLabel()
        self.lbl_loading.setFixedSize(24, 24)
        self.lbl_loading.setVisible(False)
        btn_row_text.addWidget(self.lbl_loading)

        self.btn_stop = QPushButton("停止")
        self.btn_stop.setFixedHeight(34)
        self.btn_stop.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_stop.setFont(QFont("Microsoft YaHei", 10))
        self.btn_stop.setStyleSheet("QPushButton{background:#e74c3c;color:white;border:none;border-radius:6px}QPushButton:hover{background:#c0392b}")
        self.btn_stop.clicked.connect(self.on_stop_translate)
        self.btn_stop.setVisible(False)
        btn_row_text.addWidget(self.btn_stop)

        self.btn_translate = QPushButton("翻译")
        self.btn_translate.setFixedHeight(34)
        self.btn_translate.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_translate.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.btn_translate.setStyleSheet("QPushButton{background:#00AEFF;color:white;border:none;border-radius:6px}QPushButton:hover{background:#0096d6}QPushButton:pressed{background:#0080bf}")
        self.btn_translate.clicked.connect(self.on_translate_text)
        btn_row_text.addWidget(self.btn_translate)

        self._loading_angle = 0
        self._loading_timer = QTimer()
        self._loading_timer.timeout.connect(self._rotate_loading)

        layout.addLayout(btn_row_text)
        return page

    # -- 图片翻译页 --
    def _build_image_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.img_label = QLabel("拖拽图片到此处，或点击下方按钮选择")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMinimumHeight(160)
        self.img_label.setFont(QFont("Microsoft YaHei", 10))
        self.img_label.setStyleSheet("QLabel{background:white;border:2px dashed #ccc;border-radius:8px;color:#999}")
        self.img_label.setAcceptDrops(True)
        self.img_label.dragEnterEvent = self._drag_enter
        self.img_label.dropEvent = self._drop_image
        layout.addWidget(self.img_label)

        btn_row = QHBoxLayout()
        btn_open = QPushButton("选择图片")
        btn_open.setFixedHeight(34)
        btn_open.setCursor(QCursor(Qt.PointingHandCursor))
        btn_open.setFont(QFont("Microsoft YaHei", 10))
        btn_open.setStyleSheet("QPushButton{background:#00AEFF;color:white;border:none;border-radius:6px;padding:0 20px}QPushButton:hover{background:#0096d6}")
        btn_open.clicked.connect(self.on_select_image)
        btn_row.addStretch()
        btn_row.addWidget(btn_open)

        self.btn_translate_img = QPushButton("翻译")
        self.btn_translate_img.setFixedHeight(34)
        self.btn_translate_img.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_translate_img.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.btn_translate_img.setStyleSheet("QPushButton{background:#00AEFF;color:white;border:none;border-radius:6px;padding:0 20px}QPushButton:hover{background:#0096d6}")
        self.btn_translate_img.clicked.connect(self.on_translate_image)
        btn_row.addWidget(self.btn_translate_img)

        self.lbl_loading_img = QLabel()
        self.lbl_loading_img.setFixedSize(24, 24)
        self.lbl_loading_img.setVisible(False)
        btn_row.addWidget(self.lbl_loading_img)

        self.btn_stop_img = QPushButton("停止")
        self.btn_stop_img.setFixedHeight(34)
        self.btn_stop_img.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_stop_img.setFont(QFont("Microsoft YaHei", 10))
        self.btn_stop_img.setStyleSheet("QPushButton{background:#e74c3c;color:white;border:none;border-radius:6px;padding:0 20px}QPushButton:hover{background:#c0392b}")
        self.btn_stop_img.clicked.connect(self.on_stop_translate)
        self.btn_stop_img.setVisible(False)
        btn_row.addWidget(self.btn_stop_img)

        btn_row.addStretch()
        layout.addLayout(btn_row)
        return page

    # -- Tab 切换 --
    def switch_tab(self, index):
        self.stack.setCurrentIndex(index)
        active = "QPushButton{background:#00AEFF;color:white;border:none;border-radius:6px 6px 0 0;font-weight:bold}"
        inactive = "QPushButton{background:#e0e0e0;color:#555;border:none;border-radius:6px 6px 0 0}"
        self.btn_text_tab.setStyleSheet(active if index == 0 else inactive)
        self.btn_img_tab.setStyleSheet(active if index == 1 else inactive)

    # -- 图片拖放 --
    def _drag_enter(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def _drop_image(self, event):
        urls = event.mimeData().urls()
        if urls:
            self._load_image(urls[0].toLocalFile())

    def on_select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp);;所有文件 (*)")
        if path:
            self._load_image(path)

    def _load_image(self, path):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.lbl_status.setText("图片加载失败")
            return
        scaled = pixmap.scaled(self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_label.setPixmap(scaled)
        self.current_image_path = path

    def _update_char_count(self):
        count = len(self.txt_input.toPlainText())
        self.lbl_char_count.setText(f"{count} 字符")

    def on_translate_text(self):
        text = self.txt_input.toPlainText().strip()
        if not text:
            return
        self.txt_result.clear()
        self.lbl_status.setText("翻译中...")
        self.show_loading(True)
        self.translate_requested.emit("text", text, None)

    def on_translate_image(self):
        if not hasattr(self, 'current_image_path'):
            self.lbl_status.setText("请先选择图片")
            return
        img = QImage(self.current_image_path)
        if img.isNull():
            self.lbl_status.setText("图片读取失败")
            return
        img = img.convertToFormat(QImage.Format.Format_RGB888)
        w, h = img.width(), img.height()
        bpl = img.bytesPerLine()
        ptr = img.bits()
        arr = np.frombuffer(bytes(ptr), dtype=np.uint8).reshape(h, bpl)[:, :w * 3].reshape(h, w, 3)
        self.txt_result.clear()
        self.lbl_status.setText("OCR 识别中...")
        self.show_loading(True)
        self.translate_requested.emit("image", None, arr)

    # -- 加载动画 & 停止 --
    def show_loading(self, loading):
        """切换翻译按钮为加载动画+停止按钮"""
        if loading:
            self._loading_angle = 0
            self._loading_timer.start(50)
        else:
            self._loading_timer.stop()

        # 文本翻译按钮行
        self.btn_translate.setVisible(not loading)
        self.lbl_loading.setVisible(loading)
        self.btn_stop.setVisible(loading)

        # 图片翻译按钮行
        self.btn_translate_img.setVisible(not loading)
        self.lbl_loading_img.setVisible(loading)
        self.btn_stop_img.setVisible(loading)

    def _rotate_loading(self):
        """旋转加载图标"""
        self._loading_angle = (self._loading_angle + 30) % 360
        pix = QPixmap(24, 24)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        p.translate(12, 12)
        p.rotate(self._loading_angle)
        p.setPen(QPen(QColor("#00AEFF"), 3, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(0, -8, 0, 8)
        p.end()
        self.lbl_loading.setPixmap(pix)

    def on_stop_translate(self):
        """停止翻译，发出信号"""
        self.translate_stop_requested.emit()

    # -- 历史记录 --
    def add_record(self, record):
        """添加一条翻译记录（插入顶部，最新在最上）"""
        self.records.insert(0, record)

        item = QListWidgetItem()
        item.setSizeHint(QSize(0, HISTORY_ITEM_HEIGHT))
        self.history_list.insertItem(0, item)

        self._render_item_if_visible(0)
        self.history_list.setCurrentRow(0)
        self.save_history()

    def _render_item_if_visible(self, index):
        """仅当 item 在可见区域时才创建 Widget"""
        item = self.history_list.item(index)
        if item is None:
            return
        if self.history_list.itemWidget(item):
            return
        widget = HistoryItemWidget(self.records[index], index)
        widget.clicked.connect(lambda idx: self.history_list.setCurrentRow(idx))
        widget.delete_requested.connect(self.delete_record)
        self.history_list.setItemWidget(item, widget)

    def _render_visible_items(self):
        """渲染当前可见区域的 item"""
        for i in range(self.history_list.count()):
            item = self.history_list.item(i)
            if item is None:
                continue
            rect = self.history_list.visualItemRect(item)
            # 判断是否在可见区域
            view_rect = self.history_list.viewport().rect()
            if rect.intersects(view_rect):
                self._render_item_if_visible(i)

    def on_history_select(self, row):
        """点击历史记录，显示详情"""
        if row < 0 or row >= len(self.records):
            return
        self.current_record_index = row
        rec = self.records[row]
        self.txt_result.setPlainText(rec.translated_text)
        if rec.mode == "text":
            self.switch_tab(0)
            self.txt_input.setPlainText(rec.original_text)
        else:
            self.switch_tab(1)
            if rec.image_path:
                self._load_image(rec.image_path)

    def delete_record(self, index):
        """删除单条翻译记录"""
        if 0 <= index < len(self.records):
            self.records.pop(index)
            self.history_list.takeItem(index)
            self._refresh_history_indices()
            self.save_history()

    def _refresh_history_indices(self):
        """删除记录后刷新所有 item widget 的索引"""
        for i in range(self.history_list.count()):
            item = self.history_list.item(i)
            widget = self.history_list.itemWidget(item)
            if widget:
                widget.index = i

    def clear_history(self):
        """删除全部翻译历史（带确认）"""
        if not self.records:
            return
        reply = QMessageBox.question(
            self, "确认删除",
            "确定要删除全部翻译历史吗？此操作不可撤销。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        self.records.clear()
        self.history_list.clear()
        self.txt_result.clear()
        self.txt_input.clear()
        self.lbl_status.setText("历史已清空")
        self.save_history()

    def show_about(self):
        QMessageBox.about(self, "关于 AceLingo",
            "<h3>AceLingo 本地翻译工具</h3>"
            "<p>基于 RapidOCR + CTranslate2</p>"
            "<p>支持文本翻译和图片翻译</p>"
            "<p>模型: Helsinki-NLP/opus-mt-en-zh (INT8)</p>"
            '<p><a href="https://github.com/dingtongbin/AceLingo">https://github.com/dingtongbin/AceLingo</a></p>')

    def show_qq_group(self):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("QQ 群")
        dlg.setText("QQ 群号: 1075801515")
        dlg.setFixedSize(280, 120)
        # 使文本可选中复制
        for child in dlg.findChildren(QLabel):
            child.setTextInteractionFlags(Qt.TextSelectableByMouse)
        dlg.exec()

    # 重写滚动事件，触发懒渲染
    def showEvent(self, event):
        super().showEvent(event)
        self.history_list.verticalScrollBar().valueChanged.connect(self._render_visible_items)


# ================= 6. 托盘主程序 =================
class TrayApp(QObject):
    def __init__(self, icon=None):
        super().__init__()
        self.translator_model = None
        self.ocr_model = None
        self.worker = None
        self.models_ready = False
        self._total_translated = ""
        self._current_mode = "text"
        self._current_image_path = None

        self.app = QApplication.instance()

        self.win = MainWindow()
        if icon:
            self.win.setWindowIcon(icon)
        self.win.translate_requested.connect(self.on_translate)
        self.win.translate_stop_requested.connect(self.on_stop)
        self.win.show()

        self.tray = QSystemTrayIcon(icon if icon else QIcon())
        self.tray.setToolTip("AceLingo")
        self.tray.activated.connect(self.on_tray_click)
        self.tray.setVisible(True)

        tray_menu = QMenu()
        act_show = QAction("打开主窗口", self.app)
        act_show.triggered.connect(self.show_window)
        tray_menu.addAction(act_show)
        tray_menu.addSeparator()
        act_quit = QAction("退出", self.app)
        act_quit.triggered.connect(self.app.quit)
        tray_menu.addAction(act_quit)
        self.tray.setContextMenu(tray_menu)

        self.loader = ModelLoaderThread()
        self.loader.finished.connect(self.on_models_loaded)
        self.loader.error.connect(self.on_models_error)
        self.loader.start()

        self.tray.showMessage("AceLingo", "已启动，正在后台加载模型...",
                               QSystemTrayIcon.Information, 3000)

    def on_models_loaded(self, translator, ocr):
        self.translator_model = translator
        self.ocr_model = ocr
        self.worker = TranslateThread(self.translator_model, self.ocr_model)
        self.worker.finished.connect(self.on_translate_done)
        self.worker.segment_signal.connect(self.on_segment_done)
        self.models_ready = True
        self.win.lbl_status.setText("就绪")
        self.tray.showMessage("AceLingo", "模型加载完成！", QSystemTrayIcon.Information, 2000)

    def on_models_error(self, msg):
        self.win.lbl_status.setText(f"模型加载失败: {msg}")
        self.tray.showMessage("AceLingo", f"模型加载失败: {msg}",
                               QSystemTrayIcon.Critical, 5000)

    def on_tray_click(self, reason):
        if reason == QSystemTrayIcon.Trigger:
            self.show_window()

    def show_window(self):
        self.win.show()
        self.win.raise_()
        self.win.activateWindow()

    def on_translate(self, mode, text, img_array):
        if not self.models_ready:
            self.win.txt_result.setPlainText("模型加载中，请稍候...")
            return
        self._current_mode = mode
        self._current_image_path = getattr(self.win, 'current_image_path', None)
        self.win.txt_result.clear()
        self._total_translated = ""
        self._total_original = ""
        if mode == "text":
            self.worker.set_text(text)
        else:
            self.worker.set_image(img_array)
        self.worker.start()

    def on_stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()

    def on_segment_done(self, orig_seg, trans_seg, is_last):
        self._total_translated += trans_seg
        self._total_original += (orig_seg + " ")
        self.win.txt_result.setPlainText(self._total_translated)
        sb = self.win.txt_result.verticalScrollBar()
        sb.setValue(sb.maximum())
        if is_last:
            self.win.lbl_status.setText("翻译完成")
            self.win.show_loading(False)
            record = TranslateRecord(
                mode=self._current_mode,
                original_text=self._total_original.strip(),
                translated_text=self._total_translated,
                image_path=self._current_image_path
            )
            self.win.add_record(record)

    def on_translate_done(self, original, translated):
        self.win.show_loading(False)
        if not original:
            self.win.txt_result.setPlainText(translated)
            self.win.lbl_status.setText(translated)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("AceLingo")
    app.setQuitOnLastWindowClosed(True)
    icon = QIcon(os.path.join(BASE_DIR, "logo.png"))
    app.setWindowIcon(icon)
    controller = TrayApp(icon)
    sys.exit(app.exec())

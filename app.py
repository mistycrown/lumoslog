import sys
import os
import subprocess
import time
import json
import datetime
import threading
import gc
import math
import base64
from io import BytesIO
from pathlib import Path
import requests  # For direct API testing
import win32gui
import win32process
import win32api
import win32con
import psutil

from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QTextEdit, QLabel, QSystemTrayIcon, 
                               QMenu, QMessageBox, QDialog, QSizePolicy, 
                               QLineEdit, QFormLayout, QComboBox, QCheckBox, QGroupBox,
                               QListWidget, QListWidgetItem, QScrollArea, QInputDialog)

from PySide6.QtCore import QThread, Signal, Qt, QTimer, Slot, QSettings, QEvent
from PySide6.QtGui import QIcon, QAction, QCloseEvent, QPixmap
from qt_material import apply_stylesheet
import mss
from PIL import Image, ImageChops
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# Load environment variables (fallback)
load_dotenv()

# Constants
DEFAULT_INTERVAL = 60
DEFAULT_THRESHOLD = 10
MAX_IMAGE_DIM = (1024, 1024)
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
SCREENSHOT_DIR = Path("screenshots")
SCREENSHOT_DIR.mkdir(exist_ok=True)
ICON_PATH = Path("assets/icon.png")
ICON_PATH.parent.mkdir(exist_ok=True)

# Application Settings Keys
KEY_API_KEY = "api_key"
KEY_BASE_URL = "base_url"
KEY_MODEL = "model_name"
KEY_INTERVAL = "interval"
KEY_DEBUG = "debug_mode"
KEY_MONITOR_INDEX = "monitor_index"
KEY_REPORT_API_KEY = "report_api_key"
KEY_REPORT_BASE_URL = "report_base_url"
KEY_REPORT_MODEL = "report_model"
KEY_RESOLUTION = "resolution_preset"
KEY_SAVED_PRESETS = "saved_presets"
KEY_PROVIDER_PRESET = "provider_preset"
KEY_THRESHOLD = "img_threshold"

RESOLUTION_PRESETS = {
    "低 (512px) - 节省Token": 512,
    "中 (1024px) - 默认推荐": 1024,
    "高 (1536px) - 清晰细节": 1536, 
    "超高 (2048px) - 最大细节": 2048,
    "原始分辨率 (不缩放)": 0
}

def resize_image(img, target_size):
    """Resize image based on target max dimension while keeping aspect ratio."""
    if target_size and target_size > 0:
        img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
    return img

def get_active_window_info():
    """获取当前活动窗口的标题和进程信息"""
    try:
        hwnd = win32gui.GetForegroundWindow()
        window_title = win32gui.GetWindowText(hwnd)
        
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        try:
            process = psutil.Process(pid)
            process_name = process.name()
        except:
            process_name = "Unknown"
        
        return {
            "title": window_title if window_title else "Unknown",
            "process": process_name
        }
    except Exception:
        return {
            "title": "获取失败",
            "process": "Unknown"
        }

def get_active_monitor_index(sct, last_index=1):
    """
    Determine which monitor contains the center of the active window.
    Returns the mss monitor index (1-based).
    Fallback to last_index if failed.
    """
    try:
        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return last_index

        # Use Windows API to find the monitor handle for the window
        monitor_handle = win32api.MonitorFromWindow(hwnd, win32con.MONITOR_DEFAULTTONEAREST)
        monitor_info = win32api.GetMonitorInfo(monitor_handle)
        mon_rect = monitor_info['Monitor'] # (left, top, right, bottom)
        
        # Match against mss monitors
        # mss monitors[0] is 'all', 1+ are individual
        # mss struct: {'left': x, 'top': y, 'width': w, 'height': h}
        
        # Windows API rect corresponds to mss coordinates usually
        # We look for the best overlap or exact match of top-left
        
        for i, m in enumerate(sct.monitors):
            if i == 0: continue
            
            # Simple check: does the monitor start at the same point?
            # This is usually sufficient for standard setups
            if m['left'] == mon_rect[0] and m['top'] == mon_rect[1]:
                return i
                
        # If no exact match (e.g. DPI scaling diffs), try center point
        rect = win32gui.GetWindowRect(hwnd)
        cx = (rect[0] + rect[2]) // 2
        cy = (rect[1] + rect[3]) // 2
        
        for i, m in enumerate(sct.monitors):
            if i == 0: continue
            if (m['left'] <= cx < m['left'] + m['width'] and 
                m['top'] <= cy < m['top'] + m['height']):
                return i

        return last_index
    except Exception:
        return last_index

class LogManager:
    """Helper to manage logs being sent to UI"""
    def __init__(self, signal=None):
        self.signal = signal
        self.debug_mode = False

    def info(self, msg):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] [INFO] {msg}"
        self.signal.emit(log_msg)
        print(log_msg)

    def debug(self, msg):
        if self.debug_mode:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            log_msg = f"[{timestamp}] [DEBUG] {msg}"
            self.signal.emit(log_msg)
            print(log_msg)

    def error(self, msg):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.signal.emit(f"[{timestamp}] [ERROR] <font color='red'>{msg}</font>")
        print(f"[{timestamp}] [ERROR] {msg}")

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.resize(500, 400)
        self.settings = QSettings("Lumos", "LumosLog")
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        
        # Scroll Area Setup
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)

        # API Config Group
        api_group = QGroupBox("API 配置")
        api_group = QGroupBox("API 配置")
        form_layout = QFormLayout()

        # Provider Preset
        preset_layout = QHBoxLayout()
        self.provider_combo = QComboBox()
        self.refresh_presets()
        self.provider_combo.currentIndexChanged.connect(self.on_provider_changed)
        preset_layout.addWidget(self.provider_combo)
        
        save_preset_btn = QPushButton("保存为预设")
        save_preset_btn.clicked.connect(self.save_new_preset)
        preset_layout.addWidget(save_preset_btn)
        
        form_layout.addRow("快捷预设:", preset_layout)

        self.base_url_input = QLineEdit()
        self.base_url_input.setText(self.settings.value(KEY_BASE_URL, "https://api.siliconflow.cn/v1"))
        self.base_url_input.setPlaceholderText("https://api.siliconflow.cn/v1")
        form_layout.addRow("Base URL:", self.base_url_input)

        self.api_key_input = QLineEdit()
        self.api_key_input.setText(self.settings.value(KEY_API_KEY, os.getenv("OPENAI_API_KEY", "")))
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("sk-...")
        form_layout.addRow("API Key:", self.api_key_input)

        self.model_input = QComboBox()
        self.model_input.setEditable(True)
        models = [
            "glm-4.6v-flash",
            "glm-4v-flash",
            "Qwen/Qwen2.5-VL-72B-Instruct",
        ]
        self.model_input.addItems(models)
        self.model_input.setCurrentText(self.settings.value(KEY_MODEL, "Qwen/Qwen2.5-VL-72B-Instruct"))
        form_layout.addRow("Model:", self.model_input)
        
        api_group.setLayout(form_layout)
        layout.addWidget(api_group)

        # Report Model Config
        report_group = QGroupBox("日报生成模型配置 (可选)")
        report_layout = QFormLayout()

        self.report_base_url_input = QLineEdit()
        self.report_base_url_input.setText(self.settings.value(KEY_REPORT_BASE_URL, ""))
        self.report_base_url_input.setPlaceholderText("留空则使用上方配置")
        report_layout.addRow("Report Base URL:", self.report_base_url_input)

        self.report_api_key_input = QLineEdit()
        self.report_api_key_input.setText(self.settings.value(KEY_REPORT_API_KEY, ""))
        self.report_api_key_input.setEchoMode(QLineEdit.Password)
        self.report_api_key_input.setPlaceholderText("留空则使用上方配置")
        report_layout.addRow("Report API Key:", self.report_api_key_input)

        self.report_model_input = QComboBox()
        self.report_model_input.setEditable(True)
        report_models = [
            "gemini-2.0-flash-exp",
            "gemini-1.5-flash",
            "gpt-4o",
            "claude-3-5-sonnet-20241022"
        ]
        self.report_model_input.addItems(report_models)
        self.report_model_input.setCurrentText(self.settings.value(KEY_REPORT_MODEL, "gemini-2.0-flash-exp"))
        report_layout.addRow("Report Model:", self.report_model_input)
        
        # Report Model Test Button
        report_test_btn = QPushButton("测试日报模型连接")
        report_test_btn.clicked.connect(self.test_report_connection)
        report_layout.addRow("", report_test_btn)
        
        report_group.setLayout(report_layout)
        layout.addWidget(report_group)

        # Test Button - Changed to Screenshot Test
        test_btn = QPushButton("📸 立即截图测试 - 查看AI能识别多少信息")
        test_btn.clicked.connect(self.test_screenshot_analysis)
        layout.addWidget(test_btn)
        
        self.test_result_area = QTextEdit()
        self.test_result_area.setMaximumHeight(150)
        self.test_result_area.setReadOnly(True)
        layout.addWidget(self.test_result_area)
        
        # View Full Text Button
        view_full_btn = QPushButton("查看全文")
        view_full_btn.clicked.connect(self.view_full_result)
        layout.addWidget(view_full_btn)

        # Other Settings
        other_group = QGroupBox("监控设置")
        other_layout = QFormLayout()
        
        self.interval_input = QLineEdit()
        self.interval_input.setText(str(self.settings.value(KEY_INTERVAL, DEFAULT_INTERVAL)))
        other_layout.addRow("截图间隔 (秒):", self.interval_input)
        
        # Threshold Input
        self.threshold_input = QLineEdit()
        self.threshold_input.setText(str(self.settings.value(KEY_THRESHOLD, DEFAULT_THRESHOLD)))
        self.threshold_input.setPlaceholderText("默认: 10 (越小越灵敏)")
        other_layout.addRow("变化检测阈值:", self.threshold_input)

        # Monitor Selection
        self.monitor_combo = QComboBox()
        self.monitor_combo.addItem("🔄 自动跟随活动窗口 (推荐)", -1)
        
        with mss.mss() as sct:
            for i, monitor in enumerate(sct.monitors):
                if i == 0: continue # Skip 'all in one'
                self.monitor_combo.addItem(f"Monitor {i} ({monitor['width']}x{monitor['height']})", i)
        
        current_idx = int(self.settings.value(KEY_MONITOR_INDEX, 1))
        # Find index in combobox
        idx_to_set = 0
        for i in range(self.monitor_combo.count()):
            if(int(self.monitor_combo.itemData(i)) == current_idx):
                idx_to_set = i
                break
        self.monitor_combo.setCurrentIndex(idx_to_set)
        other_layout.addRow("选择显示器:", self.monitor_combo)

        # Resolution Selection
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(list(RESOLUTION_PRESETS.keys()))
        
        current_res_val = int(self.settings.value(KEY_RESOLUTION, 1024))
        # Find preset matching value
        res_text = "中 (1024px) - 默认推荐" # default
        for k, v in RESOLUTION_PRESETS.items():
            if v == current_res_val:
                res_text = k
                break
        self.resolution_combo.setCurrentText(res_text)
        other_layout.addRow("截图清晰度:", self.resolution_combo)
        
        other_group.setLayout(other_layout)
        layout.addWidget(other_group)

        # Buttons
        btn_box = QHBoxLayout()
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self.save_settings)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        
        btn_box.addWidget(save_btn)
        btn_box.addWidget(cancel_btn)
        layout.addLayout(btn_box)
        
        layout.addLayout(btn_box)
        
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

    def refresh_presets(self):
        self.provider_combo.blockSignals(True)
        self.provider_combo.clear()
        
        # Default Presets (Simplified)
        self.provider_combo.addItem("自定义 / 其他", "custom")
        self.provider_combo.addItem("SiliconFlow (硅基流动)", "silicon")
        self.provider_combo.addItem("Zhipu AI (智谱)", "zhipu")
        
        # User Saved Presets
        saved_presets = self.settings.value(KEY_SAVED_PRESETS, {})
        for name, data in saved_presets.items():
            self.provider_combo.addItem(f"⭐ {name}", data)
            
        self.provider_combo.blockSignals(False)

    def save_new_preset(self):
        name, ok = QInputDialog.getText(self, "保存预设", "请输入预设名称:")
        if ok and name:
            data = {
                "base_url": self.base_url_input.text(),
                "api_key": self.api_key_input.text(),
                "model": self.model_input.currentText()
            }
            saved_presets = self.settings.value(KEY_SAVED_PRESETS, {})
            saved_presets[name] = data
            self.settings.setValue(KEY_SAVED_PRESETS, saved_presets)
            
            QMessageBox.information(self, "成功", f"预设 '{name}' 已保存!")
            self.refresh_presets()
            # Select the new one
            idx = self.provider_combo.findText(f"⭐ {name}")
            if idx != -1:
                self.provider_combo.setCurrentIndex(idx)

        # Restore saved preset selection
        last_preset = self.settings.value(KEY_PROVIDER_PRESET, "")
        if last_preset:
             idx = self.provider_combo.findText(last_preset)
             if idx >= 0:
                 self.provider_combo.setCurrentIndex(idx)

    def test_screenshot_analysis(self):
        """立即截图并测试AI能识别多少信息"""
        api_key = self.api_key_input.text()
        base_url = self.base_url_input.text().rstrip('/')
        model = self.model_input.currentText()
        
        # Get Resolution from UI (to test effect immediately)
        res_text = self.resolution_combo.currentText()
        target_size = RESOLUTION_PRESETS.get(res_text, 1024)
        
        if not api_key:
            self.test_result_area.setText("错误: 请先输入 API Key")
            return

        self.test_result_area.setText(f"正在截图 (分辨率: {res_text})...")
        QApplication.processEvents()

        try:
            # 获取当前监控的显示器索引
            monitor_idx = int(self.settings.value(KEY_MONITOR_INDEX, 1))
            
            # 截图
            with mss.mss() as sct:
                 # Auto resolve for test
                if monitor_idx == -1:
                    monitor_idx = get_active_monitor_index(sct, 1)

                if monitor_idx >= len(sct.monitors):
                    monitor_idx = 1
                monitor = sct.monitors[monitor_idx]
                sct_img = sct.grab(monitor)
                img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                
                # Apply Resolution Setting
                img = resize_image(img, target_size)
            
            # 获取窗口信息
            window_info = get_active_window_info()
            
            # 编码图片
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            size_kb = len(img_str) / 1024 * 0.75 # approx
            
            # 构建详细分析的Prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""请详细分析截图中的**文本内容**和**用户行为**。忽略UI布局、颜色、图标等视觉细节。

当前窗口: {window_info['title']} ({window_info['process']})
图片尺寸: {img.size}

重点识别：
1. **正在阅读或编辑的核心文字**: 提取代码片段、文档标题、正文内容、网页文章等可读信息。
2. **用户意图**: 根据内容推测用户当下的具体工作或活动（如“正在修复Python缩进错误”、“正在阅读关于Transformer的论文”）。

请用中文简练描述，展示你能看清多少细节。"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_str}",
                            }
                        },
                    ],
                }
            ]
            
            # 调用API
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                # max_tokens=4096, # User requested no limit (or let model decide)
            )
            
            # DEBUG OUTPUT
            print(f"DEBUG: Full API Response: {response}")
            
            result = response.choices[0].message.content
            
            # Special handling for models that return thinking/reasoning process (e.g. some DeepSeek/GLM modes)
            if not result and hasattr(response.choices[0].message, 'reasoning_content'):
                # If content is empty but reasoning is present, show reasoning (or it might be simply truncated)
                result = getattr(response.choices[0].message, 'reasoning_content', "")
                if result:
                    result = f"[Reasoning/Thinking Process]:\n{result}"

            print(f"DEBUG: Extracted Content: {result}")

            if not result:
                result = f"⚠️ 警告: 返回内容为空。原始响应:\n{response}"

            self.test_result_area.setText(f"✅ 分析成功 ({int(size_kb)}KB | {img.size[0]}x{img.size[1]}):\n\n{result}")
            
        except Exception as e:
            self.test_result_area.setText(f"❌ 测试失败:\n{str(e)}")

    def on_provider_changed(self, index):
        data = self.provider_combo.currentData()
        
        if data == "silicon":
            self.base_url_input.setText("https://api.siliconflow.cn/v1")
        elif data == "zhipu":
            self.base_url_input.setText("https://open.bigmodel.cn/api/paas/v4")
        elif data == "deepseek":
            self.base_url_input.setText("https://api.deepseek.com")
        elif data == "openai":
            self.base_url_input.setText("https://api.openai.com/v1")
        elif isinstance(data, dict):
            # User preset
            self.base_url_input.setText(data.get("base_url", ""))
            self.api_key_input.setText(data.get("api_key", ""))
            self.model_input.setCurrentText(data.get("model", ""))

    def test_connection(self):
        """保留原有的纯文本API测试功能(备用)"""
        url = self.base_url_input.text().rstrip('/') + "/chat/completions"
        api_key = self.api_key_input.text()
        model = self.model_input.currentText()
        
        if not api_key:
            self.test_result_area.setText("错误: 请先输入 API Key")
            return

        self.test_result_area.setText("正在发送测试请求...")
        QApplication.processEvents()

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "What opportunities and challenges will the Chinese large model industry face in 2025?"
                }
            ],
            "stream": False,
            "max_tokens": 100, # Limit for test
            "n": 1
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                self.test_result_area.setText(f"成功!\n{response.text[:200]}...")
            else:
                self.test_result_area.setText(f"失败 (Code {response.status_code}):\n{response.text}")
        except Exception as e:
            self.test_result_area.setText(f"请求异常:\n{str(e)}")

    def test_report_connection(self):
        """测试日报生成模型的连接"""
        # 使用日报模型配置，如果为空则fallback到主配置
        api_key = self.report_api_key_input.text() or self.api_key_input.text()
        base_url = self.report_base_url_input.text() or self.base_url_input.text()
        model = self.report_model_input.currentText()
        
        if not api_key:
            QMessageBox.warning(self, "错误", "请先配置 API Key（主配置或日报配置）")
            return
        
        if not base_url:
            QMessageBox.warning(self, "错误", "请先配置 Base URL（主配置或日报配置）")
            return

        try:
            self.test_result_area.setText("正在测试日报模型连接...")
            QApplication.processEvents()
            
            url = base_url.rstrip('/') + "/chat/completions"
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": "请用一句话介绍你自己。"
                    }
                ],
                "stream": False,
                "max_tokens": 50,
                "n": 1
            }
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            response = requests.post(url, json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                self.test_result_area.setText(f"✅ 日报模型连接成功!\n模型: {model}\n响应: {response.text[:150]}...")
            else:
                self.test_result_area.setText(f"❌ 连接失败 (Code {response.status_code}):\n{response.text[:200]}")
        except Exception as e:
            self.test_result_area.setText(f"❌ 测试失败:\n{str(e)}")

    def save_settings(self):
        self.settings.setValue(KEY_BASE_URL, self.base_url_input.text())
        self.settings.setValue(KEY_API_KEY, self.api_key_input.text())
        self.settings.setValue(KEY_MODEL, self.model_input.currentText())
        self.settings.setValue(KEY_MODEL, self.model_input.currentText())
        self.settings.setValue(KEY_INTERVAL, self.interval_input.text())
        self.settings.setValue(KEY_THRESHOLD, self.threshold_input.text())
        
        selected_monitor_idx = self.monitor_combo.currentData()
        self.settings.setValue(KEY_MONITOR_INDEX, selected_monitor_idx)

        res_text = self.resolution_combo.currentText()
        res_val = RESOLUTION_PRESETS.get(res_text, 1024)
        self.settings.setValue(KEY_RESOLUTION, res_val)
        
        self.settings.setValue(KEY_REPORT_BASE_URL, self.report_base_url_input.text())
        self.settings.setValue(KEY_REPORT_API_KEY, self.report_api_key_input.text())
        self.settings.setValue(KEY_REPORT_MODEL, self.report_model_input.currentText())
        
        # Save Provider Preset
        self.settings.setValue(KEY_PROVIDER_PRESET, self.provider_combo.currentText())
        
        self.accept()

    def view_full_result(self):
        content = self.test_result_area.toPlainText()
        if not content:
            return
        
        dlg = QDialog(self)
        dlg.setWindowTitle("分析结果全文")
        dlg.resize(800, 600)
        vbox = QVBoxLayout()
        text = QTextEdit()
        text.setPlainText(content)
        text.setReadOnly(True)
        vbox.addWidget(text)
        dlg.setLayout(vbox)
        dlg.exec()

class MonitorWorker(QThread):
    log_signal = Signal(str)
    status_signal = Signal(bool)
    
    # ... existing init ...

    def __init__(self):
        super().__init__()
        self.running = False
        self.client = None
        self.logger = LogManager(self.log_signal)
        self.settings = QSettings("Lumos", "LumosLog")
        self.sct = None # Initialized in run

    def _init_client(self):
        api_key = self.settings.value(KEY_API_KEY, "")
        base_url = self.settings.value(KEY_BASE_URL, "https://api.siliconflow.cn/v1")
        
        if api_key:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.logger.debug(f"OpenAI Client 初始化成功. BaseURL: {base_url}")
        else:
            self.client = None
            self.logger.error("API Key 未配置")

    def run(self):
        try:
            print("DEBUG: Worker execution started")
            self.sct = mss.mss()
            print("DEBUG: MSS initialized")
            
            self.running = True
            self.status_signal.emit(True)
            self.last_image = None # Reset
            
            interval = int(self.settings.value(KEY_INTERVAL, DEFAULT_INTERVAL))
            self.logger.debug_mode = self.settings.value(KEY_DEBUG, False, type=bool)
            
            self._init_client()
            monitor_idx = int(self.settings.value(KEY_MONITOR_INDEX, 1))
            
            display_mode = f"显示器 {monitor_idx}" if monitor_idx != -1 else "自动跟随活动窗口"
            self.logger.info(f"监控已启动 - 间隔: {interval}秒 - 模式: {display_mode}")

            while self.running:
                try:
                    self._process_cycle()
                except Exception as e:
                    self.logger.error(f"循环异常: {str(e)}")
                    import traceback
                    self.logger.debug(traceback.format_exc())
                
                # Wait loop
                interval = int(self.settings.value(KEY_INTERVAL, DEFAULT_INTERVAL))
                for _ in range(interval):
                    if not self.running:
                        break
                    time.sleep(1)
            
            self.status_signal.emit(False)
            self.logger.info("监控已停止")
            
        except Exception as e:
            print(f"DEBUG: Worker crashed: {e}")
            self.logger.error(f"监控线程崩溃: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            self.running = False
            self.status_signal.emit(False)

    def stop(self):
        self.running = False

    def _process_cycle(self):
        self.logger.debug(">>> 开始新一轮监控循环")
        
        # 1. Capture
        setting_monitor_idx = int(self.settings.value(KEY_MONITOR_INDEX, 1))
        
        # Resolve Actual Monitor
        if setting_monitor_idx == -1:
            monitor_idx = get_active_monitor_index(self.sct, getattr(self, 'last_monitor_idx', 1))
        else:
            monitor_idx = setting_monitor_idx
            
        self.last_monitor_idx = monitor_idx # Cache for fallback
        
        try:
            if monitor_idx >= len(self.sct.monitors):
                self.logger.error(f"显示器索引 {monitor_idx} 超出范围，重置为 1")
                monitor_idx = 1
                
            start_time = time.time()
            monitor = self.sct.monitors[monitor_idx]
            sct_img = self.sct.grab(monitor)
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            
            # Resize based on settings
            target_res = int(self.settings.value(KEY_RESOLUTION, 1024))
            img = resize_image(img, target_res)
            
            self.logger.debug(f"截图预处理完成. 尺寸: {img.size} 耗时: {time.time() - start_time:.2f}s")
        except Exception as e:
            self.logger.error(f"截图失败: {e}")
            return
        
        # 1.5 Get active window info
        window_info = get_active_window_info()
        self.logger.debug(f"活动窗口: {window_info['title']} ({window_info['process']})")

        # 2. Local Diff
        is_static = False
        if hasattr(self, 'last_image') and self.last_image:
            # Check size match properly
            if self.last_image.size != img.size:
                self.logger.debug(f"分辨率变化 ({self.last_image.size} -> {img.size})，强制分析")
                # Treat as changed (not static), so we just fall through to analysis
                is_static = False
            else:
                diff_val = self._calculate_rms(self.last_image, img)
                threshold = int(self.settings.value(KEY_THRESHOLD, DEFAULT_THRESHOLD))
                self.logger.debug(f"图像差异 RMS: {diff_val:.2f} (阈值: {threshold})")
                if diff_val < threshold:
                    is_static = True
        else:
            self.logger.debug("首张图片，跳过对比")
        
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        activity = "同上 (静止)"
        is_api_call = False

        # 3. Handle Activity
        if is_static:
            self.logger.info(f"画面静止，跳过识别。")
        else:
            if not self.client:
                activity = "API Key 缺失"
                self.logger.error(activity)
            else:
                self.logger.info("画面变化，正在分析...")
                try:
                    activity = self._analyze_image_with_ai(img, window_info)
                    is_api_call = True
                    self.logger.info(f"识别结果: {activity}")
                except Exception as e:
                    activity = f"分析失败: {str(e)}"
                    self.logger.error(activity)
        
        self.last_image = img

        # 4. Log
        # Save Screenshot

        try:
            date_folder = datetime.datetime.now().strftime("%Y-%m-%d")
            img_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            daily_folder = SCREENSHOT_DIR / date_folder
            daily_folder.mkdir(exist_ok=True)
            screenshot_path = daily_folder / f"{img_time_str}.jpg"
            
            img.save(screenshot_path, quality=85)
            screenshot_path_str = str(screenshot_path)
        except Exception as e:
            self.logger.error(f"截图保存失败: {str(e)}")
            screenshot_path_str = ""

        log_entry = {
            "timestamp": timestamp,
            "activity": activity,
            "window_title": window_info['title'],
            "process": window_info['process'],
            "is_api_call": is_api_call,
            "screenshot": screenshot_path_str
        }
        self._save_log(log_entry)
        gc.collect()

    def _calculate_rms(self, img1, img2):
        arr1 = np.array(img1).astype(float)
        arr2 = np.array(img2).astype(float)
        diff = arr1 - arr2
        rms = math.sqrt(np.mean(diff**2))
        return rms

    def _analyze_image_with_ai(self, img, window_info):
        model_name = self.settings.value(KEY_MODEL, "Qwen/Qwen2.5-VL-72B-Instruct")
        self.logger.debug(f"调用 AI 模型: {model_name}")
        
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        self.logger.debug("图片已编码,发送请求...")
        
        # 构建提示词 - 让AI专注于内容而非窗口信息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": f"""分析屏幕截图,用20-50字详细描述用户正在做什么。**不要重复窗口标题信息**。

参考信息(已知): 窗口 {window_info['title']} - {window_info['process']}

**你需要识别的是屏幕内容本身**,例如:
- 代码编辑器: 识别正在编写的函数名/功能/代码逻辑(如"实现用户登录验证逻辑,包含密码加密和Session管理")
- 浏览器: 识别网页主要内容/URL/关键信息(如"阅读PyTorch官方教程-卷积神经网络章节,学习CNN架构")
- 文档/笔记: 识别正在写的主题/关键词/核心内容(如"整理机器学习笔记-梯度下降算法原理及实现")
- 聊天软件: 识别对话主题和关键内容(如"讨论项目需求-用户权限管理模块设计")

**输出格式**: [动作] - [具体内容/主题]
**要求**: 尽可能提取屏幕中的关键信息、文件名、函数名、主题词等,20-50字

**示例**:
- 编写代码 - 实现窗口信息捕获功能,使用win32gui获取前台窗口
- 阅读文档 - OpenAI Vision API使用说明,学习图像分析参数
- 调试程序 - 修复截图保存bug,检查文件路径和权限问题
- 浏览网页 - GitHub开源项目研究,Star数10k+的深度学习框架
- 编辑笔记 - 深度学习知识点总结,反向传播算法推导过程

**禁止**: 不要输出应用名称、进程名(这些已知)"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_str}",
                        }
                    },
                ],
            }
        ]

        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=4096, # Increased to support reasoning models
        )
        
        result = response.choices[0].message.content
        if not result and hasattr(response.choices[0].message, 'reasoning_content'):
             # If content is empty but reasoning exists, try to use it or just log it
             # For log, we prefer short summary, but better than nothing
             reasoning = getattr(response.choices[0].message, 'reasoning_content', "")
             # Try to extract the last part or just return "AI Thinking..."
             # Or just return the first 50 chars of reasoning?
             # Let's return the reasoning for now so user sees something is happening
             return f"[Thinking] {reasoning[:50]}..."
             
        return result.strip() if result else ""

    def _save_log(self, entry):
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        filename = LOG_DIR / f"daily_log_{date_str}.jsonl"
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

class ReportWindow(QDialog):
    def __init__(self, parent=None, markdown_content=""):
        super().__init__(parent)
        self.setWindowTitle("今日活动日报")
        self.resize(600, 800)
        
        layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setMarkdown(markdown_content)
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        self.setLayout(layout)

class SearchDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("搜索日志")
        self.resize(800, 600)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Search Bar
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("输入关键词搜索日志内容...")
        self.search_input.returnPressed.connect(self.do_search)
        search_layout.addWidget(self.search_input)
        
        search_btn = QPushButton("搜索")
        search_btn.clicked.connect(self.do_search)
        search_layout.addWidget(search_btn)
        layout.addLayout(search_layout)
        
        # Results List
        self.result_list = QListWidget()
        self.result_list.itemDoubleClicked.connect(self.open_screenshot)
        layout.addWidget(self.result_list)
        
        self.setLayout(layout)

    def do_search(self):
        keyword = self.search_input.text().strip()
        if not keyword:
            return
            
        self.result_list.clear()
        
        # Search in JSONL files
        log_files = sorted(LOG_DIR.glob("daily_log_*.jsonl"), reverse=True)
        results = []
        
        for log_file in log_files:
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            # Full text match construction
                            content = f"{entry.get('activity', '')} {entry.get('window_title', '')} {entry.get('process', '')}"
                            if keyword.lower() in content.lower():
                                results.append(entry)
                        except:
                            continue
            except:
                continue
                
        # Display results
        for entry in results:
             text = f"[{entry['timestamp']}] {entry['activity']} - {entry.get('window_title', '')}"
             item = QListWidgetItem(text)
             item.setData(Qt.UserRole, entry.get('screenshot')) # Store screenshot path
             self.result_list.addItem(item)
             
        if not results:
             self.result_list.addItem("无搜索结果")

    def open_screenshot(self, item):
        path = item.data(Qt.UserRole)
        if path and os.path.exists(path):
            try:
                os.startfile(path) # Windows only
            except Exception as e:
                QMessageBox.warning(self, "错误", f"无法打开截图: {e}")
        else:
            if path:
                QMessageBox.information(self, "提示", f"截图文件已丢失: {path}")
            else:
                 QMessageBox.information(self, "提示", "该日志未关联截图")

class AppWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = MonitorWorker()
        self.settings = QSettings("Lumos", "LumosLog")
        self.init_ui()
        self.init_tray()
        self.setup_connections()
        self.load_custom_icon()

    def load_custom_icon(self):
        if ICON_PATH.exists():
            icon = QIcon(str(ICON_PATH))
            self.setWindowIcon(icon)
            if hasattr(self, 'tray_icon'):
                self.tray_icon.setIcon(icon)

    def init_ui(self):
        self.setWindowTitle("LumosLog - AI Screen Logger")
        self.resize(550, 650)
        
        layout = QVBoxLayout()

        # Header Row: Status + Settings
        header_layout = QHBoxLayout()
        self.status_label = QLabel("状态: 待机")
        self.status_label.setStyleSheet("color: gray; font-weight: bold;")
        header_layout.addWidget(self.status_label)
        header_layout.addStretch()
        
        settings_btn = QPushButton("设置 / API")
        settings_btn.clicked.connect(self.open_settings)
        header_layout.addWidget(settings_btn)

        search_btn = QPushButton("🔍 搜索日志")
        search_btn.clicked.connect(self.open_search)
        header_layout.addWidget(search_btn)
        layout.addLayout(header_layout)

        # Control Row
        btn_layout = QHBoxLayout()
        self.toggle_btn = QPushButton("开始监控")
        # self.toggle_btn.setCheckable(True) # Disable checkable to control state manually
        self.toggle_btn.clicked.connect(self.toggle_monitor)
        self.toggle_btn.setMinimumHeight(40)
        btn_layout.addWidget(self.toggle_btn)

        self.report_btn = QPushButton("生成今日日报")
        self.report_btn.clicked.connect(self.generate_report)
        self.report_btn.setMinimumHeight(40)
        btn_layout.addWidget(self.report_btn)
        
        self.copy_log_btn = QPushButton("复制日志+Prompt")
        self.copy_log_btn.clicked.connect(self.copy_log_with_prompt)
        self.copy_log_btn.setMinimumHeight(40)
        btn_layout.addWidget(self.copy_log_btn)
        
        layout.addLayout(btn_layout)

        # Debug Toggle
        self.debug_chk = QCheckBox("显示详细调试信息")
        self.debug_chk.setChecked(self.settings.value(KEY_DEBUG, False, type=bool))
        self.debug_chk.stateChanged.connect(self.toggle_debug)
        layout.addWidget(self.debug_chk)

        # Log Display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        layout.addWidget(QLabel("运行日志:"))
        layout.addWidget(self.log_display)

        self.setLayout(layout)

    def init_tray(self):
        self.tray_icon = QSystemTrayIcon(self)
        if ICON_PATH.exists():
            self.tray_icon.setIcon(QIcon(str(ICON_PATH)))
        else:
            self.tray_icon.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_ComputerIcon))
        
        tray_menu = QMenu()
        show_action = QAction("显示主界面", self)
        show_action.triggered.connect(self.show_normal)
        restart_action = QAction("⚡ 重启程序", self)
        restart_action.triggered.connect(self.restart_app)
        
        quit_action = QAction("退出程序", self)
        quit_action.triggered.connect(self.quit_app)
        
        tray_menu.addAction(show_action)
        tray_menu.addAction(restart_action)
        tray_menu.addSeparator()
        tray_menu.addAction(quit_action)
        
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self.on_tray_activated)
        self.tray_icon.show()

    def setup_connections(self):
        self.worker.log_signal.connect(self.update_log)
        self.worker.status_signal.connect(self.update_status)

    def open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec():
            # Reload connection if running?
            # Ideally restart worker if running to apply changes immediately
            if self.worker.running:
                self.update_log("[INFO] 配置已保存，将在下一轮循环生效")
            self.load_custom_icon() # Refresh icon immediately

    def open_search(self):
        dlg = SearchDialog(self)
        dlg.exec()

    def toggle_debug(self, state):
        is_debug = (state == Qt.Checked.value) # Fix comparison for PySide6 enum or int
        if isinstance(state, int):
            is_debug = (state == 2) # 2 is Checked

        self.settings.setValue(KEY_DEBUG, is_debug)
        self.worker.logger.debug_mode = is_debug
        
        if is_debug:
            self.update_log("[INFO] 调试模式已开启 - 将显示详细日志")

    def toggle_monitor(self):
        # Toggle based on actual worker intent
        if not self.worker.isRunning():
            # Start Monitor
            api_key = self.settings.value(KEY_API_KEY)
            
            if not api_key:
                QMessageBox.warning(self, "警告", "请先在设置中配置 API Key！")
                return
            
            try:
                self.worker.start()
                self.toggle_btn.setText("停止监控")
                self.toggle_btn.setStyleSheet("background-color: #ff4d4f; border: 1px solid #ff4d4f;")
                self.update_log("[INFO] 正在启动监控...")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"启动失败: {str(e)}")
        else:
            # Stop Monitor
            self.worker.stop()
            self.toggle_btn.setText("开始监控")
            self.toggle_btn.setStyleSheet("")
            self.update_log("[INFO]正在停止监控...")

    def update_log(self, text):
        import html
        # Escape HTML to prevent tag rendering issues and ensure special chars like < > are visible
        # Check if text is already HTML formatted (contains <font ...)? 
        # LogManager uses <font color='red'> for errors.
        # We should only escape if it's NOT a formatted error, or handle specifically.
        
        # Simple heuristic: if it contains <font, assume it's pre-formatted safely by us.
        # Otherwise escape.
        if "<font" in text:
            # It's likely our own formatted error. 
            pass 
        else:
            # Escape content to treat as plain text
            text = html.escape(text)
            
        self.log_display.append(text)
        sb = self.log_display.verticalScrollBar()
        sb.setValue(sb.maximum())

    def update_status(self, is_running):
        if is_running:
            self.status_label.setText("状态: 运行中")
            self.status_label.setStyleSheet("color: #76ff03; font-weight: bold;")
        else:
            self.toggle_btn.setChecked(False)
            self.toggle_btn.setText("开始监控")
            self.status_label.setText("状态: 待机")
            self.status_label.setStyleSheet("color: gray; font-weight: bold;")
    
    def compress_logs(self, logs):
        """智能压缩日志,过滤和合并相似活动,减少token消耗"""
        if not logs:
            return []
        
        compressed = []
        
        # 第一步: 过滤掉"同上(静止)"等无效记录
        filtered_logs = []
        for log in logs:
            activity = log.get('activity', '').strip()
            # 跳过静止状态或API失败的记录
            if activity in ['同上 (静止)', 'API Key 缺失', ''] or '分析失败' in activity:
                continue
            filtered_logs.append(log)
        
        if not filtered_logs:
            return []
        
        # 第二步: 合并连续的相似活动
        i = 0
        while i < len(filtered_logs):
            current = filtered_logs[i]
            start_time = current['timestamp']
            end_time = start_time
            
            # 收集所有相似活动的描述
            activities = [current['activity']]
            
            # 向前查找相似活动
            j = i + 1
            similar_count = 0
            while j < len(filtered_logs):
                next_log = filtered_logs[j]
                # 判断活动是否相似（窗口相同 + 活动描述相似度高）
                if (self._is_similar_activity(current, next_log)):
                    end_time = next_log['timestamp']
                    activities.append(next_log['activity'])
                    similar_count += 1
                    j += 1
                else:
                    break
            
            # 合并活动描述
            merged_activity = self._merge_activities(activities) if similar_count > 0 else current['activity']
            
            # 创建压缩后的日志条目
            merged_log = {
                'timestamp': f"{start_time} - {end_time}" if similar_count > 0 else start_time,
                'activity': merged_activity,
                'window_title': current.get('window_title', 'Unknown'),
                'process': current.get('process', 'Unknown'),
                'duration_minutes': similar_count # 持续分钟数（粗略估计）
            }
            compressed.append(merged_log)
            
            i = j if j > i else i + 1
        
        # 统计信息
        filtered_count = len(filtered_logs)
        skipped_count = len(logs) - filtered_count
        compression_ratio = len(compressed) / len(logs) * 100 if logs else 0
        
        # 详细的调试信息
        self.log_display.append(f"[INFO] ━━━ 日志压缩统计 ━━━")
        self.log_display.append(f"[INFO] 原始日志: {len(logs)} 条")
        self.log_display.append(f"[INFO] 过滤掉静止/无效: {skipped_count} 条")
        self.log_display.append(f"[INFO] 有效记录: {filtered_count} 条")
        self.log_display.append(f"[INFO] 合并后: {len(compressed)} 条")
        self.log_display.append(f"[INFO] 压缩率: {compression_ratio:.1f}% (节省token约 {100-compression_ratio:.1f}%)")
        self.log_display.append(f"[INFO] ━━━━━━━━━━━━━━━━━━")
        
        return compressed
    
    def _merge_activities(self, activities):
        """合并多条相似活动的描述,去除重复部分,保留差异信息"""
        if len(activities) == 1:
            return activities[0]
        
        # 找出共同前缀
        common_prefix = ""
        first_parts = activities[0].split(' - ', 1)
        if len(first_parts) > 1 and all(' - ' in act for act in activities):
            # 检查是否所有活动都有相同的前缀部分
            prefix_candidate = first_parts[0]
            if all(act.startswith(prefix_candidate) for act in activities):
                common_prefix = prefix_candidate + ' - '
        
        # 提取差异部分
        details = []
        for act in activities:
            if common_prefix:
                detail = act[len(common_prefix):].strip()
            else:
                # 如果没有共同前缀,尝试提取' - '后的部分
                parts = act.split(' - ', 1)
                detail = parts[1] if len(parts) > 1 else act
            
            # 去重
            if detail and detail not in details:
                details.append(detail)
        
        # 合并结果
        if common_prefix:
            return common_prefix + '; '.join(details)
        else:
            return activities[0].split(' - ')[0] + ' - ' + '; '.join(details) if details else activities[0]
    
    def _is_similar_activity(self, log1, log2):
        """判断两个日志条目是否为相似活动"""
        # 窗口标题必须相同
        if log1.get('window_title') != log2.get('window_title'):
            return False
        
        # 进程名必须相同
        if log1.get('process') != log2.get('process'):
            return False
        
        # 活动描述相似度检查（简单的关键词匹配）
        act1 = log1.get('activity', '').lower()
        act2 = log2.get('activity', '').lower()
        
        # 提取关键词（去除常见动词）
        stop_words = {'编写', '阅读', '浏览', '查看', '调试', '编辑', '操作', '配置'}
        words1 = set(act1.split()) - stop_words
        words2 = set(act2.split()) - stop_words
        
        if not words1 or not words2:
            return act1 == act2
        
        # 计算Jaccard相似度
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        similarity = intersection / union if union > 0 else 0
        
        # 相似度阈值设为0.5
        return similarity >= 0.5

    def generate_report(self):
        # Use separate model config for report generation
        api_key = self.settings.value(KEY_REPORT_API_KEY) or self.settings.value(KEY_API_KEY)
        base_url = self.settings.value(KEY_REPORT_BASE_URL) or self.settings.value(KEY_BASE_URL, "https://api.siliconflow.cn/v1")
        model = self.settings.value(KEY_REPORT_MODEL, "gemini-2.0-flash-exp")
        
        if not api_key:
            QMessageBox.warning(self, "错误", "API Key 未配置")
            return

        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        filename = LOG_DIR / f"daily_log_{date_str}.jsonl"
        
        if not filename.exists():
            QMessageBox.information(self, "提示", "今日暂无日志数据。")
            return

        logs = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                logs.append(json.loads(line))
        
        if not logs:
            QMessageBox.information(self, "提示", "日志数据为空。")
            return
        
        # 智能压缩日志,减少token消耗
        self.log_display.append(f"[INFO] 📊 开始生成日报... (原始: {len(logs)} 条)")
        compressed_logs = self.compress_logs(logs)
        
        if not compressed_logs:
            QMessageBox.information(self, "提示", "压缩后无有效日志数据（可能全是静止状态）。")
            return
            
        # 构建表格格式的精简日志
        context = "| 时间 | 窗口 | 应用 | 日志 |\n"
        context += "| --- | --- | --- | --- |\n"
        for log in compressed_logs:
            timestamp = log.get('timestamp', '')
            window = log.get('window_title', 'Unknown')
            process = log.get('process', 'Unknown')
            activity = log.get('activity', '')
            context += f"| {timestamp} | {window} | {process} | {activity} |\n"
            
        try:
            self.log_display.append("[INFO] 正在生成日报...")
            QApplication.processEvents()
            
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"""请根据以下用户今日的活动日志，生成一份结构化的日报。

原始日志（表格格式）：
{context}

输出要求：
1. 使用时间轴格式，而非段落叙述
2. 格式：**HH:MM - HH:MM**: 活动描述
3. 合并连续的相似活动
4. 突出重点项目和成果
5. 最后添加一句总结

示例格式：
## 📅 今日活动时间轴

**09:00 - 10:30**: 开发 LumosLog 项目 - 实现窗口信息捕获功能
**10:30 - 11:00**: 查阅 Python pywin32 官方文档
**11:00 - 12:00**: 编写代码测试与调试
...

---
💡 **今日总结**: ...
"""
                    }
                ]
            )
            report_content = response.choices[0].message.content
            report_window = ReportWindow(self, report_content)
            report_window.exec()
            self.log_display.append("[INFO] 日报已生成。")

        except Exception as e:
            QMessageBox.critical(self, "生成失败", str(e))
            self.log_display.append(f"[ERROR] 生成失败: {str(e)}")

    def copy_log_with_prompt(self):
        """复制结构化日志和提示词到剪贴板"""
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        filename = LOG_DIR / f"daily_log_{date_str}.jsonl"
        
        if not filename.exists():
            QMessageBox.information(self, "提示", "今日暂无日志数据。")
            return

        logs = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                logs.append(json.loads(line))
        
        if not logs:
            QMessageBox.information(self, "提示", "日志数据为空。")
            return
        
        # 智能压缩日志,减少token消耗
        self.log_display.append(f"[INFO] 📋 开始复制日志... (原始: {len(logs)} 条)")
        compressed_logs = self.compress_logs(logs)
        
        if not compressed_logs:
            QMessageBox.information(self, "提示", "压缩后无有效日志数据（可能全是静止状态）。")
            return
        
        # 构建表格格式的精简日志
        structured_log = f"# {date_str} 活动日志\n\n"
        structured_log += "| 时间 | 窗口 | 应用 | 日志 |\n"
        structured_log += "| --- | --- | --- | --- |\n"
        for log in compressed_logs:
            timestamp = log.get('timestamp', '')
            window = log.get('window_title', 'Unknown')
            process = log.get('process', 'Unknown')
            activity = log.get('activity', '')
            structured_log += f"| {timestamp} | {window} | {process} | {activity} |\n"
        
        # 添加提示词模板
        prompt_template = """

---

**请根据以上日志，生成一份结构化的工作日报。要求：**

1. 使用时间轴格式：**HH:MM - HH:MM**: 活动描述
2. 合并连续的相似活动
3. 突出重点项目和成果
4. 最后添加一句总结

示例格式：
## 📅 今日活动时间轴

**09:00 - 10:30**: 开发 XXX 项目 - 实现 YYY 功能
**10:30 - 11:00**: 查阅技术文档
...

---
💡 **今日总结**: ...
"""
        
        full_text = structured_log + prompt_template
        
        # 复制到剪贴板
        clipboard = QApplication.clipboard()
        clipboard.setText(full_text)
        
        QMessageBox.information(self, "成功", f"已复制日志到剪贴板！\n原始: {len(logs)} 条 → 压缩后: {len(compressed_logs)} 条\n可直接粘贴到 AI 对话框。")
        self.log_display.append(f"[INFO] 已复制日志到剪贴板 (压缩后 {len(compressed_logs)} 条)")


    def closeEvent(self, event: QCloseEvent):
        # 恢复关闭按钮的原始含义：退出程序
        self.quit_app()
        event.accept()

    def changeEvent(self, event):
        # 只有点击最小化时，才缩小到托盘（隐藏窗口）
        if event.type() == QEvent.WindowStateChange:
            if self.windowState() & Qt.WindowMinimized:
                event.accept()
                QTimer.singleShot(0, self.hide)
                return
        super().changeEvent(event)

    def show_normal(self):
        self.show()
        self.setWindowState(Qt.WindowActive)
        self.activateWindow()

    def on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.Trigger:
            self.show_normal()

    def restart_app(self):
        """重启应用程序"""
        self.worker.stop()
        self.tray_icon.hide()
        QApplication.quit()
        # Relaunch
        subprocess.Popen([sys.executable] + sys.argv)

    def quit_app(self):
        self.worker.stop()
        self.worker.wait()
        QApplication.quit()

def main():
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_teal.xml')
    window = AppWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

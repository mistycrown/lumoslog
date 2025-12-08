import sys
import os
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

from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QTextEdit, QLabel, QSystemTrayIcon, 
                               QMenu, QMessageBox, QDialog, QSizePolicy, 
                               QLineEdit, QFormLayout, QComboBox, QCheckBox, QGroupBox)
from PySide6.QtCore import QThread, Signal, Qt, QTimer, Slot, QSettings
from PySide6.QtGui import QIcon, QAction, QCloseEvent
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
IMAGE_SIZE = (512, 512)
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Application Settings Keys
KEY_API_KEY = "api_key"
KEY_BASE_URL = "base_url"
KEY_MODEL = "model_name"
KEY_INTERVAL = "interval"
KEY_DEBUG = "debug_mode"
KEY_MONITOR_INDEX = "monitor_index"

class LogManager:
    """Helper to manage logs being sent to UI"""
    def __init__(self, signal=None):
        self.signal = signal
        self.debug_mode = False

    def info(self, msg):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.signal.emit(f"[{timestamp}] [INFO] {msg}")

    def debug(self, msg):
        if self.debug_mode:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.signal.emit(f"[{timestamp}] [DEBUG] {msg}")

    def error(self, msg):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.signal.emit(f"[{timestamp}] [ERROR] <font color='red'>{msg}</font>")

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.resize(500, 400)
        self.settings = QSettings("Lumos", "LumosLog")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # API Config Group
        api_group = QGroupBox("API 配置")
        form_layout = QFormLayout()

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
            "Qwen/Qwen2.5-VL-72B-Instruct",
            "Qwen/Qwen3-VL-8B-Instruct",
            "gpt-4o-mini",
            "deepseek-ai/DeepSeek-V3"
        ]
        self.model_input.addItems(models)
        self.model_input.setCurrentText(self.settings.value(KEY_MODEL, "Qwen/Qwen2.5-VL-72B-Instruct"))
        form_layout.addRow("Model:", self.model_input)
        
        api_group.setLayout(form_layout)
        layout.addWidget(api_group)

        # Test Button
        test_btn = QPushButton("测试 API 连接 (纯文本)")
        test_btn.clicked.connect(self.test_connection)
        layout.addWidget(test_btn)
        
        self.test_result_area = QTextEdit()
        self.test_result_area.setMaximumHeight(80)
        self.test_result_area.setReadOnly(True)
        layout.addWidget(self.test_result_area)

        # Other Settings
        other_group = QGroupBox("监控设置")
        other_layout = QFormLayout()
        
        self.interval_input = QLineEdit()
        self.interval_input.setText(str(self.settings.value(KEY_INTERVAL, DEFAULT_INTERVAL)))
        other_layout.addRow("截图间隔 (秒):", self.interval_input)

        # Monitor Selection
        self.monitor_combo = QComboBox()
        with mss.mss() as sct:
            for i, monitor in enumerate(sct.monitors):
                if i == 0: continue # Skip 'all in one'
                self.monitor_combo.addItem(f"Monitor {i} ({monitor['width']}x{monitor['height']})", i)
        
        current_idx = int(self.settings.value(KEY_MONITOR_INDEX, 1))
        # Find index in combobox
        idx_to_set = 0
        for i in range(self.monitor_combo.count()):
            if self.monitor_combo.itemData(i) == current_idx:
                idx_to_set = i
                break
        self.monitor_combo.setCurrentIndex(idx_to_set)
        other_layout.addRow("选择显示器:", self.monitor_combo)
        
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
        
        self.setLayout(layout)

    def test_connection(self):
        """
        Runs the minimal unit test as requested by user using 'requests'.
        """
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
        
        # Add some specific SiliconFlow params if needed, but keep it simple for connectivity test
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                self.test_result_area.setText(f"成功!\n{response.text[:200]}...") # Show first 200 chars
            else:
                self.test_result_area.setText(f"失败 (Code {response.status_code}):\n{response.text}")
        except Exception as e:
            self.test_result_area.setText(f"请求异常:\n{str(e)}")

    def save_settings(self):
        self.settings.setValue(KEY_BASE_URL, self.base_url_input.text())
        self.settings.setValue(KEY_API_KEY, self.api_key_input.text())
        self.settings.setValue(KEY_MODEL, self.model_input.currentText())
        self.settings.setValue(KEY_INTERVAL, self.interval_input.text())
        
        selected_monitor_idx = self.monitor_combo.currentData()
        self.settings.setValue(KEY_MONITOR_INDEX, selected_monitor_idx)
        
        self.accept()

class MonitorWorker(QThread):
    log_signal = Signal(str)
    status_signal = Signal(bool)

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
        self.sct = mss.mss()
        self.running = True
        self.status_signal.emit(True)
        
        interval = int(self.settings.value(KEY_INTERVAL, DEFAULT_INTERVAL))
        self.logger.debug_mode = self.settings.value(KEY_DEBUG, False, type=bool)
        
        self._init_client()
        monitor_idx = int(self.settings.value(KEY_MONITOR_INDEX, 1))
        self.logger.info(f"监控已启动 - 间隔: {interval}秒 - 显示器: {monitor_idx}")

        while self.running:
            try:
                self._process_cycle()
            except Exception as e:
                self.logger.error(f"循环异常: {str(e)}")
                import traceback
                self.logger.debug(traceback.format_exc())
            
            # Wait loop
            interval = int(self.settings.value(KEY_INTERVAL, DEFAULT_INTERVAL)) # Update interval dynmically
            for _ in range(interval):
                if not self.running:
                    break
                time.sleep(1)
        
        self.status_signal.emit(False)
        self.logger.info("监控已停止")

    def stop(self):
        self.running = False

    def _process_cycle(self):
        self.logger.debug(">>> 开始新一轮监控循环")
        
        # 1. Capture
        monitor_idx = int(self.settings.value(KEY_MONITOR_INDEX, 1))
        try:
            # Validate index
            if monitor_idx >= len(self.sct.monitors):
                self.logger.error(f"显示器索引 {monitor_idx} 超出范围，重置为 1")
                monitor_idx = 1
                
            start_time = time.time()
            monitor = self.sct.monitors[monitor_idx]
            sct_img = self.sct.grab(monitor)
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            img = img.resize(IMAGE_SIZE)
            self.logger.debug(f"截图完成 (Monitor {monitor_idx}). 耗时: {time.time() - start_time:.2f}s")
        except Exception as e:
            self.logger.error(f"截图失败: {e}")
            return

        # 2. Local Diff
        is_static = False
        if hasattr(self, 'last_image') and self.last_image:
            diff_val = self._calculate_rms(self.last_image, img)
            self.logger.debug(f"图像差异 RMS: {diff_val:.2f} (阈值: {DEFAULT_THRESHOLD})")
            if diff_val < DEFAULT_THRESHOLD:
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
                    activity = self._analyze_image_with_ai(img)
                    is_api_call = True
                    self.logger.info(f"识别结果: {activity}")
                except Exception as e:
                    activity = f"分析失败: {str(e)}"
                    self.logger.error(activity)
        
        self.last_image = img

        # 4. Log
        log_entry = {
            "timestamp": timestamp,
            "activity": activity,
            "is_api_call": is_api_call
        }
        self._save_log(log_entry)
        gc.collect()

    def _calculate_rms(self, img1, img2):
        arr1 = np.array(img1).astype(float)
        arr2 = np.array(img2).astype(float)
        diff = arr1 - arr2
        rms = math.sqrt(np.mean(diff**2))
        return rms

    def _analyze_image_with_ai(self, img):
        model_name = self.settings.value(KEY_MODEL, "Qwen/Qwen2.5-VL-72B-Instruct")
        self.logger.debug(f"调用 AI 模型: {model_name}")
        
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        self.logger.debug("图片已编码，发送请求...")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "用5-10个字简短描述屏幕上正在进行的主要任务（例如：编写Python代码、浏览B站视频、空闲待机）。"},
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
            max_tokens=30,
        )
        return response.choices[0].message.content.strip()

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

class AppWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = MonitorWorker()
        self.settings = QSettings("Lumos", "LumosLog")
        self.init_ui()
        self.init_tray()
        self.setup_connections()

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
        layout.addLayout(header_layout)

        # Control Row
        btn_layout = QHBoxLayout()
        self.toggle_btn = QPushButton("开始监控")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self.toggle_monitor)
        self.toggle_btn.setMinimumHeight(40)
        btn_layout.addWidget(self.toggle_btn)

        self.report_btn = QPushButton("生成今日日报")
        self.report_btn.clicked.connect(self.generate_report)
        self.report_btn.setMinimumHeight(40)
        btn_layout.addWidget(self.report_btn)
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
        self.tray_icon.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_ComputerIcon))
        
        tray_menu = QMenu()
        show_action = QAction("显示主界面", self)
        show_action.triggered.connect(self.show_normal)
        quit_action = QAction("退出程序", self)
        quit_action.triggered.connect(self.quit_app)
        
        tray_menu.addAction(show_action)
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

    def toggle_debug(self, state):
        is_debug = (state == Qt.Checked.value) # Fix comparison for PySide6 enum or int
        if isinstance(state, int):
            is_debug = (state == 2) # 2 is Checked

        self.settings.setValue(KEY_DEBUG, is_debug)
        self.worker.logger.debug_mode = is_debug
        
        if is_debug:
            self.update_log("[INFO] 调试模式已开启 - 将显示详细日志")

    def toggle_monitor(self):
        if self.toggle_btn.isChecked():
            # Check API key first
            if not self.settings.value(KEY_API_KEY):
                QMessageBox.warning(self, "警告", "请先在设置中配置 API Key！")
                self.toggle_btn.setChecked(False)
                return
            
            self.worker.start()
            self.toggle_btn.setText("停止监控")
        else:
            self.worker.stop()
            self.toggle_btn.setText("开始监控")

    def update_log(self, text):
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

    def generate_report(self):
        # ... logic similar to before, but read api key from settings ...
        api_key = self.settings.value(KEY_API_KEY)
        base_url = self.settings.value(KEY_BASE_URL, "https://api.siliconflow.cn/v1")
        model = self.settings.value(KEY_MODEL, "Qwen/Qwen2.5-VL-72B-Instruct")
        
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
            
        context = ""
        for log in logs:
            context += f"[{log['timestamp']}] {log['activity']}\n"
            
        try:
            self.log_display.append("[INFO] 正在生成日报...")
            QApplication.processEvents()
            
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"请根据以下用户今日的活动日志，生成一份日报摘要：\n{context}"
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

    def closeEvent(self, event: QCloseEvent):
        if self.tray_icon.isVisible():
            self.hide()
            self.tray_icon.showMessage("LumosLog", "程序已最小化到托盘", QSystemTrayIcon.Information, 2000)
            event.ignore()
        else:
            event.accept()

    def show_normal(self):
        self.show()
        self.activateWindow()

    def on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.Trigger:
            self.show_normal()

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

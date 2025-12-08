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
import win32gui
import win32process
import psutil

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
MAX_IMAGE_DIM = (1024, 1024)
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

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
    except Exception as e:
        return {
            "title": "获取失败",
            "process": "Unknown"
        }

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

    def test_screenshot_analysis(self):
        """立即截图并测试AI能识别多少信息"""
        api_key = self.api_key_input.text()
        base_url = self.base_url_input.text().rstrip('/')
        model = self.model_input.currentText()
        
        if not api_key:
            self.test_result_area.setText("错误: 请先输入 API Key")
            return

        self.test_result_area.setText("正在截图并分析,请稍候...")
        QApplication.processEvents()

        try:
            # 获取当前监控的显示器索引
            monitor_idx = int(self.settings.value(KEY_MONITOR_INDEX, 1))
            
            # 截图
            with mss.mss() as sct:
                if monitor_idx >= len(sct.monitors):
                    monitor_idx = 1
                monitor = sct.monitors[monitor_idx]
                sct_img = sct.grab(monitor)
                img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                img.thumbnail(MAX_IMAGE_DIM, Image.Resampling.LANCZOS)
            
            # 获取窗口信息
            window_info = get_active_window_info()
            
            # 编码图片
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # 构建详细分析的Prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""请尽可能详细地描述这张屏幕截图中的所有内容。

当前窗口信息: {window_info['title']} ({window_info['process']})

请列出你能看到的:
1. **主要应用/界面**: 是什么软件?布局如何?
2. **文本内容**: 能识别的标题、段落、代码、命令等(尽可能多)
3. **UI元素**: 按钮、菜单、选项卡、输入框等
4. **视觉细节**: 颜色、图标、布局风格
5. **用户正在做什么**: 推测具体活动
6. **其他细节**: 任何你能观察到的信息

请用中文详细描述,不要遗漏细节。"""
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
                max_tokens=500,  # 增加token限制以获取更详细的描述
            )
            
            result = response.choices[0].message.content
            self.test_result_area.setText(f"✅ 分析成功!\n\n{result}")
            
        except Exception as e:
            self.test_result_area.setText(f"❌ 测试失败:\n{str(e)}")

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
        self.settings.setValue(KEY_INTERVAL, self.interval_input.text())
        
        selected_monitor_idx = self.monitor_combo.currentData()
        self.settings.setValue(KEY_MONITOR_INDEX, selected_monitor_idx)
        
        self.settings.setValue(KEY_REPORT_BASE_URL, self.report_base_url_input.text())
        self.settings.setValue(KEY_REPORT_API_KEY, self.report_api_key_input.text())
        self.settings.setValue(KEY_REPORT_MODEL, self.report_model_input.currentText())
        
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
            interval = int(self.settings.value(KEY_INTERVAL, DEFAULT_INTERVAL))
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
            if monitor_idx >= len(self.sct.monitors):
                self.logger.error(f"显示器索引 {monitor_idx} 超出范围，重置为 1")
                monitor_idx = 1
                
            start_time = time.time()
            monitor = self.sct.monitors[monitor_idx]
            sct_img = self.sct.grab(monitor)
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            
            # Resize (Thumbnail - Keep Aspect Ratio)
            img.thumbnail(MAX_IMAGE_DIM, Image.Resampling.LANCZOS)
            
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
                    activity = self._analyze_image_with_ai(img, window_info)
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
            "window_title": window_info['title'],
            "process": window_info['process'],
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
            max_tokens=80,  # 增加到80以支持50字中文输出
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
            
        context = ""
        for log in logs:
            window_info = f" ({log.get('window_title', 'Unknown')})"
            context += f"[{log['timestamp']}] {log['activity']}{window_info}\n"
            
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

原始日志：
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
        
        # 构建结构化日志
        structured_log = f"# {date_str} 活动日志\n\n"
        for log in logs:
            timestamp = log['timestamp']
            activity = log['activity']
            window = log.get('window_title', 'Unknown')
            process = log.get('process', 'Unknown')
            structured_log += f"**[{timestamp}]** {activity}\n"
            structured_log += f"  - 窗口: {window}\n"
            structured_log += f"  - 应用: {process}\n\n"
        
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
        
        QMessageBox.information(self, "成功", f"已复制 {len(logs)} 条日志和提示词到剪贴板！\n可直接粘贴到 AI 对话框。")
        self.log_display.append(f"[INFO] 已复制日志到剪贴板 ({len(logs)} 条)")


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

"""步態分析分頁：載入側拍行走影片 → 背景萃取關節點與計算指標 → 播放展示畫面（與 demo 影片相同）。"""
import os
import subprocess
import sys

import cv2
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QFileDialog, QGroupBox, QHBoxLayout, QLabel, QMessageBox, QProgressBar,
                             QPushButton, QSizePolicy, QSlider, QVBoxLayout, QWidget)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gait_analysis.pipeline import analyze_video          # noqa: E402
from gait_analysis.render_demo import render              # noqa: E402


class AnalysisThread(QThread):
    progress = pyqtSignal(int, int)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, model, video_path):
        super().__init__()
        self.model = model
        self.video_path = video_path

    def run(self):
        try:
            analysis = analyze_video(self.video_path, self.model, progress=self.progress.emit,
                                     should_stop=self.isInterruptionRequested)
        except Exception as error:      # 顯示給使用者，避免背景執行緒默默失敗
            self.failed.emit(f'{type(error).__name__}: {error}')
            return
        if analysis is not None:
            self.succeeded.emit(analysis)


class ExportThread(QThread):
    progress = pyqtSignal(int, int)
    succeeded = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, analysis, out_path):
        super().__init__()
        self.analysis = analysis
        self.out_path = out_path

    def run(self):
        try:
            from gait_analysis import plots
            analysis, base = self.analysis, os.path.splitext(self.out_path)[0]
            footer = '' if analysis.trial is None else plots.VIDIMU_FOOTER
            plots.number_panel(analysis.values, f'{analysis.header} 步態數字面板', f'{base}_panel.png', footer)
            if analysis.agreement is not None:
                plots.knee_overlay(analysis.curves, analysis.result.strides, analysis.agreement,
                                   f'{base}_knee_overlay.png')
            if render(analysis.renderer(), analysis.video_path, self.out_path,
                      progress=self.progress.emit, should_stop=self.isInterruptionRequested):
                self.succeeded.emit(self.out_path)
        except Exception as error:
            self.failed.emit(f'{type(error).__name__}: {error}')


class GaitTabControl(QWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self.analysis = None
        self.renderer = None
        self.cap = None
        self.current_frame = -1
        self.pixmap = None
        self.worker = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.nextFrame)
        self.setupUi()
        self.bindUi()
        self.setLoaded(False)

    # ------------------------------------------------------------------ UI

    def setupUi(self):
        self.view = QLabel('請載入側拍行走影片')
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(640, 360)
        self.view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.view.setStyleSheet('background-color: #f2f2f2; color: #888888; font-size: 18px;')

        self.backBtn = QPushButton('<<')
        self.playBtn = QPushButton('▶︎')
        self.forwardBtn = QPushButton('>>')
        for button in (self.backBtn, self.playBtn, self.forwardBtn):
            button.setFixedSize(50, 30)
        self.frameSlider = QSlider(Qt.Horizontal)
        self.frameNumLabel = QLabel('0/0')
        controls = QHBoxLayout()
        for widget in (self.backBtn, self.playBtn, self.forwardBtn, self.frameSlider, self.frameNumLabel):
            controls.addWidget(widget)

        left = QVBoxLayout()
        left.addWidget(self.view, 1)
        left.addLayout(controls)

        self.loadBtn = QPushButton('載入影片')
        self.exportBtn = QPushButton('匯出展示影片')
        self.openFolderBtn = QPushButton('開啟輸出資料夾')
        self.cancelBtn = QPushButton('取消')
        for button in (self.loadBtn, self.exportBtn, self.openFolderBtn, self.cancelBtn):
            button.setMinimumHeight(40)
        self.fileLabel = QLabel('檔名：')
        self.statusLabel = QLabel('')
        self.statusLabel.setWordWrap(True)
        self.progressBar = QProgressBar()
        self.progressBar.setVisible(False)
        self.infoLabel = QLabel('')
        self.infoLabel.setWordWrap(True)
        self.infoLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)

        panel = QGroupBox('步態分析')
        panel.setFixedWidth(320)
        side = QVBoxLayout(panel)
        for widget in (self.fileLabel, self.loadBtn, self.exportBtn, self.openFolderBtn, self.cancelBtn,
                       self.progressBar, self.statusLabel, self.infoLabel):
            side.addWidget(widget)
        side.addStretch(1)
        hint = QLabel('影片需為側面拍攝、受試者在畫面中左右來回行走。檔名為 VIDIMU 格式'
                      '（如 S40_A01_T01.mp4）且有 IMU 資料時，會自動疊上 IMU 膝角。')
        hint.setWordWrap(True)
        hint.setStyleSheet('color: #777777;')
        side.addWidget(hint)

        layout = QHBoxLayout(self)
        layout.addLayout(left, 1)
        layout.addWidget(panel)

    def bindUi(self):
        self.loadBtn.clicked.connect(self.loadVideo)
        self.exportBtn.clicked.connect(self.exportDemo)
        self.openFolderBtn.clicked.connect(self.openOutputFolder)
        self.cancelBtn.clicked.connect(self.cancelWorker)
        self.playBtn.clicked.connect(self.togglePlay)
        self.backBtn.clicked.connect(lambda: self.frameSlider.setValue(self.frameSlider.value() - 1))
        self.forwardBtn.clicked.connect(lambda: self.frameSlider.setValue(self.frameSlider.value() + 1))
        self.frameSlider.valueChanged.connect(self.showFrame)

    def setLoaded(self, loaded):
        busy = self.worker is not None
        for widget in (self.playBtn, self.backBtn, self.forwardBtn, self.frameSlider):
            widget.setEnabled(loaded)
        self.exportBtn.setEnabled(loaded and not busy)
        self.openFolderBtn.setEnabled(loaded)
        self.loadBtn.setEnabled(not busy)
        self.cancelBtn.setVisible(busy)

    def keyPressEvent(self, event):
        if event.key() in (ord('D'), ord('d')):
            self.frameSlider.setValue(self.frameSlider.value() + 1)
        elif event.key() in (ord('A'), ord('a')):
            self.frameSlider.setValue(self.frameSlider.value() - 1)
        elif event.key() == Qt.Key_Space and self.playBtn.isEnabled():
            self.togglePlay()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.updateView()

    # ------------------------------------------------------------- 背景工作

    def startWorker(self, worker, status, on_success):
        # 所有訊號都要在 start() 前連好，否則讀快取很快完成時 succeeded 可能先送出而遺失
        self.worker = worker
        worker.progress.connect(self.onProgress)
        worker.succeeded.connect(on_success)
        worker.failed.connect(self.onFailed)
        worker.finished.connect(self.onWorkerFinished)
        self.progressBar.setValue(0)
        self.progressBar.setVisible(True)
        self.statusLabel.setText(status)
        self.setLoaded(self.analysis is not None)
        worker.start()

    def onProgress(self, done, total):
        self.progressBar.setMaximum(max(total, 1))
        self.progressBar.setValue(done)

    def onFailed(self, message):
        self.statusLabel.setText('')
        QMessageBox.warning(self, '步態分析失敗', message)

    def onWorkerFinished(self):
        if self.worker is not None and self.worker.isInterruptionRequested():
            self.statusLabel.setText('已取消')
        self.worker = None
        self.progressBar.setVisible(False)
        self.setLoaded(self.analysis is not None)

    def cancelWorker(self):
        if self.worker is not None:
            self.worker.requestInterruption()
            self.statusLabel.setText('取消中…')

    def shutdown(self):
        """主視窗關閉時呼叫：停止播放並等背景工作結束，避免執行緒在執行中被銷毀。"""
        self.timer.stop()
        if self.worker is not None:
            self.worker.requestInterruption()
            self.worker.wait()
        if self.cap is not None:
            self.cap.release()

    # ----------------------------------------------------------------- 分析

    def loadVideo(self):
        path, _ = QFileDialog.getOpenFileName(self, '選擇側拍行走影片', '', 'Video Files (*.mp4 *.avi *.mov);;All Files (*)')
        if not path:
            return
        stem, ext = os.path.splitext(os.path.basename(path))
        for suffix, kind in (('_demo', '匯出的展示影片'), ('_Sk26', '已疊上骨架的影片')):
            if stem.endswith(suffix):
                # 畫面已疊上圖表或骨架，再分析一次得到的數字沒有意義
                original = os.path.join(os.path.dirname(path), stem[:-len(suffix)] + ext)
                hint = f'\n同資料夾的原始影片：{os.path.basename(original)}' if os.path.isfile(original) else ''
                QMessageBox.warning(self, '請載入原始影片', f'{os.path.basename(path)} 是{kind}，不能再拿來分析。{hint}')
                return
        self.stopPlay()
        self.fileLabel.setText(f'檔名：{os.path.basename(path)}')
        self.infoLabel.setText('')
        self.startWorker(AnalysisThread(self.model, path), '萃取關節點與計算步態指標中…（處理過的影片會直接讀取快取）',
                         self.onAnalyzed)

    def onAnalyzed(self, analysis):
        if self.cap is not None:
            self.cap.release()
        self.analysis = analysis
        self.renderer = analysis.renderer()
        self.cap = cv2.VideoCapture(analysis.video_path)
        self.current_frame = -1
        n_frames = self.renderer.n_frames
        blocked = self.frameSlider.blockSignals(True)
        self.frameSlider.setRange(0, max(n_frames - 1, 0))
        self.frameSlider.setValue(0)
        self.frameSlider.blockSignals(blocked)
        self.timer.setInterval(max(1, int(round(1000 / analysis.result.fps))))
        self.statusLabel.setText('分析完成')
        self.infoLabel.setText(self.summaryText(analysis))
        self.showFrame(0)

    @staticmethod
    def summaryText(analysis):
        values = analysis.values
        lines = [f'行走趟數：{values["passes"]}　近側有效週期：{values["near_strides"]}']
        if values['passes'] == 0:
            lines.append('⚠ 沒有偵測到直線行走段，請確認影片為側拍、受試者在畫面中左右行走。')
        if analysis.agreement is not None:
            a = analysis.agreement
            lines.append(f'與 IMU 一致性（膝角）：r = {a["knee_pearson_r"]:.2f}、'
                         f'扣除平均差後 RMSE = {a["knee_rmse_centered"]:.1f}°')
        elif analysis.trial is not None:
            lines.append('此 VIDIMU 試驗沒有 IMU 資料（只有 S40–S57 有），不做 IMU 對照。')
        else:
            lines.append('非 VIDIMU 影片，沒有 IMU 對照。')
        lines.append(f'CSV 已輸出至：{analysis.out_dir}')
        return '\n'.join(lines)

    def exportDemo(self):
        if self.analysis is None:
            return
        self.stopPlay()
        default = os.path.join(self.analysis.out_dir, f'{self.analysis.name}_demo.mp4')
        os.makedirs(os.path.dirname(default), exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(self, '匯出展示影片', default, 'MP4 (*.mp4)')
        if not path:
            return
        self.startWorker(ExportThread(self.analysis, path), '匯出展示影片中…',
                         lambda out: self.statusLabel.setText(f'已匯出：{out}'))

    def openOutputFolder(self):
        if self.analysis is not None and os.path.isdir(self.analysis.out_dir):
            subprocess.Popen(['explorer', os.path.normpath(self.analysis.out_dir)])

    # ----------------------------------------------------------------- 播放

    def togglePlay(self):
        if self.timer.isActive():
            self.stopPlay()
            return
        if self.frameSlider.value() >= self.frameSlider.maximum():
            self.frameSlider.setValue(0)
        self.playBtn.setText('||')
        self.timer.start()

    def stopPlay(self):
        self.timer.stop()
        self.playBtn.setText('▶︎')

    def nextFrame(self):
        if self.frameSlider.value() >= self.frameSlider.maximum():
            self.stopPlay()
            return
        self.frameSlider.setValue(self.frameSlider.value() + 1)

    def showFrame(self, frame_idx):
        if self.renderer is None or not 0 <= frame_idx < self.renderer.n_frames:
            return
        if frame_idx != self.current_frame + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok:
            self.stopPlay()
            return
        self.current_frame = frame_idx
        canvas = self.renderer.render_frame(frame_idx, frame)
        height, width = canvas.shape[:2]
        image = QImage(canvas.data, width, height, 3 * width, QImage.Format_RGB888).rgbSwapped()
        self.pixmap = QPixmap.fromImage(image)
        self.frameNumLabel.setText(f'{frame_idx}/{self.renderer.n_frames - 1}')
        self.updateView()

    def updateView(self):
        if self.pixmap is not None:
            self.view.setPixmap(self.pixmap.scaled(self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

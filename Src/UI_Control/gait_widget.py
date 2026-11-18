"""步態分析分頁：載入側拍行走影片 → 背景萃取關節點與計算指標 → 以 Qt 元件顯示骨架、角度曲線與數字面板。

畫面中有多人時可點選目標人物（記住選擇並重算）；也可批量分析每位受試者的步態影片（A01），產生快取與輸出檔。

螢幕顯示直接用 Qt 繪製（gait_views.py），任何視窗大小都清晰；「匯出展示影片」才用 render_demo 合成固定畫面。
"""
import os
import subprocess
import sys

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QCheckBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QMessageBox, QProgressBar,
                             QPushButton, QSlider, QSplitter, QVBoxLayout, QWidget)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gait_analysis import vidimu                          # noqa: E402
from gait_analysis.pipeline import (DERIVED_VIDEOS, SAVED, analyze_video, derived_suffix,   # noqa: E402
                                    has_current_cache)
from gait_analysis.render_demo import blur_face, render   # noqa: E402
from gait_analysis.style import IMU_COLOR, VIDEO_COLOR, VIDIMU_FOOTER   # noqa: E402
from gait_analysis.vidimu import OUTPUT_ROOT, PROJECT_ROOT, VIDEO_DIR, gait_videos   # noqa: E402
from gait_views import AnglePlot, FrameView, SummaryPanel, VideoRow   # noqa: E402

FOLLOW_BEFORE_S, FOLLOW_AFTER_S = 4.0, 2.0     # 播放時曲線顯示目前時間前 4 秒、後 2 秒
TOP_SHARE = 0.56                               # 上排（影片、數字、按鈕）預設佔可用高度的比例
PLOT_MIN_HEIGHT = 175                          # 每張曲線圖最小高度（像素），刻度與最大伸展標籤才放得下


class AnalysisThread(QThread):
    progress = pyqtSignal(int, int)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, model, video_path, person_id=SAVED):
        super().__init__()
        self.model = model
        self.video_path = video_path
        self.person_id = person_id

    def run(self):
        try:
            analysis = analyze_video(self.video_path, self.model, progress=self.progress.emit,
                                     should_stop=self.isInterruptionRequested, person_id=self.person_id)
        except Exception as error:      # 顯示給使用者，避免背景執行緒默默失敗
            self.failed.emit(f'{type(error).__name__}: {error}')
            return
        if analysis is not None:
            self.succeeded.emit(analysis)


class BatchThread(QThread):
    """依序分析多支影片；單支失敗不中斷，最後回報每支結果 [(檔名, 錯誤訊息或 None)]。"""
    progress = pyqtSignal(int, int)
    itemStarted = pyqtSignal(int, int, str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, model, video_paths):
        super().__init__()
        self.model = model
        self.video_paths = list(video_paths)

    def run(self):
        results = []
        for index, path in enumerate(self.video_paths, 1):
            if self.isInterruptionRequested():
                break
            self.itemStarted.emit(index, len(self.video_paths), os.path.basename(path))
            try:
                done = analyze_video(path, self.model, progress=self.progress.emit,
                                     should_stop=self.isInterruptionRequested)
                if done is None:
                    break
                results.append((os.path.basename(path), None))
            except Exception as error:
                results.append((os.path.basename(path), f'{type(error).__name__}: {error}'))
        self.succeeded.emit(results)


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
        except PermissionError as error:
            locked = os.path.basename(error.filename2 or error.filename or self.out_path)
            self.failed.emit(f'{locked} 正被其他程式開啟（例如播放器），請關閉後再匯出。')
        except Exception as error:
            self.failed.emit(f'{type(error).__name__}: {error}')


class GaitTabControl(QWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self.analysis = None
        self.renderer = None        # DemoRenderer：提供近側腿、曲線、最大伸展點與事件等逐格資料
        self.cap = None
        self.current_frame = -1
        self.worker = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.nextFrame)
        self.setupUi()
        self.bindUi()
        self.setLoaded(False)

    # ------------------------------------------------------------------ UI

    def setupUi(self):
        """版面：上排「影片（左上）│ 數字區 │ 功能按鈕」，下方髖、膝曲線全寬堆疊，最底下是播放進度條。"""
        self.view = FrameView()
        self.view.placeholder = ('請載入側拍行走影片\n\n'
                                 '影片需為側面拍攝、受試者在畫面中左右來回行走。檔名為 VIDIMU 格式'
                                 '（如 S40_A01_T01.mp4）且有 IMU 資料時，會自動疊上 IMU 膝角。\n'
                                 '畫面中有多人時會顯示每個人的框，點選要分析的人即可重算。')
        self.hipPlot = AnglePlot('髖關節屈伸角度（正值＝屈曲，負值＝伸展）', (-35, 50))
        self.kneePlot = AnglePlot('膝關節屈曲角度', (-10, 80))
        self.kneePlot.setXLink(self.hipPlot)
        for plot in (self.hipPlot, self.kneePlot):
            plot.setMinimumHeight(PLOT_MIN_HEIGHT)      # 小視窗時曲線不被壓扁，改由上排讓出空間

        # 數字區：檔頭、四項指標、分析摘要、資料來源
        self.headerLabel = QLabel('')
        self.headerLabel.setStyleSheet('font-size: 17pt; font-weight: bold; color: #444;')
        self.headerLabel.setWordWrap(True)
        self.summaryPanel = SummaryPanel()
        self.infoLabel = QLabel('')
        self.infoLabel.setStyleSheet('font-size: 12pt;')
        self.infoLabel.setWordWrap(True)
        self.infoLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.footerLabel = QLabel('')
        self.footerLabel.setStyleSheet('font-size: 11pt; color: #777;')
        self.footerLabel.setWordWrap(True)
        numbers = QWidget()
        numbersLayout = QVBoxLayout(numbers)
        numbersLayout.setContentsMargins(0, 0, 0, 0)
        numbersLayout.addWidget(self.headerLabel)
        numbersLayout.addWidget(self.summaryPanel)
        numbersLayout.addWidget(self.infoLabel)
        numbersLayout.addStretch(1)
        numbersLayout.addWidget(self.footerLabel)

        self.backBtn = QPushButton('<<')
        self.playBtn = QPushButton('▶︎')
        self.forwardBtn = QPushButton('>>')
        for button in (self.backBtn, self.playBtn, self.forwardBtn):
            button.setFixedSize(50, 30)
        self.frameSlider = QSlider(Qt.Horizontal)
        self.frameNumLabel = QLabel('0/0')
        self.followCheckBox = QCheckBox('曲線跟隨播放')
        self.followCheckBox.setChecked(True)
        for widget in (self.frameNumLabel, self.followCheckBox):
            widget.setStyleSheet('font-size: 12pt;')
        controls = QHBoxLayout()
        for widget in (self.backBtn, self.playBtn, self.forwardBtn, self.frameSlider, self.frameNumLabel,
                       self.followCheckBox):
            controls.addWidget(widget)

        self.loadBtn = QPushButton('載入影片')
        self.imuBtn = QPushButton('分析含 IMU 步態影片')
        self.batchBtn = QPushButton('批量分析步態影片')
        self.autoTargetBtn = QPushButton('改回自動選擇目標')
        self.exportBtn = QPushButton('匯出展示影片')
        self.openFolderBtn = QPushButton('開啟輸出資料夾')
        self.cancelBtn = QPushButton('取消')
        for button in (self.loadBtn, self.imuBtn, self.batchBtn, self.autoTargetBtn, self.exportBtn,
                       self.openFolderBtn, self.cancelBtn):
            button.setMinimumHeight(38)
        self.fileLabel = QLabel('檔名：')
        self.fileLabel.setWordWrap(True)
        self.targetLabel = QLabel('')
        self.targetLabel.setWordWrap(True)
        self.statusLabel = QLabel('')
        self.statusLabel.setWordWrap(True)
        self.progressBar = QProgressBar()
        self.progressBar.setVisible(False)

        panel = QGroupBox('步態分析')
        panel.setFixedWidth(270)
        panel.setStyleSheet('QGroupBox, QPushButton, QLabel { font-size: 12pt; }')
        side = QVBoxLayout(panel)
        for widget in (self.fileLabel, self.loadBtn, self.imuBtn, self.batchBtn, self.targetLabel, self.autoTargetBtn,
                       self.exportBtn, self.openFolderBtn, self.cancelBtn, self.progressBar, self.statusLabel):
            side.addWidget(widget)
        side.addStretch(1)

        self.topRow = VideoRow(self.view, numbers, panel)
        curves = QWidget()
        curvesLayout = QVBoxLayout(curves)
        curvesLayout.setContentsMargins(0, 0, 0, 0)
        curvesLayout.addWidget(self.hipPlot)
        curvesLayout.addWidget(self.kneePlot)
        # 上排（影片）與曲線之間可拖曳分隔線調整比例；預設上排佔 TOP_SHARE
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.addWidget(self.topRow)
        self.splitter.addWidget(curves)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(8)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)
        self.splitterInitialized = False
        layout = QVBoxLayout(self)
        layout.addWidget(self.splitter, 1)
        layout.addLayout(controls)

    def showEvent(self, event):
        super().showEvent(event)
        if not self.splitterInitialized:
            # 只在第一次顯示（版面算好之後）套用預設比例；之後縮放視窗保留使用者拖曳的位置
            self.splitterInitialized = True
            QTimer.singleShot(0, self.applyDefaultSplit)

    def applyDefaultSplit(self):
        height = sum(self.splitter.sizes())
        self.splitter.setSizes([int(height * TOP_SHARE), height - int(height * TOP_SHARE)])

    def bindUi(self):
        self.loadBtn.clicked.connect(self.loadVideo)
        self.imuBtn.clicked.connect(self.analyzeImuVideos)
        self.batchBtn.clicked.connect(self.batchAnalyze)
        self.autoTargetBtn.clicked.connect(lambda: self.selectTarget(None))
        self.view.imageClicked.connect(self.onImageClicked)
        self.exportBtn.clicked.connect(self.exportDemo)
        self.openFolderBtn.clicked.connect(self.openOutputFolder)
        self.cancelBtn.clicked.connect(self.cancelWorker)
        self.playBtn.clicked.connect(self.togglePlay)
        self.backBtn.clicked.connect(lambda: self.frameSlider.setValue(self.frameSlider.value() - 1))
        self.forwardBtn.clicked.connect(lambda: self.frameSlider.setValue(self.frameSlider.value() + 1))
        self.frameSlider.valueChanged.connect(self.showFrame)
        for plot in (self.hipPlot, self.kneePlot):
            plot.seekRequested.connect(self.seekTime)
        self.followCheckBox.toggled.connect(self.followCurrent)

    def setLoaded(self, loaded):
        busy = self.worker is not None
        for widget in (self.playBtn, self.backBtn, self.forwardBtn, self.frameSlider):
            widget.setEnabled(loaded)
        self.exportBtn.setEnabled(loaded and not busy)
        self.openFolderBtn.setEnabled(loaded)
        self.loadBtn.setEnabled(not busy)
        self.imuBtn.setEnabled(not busy)
        self.batchBtn.setEnabled(not busy)
        manual = loaded and self.analysis is not None and self.analysis.person_id is not None
        self.autoTargetBtn.setVisible(manual)
        self.autoTargetBtn.setEnabled(manual and not busy)
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
        start = OUTPUT_ROOT if os.path.isdir(OUTPUT_ROOT) else ''        # 預設開在 Db/Record
        path, _ = QFileDialog.getOpenFileName(self, '選擇側拍行走影片', start, 'Video Files (*.mp4 *.avi *.mov);;All Files (*)')
        if path:
            self.openVideo(path)

    def openVideo(self, path):
        suffix = derived_suffix(path)
        if suffix:
            # 畫面已疊上圖表或骨架，再分析一次得到的數字沒有意義
            stem, ext = os.path.splitext(os.path.basename(path))
            original = os.path.join(os.path.dirname(path), stem[:-len(suffix)] + ext)
            hint = f'\n同資料夾的原始影片：{os.path.basename(original)}' if os.path.isfile(original) else ''
            QMessageBox.warning(self, '請載入原始影片',
                                f'{os.path.basename(path)} 是{DERIVED_VIDEOS[suffix]}，不能再拿來分析。{hint}')
            return
        self.stopPlay()
        self.fileLabel.setText(f'檔名：{os.path.basename(path)}')
        self.infoLabel.setText('')
        self.startWorker(AnalysisThread(self.model, path), '萃取關節點與計算步態指標中…（處理過的影片會直接讀取快取）',
                         self.onAnalyzed)

    def onImageClicked(self, x, y):
        """點到某人的框就改以他為目標重算（框重疊時取面積最小的人）。"""
        if self.analysis is None or self.worker is not None or not self.analysis.can_select:
            return
        hits = [(float((b[2] - b[0]) * (b[3] - b[1])), person_id)
                for person_id, b, _ in self.analysis.persons_at(self.current_frame)
                if b[0] <= x <= b[2] and b[1] <= y <= b[3]]
        if not hits:
            return
        person_id = min(hits)[1]
        if person_id != self.analysis.person_id:
            self.selectTarget(person_id)

    def selectTarget(self, person_id):
        if self.analysis is None or self.worker is not None:
            return
        self.stopPlay()
        frame = self.current_frame
        text = '自動選擇' if person_id is None else f'ID {person_id}'
        self.startWorker(AnalysisThread(self.model, self.analysis.video_path, person_id),
                         f'改以 {text} 為目標重新計算，並更新骨架影片…',
                         lambda analysis: self.onAnalyzed(analysis, frame))

    @staticmethod
    def timeEstimate(videos):
        cached = sum(map(has_current_cache, videos))
        new = len(videos) - cached
        if not new:
            return '全部已有快取，只需數秒。'
        return (f'其中 {cached} 支已有快取（數秒），其餘 {new} 支需萃取關節點，每支約 30 秒'
                f'（共約 {max(1, round(new * 0.5))} 分鐘）。')

    def gaitVideoRoot(self):
        """步態影片根目錄（底下為 S01、S02… 資料夾）；資料集不在預設位置時請使用者選擇。"""
        if os.path.isdir(VIDEO_DIR):
            return VIDEO_DIR
        return QFileDialog.getExistingDirectory(self, '選擇步態影片根目錄（底下為 S01、S02… 資料夾）', PROJECT_ROOT)

    def analyzeImuVideos(self):
        """與批量分析相同，但只分析有對應 IMU 檔的步態影片；新增影片與 IMU 檔後會自動納入。"""
        root = self.gaitVideoRoot()
        if not root:
            return
        videos = [path for path in gait_videos(root)[0] if vidimu.video_has_imu(path)]
        if not videos:
            QMessageBox.warning(self, '無法分析', f'{root} 底下的步態影片在 {vidimu.IMU_DIR} 中都沒有對應的 IMU 檔。')
            return
        subjects = len({os.path.basename(os.path.dirname(path)) for path in videos})
        self.runBatch('分析含 IMU 步態影片', videos,
                      f'將分析 {root}\n底下有 IMU 資料的 {subjects} 位受試者、{len(videos)} 支步態影片（A01），'
                      f'{self.timeEstimate(videos)}')

    def batchAnalyze(self):
        """批量分析步態影片：每位受試者（Sxx 資料夾）的 A01（T01／T02／T03，有幾趟跑幾趟）。"""
        root = self.gaitVideoRoot()
        if not root:
            return
        videos, missing = gait_videos(root)
        if not videos:
            QMessageBox.warning(self, '無法批量分析',
                                f'{root} 底下找不到 Sxx/Sxx_A01_Tnn.mp4 這類步態影片。')
            return
        subjects = len({os.path.basename(os.path.dirname(path)) for path in videos})
        text = (f'將分析 {root}\n底下 {subjects} 位受試者的 {len(videos)} 支步態影片（A01），'
                f'{self.timeEstimate(videos)}')
        if missing:
            text += f'\n\n沒有步態影片、將略過：' + '、'.join(missing)
        self.runBatch('批量分析步態影片', videos, text)

    def runBatch(self, title, videos, text):
        if QMessageBox.question(self, title, text + '\n\n要開始嗎？') != QMessageBox.Yes:
            return
        self.stopPlay()
        worker = BatchThread(self.model, videos)
        worker.itemStarted.connect(
            lambda i, n, name: self.statusLabel.setText(f'{title} {i}/{n}：{name}\n（有快取的影片很快，新影片需萃取關節點）'))
        self.startWorker(worker, f'{title}：{len(videos)} 支影片…', self.onBatchDone)

    def onBatchDone(self, results):
        failed = [(name, error) for name, error in results if error]
        lines = [f'完成 {len(results) - len(failed)}/{len(results)} 支，輸出在 {OUTPUT_ROOT}']
        lines += [f'✗ {name}：{error}' for name, error in failed[:10]]
        self.statusLabel.setText(lines[0])
        QMessageBox.information(self, '批量分析結束', '\n'.join(lines))

    def onAnalyzed(self, analysis, frame=0):
        if self.cap is not None:
            self.cap.release()
        self.analysis = analysis
        self.renderer = renderer = analysis.renderer()
        self.cap = cv2.VideoCapture(analysis.video_path)
        self.current_frame = -1
        fps = analysis.result.fps
        times = np.arange(renderer.n_frames) / fps

        hip_markers = [(f / fps, v, f'最大伸展 {-v:.0f}°') for f, v in renderer.hip_markers]
        self.hipPlot.setSeries(times, [(renderer.hip, VIDEO_COLOR, 2.5, '影片', False)], hip_markers)
        knee = [(renderer.knee, VIDEO_COLOR, 2.5, '影片（單一鏡頭）', False)]
        if renderer.imu_knee is not None:
            knee.append((renderer.imu_knee, IMU_COLOR, 2, '穿戴式感測器 IMU', True))
        self.kneePlot.setSeries(times, knee)
        self.summaryPanel.setValues(analysis.values)
        self.headerLabel.setText(analysis.header)
        self.footerLabel.setText(VIDIMU_FOOTER if analysis.trial is not None else '')

        blocked = self.frameSlider.blockSignals(True)
        self.frameSlider.setRange(0, max(renderer.n_frames - 1, 0))
        frame = min(max(frame, 0), max(renderer.n_frames - 1, 0))
        self.frameSlider.setValue(frame)
        self.frameSlider.blockSignals(blocked)
        self.timer.setInterval(max(1, int(round(1000 / fps))))
        self.statusLabel.setText('分析完成')
        self.infoLabel.setText(self.summaryText(analysis))
        self.targetLabel.setText(self.targetText(analysis))
        self.hipPlot.setXRange(0, FOLLOW_BEFORE_S + FOLLOW_AFTER_S, padding=0)     # 新影片從預設 6 秒寬開始
        self.showFrame(frame)
        self.setLoaded(True)

    @staticmethod
    def targetText(analysis):
        if not analysis.can_select:
            return '目標人物：自動（這支影片沒有多人資料，無法點選）'
        if analysis.person_id is not None:
            return f'目標人物：ID {analysis.person_id}（手動點選）'
        auto = f'，主要為 ID {analysis.auto_id}' if analysis.auto_id is not None else ''
        return f'目標人物：自動（畫面中最大的人{auto}）'

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
            lines.append('此 VIDIMU 試驗在資料集中沒有對應的 IMU 檔，不做 IMU 對照。')
        else:
            lines.append('非 VIDIMU 影片，沒有 IMU 對照。')
        lines.append(f'已輸出至：{analysis.out_dir}')
        lines += [f'⚠ {warning}' for warning in analysis.warnings]
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

    def seekTime(self, t):
        if self.renderer is not None:
            self.frameSlider.setValue(int(round(max(t, 0.0) * self.analysis.result.fps)))

    def showFrame(self, frame_idx):
        renderer = self.renderer
        if renderer is None or not 0 <= frame_idx < renderer.n_frames:
            return
        if frame_idx != self.current_frame + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok:
            self.stopPlay()
            return
        self.current_frame = frame_idx
        result = renderer.result
        bbox = renderer.bboxes[frame_idx] if renderer.bboxes is not None else np.full(4, np.nan)
        frame = blur_face(frame, result.keypoints[frame_idx], bbox)
        self.view.setFrame(frame, result.xy[frame_idx], renderer.near_by_frame.get(frame_idx),
                           renderer.event_labels(frame_idx), self.analysis.persons_at(frame_idx))
        t = frame_idx / result.fps
        for plot in (self.hipPlot, self.kneePlot):
            plot.setTimeCursor(t)
        self.followCurrent()
        self.frameNumLabel.setText(f'{frame_idx}/{renderer.n_frames - 1}　{t:.2f} 秒')

    def followCurrent(self):
        if self.renderer is None or not self.followCheckBox.isChecked() or self.current_frame < 0:
            return
        # 沿用使用者目前的縮放寬度，只平移：游標保持在畫面前 4／後 2 的比例位置
        t = self.current_frame / self.analysis.result.fps
        x0, x1 = self.hipPlot.viewRange()[0]
        width = x1 - x0
        start = max(0.0, t - width * FOLLOW_BEFORE_S / (FOLLOW_BEFORE_S + FOLLOW_AFTER_S))
        self.hipPlot.setXRange(start, start + width, padding=0)

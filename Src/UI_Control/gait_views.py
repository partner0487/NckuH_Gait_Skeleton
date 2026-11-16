"""步態分析分頁的顯示元件：直接以 Qt 繪製，任何視窗大小都保持清晰。

匯出影片仍用 gait_analysis.render_demo 合成固定 1280×720 畫面；這裡的元件只負責螢幕顯示。
"""
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from gait_analysis.gait import KPT, SKELETON_LINKS, side_index
from gait_analysis.style import panel_items

VIDEO_COLOR = '#1f77b4'
IMU_COLOR = '#d62728'
NEAR_COLOR = QColor(255, 140, 0)
FAR_COLOR = QColor(170, 170, 170)
TRUNK_COLOR = QColor(90, 90, 90)
FULL_COLOR = QColor(225, 225, 225, 200)        # 完整骨架底層：淡灰
FULL_EDGE = QColor(0, 0, 0, 90)
MARK_COLOR = '#00a000'


class FrameView(QWidget):
    """影片畫面：影像等比例縮放後，骨架與標記在螢幕解析度上繪製。

    畫面中有兩人以上時畫出每個人的框與 ID；點擊畫面發出 imageClicked(影像 x, 影像 y)。
    """
    imageClicked = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.xy = None
        self.near_side = None
        self.labels = []
        self.persons = []           # [(person_id, bbox, is_target)]
        self.placeholder = '請載入側拍行走影片'
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def setFrame(self, frame_bgr, xy, near_side, labels=(), persons=()):
        """xy 為 (26, 2) 影像座標；labels 為 [(文字, 影像 x, 影像 y)]；persons 為 [(ID, 框, 是否目標)]。"""
        height, width = frame_bgr.shape[:2]
        self.image = QImage(frame_bgr.data, width, height, 3 * width, QImage.Format_RGB888).rgbSwapped()
        self.xy, self.near_side, self.labels, self.persons = xy, near_side, list(labels), list(persons)
        self.update()

    def mousePressEvent(self, event):
        target = self.targetRect()
        if self.image is None or event.button() != Qt.LeftButton or not target.contains(event.pos()):
            return super().mousePressEvent(event)
        scale = target.width() / self.image.width()
        self.imageClicked.emit((event.pos().x() - target.x()) / scale, (event.pos().y() - target.y()) / scale)

    def clear(self, placeholder=''):
        self.image = None
        self.placeholder = placeholder
        self.update()

    def targetRect(self):
        """影像在元件內等比例置中的位置。"""
        if self.image is None:
            return QRectF()
        scale = min(self.width() / self.image.width(), self.height() / self.image.height())
        w, h = self.image.width() * scale, self.image.height() * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        if self.image is None:
            painter.setPen(QColor(160, 160, 160))
            painter.drawText(self.rect(), Qt.AlignCenter, self.placeholder)
            return
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        target = self.targetRect()
        painter.drawImage(target, self.image)
        scale = target.width() / self.image.width()

        def pt(k):
            if self.xy is None or not np.isfinite(self.xy[k]).all():
                return None
            return QPointF(target.x() + self.xy[k, 0] * scale, target.y() + self.xy[k, 1] * scale)

        # 線寬隨顯示大小縮放，小視窗不會過粗、大視窗不會過細
        base = max(1.5, target.width() / 320)
        if len(self.persons) >= 2:          # 多人時才畫框，方便點選目標
            font = QFont(self.font())
            font.setPixelSize(int(max(11, target.height() / 40)))
            painter.setFont(font)
            for person_id, box, is_target in self.persons:
                color = NEAR_COLOR if is_target else QColor(255, 255, 255)
                rect = QRectF(target.x() + box[0] * scale, target.y() + box[1] * scale,
                              (box[2] - box[0]) * scale, (box[3] - box[1]) * scale)
                painter.setPen(QPen(color, 2 if is_target else 1, Qt.SolidLine if is_target else Qt.DashLine))
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(rect)
                text = f'ID {person_id}' + ('（目標）' if is_target else '')
                tag = QRectF(rect.x(), rect.y() - font.pixelSize() - 6,
                             painter.fontMetrics().horizontalAdvance(text) + 8, font.pixelSize() + 6)
                painter.fillRect(tag, QColor(40, 40, 40, 190))
                painter.setPen(color)
                painter.drawText(tag, Qt.AlignCenter, text)
        # 底層：完整骨架細線；上層再疊計算用到的軀幹與雙腿
        # 淡灰線外加深色細邊，白牆或深色背景都看得到
        links = [(pt(a), pt(b)) for a, b in SKELETON_LINKS]
        links = [(pa, pb) for pa, pb in links if pa and pb]
        for color, width in ((FULL_EDGE, base * 0.7 + 2), (FULL_COLOR, base * 0.7)):
            painter.setPen(QPen(color, width, Qt.SolidLine, Qt.RoundCap))
            for pa, pb in links:
                painter.drawLine(pa, pb)
        neck = [pt(k) for k in KPT['shoulder']]
        pelvis = [pt(k) for k in KPT['hip']]
        if all(neck) and all(pelvis):
            painter.setPen(QPen(TRUNK_COLOR, base * 1.5, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine((neck[0] + neck[1]) / 2, (pelvis[0] + pelvis[1]) / 2)
        for side in sorted(('L', 'R'), key=lambda s: s == self.near_side):      # 近側腿最後畫
            near = side == self.near_side
            color, width = (NEAR_COLOR, base * 2) if near else (FAR_COLOR, base)
            chain = [side_index(p, side) for p in ('hip', 'knee', 'ankle', 'heel', 'toe')]
            painter.setPen(QPen(color, width, Qt.SolidLine, Qt.RoundCap))
            for a, b in list(zip(chain, chain[1:])) + [(chain[2], chain[4])]:
                pa, pb = pt(a), pt(b)
                if pa and pb:
                    painter.drawLine(pa, pb)
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            for k in chain:
                p = pt(k)
                if p:
                    painter.drawEllipse(p, width * 1.1, width * 1.1)

        font = QFont(self.font())
        font.setPixelSize(int(max(12, target.height() / 34)))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(0, 110, 230))
        for text, x, y in self.labels:
            painter.drawText(QPointF(target.x() + x * scale - font.pixelSize(),
                                     target.y() + y * scale + font.pixelSize() * 1.6), text)
        if self.near_side:
            tag = f'近側腿：{"左" if self.near_side == "L" else "右"}腳（橘色）'
            metrics = painter.fontMetrics()
            box = QRectF(target.x() + 8, target.y() + 8, metrics.horizontalAdvance(tag) + 16, metrics.height() + 8)
            painter.fillRect(box, QColor(50, 50, 50, 200))
            painter.setPen(Qt.white)
            painter.drawText(box, Qt.AlignCenter, tag)


class AnglePlot(pg.PlotWidget):
    """角度對時間曲線，含可拖曳的時間游標；拖曳或點擊會發出 seekRequested(秒)。"""
    seekRequested = pyqtSignal(float)

    def __init__(self, title, y_range, parent=None):
        super().__init__(parent, background='w')
        self.setTitle(f'<span style="font-size: 12pt; color: #333">{title}</span>')
        self.setLabel('left', '角度（度）')
        self.setLabel('bottom', '時間（秒）')
        self.showGrid(x=True, y=True, alpha=0.25)
        self.setYRange(*y_range, padding=0)
        self.setMouseEnabled(x=True, y=False)
        self.setMenuEnabled(False)
        self.hideButtons()
        self.legend = self.addLegend(offset=(-10, 10), labelTextSize='10pt')
        self.cursor = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#444444', width=1.5))
        self.cursor.setZValue(10)
        self.cursor.sigDragged.connect(lambda line: self.seekRequested.emit(line.value()))
        self.addItem(self.cursor)
        self.scene().sigMouseClicked.connect(self._onClick)
        self.dynamic = []

    def _onClick(self, event):
        if event.button() == Qt.LeftButton and self.plotItem.vb.sceneBoundingRect().contains(event.scenePos()):
            self.seekRequested.emit(self.plotItem.vb.mapSceneToView(event.scenePos()).x())

    def setSeries(self, times, series, markers=()):
        """series 為 [(數值陣列, 顏色, 線寬, 圖例, 虛線)]；markers 為 [(秒, 數值, 文字)]。"""
        for item in self.dynamic:
            self.removeItem(item)
        self.legend.clear()
        self.dynamic = []
        for values, color, width, name, dashed in series:
            pen = pg.mkPen(color, width=width, style=Qt.DashLine if dashed else Qt.SolidLine)
            self.dynamic.append(self.plot(times, values, pen=pen, name=name, connect='finite'))
        if markers:
            scatter = pg.ScatterPlotItem([m[0] for m in markers], [m[1] for m in markers], size=9,
                                         brush=pg.mkBrush(MARK_COLOR), pen=pg.mkPen('w', width=1))
            self.addItem(scatter)
            self.dynamic.append(scatter)
            for t, value, text in markers:
                label = pg.TextItem(text, color=MARK_COLOR, anchor=(0.5, -0.2))
                label.setPos(t, value)
                self.addItem(label)
                self.dynamic.append(label)

    def setTimeCursor(self, t):
        # 不能叫 setCursor：會蓋掉 QWidget.setCursor（滑鼠游標）
        self.cursor.setValue(t)


class SummaryPanel(QWidget):
    """數字面板：步頻、膝活動範圍、最大髖伸展、軀幹前傾角。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.cards = []
        for _ in range(4):
            card = QFrame()
            card.setStyleSheet('QFrame { background: #f5f7fa; border: 1px solid #d0d7de; border-radius: 6px; }'
                               'QLabel { border: none; background: transparent; }')
            box = QVBoxLayout(card)
            labels = [QLabel(), QLabel(), QLabel()]
            for label, style in zip(labels, ('font-size: 11pt; color: #333;',
                                             'font-size: 24pt; font-weight: bold; color: #222;',
                                             'font-size: 9pt; color: #666;')):
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet(style)
                label.setWordWrap(True)
                box.addWidget(label)
            layout.addWidget(card)
            self.cards.append(labels)
        self.setValues(None)

    def setValues(self, values):
        items = panel_items(values) if values else [
            {'label': name, 'value': '—', 'note': ''} for name in ('步頻', '膝活動範圍', '最大髖伸展', '軀幹前傾角')]
        for (title, value, note), item in zip(self.cards, items):
            title.setText(item['label'])
            value.setText(item['value'])
            note.setText(item['note'])

    def texts(self):
        return [value.text() for _, value, _ in self.cards]

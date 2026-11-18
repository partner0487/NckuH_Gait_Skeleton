"""Run with: conda run -n Gait python -m unittest discover -s tests -v

需要 VIDIMU 資料與 S52_A01_T02 關節點快取的測試，在資料不存在時會略過。
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
_yolo_config = tempfile.TemporaryDirectory()
os.environ['YOLO_CONFIG_DIR'] = _yolo_config.name
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Src' / 'UI_Control'))
sys.path.insert(0, str(ROOT / 'Src'))

import numpy as np
from PyQt5.QtWidgets import QApplication

from gait_analysis import gait, pipeline
from gait_analysis.render_demo import DemoRenderer, H, W, draw_skeleton
from gait_views import FrameView
from gait_widget import AnalysisThread, GaitTabControl

APP = QApplication.instance() or QApplication([])
DEMO_VIDEO = ROOT / 'Db_NckuH_Gait' / 'Db_VideoSmallSize' / 'Db_VideoSOriginal' / 'S52' / 'S52_A01_T02.mp4'
HAS_DEMO = DEMO_VIDEO.is_file() and Path(pipeline.cache_path(str(DEMO_VIDEO))).is_file()


def standing_keypoints(n_frames=90):
    """站著不動的合成關節點：沒有行走段。"""
    kpts = np.zeros((n_frames, 26, 3))
    base = {5: (300, 150), 6: (300, 150), 11: (300, 250), 12: (300, 250), 13: (300, 330), 14: (300, 330),
            15: (300, 410), 16: (300, 410), 24: (290, 420), 25: (290, 420), 20: (330, 420), 21: (330, 420)}
    for k, (x, y) in base.items():
        kpts[:, k, :2] = (x, y)
    kpts[:, :, 2] = 0.9
    return kpts


class PipelineTests(unittest.TestCase):
    def test_non_vidimu_name_has_no_trial(self):
        self.assertIsNone(pipeline.find_vidimu_trial('D:/videos/my_walk.mp4'))

    def test_vidimu_style_name_outside_dataset_is_not_vidimu(self):
        # 醫院資料可能沿用 VIDIMU 命名；沒有 IMU 且不在資料集中就不當成 VIDIMU（不固定 30 fps、不加資料來源註記）
        self.assertIsNone(pipeline.find_vidimu_trial('D:/hospital/S01_A01_T01.mp4'))

    def test_standing_video_has_no_passes_but_renders(self):
        result = gait.analyze('standing', standing_keypoints())
        values = gait.trial_summary(result)
        self.assertEqual(values['passes'], 0)
        self.assertTrue(np.isnan(values['cadence']))
        canvas = DemoRenderer(result, values, footer='').render_frame(0, np.zeros((480, 640, 3), np.uint8))
        self.assertEqual(canvas.shape, (H, W, 3))

    def test_full_skeleton_is_drawn_underneath(self):
        # 手臂沒參與計算，但完整骨架要以淡灰色畫出；缺點（NaN）的連線不畫
        xy = np.full((26, 2), np.nan)
        xy[18], xy[6], xy[8], xy[10] = (100, 50), (100, 60), (100, 120), (100, 180)
        frame = draw_skeleton(np.zeros((240, 200, 3), np.uint8), xy, 'R')
        self.assertGreater(frame[150, 100].mean(), 100)        # 手肘到手腕中間
        self.assertEqual(frame[150, 60].mean(), 0)
        view = FrameView()
        view.setFrame(np.zeros((240, 200, 3), np.uint8), xy, 'R')
        view.resize(400, 480)
        rect = view.targetRect()
        scale = rect.width() / 200
        x, y = int(rect.x() + 100 * scale), int(rect.y() + 150 * scale)
        image = view.grab().toImage()
        self.assertGreater(max(image.pixelColor(x + dx, y).red() for dx in (-1, 0, 1)), 60)


@unittest.skipUnless(HAS_DEMO, 'VIDIMU S52_A01_T02 資料或關節點快取不存在')
class LayoutTests(unittest.TestCase):
    def show(self, frame_size, window_size):
        widget = GaitTabControl(Mock())
        widget.resize(*window_size)
        widget.show()
        height, width = frame_size
        widget.view.setFrame(np.zeros((height, width, 3), np.uint8), np.full((26, 2), np.nan), None)
        APP.processEvents()
        self.addCleanup(widget.close)
        return widget

    def test_video_sits_top_left_without_bars(self):
        # 教授要求：影片在左上角、不裁切不留黑邊 → 影片框比例與影片相同；數字區在右、按鈕最右，互不重疊
        for frame_size in ((480, 640), (1080, 1920), (1920, 1080)):
            for window_size in ((1600, 900), (1250, 780)):
                widget = self.show(frame_size, window_size)
                view, numbers, side = (w.geometry() for w in (widget.view, widget.summaryPanel.parentWidget(),
                                                              widget.loadBtn.parentWidget()))
                with self.subTest(frame=frame_size, window=window_size):
                    self.assertEqual((view.x(), view.y()), (0, 0))
                    self.assertAlmostEqual(view.width() / view.height(), frame_size[1] / frame_size[0], delta=0.02)
                    self.assertLess(view.right(), numbers.left())
                    self.assertLess(numbers.right(), side.left())
                    self.assertLessEqual(side.right(), widget.topRow.width())

    def test_curves_sit_between_top_row_and_slider(self):
        widget = self.show((480, 640), (1250, 780))
        ys = [w.mapTo(widget, w.rect().topLeft()).y()
              for w in (widget.topRow, widget.hipPlot, widget.kneePlot, widget.frameSlider)]
        self.assertEqual(ys, sorted(ys))


class GaitWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = pipeline.analyze_video(str(DEMO_VIDEO), write_outputs=False)

    def test_vidimu_without_imu_is_still_vidimu(self):
        video = DEMO_VIDEO.parents[1] / 'S01' / 'S01_A01_T01.mp4'
        if not video.is_file():
            self.skipTest('S01 影片不存在')
        trial = pipeline.find_vidimu_trial(str(video))
        self.assertIsNotNone(trial)
        self.assertFalse(trial.has_imu)
        self.assertEqual(pipeline.video_fps(str(video), trial), 30.0)

    def test_vidimu_trial_gets_imu(self):
        self.assertIsNotNone(self.analysis.trial)
        self.assertIsNotNone(self.analysis.imu_knee)
        self.assertGreater(self.analysis.agreement['knee_pearson_r'], 0.9)

    def test_renderer_letterboxes_wide_frames(self):
        canvas = self.analysis.renderer().render_frame(100, np.zeros((1080, 1920, 3), np.uint8))
        self.assertEqual(canvas.shape, (H, W, 3))

    def test_rendered_videos_are_rejected(self):
        widget = GaitTabControl(Mock())
        for name in ('S52_A01_T02_demo.mp4', 'S52_A01_T02_Sk26.mp4'):
            with patch('gait_widget.QFileDialog.getOpenFileName', return_value=(f'D:/x/{name}', '')), \
                    patch('gait_widget.QMessageBox.warning') as warning:
                widget.loadVideo()
            warning.assert_called_once()
            self.assertIsNone(widget.worker)

    def test_widget_starts_disabled(self):
        widget = GaitTabControl(Mock())
        self.assertFalse(widget.playBtn.isEnabled())
        self.assertFalse(widget.exportBtn.isEnabled())
        self.assertTrue(widget.loadBtn.isEnabled())

    def test_widget_shows_and_seeks_frames(self):
        widget = GaitTabControl(Mock())
        widget.onAnalyzed(self.analysis)
        widget.setLoaded(True)
        self.assertEqual(widget.frameSlider.maximum(), self.analysis.result.xy.shape[0] - 1)
        self.assertIsNotNone(widget.view.image)
        widget.frameSlider.setValue(300)
        self.assertEqual(widget.current_frame, 300)
        self.assertAlmostEqual(widget.hipPlot.cursor.value(), 10.0)
        self.assertIn('與 IMU 一致性', widget.infoLabel.text())
        self.assertNotIn('—', widget.summaryPanel.texts())
        # 曲線上拖曳／點擊會跳到該時間；跟隨播放時曲線視窗包含目前時間
        widget.hipPlot.seekRequested.emit(20.0)
        self.assertEqual(widget.current_frame, 600)
        x_min, x_max = widget.hipPlot.viewRange()[0]
        self.assertTrue(x_min <= 20.0 <= x_max)
        widget.shutdown()

    def test_follow_keeps_user_zoom(self):
        # 縮放曲線後拖動時間軸，只平移、不變回預設 6 秒寬；換新影片才回到預設寬度
        widget = GaitTabControl(Mock())
        widget.onAnalyzed(self.analysis)
        x_min, x_max = widget.hipPlot.viewRange()[0]
        self.assertAlmostEqual(x_max - x_min, 6.0, places=2)
        widget.hipPlot.setXRange(5.0, 17.0, padding=0)        # 使用者縮放成 12 秒寬
        widget.frameSlider.setValue(900)                      # 30 秒
        x_min, x_max = widget.hipPlot.viewRange()[0]
        self.assertAlmostEqual(x_max - x_min, 12.0, places=2)
        self.assertAlmostEqual(x_min, 30.0 - 8.0, places=2)   # 游標仍在前 2/3 處
        self.assertAlmostEqual(widget.kneePlot.viewRange()[0][1] - widget.kneePlot.viewRange()[0][0], 12.0, places=2)
        widget.onAnalyzed(self.analysis)
        x_min, x_max = widget.hipPlot.viewRange()[0]
        self.assertAlmostEqual(x_max - x_min, 6.0, places=2)
        widget.shutdown()

    def test_frame_view_paints_at_any_size(self):
        view = FrameView()
        frame = np.full((480, 640, 3), 128, np.uint8)
        view.setFrame(frame, self.analysis.result.xy[300], 'R', [('著地', 300.0, 400.0)])
        for size in ((400, 300), (1600, 900)):
            view.resize(*size)
            image = view.grab().toImage()
            self.assertEqual((image.width(), image.height()), size)
            # 影像置中、上下或左右留邊：中心是影像灰色，角落是背景深灰
            self.assertEqual(image.pixelColor(size[0] // 2, size[1] // 2).red(), 128)

    def test_analysis_thread_reads_cache_without_model(self):
        thread = AnalysisThread(None, str(DEMO_VIDEO))
        received = []
        thread.succeeded.connect(received.append)
        thread.start()
        thread.wait(60000)
        APP.processEvents()
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].name, 'S52_A01_T02')


if __name__ == '__main__':
    unittest.main()

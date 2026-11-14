"""Run with: conda run -n Gait python -m unittest discover -s tests -v

需要 VIDIMU 資料與 S52_A01_T02 關節點快取的測試，在資料不存在時會略過。
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
_yolo_config = tempfile.TemporaryDirectory()
os.environ['YOLO_CONFIG_DIR'] = _yolo_config.name
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Src' / 'UI_Control'))
sys.path.insert(0, str(ROOT / 'Src'))

import numpy as np
from PyQt5.QtWidgets import QApplication

from gait_analysis import gait, pipeline
from gait_analysis.render_demo import DemoRenderer, H, W
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


@unittest.skipUnless(HAS_DEMO, 'VIDIMU S52_A01_T02 資料或關節點快取不存在')
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
        self.assertIsNotNone(widget.pixmap)
        widget.frameSlider.setValue(300)
        self.assertEqual(widget.current_frame, 300)
        self.assertIn('與 IMU 一致性', widget.infoLabel.text())
        widget.shutdown()

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

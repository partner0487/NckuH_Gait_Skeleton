"""步態分頁的多人選擇、基本輸出檔與批量清單。不需要模型或 VIDIMU 資料。

Run with: conda run -n Gait python -m unittest discover -s tests -v
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
_yolo_config = tempfile.TemporaryDirectory()
os.environ['YOLO_CONFIG_DIR'] = _yolo_config.name
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Src' / 'UI_Control'))
sys.path.insert(0, str(ROOT / 'Src'))

import cv2
import numpy as np
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtWidgets import QApplication

from gait_analysis import pipeline, vidimu
from gait_views import FrameView
from gait_widget import GaitTabControl

APP = QApplication.instance() or QApplication([])
T = 60


def fake_cache(folder, video_path):
    """兩個人：ID 1 在左（較大、為自動目標）、ID 2 在右；第 10 格起 ID 2 才出現。"""
    kpts = np.zeros((T, 26, 3))
    kpts[:, :, :2] = 100
    kpts[:, :, 2] = 0.9
    boxes_1 = np.tile([50.0, 20.0, 150.0, 220.0], (T, 1))
    rows = {'frame': [], 'id': [], 'bbox': [], 'kpts': []}
    for t in range(T):
        people = [(1, boxes_1[t], kpts[t])]
        if t >= 10:
            other = kpts[t].copy()
            other[:, 0] += 200
            people.append((2, np.array([260.0, 40.0, 330.0, 200.0]), other))
        for person_id, box, k in people:
            rows['frame'].append(t)
            rows['id'].append(person_id)
            rows['bbox'].append(box)
            rows['kpts'].append(k)
    path = pipeline.cache_path(video_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, format=2, keypoints=kpts, bboxes=boxes_1, image_size=np.array([400, 240]),
                        video_path=video_path, video_bytes=os.path.getsize(video_path),
                        person_frame=np.array(rows['frame']), person_id=np.array(rows['id']),
                        person_bbox=np.array(rows['bbox']), person_kpts=np.array(rows['kpts']))
    return path


def write_video(path, n=T):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (400, 240))
    for _ in range(n):
        writer.write(np.zeros((240, 400, 3), np.uint8))
    writer.release()


class OutputRootMixin:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        out = os.path.join(self.tmp.name, 'Record')
        patches = [patch.object(pipeline, 'CACHE_DIR', os.path.join(out, '_cache', 'keypoints')),
                   patch.object(pipeline, 'TARGETS_PATH', os.path.join(out, '_cache', 'targets.json')),
                   patch.object(vidimu, 'OUTPUT_ROOT', out)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)
        self.out = out
        self.video = os.path.join(self.tmp.name, 'walk.mp4')
        write_video(self.video)
        fake_cache(self.tmp.name, self.video)


class TargetSelectionTests(OutputRootMixin, unittest.TestCase):
    def test_series_auto_and_selected(self):
        data = pipeline.load_cache(pipeline.cache_path(self.video))
        self.assertEqual(pipeline.auto_person_id(data), 1)
        kpts, boxes = pipeline.target_series(data, 2)
        self.assertTrue(np.isnan(kpts[:10]).all())          # ID 2 第 10 格才出現
        self.assertEqual(kpts[10, 0, 0], 300)
        np.testing.assert_array_equal(boxes[20], [260, 40, 330, 200])

    def test_selection_is_remembered_and_outputs_written(self):
        auto = pipeline.analyze_video(self.video)
        self.assertIsNone(auto.person_id)
        self.assertEqual(auto.auto_id, 1)
        folder = os.path.join(self.out, 'walk')
        for suffix in ('.mp4', '.json', '.csv', '_Sk26.mp4', '_keypoints.csv', '_curves.csv'):
            self.assertTrue(os.path.isfile(os.path.join(folder, f'walk{suffix}')), suffix)

        chosen = pipeline.analyze_video(self.video, person_id=2)
        self.assertEqual(chosen.person_id, 2)
        self.assertEqual(pipeline.analyze_video(self.video).person_id, 2)      # 下次載入沿用
        self.assertEqual(pipeline.analyze_video(self.video, person_id=None).person_id, None)
        self.assertIsNone(pipeline.load_target('walk'))
        # 不存在的 ID 改回自動
        self.assertIsNone(pipeline.analyze_video(self.video, person_id=99).person_id)

    def test_skeleton_json_format(self):
        pipeline.analyze_video(self.video)
        records = json.load(open(os.path.join(self.out, 'walk', 'walk.json'), encoding='utf-8'))
        self.assertEqual(len(records), T + (T - 10))
        self.assertEqual(set(records[0]), {'person_id', 'bbox', 'keypoints', 'frame_number'})
        self.assertEqual(len(records[0]['keypoints'][0]), 4)
        self.assertEqual(sorted({r['person_id'] for r in records}), [1, 2])

    def test_record_copy_reads_same_cache(self):
        pipeline.analyze_video(self.video, person_id=2)
        copy = os.path.join(self.out, 'walk', 'walk.mp4')
        again = pipeline.analyze_video(copy)        # 沒有 model：要讀得到快取才不會丟例外
        self.assertEqual(again.person_id, 2)

    def test_skeleton_video_redrawn_only_when_target_changes(self):
        with patch('gait_analysis.render_demo.render_skeleton_video', return_value=True) as render:
            pipeline.analyze_video(self.video)
            open(os.path.join(self.out, 'walk', 'walk_Sk26.mp4'), 'wb').close()
            pipeline.analyze_video(self.video)
            self.assertEqual(render.call_count, 1)
            pipeline.analyze_video(self.video, person_id=2)
            self.assertEqual(render.call_count, 2)

    def test_locked_output_is_a_warning_not_a_failure(self):
        error = PermissionError(13, 'denied', 'walk_Sk26.mp4.part.mp4', None, 'walk_Sk26.mp4')
        with patch('gait_analysis.render_demo.render_skeleton_video', side_effect=error):
            analysis = pipeline.analyze_video(self.video, person_id=2)
        self.assertIsNotNone(analysis)
        self.assertIn('walk_Sk26.mp4', analysis.warnings[0])
        # 上次沒畫成，下次載入會補畫
        with patch('gait_analysis.render_demo.render_skeleton_video', return_value=True) as render:
            pipeline.analyze_video(self.video)
        render.assert_called_once()

    def test_render_skeleton_video_keeps_fps_and_frames(self):
        analysis = pipeline.analyze_video(self.video, write_outputs=False)
        out = os.path.join(self.tmp.name, 'sk.mp4')
        from gait_analysis.render_demo import render_skeleton_video
        self.assertTrue(render_skeleton_video(analysis, out))
        cap = cv2.VideoCapture(out)
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), T)
        self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 25.0)
        cap.release()
        self.assertFalse(os.path.exists(out + '.part.mp4'))


class BatchListTests(unittest.TestCase):
    def test_gait_videos_takes_every_a01_trial_of_each_subject(self):
        from gait_analysis.vidimu import gait_videos
        with tempfile.TemporaryDirectory() as root:
            files = {'S10': ('S10_A01_T10', 'S10_A01_T02', 'S10_A01_T02_Npose', 'S10_A02_T01'),
                     'S2': ('S2_A01_T03',), 'S7': ('S7_A02_T01',), 'Sx': ('Sx_A01_T01',)}
            for subject, names in files.items():
                os.mkdir(os.path.join(root, subject))
                for name in names:
                    open(os.path.join(root, subject, name + '.mp4'), 'wb').close()
            open(os.path.join(root, 'S3'), 'wb').close()                        # 不是資料夾
            found, missing = gait_videos(root)
        self.assertEqual([os.path.basename(p) for p in found], ['S2_A01_T03.mp4', 'S10_A01_T02.mp4', 'S10_A01_T10.mp4'])
        self.assertEqual(missing, ['S7'])


class ClickTests(OutputRootMixin, unittest.TestCase):
    def test_frame_view_click_maps_to_image_coordinates(self):
        view = FrameView()
        view.resize(800, 480)
        view.setFrame(np.zeros((240, 400, 3), np.uint8), np.full((26, 2), np.nan), None)
        got = []
        view.imageClicked.connect(lambda x, y: got.append((x, y)))
        rect = view.targetRect()
        from PyQt5.QtTest import QTest
        QTest.mouseClick(view, Qt.LeftButton, pos=QPoint(int(rect.x() + 300 * 2), int(rect.y() + 100 * 2)))
        self.assertAlmostEqual(got[0][0], 300, delta=1)
        self.assertAlmostEqual(got[0][1], 100, delta=1)

    def test_click_on_other_person_selects_them(self):
        widget = GaitTabControl(Mock())
        widget.onAnalyzed(pipeline.analyze_video(self.video))
        widget.frameSlider.setValue(20)
        self.assertEqual(len(widget.view.persons), 2)
        widget.selectTarget = Mock()
        widget.onImageClicked(10, 10)                   # 沒點到人
        widget.selectTarget.assert_not_called()
        widget.onImageClicked(300, 100)                 # ID 2 的框
        widget.selectTarget.assert_called_once_with(2)
        widget.shutdown()

    def test_auto_button_only_after_manual_selection(self):
        widget = GaitTabControl(Mock())
        widget.onAnalyzed(pipeline.analyze_video(self.video))
        self.assertFalse(widget.autoTargetBtn.isVisibleTo(widget))
        self.assertIn('自動', widget.targetLabel.text())
        widget.onAnalyzed(pipeline.analyze_video(self.video, person_id=2))
        self.assertTrue(widget.autoTargetBtn.isVisibleTo(widget))
        self.assertIn('ID 2', widget.targetLabel.text())
        widget.shutdown()


if __name__ == '__main__':
    unittest.main()

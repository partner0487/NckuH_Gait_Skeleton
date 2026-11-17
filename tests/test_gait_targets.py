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
        with open(os.path.join(self.out, 'walk', 'walk.json'), encoding='utf-8') as f:
            records = json.load(f)
        self.assertEqual(len(records), T + (T - 10))
        self.assertEqual(set(records[0]), {'person_id', 'bbox', 'keypoints', 'frame_number'})
        self.assertEqual(len(records[0]['keypoints'][0]), 4)
        self.assertEqual(sorted({r['person_id'] for r in records}), [1, 2])

    def test_persons_at_marks_only_the_target(self):
        manual = pipeline.analyze_video(self.video, person_id=2, write_outputs=False)
        self.assertEqual({pid: t for pid, _, t in manual.persons_at(20)}, {1: False, 2: True})
        auto = pipeline.analyze_video(self.video, person_id=None, write_outputs=False)
        self.assertEqual(sum(t for _, _, t in auto.persons_at(20)), 1)
        self.assertEqual([t for _, _, t in auto.persons_at(5)], [True])       # 只有一人的格

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


class RunDemoTests(unittest.TestCase):
    def test_trial_without_valid_strides_is_skipped(self):
        from gait_analysis import run_demo
        trial = vidimu.Trial('S99', 'A01', 'T01')
        empty = SimpleNamespace(agreement=None, values={})
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(vidimu, 'find_trials', return_value=[trial]), \
                patch.object(vidimu, 'SUMMARY_DIR', tmp), \
                patch.object(run_demo, 'analyze_video', return_value=empty), \
                patch.object(sys, 'argv', ['run_demo.py', '--no-video']), \
                patch('builtins.print') as printed:
            run_demo.main()                     # 不能因為 agreement 是 None 而中斷
            self.assertEqual(os.listdir(tmp), [])
        lines = ' '.join(str(c.args[0]) for c in printed.call_args_list)
        self.assertIn('S99_A01_T01: 沒有有效步態週期', lines)


class DemoExportTests(OutputRootMixin, unittest.TestCase):
    """匯出展示影片：先寫暫存檔，取消或失敗都不留下半個檔案、不動到舊檔。"""

    def setUp(self):
        super().setUp()
        from gait_analysis.render_demo import render
        self.render = render
        self.analysis = pipeline.analyze_video(self.video, write_outputs=False)
        self.demo = os.path.join(self.tmp.name, 'walk_demo.mp4')
        with open(self.demo, 'wb') as f:
            f.write(b'old')

    def old_kept(self):
        self.assertFalse(os.path.exists(self.demo + '.part.mp4'))
        with open(self.demo, 'rb') as f:
            return f.read() == b'old'

    def test_finished_export_replaces_old_file(self):
        self.assertTrue(self.render(self.analysis.renderer(), self.video, self.demo))
        self.assertFalse(self.old_kept())
        cap = cv2.VideoCapture(self.demo)
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), T)
        cap.release()

    def test_cancel_keeps_old_file(self):
        self.assertFalse(self.render(self.analysis.renderer(), self.video, self.demo, should_stop=lambda: True))
        self.assertTrue(self.old_kept())

    def test_unwritable_folder_raises(self):
        with self.assertRaises(OSError):
            self.render(self.analysis.renderer(), self.video, os.path.join(self.tmp.name, 'missing', 'x.mp4'))

    def test_old_file_open_elsewhere_raises_and_is_kept(self):
        with open(self.demo, 'rb'):              # Windows 上開著的檔案不能被取代，模擬播放器占用
            with self.assertRaises(PermissionError):
                self.render(self.analysis.renderer(), self.video, self.demo)
        self.assertTrue(self.old_kept())


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


def imu_row(trial, knee_r, rmse=3.0):
    return {'subject': trial[:3], 'trial': trial, 'near_strides': 8, 'knee_pearson_r': knee_r,
            'knee_rmse_centered': rmse, 'hip_pearson_r': 0.9, 'hip_rmse_centered': 4.0, 'cadence': 100.0}


class ImuAnalysisTests(unittest.TestCase):
    """「分析含 IMU 步態影片」：結果表、彙總表、取消時不覆寫。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = patch.object(vidimu, 'SUMMARY_DIR', self.tmp.name)
        p.start()
        self.addCleanup(p.stop)
        self.widget = GaitTabControl(None)
        self.videos = ['D:/v/S40_A01_T01.mp4', 'D:/v/S41_A01_T02.mp4', 'D:/v/S42_A01_T01.mp4']

    def test_agreement_row(self):
        trial = vidimu.Trial('S40', 'A01', 'T01')
        no_imu = SimpleNamespace(agreement=None, trial=trial, values={})
        self.assertIsNone(pipeline.agreement_row(no_imu))
        ok = SimpleNamespace(agreement={'trial': trial.name, 'knee_pearson_r': 0.9}, trial=trial,
                             values={'cadence': 100.0})
        self.assertEqual(pipeline.agreement_row(ok),
                         {'subject': 'S40', 'trial': 'S40_A01_T01', 'knee_pearson_r': 0.9, 'cadence': 100.0})

    def test_batch_thread_returns_imu_rows(self):
        from gait_widget import BatchThread
        trial = vidimu.Trial('S40', 'A01', 'T01')
        done = SimpleNamespace(agreement={'trial': trial.name}, trial=trial, values={})
        worker = BatchThread(None, self.videos[:2])
        received = []
        worker.succeeded.connect(received.append)
        with patch('gait_widget.analyze_video', side_effect=[done, RuntimeError('壞檔')]):
            worker.run()
        (results,) = received
        self.assertEqual(results[0], ('S40_A01_T01.mp4', None, {'subject': 'S40', 'trial': 'S40_A01_T01'}))
        self.assertEqual(results[1], ('S41_A01_T02.mp4', 'RuntimeError: 壞檔', None))

    def run_done(self, results):
        with patch('gait_widget.ImuSummaryDialog') as dialog:
            dialog.return_value.exec_.return_value = 0
            self.widget.onImuDone(results, self.videos)
        rows, paths, notes, _ = dialog.call_args.args
        return rows, paths, notes

    def test_complete_run_writes_summary_and_lists_problems(self):
        results = [('S40_A01_T01.mp4', None, imu_row('S40_A01_T01', 0.95)),
                   ('S41_A01_T02.mp4', None, None),
                   ('S42_A01_T01.mp4', 'OSError: x', None)]
        rows, paths, notes = self.run_done(results)
        self.assertEqual([r['trial'] for r in rows], ['S40_A01_T01'])
        self.assertEqual(paths['S42_A01_T01'], 'D:/v/S42_A01_T01.mp4')
        self.assertTrue(any('S42_A01_T01.mp4：OSError' in n for n in notes))
        self.assertTrue(any('S41_A01_T02.mp4：沒有有效步態週期' in n for n in notes))
        table = os.path.join(self.tmp.name, 'agreement_A01.csv')
        self.assertTrue(os.path.isfile(table))
        import pandas as pd
        self.assertEqual(pd.read_csv(table)['trial'].tolist(), ['S40_A01_T01'])

    def test_cancelled_run_keeps_existing_summary(self):
        table = os.path.join(self.tmp.name, 'agreement_A01.csv')
        with open(table, 'w') as f:
            f.write('old')
        _, _, notes = self.run_done([('S40_A01_T01.mp4', None, imu_row('S40_A01_T01', 0.95))])
        with open(table) as f:
            self.assertEqual(f.read(), 'old')
        self.assertTrue(any('已取消' in n for n in notes))

    def test_dialog_sorts_numbers_and_opens_on_double_click(self):
        from gait_views import ImuSummaryDialog
        rows = [imu_row('S40_A01_T01', 0.95), imu_row('S41_A01_T02', float('nan')), imu_row('S42_A01_T01', 0.6)]
        paths = {r['trial']: f'D:/v/{r["trial"]}.mp4' for r in rows}
        dialog = ImuSummaryDialog(rows[::-1], paths)
        self.assertEqual(dialog.table.item(0, 0).text(), 'S40_A01_T01')          # 預設依試驗名稱遞增
        dialog.table.sortItems(2, Qt.AscendingOrder)
        self.assertEqual([dialog.table.item(i, 0).text() for i in range(3)],
                         ['S42_A01_T01', 'S40_A01_T01', 'S41_A01_T02'])       # 沒有數值的排最後
        self.assertEqual(dialog.table.item(2, 2).text(), '—')
        dialog.table.cellDoubleClicked.emit(1, 3)
        self.assertEqual(dialog.selected_path, 'D:/v/S40_A01_T01.mp4')

    def test_imu_button_sits_between_load_and_batch(self):
        layout = self.widget.loadBtn.parentWidget().layout()
        order = [layout.indexOf(w) for w in (self.widget.loadBtn, self.widget.imuBtn, self.widget.batchBtn)]
        self.assertEqual(order, sorted(order))
        self.assertEqual(order[1] - order[0], 1)

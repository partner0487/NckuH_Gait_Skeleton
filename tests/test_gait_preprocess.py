"""腿部跳點剔除（兩腳交錯時腳部點被接到另一隻腳）。不需要模型或資料集。

Run with: conda run -n Gait python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Src'))

from gait_analysis import gait  # noqa: E402

T = 120
LEFT, RIGHT = gait.LEG_KPTS['L'], gait.LEG_KPTS['R']


def walking(n=T):
    """側面往右走：髖等速前進，兩腳膝、踝、腳以相反相位前後擺動（腿長約 100 像素）。"""
    t = np.arange(n)
    xy = np.full((n, 26, 2), np.nan)
    hip_x = 100 + 3.0 * t
    for side, phase in (('L', 0.0), ('R', np.pi)):
        swing = 25 * np.sin(2 * np.pi * t / 30 + phase)
        hip, knee, ankle = (gait.side_index(p, side) for p in ('hip', 'knee', 'ankle'))
        xy[:, hip] = np.stack([hip_x, np.full(n, 200.0)], 1)
        xy[:, knee] = np.stack([hip_x + 0.5 * swing, np.full(n, 250.0)], 1)
        xy[:, ankle] = np.stack([hip_x + swing, np.full(n, 300.0)], 1)
        for k, dx in zip(gait.LEG_KPTS[side][2:], (14.0, 10.0, -6.0)):       # 大趾、小趾、腳跟
            xy[:, k] = xy[:, ankle] + (dx, 5.0)
    return xy


def keypoints(xy):
    return np.concatenate([np.nan_to_num(xy), np.where(np.isnan(xy[:, :, :1]), 0.0, 0.9)], axis=2)


def config():
    cfg = gait.load_config()
    cfg['fps'] = 30.0
    return cfg


class LegOutlierTests(unittest.TestCase):
    def test_clean_walking_is_untouched(self):
        xy = walking()
        cleaned = gait.reject_leg_outliers(xy, 7, 0.12)
        self.assertEqual(np.isnan(cleaned).sum(), np.isnan(xy).sum())

    def test_foot_jumping_to_other_leg_removes_whole_leg_with_margin(self):
        xy = walking()
        truth = xy.copy()
        # 第 67、68 格（兩腳分最開）：右腳的踝與腳部點被接到左腳上（膝蓋沒跳）
        for frame in (67, 68):
            xy[frame, list(RIGHT[1:])] = truth[frame, list(LEFT[1:])]
        cleaned = gait.reject_leg_outliers(xy, 7, 0.12, dilate=1)
        removed = np.flatnonzero(np.isnan(cleaned[:, RIGHT[0], 0]))
        self.assertTrue({66, 67, 68, 69} <= set(removed))                 # 整隻腿（含膝），前後各多 1 格
        self.assertLessEqual(len(removed), 8)                              # 只剔除跳點附近
        self.assertTrue(np.isnan(cleaned[66:70][:, list(RIGHT)]).all())
        self.assertFalse(np.isnan(cleaned[:, list(LEFT)]).any())           # 另一隻腳不受影響

        # 補值後腳踝回到原本軌跡，而不是另一隻腳的位置
        out = gait.preprocess(keypoints(xy), config())
        ankle = gait.side_index('ankle', 'R')
        error = np.abs(out[65:71, ankle, 0] - truth[65:71, ankle, 0]).max()
        jump = np.abs(truth[67, gait.side_index('ankle', 'L'), 0] - truth[67, ankle, 0])
        self.assertLess(error, 0.2 * jump)

    def test_rejected_gap_gets_longer_fill_limit_than_missing_data(self):
        cfg = config()
        xy = walking()
        ankle = gait.side_index('ankle', 'R')
        missing = xy.copy()
        missing[40:47, ankle] = np.nan                                     # 低信心缺 7 格：超過 5 格不補
        out = gait.preprocess(keypoints(missing), cfg)
        self.assertTrue(np.isnan(out[43, ankle]).all())
        # 跳點 5 格（剔除後含前後各 1 格共 7 格）：在 9 格上限內，會補值
        jumped = xy.copy()
        jumped[40:45, ankle] = xy[40:45, gait.side_index('ankle', 'L')]
        out = gait.preprocess(keypoints(jumped), cfg)
        self.assertTrue(np.isfinite(out[39:46, ankle]).all())

    def test_can_be_turned_off(self):
        cfg = config()
        cfg['preprocess']['outlier_window'] = 0
        xy = walking()
        ankle = gait.side_index('ankle', 'R')
        xy[60, ankle] = xy[60, gait.side_index('ankle', 'L')]
        out = gait.preprocess(keypoints(xy), cfg)
        self.assertTrue(np.isfinite(out[60, ankle]).all())                # 沒剔除：跳點保留並被濾波


if __name__ == '__main__':
    unittest.main()

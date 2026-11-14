"""單支影片的完整步態分析流程（GUI 與 run_demo.py 共用）。

萃取關節點（有快取就沿用）→ 計算指標 → 若是 VIDIMU 有 IMU 的試驗則與 IMU 比對 → 寫出 CSV。
"""
import os
import re
from dataclasses import dataclass

import cv2
import numpy as np

from gait_analysis import gait, imu_compare, vidimu
from gait_analysis.render_demo import DemoRenderer, near_series
from gait_analysis.style import ACTIVITY_NAMES, VIDIMU_FOOTER

CACHE_DIR = vidimu.CACHE_DIR
VIDIMU_NAME = re.compile(r'^(S\d+)_(A\d+)_(T\d+)$')


def find_vidimu_trial(video_path):
    """VIDIMU 影片回傳 Trial（不一定有 IMU，看 trial.has_imu），否則 None。

    只看檔名不夠（醫院收案資料可能沿用同樣命名），所以還要求影片就是資料集內的檔案，
    或資料集中有同名試驗的 IMU（例如從 Db/Record 載入重新編碼過的 VIDIMU 影片）。
    """
    match = VIDIMU_NAME.match(os.path.splitext(os.path.basename(video_path))[0])
    if not match:
        return None
    trial = vidimu.Trial(*match.groups())
    return trial if trial.has_imu or _same_file(trial.video_path, video_path) else None


def video_fps(video_path, trial=None):
    """VIDIMU 固定 30 fps（metadata 不可靠）；其他影片讀 metadata，讀不到時用 30。"""
    if trial is not None:
        return vidimu.VIDEO_FPS
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 1 else 30.0


def cache_path(video_path):
    return os.path.join(CACHE_DIR, f'{os.path.splitext(os.path.basename(video_path))[0]}.npz')


def _same_file(a, b):
    return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


def load_or_extract(video_path, model=None, progress=None, should_stop=None):
    """回傳 npz 資料；快取不存在或來自別的影片時重新萃取（需要 model）。中止時回傳 None。"""
    path = cache_path(video_path)
    if os.path.isfile(path):
        data = np.load(path)
        if _same_file(data['video_path'], video_path):
            return data
    if model is None:
        raise FileNotFoundError(f'{os.path.basename(video_path)} 尚未萃取關節點，請先執行 run_extract.py')
    from gait_analysis.extract_keypoints import extract_video
    if extract_video(model, video_path, path, log_every=0, progress=progress, should_stop=should_stop) is None:
        return None
    return np.load(path)


@dataclass
class GaitAnalysis:
    video_path: str
    name: str
    trial: object               # VIDIMU 影片為 vidimu.Trial（不一定有 IMU），其他影片為 None
    result: gait.TrialResult
    curves: object
    agreement: dict             # 與 IMU 的一致性；沒有 IMU 時為 None
    imu_knee: np.ndarray        # 逐格近側腿 IMU 膝角；沒有 IMU 時為 None
    values: dict                # 數字面板
    bboxes: np.ndarray
    out_dir: str

    @property
    def header(self):
        if self.trial is not None:
            activity = ACTIVITY_NAMES.get(self.trial.activity, self.trial.activity)
            return f'受試者 {self.trial.subject}｜{self.trial.activity} {activity}'
        return self.name

    def renderer(self):
        return DemoRenderer(self.result, self.values, imu_knee=self.imu_knee, bboxes=self.bboxes,
                            header_right=self.header, footer=VIDIMU_FOOTER if self.trial is not None else '')


def write_trial_csvs(result, curves, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, result.name)
    gait.keypoints_table(result).to_csv(f'{prefix}_keypoints.csv', index=False, float_format='%.3f')
    result.events.to_csv(f'{prefix}_events.csv', index=False, float_format='%.3f')
    result.strides.to_csv(f'{prefix}_strides.csv', index=False, float_format='%.3f')
    curves.to_csv(f'{prefix}_curves.csv', index=False, float_format='%.3f')


def analyze_video(video_path, model=None, progress=None, should_stop=None, write_outputs=True):
    """progress(done, total) 只在萃取關節點時回報。中止時回傳 None。"""
    name = os.path.splitext(os.path.basename(video_path))[0]
    trial = find_vidimu_trial(video_path)
    data = load_or_extract(video_path, model, progress, should_stop)
    if data is None:
        return None
    cfg = gait.load_config()
    cfg['fps'] = video_fps(video_path, trial)
    result = gait.analyze(name, data['keypoints'], cfg)
    curves, agreement, imu_knee = result.curves, None, None
    if trial is not None and trial.has_imu and not result.strides.empty:
        curves, agreement, imu = imu_compare.compare_with_imu(result, trial, cfg)
        imu_knee = near_series(result, {s: imu[imu_compare.imu_column('knee', s)] for s in ('L', 'R')})
    out_dir = os.path.join(vidimu.OUTPUT_ROOT, name)
    if write_outputs:
        write_trial_csvs(result, curves, out_dir)
    return GaitAnalysis(video_path, name, trial, result, curves, agreement, imu_knee,
                        gait.trial_summary(result), data['bboxes'], out_dir)

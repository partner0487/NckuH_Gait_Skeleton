"""VIDIMU 資料存取：試驗清單、影片與 IMU 的時間偏移、IMU 關節角度、身體量測。"""
import csv
import glob
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
VIDIMU_ROOT = os.path.join(PROJECT_ROOT, 'Db_NckuH_Gait')
VIDEO_DIR = os.path.join(VIDIMU_ROOT, 'Db_VideoSmallSize', 'Db_VideoSOriginal')
IMU_DIR = os.path.join(VIDIMU_ROOT, 'Database', 'VideoAndImus')
SYNC_INFO = os.path.join(VIDIMU_ROOT, 'Analysis', 'VideoAndImuSync', 'infoToSync.csv')
# 與 GUI 共用 Db/Record：一支影片一個資料夾；底線開頭的是共用資料夾，不是影片
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'Db', 'Record')
CACHE_DIR = os.path.join(OUTPUT_ROOT, '_cache', 'keypoints')
SUMMARY_DIR = os.path.join(OUTPUT_ROOT, '_summary')

VIDEO_FPS = 30.0  # VIDIMU 標稱 30 fps；mp4 metadata 會寫 29/31，但幀數 ÷ 30 與 IMU 時長吻合


@dataclass
class Trial:
    subject: str
    activity: str
    trial: str

    @property
    def name(self):
        return f'{self.subject}_{self.activity}_{self.trial}'

    @property
    def video_path(self):
        return os.path.join(VIDEO_DIR, self.subject, f'{self.name}.mp4')

    @property
    def mot_path(self):
        return os.path.join(IMU_DIR, self.subject, f'ik_{self.name}.mot')

    @property
    def has_imu(self):
        return os.path.isfile(self.mot_path)


def imu_subjects():
    return sorted(name for name in os.listdir(IMU_DIR) if os.path.isdir(os.path.join(IMU_DIR, name)))


def find_trials(activity='A01', subjects=None):
    """有 IMU 的受試者中，影片與 .mot 都存在的試驗。"""
    trials = []
    for subject in subjects or imu_subjects():
        for mot in sorted(glob.glob(os.path.join(IMU_DIR, subject, f'ik_{subject}_{activity}_T*.mot'))):
            if mot.endswith('_orientationErrors.sto'):
                continue
            _, _, trial_id = os.path.basename(mot)[3:-4].split('_')
            trial = Trial(subject, activity, trial_id)
            if os.path.isfile(trial.video_path):
                trials.append(trial)
    return trials


def gait_videos(root=VIDEO_DIR, activity='A01'):
    """root 底下每位受試者（S01、S02…）的步態影片（A01 的每個試驗），依受試者、試驗排序。

    VIDIMU 每人的步態多半只有一趟，編號不固定（T01／T02／T03），所以取全部試驗；
    略過 _Npose 等衍生檔。回傳 (影片路徑, 沒有步態影片的受試者)。
    """
    subjects = sorted((name for name in os.listdir(root)
                       if re.fullmatch(r'S\d+', name) and os.path.isdir(os.path.join(root, name))),
                      key=lambda name: int(name[1:]))
    found, missing = [], []
    for subject in subjects:
        pattern = re.compile(rf'{subject}_{activity}_T(\d+)\.mp4', re.IGNORECASE)
        trials = sorted((int(m.group(1)), name) for name in os.listdir(os.path.join(root, subject))
                        for m in [pattern.fullmatch(name)] if m)
        if trials:
            found += [os.path.join(root, subject, name) for _, name in trials]
        else:
            missing.append(subject)
    return found, missing


def sync_offset(trial: Trial):
    """回傳 IMU 時間要加多少秒才對到影片時間（t_video = t_imu + offset）。

    infoToSync.csv 的 CutFrames 以 30 fps 影片幀計：類型是 mp4/csv 代表影片開頭多出
    CutFrames 格（影片較晚對到 IMU 起點），類型是 raw/mot 代表 IMU 開頭多出對應時間。
    沒列在表中的試驗原本就已對齊。
    """
    with open(SYNC_INFO, newline='') as f:
        for row in csv.DictReader(f):
            if (row['Subject'], row['Activity'], row['Trial']) != (trial.subject, trial.activity, trial.trial):
                continue
            cut = int(row['CutFrames']) / VIDEO_FPS
            return cut if row['Type'] in ('mp4', 'csv') else -cut
    return 0.0


def load_mot(trial: Trial):
    """OpenSim IK 結果（50 Hz，度）。第一列是 t=0 的全零初始姿勢，略過。"""
    with open(trial.mot_path) as f:
        for header_lines, line in enumerate(f, 1):
            if line.strip() == 'endheader':
                break
    df = pd.read_csv(trial.mot_path, sep='\t', skiprows=header_lines)
    df = df.iloc[1:].drop_duplicates(subset='time').reset_index(drop=True)
    return df


def imu_on_video_frames(trial: Trial, n_frames: int, columns, extra_lag_frames=0):
    """把 IMU 欄位內插到影片每一格的時間上；超出 IMU 範圍的格為 NaN。

    extra_lag_frames 可為常數或逐格陣列：第 f 格取 IMU 在 t_f − offset − lag_f / fps 的值。
    """
    mot = load_mot(trial)
    t_query = (np.arange(n_frames) - extra_lag_frames) / VIDEO_FPS - sync_offset(trial)
    return {col: np.interp(t_query, mot['time'].to_numpy(), mot[col].to_numpy(), left=np.nan, right=np.nan)
            for col in columns}

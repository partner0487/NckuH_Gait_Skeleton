"""單支影片的完整步態分析流程（GUI、批量分析與 run_demo.py 共用）。

萃取關節點（有快取就沿用）→ 計算指標 → 若是 VIDIMU 有 IMU 的試驗則與 IMU 比對 → 寫出檔案。

每支影片在 Db/Record/{名稱}/ 寫出同一組基本檔案，有 IMU 時只是 _curves.csv 多 IMU 欄位：
    {名稱}.mp4                 原始影片（之後可直接從 Db/Record 載入）
    {名稱}.json、{名稱}.csv     所有人的骨架（逐人逐格／逐關節點）
    {名稱}_Sk26.mp4            疊上目標人物骨架的影片
    _keypoints/_events/_strides/_curves.csv   步態分析結果
"""
import csv
import json
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

from gait_analysis import gait, imu_compare, vidimu
from gait_analysis.render_demo import DemoRenderer, near_series
from gait_analysis.style import ACTIVITY_NAMES, VIDIMU_FOOTER

CACHE_DIR = vidimu.CACHE_DIR
TARGETS_PATH = os.path.join(os.path.dirname(CACHE_DIR), 'targets.json')     # 每支影片手動點選的目標人物
VIDIMU_NAME = re.compile(r'^(S\d+)_(A\d+)_(T\d+)$')
VIDEO_EXTS = ('.mp4', '.avi', '.mov')
DERIVED_SUFFIXES = ('_Sk26', '_demo')         # 已疊上骨架／圖表的輸出影片，不能再拿來分析
SAVED = 'saved'                                # analyze_video 的 person_id：沿用上次點選的目標
KEYPOINT_NAMES_ZH = ['鼻子', '左眼', '右眼', '左耳', '右耳', '左肩', '右肩', '左肘', '右肘', '左腕', '右腕',
                     '左髖', '右髖', '左膝', '右膝', '左踝', '右踝', '頭部', '頸部', '臀部', '左大腳趾',
                     '右大腳趾', '左小腳趾', '右小腳趾', '左腳跟', '右腳跟']


def natural_key(name):
    """S01_A01_T2 排在 S01_A01_T10 前面。"""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', name)]


def is_derived_video(path):
    return os.path.splitext(os.path.basename(path))[0].endswith(DERIVED_SUFFIXES)


def list_videos(folder):
    """資料夾中可分析的影片（略過 _Sk26、_demo），依自然順序排列。"""
    names = [n for n in os.listdir(folder)
             if n.lower().endswith(VIDEO_EXTS) and not is_derived_video(n)]
    return [os.path.join(folder, n) for n in sorted(names, key=natural_key)]


def find_vidimu_trial(video_path):
    """VIDIMU 影片回傳 Trial（不一定有 IMU，看 trial.has_imu），否則 None。

    只看檔名不夠（醫院收案資料可能沿用同樣命名），所以還要求影片就是資料集內的檔案
    （或 Db/Record 裡複製的同一支），或資料集中有同名試驗的 IMU。
    """
    match = VIDIMU_NAME.match(os.path.splitext(os.path.basename(video_path))[0])
    if not match:
        return None
    trial = vidimu.Trial(*match.groups())
    return trial if trial.has_imu or _same_video(trial.video_path, video_path) else None


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


def _same_video(a, b):
    """同一支影片：同一路徑，或檔名與大小都相同（例如 Db/Record 裡複製的原始影片）。"""
    a, b = str(a), str(b)
    if _same_file(a, b):
        return True
    return (os.path.basename(a) == os.path.basename(b) and os.path.isfile(a) and os.path.isfile(b)
            and os.path.getsize(a) == os.path.getsize(b))


def load_cache(path):
    """整個讀進記憶體再關檔；np.load 的延遲讀取會一直占著檔案，Windows 上之後無法覆寫。"""
    with np.load(path) as npz:
        return {key: npz[key] for key in npz.files}


def has_persons(data):
    return 'person_id' in data


def load_or_extract(video_path, model=None, progress=None, should_stop=None, frame_rate=30.0):
    """回傳 (npz 資料, 是否剛萃取)；中止時回傳 (None, False)。

    快取不存在、來自別的影片，或是舊格式（沒有多人資料）而且有 model 時重新萃取。
    """
    path = cache_path(video_path)
    if os.path.isfile(path):
        data = load_cache(path)
        if _same_video(data['video_path'], video_path) and (has_persons(data) or model is None):
            return data, False
    if model is None:
        raise FileNotFoundError(f'{os.path.basename(video_path)} 尚未萃取關節點，請先執行 run_extract.py')
    from gait_analysis.extract_keypoints import extract_video
    if extract_video(model, video_path, path, frame_rate=frame_rate, log_every=0, progress=progress,
                     should_stop=should_stop) is None:
        return None, False
    return load_cache(path), True


def _load_targets():
    try:
        with open(TARGETS_PATH, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def load_target(name):
    value = _load_targets().get(name)
    return int(value) if value is not None else None


def save_target(name, person_id):
    targets = _load_targets()
    if person_id is None:
        targets.pop(name, None)
    else:
        targets[name] = int(person_id)
    os.makedirs(os.path.dirname(TARGETS_PATH), exist_ok=True)
    with open(TARGETS_PATH, 'w', encoding='utf-8') as f:
        json.dump(targets, f, ensure_ascii=False, indent=1, sort_keys=True)


def target_series(data, person_id):
    """目標人物逐格的關節點 (T, 26, 3) 與人物框 (T, 4)；person_id 為 None 時用自動目標。"""
    if person_id is None:
        return data['keypoints'], data['bboxes']
    kpts = np.full_like(data['keypoints'], np.nan)
    boxes = np.full_like(data['bboxes'], np.nan)
    rows = data['person_id'] == person_id
    kpts[data['person_frame'][rows]] = data['person_kpts'][rows]
    boxes[data['person_frame'][rows]] = data['person_bbox'][rows]
    return kpts, boxes


def auto_person_id(data):
    """自動目標最常對應到的追蹤 ID（只用來顯示）。"""
    if not has_persons(data) or not len(data['person_id']):
        return None
    auto = data['bboxes'][data['person_frame']]
    hit = np.all(np.isclose(data['person_bbox'], auto), axis=1) & (data['person_id'] >= 0)
    counts = Counter(data['person_id'][hit].tolist())
    return counts.most_common(1)[0][0] if counts else None


def _persons_by_frame(data):
    persons = {}
    if not has_persons(data):
        return persons
    for frame, person_id, box in zip(data['person_frame'], data['person_id'], data['person_bbox']):
        if person_id >= 0:
            persons.setdefault(int(frame), []).append((int(person_id), box))
    return persons


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
    person_id: object = None    # 手動點選的目標 ID；None 為自動
    auto_id: object = None      # 自動目標最常對應的 ID
    persons: dict = field(default_factory=dict)     # frame -> [(person_id, bbox)]，只含有追蹤 ID 者
    warnings: list = field(default_factory=list)    # 給使用者看的提示（例如輸出檔被占用）

    @property
    def header(self):
        if self.trial is not None:
            activity = ACTIVITY_NAMES.get(self.trial.activity, self.trial.activity)
            return f'受試者 {self.trial.subject}｜{self.trial.activity} {activity}'
        return self.name

    @property
    def can_select(self):
        return bool(self.persons)

    def persons_at(self, frame_idx):
        return self.persons.get(frame_idx, [])

    def target_id(self):
        return self.person_id if self.person_id is not None else self.auto_id

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


def analyze_video(video_path, model=None, progress=None, should_stop=None, write_outputs=True, person_id=SAVED):
    """progress(done, total) 在萃取關節點與輸出骨架影片時回報。中止時回傳 None。

    person_id：SAVED 沿用上次點選的目標（沒有則自動）、None 自動、整數為指定追蹤 ID（並記住）。
    """
    name = os.path.splitext(os.path.basename(video_path))[0]
    trial = find_vidimu_trial(video_path)
    fps = video_fps(video_path, trial)
    data, extracted = load_or_extract(video_path, model, progress, should_stop, frame_rate=fps)
    if data is None:
        return None
    explicit = person_id != SAVED
    # 重新萃取後追蹤 ID 會重編，舊的點選紀錄不再對應同一人，改回自動
    saved = None if extracted else load_target(name)
    if not explicit:
        person_id = saved
    if person_id is not None and (not has_persons(data) or person_id not in set(data['person_id'].tolist())):
        person_id = None
    if write_outputs and (explicit or extracted):
        save_target(name, person_id)
    keypoints, bboxes = target_series(data, person_id)
    cfg = gait.load_config()
    cfg['fps'] = fps
    result = gait.analyze(name, keypoints, cfg)
    curves, agreement, imu_knee = result.curves, None, None
    if trial is not None and trial.has_imu and not result.strides.empty:
        curves, agreement, imu = imu_compare.compare_with_imu(result, trial, cfg)
        imu_knee = near_series(result, {s: imu[imu_compare.imu_column('knee', s)] for s in ('L', 'R')})
    out_dir = os.path.join(vidimu.OUTPUT_ROOT, name)
    analysis = GaitAnalysis(video_path, name, trial, result, curves, agreement, imu_knee,
                            gait.trial_summary(result), bboxes, out_dir, person_id, auto_person_id(data),
                            _persons_by_frame(data))
    if write_outputs:
        try:
            write_trial_csvs(result, curves, out_dir)
            if not write_base_outputs(analysis, data, extracted, progress, should_stop):
                return None
        except PermissionError as error:
            # 輸出檔正被其他程式開著（Excel、播放器）；分析結果照常顯示，只提示沒寫成
            locked = os.path.basename(error.filename2 or error.filename or '')
            analysis.warnings.append(f'{locked} 正被其他程式開啟，未能更新；關閉後重新載入即可')
    return analysis


def write_base_outputs(analysis, data, extracted=False, progress=None, should_stop=None):
    """原始影片、骨架 JSON/CSV、_Sk26.mp4；已是最新的就略過。中止時回傳 False。

    JSON/CSV 只跟關節點快取有關；_Sk26.mp4 還跟目標人物有關，畫完記下「目標｜快取時間」，不一致才重畫
    （例如上次被播放器占用沒寫成，下次載入會補畫）。
    """
    out_dir, name = analysis.out_dir, analysis.name
    os.makedirs(out_dir, exist_ok=True)
    source = analysis.video_path
    copy = os.path.join(out_dir, name + os.path.splitext(source)[1].lower())
    if not _same_video(source, copy):
        shutil.copyfile(source, copy)
    if not has_persons(data):
        return True
    prefix = os.path.join(out_dir, name)
    if extracted or not os.path.isfile(f'{prefix}.json') or not os.path.isfile(f'{prefix}.csv'):
        write_skeleton_files(data, prefix)
    sk26 = f'{prefix}_Sk26.mp4'
    stamp_path = os.path.splitext(cache_path(analysis.video_path))[0] + '.sk26.txt'
    stamp = f'{analysis.person_id if analysis.person_id is not None else "auto"}|'             f'{os.path.getmtime(cache_path(analysis.video_path)):.0f}'
    try:
        with open(stamp_path, encoding='utf-8') as f:
            current = f.read() == stamp
    except OSError:
        current = False
    if not current or not os.path.isfile(sk26):
        from gait_analysis.render_demo import render_skeleton_video
        if not render_skeleton_video(analysis, sk26, progress, should_stop):
            return False
        with open(stamp_path, 'w', encoding='utf-8') as f:
            f.write(stamp)
    return True


def write_skeleton_files(data, prefix):
    """JSON 為逐人逐格紀錄，CSV 為逐關節點一列（沿用原 2D 影片分頁的格式）。"""
    rows = data['person_id'] >= 0
    frames, ids = data['person_frame'][rows], data['person_id'][rows]
    boxes, kpts = data['person_bbox'][rows], data['person_kpts'][rows]
    records = [{'person_id': int(i), 'bbox': [round(float(v), 3) for v in b],
                'keypoints': [[round(float(x), 2), round(float(y), 2), round(float(c), 4), 0.0] for x, y, c in k],
                'frame_number': int(f)}
               for f, i, b, k in zip(frames, ids, boxes, kpts)]
    with open(f'{prefix}.json', 'w', encoding='utf-8') as f:
        json.dump(records, f)
    with open(f'{prefix}.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_number', 'person_id', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
                         'keypoint_id', 'keypoint_name', 'x', 'y', 'confidence', 'is_modified'])
        for record in records:
            for k, (x, y, c, _) in enumerate(record['keypoints']):
                writer.writerow([record['frame_number'], record['person_id'], *record['bbox'],
                                 k, KEYPOINT_NAMES_ZH[k], x, y, c, 0])

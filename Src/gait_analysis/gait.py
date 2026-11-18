"""步態指標計算（定義書第二節共通定義 + 第四節展示版指標 A1、A2、B1、B6）。

座標為影像像素，y 軸向下為正；角度公式已考慮，不需翻轉。
"""
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import CubicSpline
from scipy.ndimage import binary_dilation
from scipy.signal import butter, filtfilt, find_peaks

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'gait_config.yaml')

# Halpe-26 索引（定義書 2.2）
KPT = {
    'shoulder': (5, 6), 'hip': (11, 12), 'knee': (13, 14), 'ankle': (15, 16),
    'heel': (24, 25), 'toe': (20, 21),
}
SIDES = ('L', 'R')
# 完整骨架連線（Halpe-26，不含臉部五官），只用於顯示；計算只用到軀幹與雙腿
SKELETON_LINKS = (
    (17, 18), (18, 5), (18, 6), (5, 7), (7, 9), (6, 8), (8, 10), (18, 19), (19, 11), (19, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (15, 24), (16, 25), (24, 20), (25, 21), (24, 22),
    (25, 23), (15, 20), (16, 21),
)
HALPE_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'head', 'neck', 'hip',
    'left_big_toe', 'right_big_toe', 'left_small_toe', 'right_small_toe', 'left_heel', 'right_heel',
]


def load_config(path=CONFIG_PATH):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def side_index(part, side):
    return KPT[part][SIDES.index(side)]


# ---------------------------------------------------------------- 2.3 前處理

def _runs(mask):
    """回傳 mask 中連續 True 段的 (start, end) 列表，end 不含。"""
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(int))
    return list(zip(np.flatnonzero(diff == 1), np.flatnonzero(diff == -1)))


def fill_gaps(series, max_gap, rejected=None, rejected_max_gap=None):
    """缺值段 ≤ max_gap 格用三次樣條補；更長的段與頭尾缺值保持 NaN。

    rejected 標出被剔除的跳點：含跳點的缺值段改用 rejected_max_gap 為上限（整腿剔除並前後
    多剔的段較長，但兩側都是好資料）。
    """
    out = series.copy()
    valid = ~np.isnan(series)
    if valid.sum() < 4:
        return out
    spline = CubicSpline(np.flatnonzero(valid), series[valid])
    for start, end in _runs(~valid):
        limit = max_gap
        if rejected is not None and rejected[start:end].any():
            limit = max(max_gap, rejected_max_gap)
        if start == 0 or end == len(series) or end - start > limit:
            continue
        out[start:end] = spline(np.arange(start, end))
    return out


def lowpass(series, fps, cutoff, order):
    """逐個連續有效段做零相位 Butterworth；過短無法濾波的段設為 NaN。"""
    b, a = butter(order, cutoff / (fps / 2))
    padlen = 3 * max(len(a), len(b))
    out = np.full_like(series, np.nan)
    for start, end in _runs(~np.isnan(series)):
        if end - start > padlen:
            out[start:end] = filtfilt(b, a, series[start:end])
    return out


LEG_KPTS = {'L': (13, 15, 20, 22, 24), 'R': (14, 16, 21, 23, 25)}    # 每隻腳的膝、踝、大趾、小趾、腳跟


def reject_leg_outliers(xy, window, max_leg_ratio, dilate=1):
    """腿部跳點：整隻腿（膝到腳）設為 NaN，之後由 fill_gaps 用前後的好資料補值。

    側拍時兩腳交錯，模型偶爾把腳踝、腳跟、腳趾（有時整隻腳）接到另一隻腳上 1～3 格；直接濾波
    會把這一跳平滑成好幾格「腳擺過去又擺回來」。各點與前後 window 格的中位數軌跡比較（中位數
    不受少於半數的離群格影響；正常的平滑動作幾乎落在中位數上），偏離超過腿長 × max_leg_ratio
    即視為跳點。同一格同一隻腿的膝蓋常一起偏掉但幅度較小、抓不到，所以整隻腿一起剔除，
    並往前後各多剔 dilate 格（16 支 IMU 影片驗證：只剔除單點時有 5 支膝角誤差變大）。
    """
    out = xy.copy()
    leg = np.nanmedian(np.linalg.norm(xy[:, list(KPT['hip'])] - xy[:, list(KPT['ankle'])], axis=2))
    if not np.isfinite(leg) or leg <= 0:
        return out
    for idx in LEG_KPTS.values():
        bad = np.zeros(len(xy), bool)
        for k in idx:
            med = pd.DataFrame(xy[:, k]).rolling(window, center=True, min_periods=window // 2 + 1).median().to_numpy()
            bad |= np.linalg.norm(xy[:, k] - med, axis=1) > max_leg_ratio * leg
        if dilate and bad.any():
            bad = binary_dilation(bad, iterations=dilate)
        out[np.ix_(bad, list(idx))] = np.nan
    return out


def preprocess(keypoints, cfg):
    """keypoints (T, 26, 3) → 濾波後 (T, 26, 2)。"""
    pre = cfg['preprocess']
    xy = keypoints[:, :, :2].copy()
    xy[keypoints[:, :, 2] < pre['min_confidence']] = np.nan
    rejected = np.zeros(xy.shape[:2], bool)
    if pre.get('outlier_window'):
        cleaned = reject_leg_outliers(xy, pre['outlier_window'], pre['outlier_leg_ratio'],
                                      pre.get('outlier_dilate_frames', 1))
        rejected = np.isnan(cleaned[:, :, 0]) & ~np.isnan(xy[:, :, 0])
        xy = cleaned
    out = np.full_like(xy, np.nan)
    for k in range(xy.shape[1]):
        for c in range(2):
            filled = fill_gaps(xy[:, k, c], pre['max_gap_frames'], rejected[:, k],
                               pre.get('outlier_max_gap_frames', pre['max_gap_frames']))
            out[:, k, c] = lowpass(filled, cfg['fps'], pre['cutoff_hz'], pre['butter_order'])
    return out


# ---------------------------------------------------------------- 趟、近側腿

def mid(xy, part):
    left, right = KPT[part]
    return (xy[:, left] + xy[:, right]) / 2


@dataclass
class Pass:
    index: int
    start: int          # 含
    end: int            # 不含
    direction: int      # d：往畫面右方 +1，往左 −1
    near_side: str = ''
    side_conf: dict = field(default_factory=dict)


def detect_passes(xy, cfg):
    """骨盆水平速度方向不變、且大於該段平均速度一定比例的直線行走段。"""
    pc = cfg['pass_detection']
    vx = np.gradient(mid(xy, 'hip')[:, 0])
    passes = []
    for sign in (1, -1):
        for start, end in _runs(np.nan_to_num(vx * sign) > 0):
            speed = np.abs(vx[start:end])
            keep = speed > pc['min_speed_ratio'] * np.nanmean(speed)
            for s, e in _runs(keep):
                if (e - s) / cfg['fps'] >= pc['min_duration_s']:
                    passes.append((start + s, start + e, sign))
    passes.sort()
    return [Pass(i, s, e, d) for i, (s, e, d) in enumerate(passes)]


def assign_near_side(passes, keypoints):
    """近側腿：膝、踝、腳跟平均信心值較高的一側。

    單趟兩腳信心值常只差 0.01，逐趟判斷會偶爾判反；相機固定時同一行進方向的近側腿必定
    是同一隻，因此把同方向各趟的信心值差加總後決定。
    """
    for p in passes:
        conf = keypoints[p.start:p.end, :, 2]
        p.side_conf = {
            side: float(np.nanmean([conf[:, side_index(part, side)] for part in ('knee', 'ankle', 'heel')]))
            for side in SIDES
        }
    for direction in (1, -1):
        same = [p for p in passes if p.direction == direction]
        margin = sum(p.side_conf['L'] - p.side_conf['R'] for p in same)
        for p in same:
            p.near_side = 'L' if margin > 0 else 'R'
    return passes


# ---------------------------------------------------------------- 2.5 角度

def segment_angle(xy, proximal, distal, d):
    """節段角：鉛直向下為 0 度，遠端往前為正。xy 為 (T, 26, 2)，d 為 (T,)。"""
    dx = d * (xy[:, distal, 0] - xy[:, proximal, 0])
    dy = xy[:, distal, 1] - xy[:, proximal, 1]
    return np.degrees(np.arctan2(dx, dy))


def joint_angles(xy, direction, side):
    """回傳某一側的髖屈曲、膝屈曲、踝背屈與軀幹前傾（度）。direction 為逐格 d。"""
    neck, pelvis = mid(xy, 'shoulder'), mid(xy, 'hip')
    trunk_xy = np.stack([neck, pelvis], axis=1)
    phi_t = segment_angle(trunk_xy, 0, 1, direction)
    idx = {part: side_index(part, side) for part in ('hip', 'knee', 'ankle', 'heel', 'toe')}
    phi_th = segment_angle(xy, idx['hip'], idx['knee'], direction)
    phi_sh = segment_angle(xy, idx['knee'], idx['ankle'], direction)
    phi_f = segment_angle(xy, idx['heel'], idx['toe'], direction)
    return {
        'hip': phi_th - phi_t,
        'knee': phi_th - phi_sh,
        'ankle': phi_f - phi_sh - 90.0,
        'trunk': -phi_t,
    }


def near_side_by_frame(passes):
    """{格: 該趟近側腿 'L'/'R'}；不在任何趟內的格不在字典中。"""
    return {f: p.near_side for p in passes for f in range(p.start, p.end)}


def direction_per_frame(passes, n_frames):
    d = np.full(n_frames, np.nan)
    for p in passes:
        d[p.start:p.end] = p.direction
    return d


# ---------------------------------------------------------------- 2.4 著地與離地

def leg_length(xy, side):
    hip, knee, ankle = (xy[:, side_index(p, side)] for p in ('hip', 'knee', 'ankle'))
    return np.linalg.norm(hip - knee, axis=1) + np.linalg.norm(knee - ankle, axis=1)


def detect_events(xy, passes, cfg):
    """Zeni 2008：HS = d·(x_heel − x_P) 局部最大，TO = d·(x_toe − x_P) 局部最小。"""
    ec = cfg['events']
    distance = max(1, int(round(ec['min_hs_interval_s'] * cfg['fps'])))
    pelvis_x = mid(xy, 'hip')[:, 0]
    rows = []
    for p in passes:
        sl = slice(p.start, p.end)
        for side in SIDES:
            prominence = ec['min_prominence_leg'] * np.nanmedian(leg_length(xy[sl], side))
            for event, part, sign in (('HS', 'heel', 1), ('TO', 'toe', -1)):
                signal = sign * p.direction * (xy[sl, side_index(part, side), 0] - pelvis_x[sl])
                if np.isnan(signal).all():
                    continue
                peaks, _ = find_peaks(np.nan_to_num(signal, nan=np.nanmin(signal)),
                                      distance=distance, prominence=prominence)
                for peak in peaks:
                    if np.isnan(signal[peak]):
                        continue
                    frame = p.start + int(peak)
                    rows.append({'pass': p.index, 'side': side, 'event': event, 'frame': frame,
                                 'time': frame / cfg['fps'],
                                 'near_far': 'near' if side == p.near_side else 'far'})
    events = pd.DataFrame(rows, columns=['pass', 'side', 'event', 'frame', 'time', 'near_far'])
    return events.sort_values(['pass', 'side', 'frame']).reset_index(drop=True)


def build_strides(events, cfg):
    """同一隻腳兩次連續 HS 為一週期；中間必須恰有一個 TO（HS → TO → HS）。"""
    ec = cfg['events']
    strides = []
    for (pass_id, side), group in events.groupby(['pass', 'side']):
        hs = group.loc[group['event'] == 'HS', 'frame'].to_numpy()
        to = group.loc[group['event'] == 'TO', 'frame'].to_numpy()
        side_strides = []
        for start, end in zip(hs[:-1], hs[1:]):
            between = to[(to > start) & (to < end)]
            duration = (end - start) / cfg['fps']
            if len(between) != 1 or not ec['min_hs_interval_s'] <= duration <= ec['max_stride_s']:
                continue
            side_strides.append({'pass': pass_id, 'side': side, 'hs_start': int(start),
                                 'to': int(between[0]), 'hs_end': int(end),
                                 'near_far': group['near_far'].iloc[0]})
        for i, stride in enumerate(side_strides):
            stride['edge_step'] = i in (0, len(side_strides) - 1)
        strides.extend(side_strides)
    return strides


def mark_edge_events(events, strides):
    """事件表的 edge_step：屬於該趟該側第一或最後一個週期的事件。"""
    events = events.copy()
    events['edge_step'] = False
    for s in strides:
        if s['edge_step']:
            mask = ((events['pass'] == s['pass']) & (events['side'] == s['side'])
                    & events['frame'].between(s['hs_start'], s['hs_end']))
            events.loc[mask, 'edge_step'] = True
    return events


# ---------------------------------------------------------------- 指標

def normalize_curve(values, start, end, n_points):
    """把 [start, end] 格的曲線重採樣成 n_points 點（0–100% 週期）。"""
    frames = np.arange(start, end + 1)
    seg = values[start:end + 1]
    if np.isnan(seg).any():
        return None
    return np.interp(np.linspace(start, end, n_points), frames, seg)


def cadence_for_pass(events, pass_id, strides, cfg):
    """B1：60 ÷ 平均步時，步時 = 左右腳相鄰兩次 HS。缺另一腳時退回近側腿 120 ÷ 週期時間。

    步時只採計週期上下限的一半之間（遠側腳偶爾重複偵測，會產生過短的步時）。
    """
    fps, ec = cfg['fps'], cfg['events']
    hs = events[(events['pass'] == pass_id) & (events['event'] == 'HS')].sort_values('frame')
    step_times = [(b.frame - a.frame) / fps
                  for a, b in zip(hs.itertuples(), hs.iloc[1:].itertuples()) if a.side != b.side]
    step_times = [t for t in step_times if ec['min_hs_interval_s'] / 2 <= t <= ec['max_stride_s'] / 2]
    if step_times:
        return 60.0 / np.mean(step_times), 'alternating_hs', len(step_times)
    near = [s for s in strides if s['pass'] == pass_id and s['near_far'] == 'near']
    if near:
        return 120.0 / np.mean([(s['hs_end'] - s['hs_start']) / fps for s in near]), 'near_stride', len(near)
    return np.nan, 'none', 0


@dataclass
class TrialResult:
    name: str
    fps: float
    keypoints: np.ndarray       # (T, 26, 3) 原始
    xy: np.ndarray              # (T, 26, 2) 濾波後
    passes: list
    events: pd.DataFrame
    strides: pd.DataFrame
    curves: pd.DataFrame
    angles: dict                # side -> {'hip','knee','ankle','trunk'} 逐格
    cadence: pd.DataFrame


def analyze(name, keypoints, cfg=None):
    cfg = cfg or load_config()
    fps = cfg['fps']
    n_points = cfg['strides']['curve_points']
    xy = preprocess(keypoints, cfg)
    passes = assign_near_side(detect_passes(xy, cfg), keypoints)
    direction = direction_per_frame(passes, len(xy))
    angles = {side: joint_angles(xy, direction, side) for side in SIDES}
    events = detect_events(xy, passes, cfg)
    strides = build_strides(events, cfg)
    events = mark_edge_events(events, strides)

    stride_rows, curve_rows = [], []
    median_frames = np.median([s['hs_end'] - s['hs_start'] for s in strides]) if strides else np.nan
    max_dev = cfg['strides']['max_time_deviation']
    for stride_id, s in enumerate(strides):
        side_angles = angles[s['side']]
        curves = {key: normalize_curve(side_angles[key], s['hs_start'], s['hs_end'], n_points)
                  for key in ('hip', 'knee', 'trunk')}
        reason = ''
        if abs((s['hs_end'] - s['hs_start']) / median_frames - 1) > max_dev:
            reason = 'stride_time_outlier'
        elif any(c is None for c in curves.values()):
            reason = 'missing_keypoints'
        valid = not reason
        row = {'stride_id': stride_id, 'subject': name.split('_')[0], 'trial': name, **s,
               'start_time': s['hs_start'] / fps, 'stride_time': (s['hs_end'] - s['hs_start']) / fps,
               'valid': valid, 'invalid_reason': reason}
        if valid:
            hip, knee, trunk = curves['hip'], curves['knee'], curves['trunk']
            swing = side_angles['knee'][s['to']:s['hs_end'] + 1]
            row.update({
                'A1_hip_max_flexion': hip.max(), 'A1_hip_max_extension': -hip.min(),
                'A1_hip_rom': hip.max() - hip.min(),
                'A2_knee_at_hs': side_angles['knee'][s['hs_start']],
                'A2_knee_swing_max_flexion': np.nanmax(swing), 'A2_knee_rom': knee.max() - knee.min(),
                'B6_trunk_mean': trunk.mean(), 'B6_trunk_range': trunk.max() - trunk.min(),
            })
            for pct in range(n_points):
                curve_rows.append({'stride_id': stride_id, 'pct': pct * 100 / (n_points - 1),
                                   'hip': hip[pct], 'knee': knee[pct], 'trunk': trunk[pct]})
        stride_rows.append(row)

    cadence_rows = []
    for p in passes:
        value, method, n = cadence_for_pass(events, p.index, strides, cfg)
        cadence_rows.append({'pass': p.index, 'start_frame': p.start, 'end_frame': p.end,
                             'direction': p.direction, 'near_side': p.near_side,
                             'B1_cadence': value, 'cadence_method': method, 'n_intervals': n})
    cadence = pd.DataFrame(cadence_rows)
    strides_df = pd.DataFrame(stride_rows)
    if not strides_df.empty:
        strides_df = strides_df.merge(cadence[['pass', 'B1_cadence']], on='pass', how='left')
    return TrialResult(name, fps, keypoints, xy, passes, events, strides_df,
                       pd.DataFrame(curve_rows), angles, cadence)


def trial_summary(result: TrialResult):
    """展示版數字面板的試驗層級數值。

    A 類取近側腿有效週期的平均。軀幹前傾先分往右、往左兩個方向各自平均再取平均：
    相機若有些微傾斜，兩個方向的偏差正負相反，可互相抵銷。
    """
    s = result.strides
    near = s[(s['near_far'] == 'near') & s['valid']] if not s.empty else s
    trunk_by_dir = []
    for direction in (1, -1):
        ids = [p.index for p in result.passes if p.direction == direction]
        values = s.loc[s['pass'].isin(ids) & s['valid'], 'B6_trunk_mean'] if not s.empty else []
        if len(values):
            trunk_by_dir.append(np.mean(values))
    # 步頻以步時加權：短趟只有一兩個步時，不應與長趟等權
    cad = result.cadence.dropna(subset=['B1_cadence']) if not result.cadence.empty else result.cadence
    cadence = (60.0 / (np.sum(cad['n_intervals'] * 60.0 / cad['B1_cadence']) / cad['n_intervals'].sum())
               if len(cad) else np.nan)
    return {
        'cadence': float(cadence),
        'knee_rom': float(near['A2_knee_rom'].mean()) if len(near) else np.nan,
        'hip_ext': float(near['A1_hip_max_extension'].mean()) if len(near) else np.nan,
        'trunk': float(np.mean(trunk_by_dir)) if trunk_by_dir else np.nan,
        'trunk_directions': len(trunk_by_dir),
        'passes': len(result.passes),
        'near_strides': int(len(near)),
    }


def keypoints_table(result: TrialResult):
    """_keypoints.csv：一列一格，含原始與濾波後座標。"""
    T = len(result.keypoints)
    data = {'frame': np.arange(T), 'time': np.arange(T) / result.fps}
    for k, name in enumerate(HALPE_NAMES):
        data[f'{name}_x'] = result.keypoints[:, k, 0]
        data[f'{name}_y'] = result.keypoints[:, k, 1]
        data[f'{name}_conf'] = result.keypoints[:, k, 2]
        data[f'{name}_x_filt'] = result.xy[:, k, 0]
        data[f'{name}_y_filt'] = result.xy[:, k, 1]
    return pd.DataFrame(data)

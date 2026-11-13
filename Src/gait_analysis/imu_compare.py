"""影片角度與 IMU（OpenSim IK）角度的比對（定義書第五節驗證方法的展示版子集）。

報告用語一律寫「與 IMU 的一致性」，IMU 不是金標準。
"""
import numpy as np

from gait_analysis import vidimu

IMU_COLUMNS = {'knee': 'knee_angle_{}', 'hip': 'hip_flexion_{}'}


def imu_column(joint, side):
    return IMU_COLUMNS[joint].format(side.lower())


def _shift(series, lag):
    """回傳 out[f] = series[f − lag]，超出範圍為 NaN。"""
    out = np.full(len(series), np.nan)
    if lag >= 0:
        out[lag:] = series[:len(series) - lag]
    else:
        out[:lag] = series[-lag:]
    return out


def _best_lag(video, imu, candidates):
    """在候選位移中找使兩條膝角相關係數最高者（相關係數不受固定偏差影響）。"""
    best_lag, best_r = None, -np.inf
    for lag in candidates:
        shifted = _shift(imu, lag)
        ok = np.isfinite(video) & np.isfinite(shifted)
        if ok.sum() < 20:
            continue
        r = np.corrcoef(video[ok], shifted[ok])[0, 1]
        if r > best_r:
            best_lag, best_r = lag, r
    return best_lag


def estimate_lag_track(result, imu0, max_lag):
    """逐趟估計 IMU 位移並以直線擬合，回傳逐格位移陣列與 (起始位移, 每分鐘漂移格數)。

    定義書指定以 VIDIMU 同步結果為準、膝角互相關在 ±max_lag 格內檢查。部分試驗（如 S41、S54）
    的影片時鐘與 IMU 有線性漂移，一段影片內位移會累積到 30 格，因此第一趟在 ±max_lag 內搜尋，
    之後每趟只在前一趟位移的 ±max_lag 內追蹤，最後用直線描述位移隨時間的變化。
    """
    fps = result.fps
    centers, lags = [], []
    previous = 0
    for p in result.passes:
        sl = slice(p.start, p.end)
        video = np.full(len(result.xy), np.nan)
        video[sl] = result.angles[p.near_side]['knee'][sl]
        lag = _best_lag(video, imu0[imu_column('knee', p.near_side)],
                        range(previous - max_lag, previous + max_lag + 1))
        if lag is None:
            continue
        centers.append((p.start + p.end) / 2 / fps)
        lags.append(lag)
        previous = lag
    frames = np.arange(len(result.xy))
    if len(lags) >= 2:
        slope, intercept = np.polyfit(centers, lags, 1)
    else:
        slope, intercept = 0.0, (lags[0] if lags else 0)
    return intercept + slope * frames / fps, (float(intercept), float(slope * 60)), lags


def curve_agreement(video, imu):
    ok = np.isfinite(video) & np.isfinite(imu)
    v, i = video[ok], imu[ok]
    if len(v) < 3:
        return {'n': len(v), 'rmse': np.nan, 'rmse_centered': np.nan, 'bias': np.nan, 'pearson_r': np.nan}
    diff = v - i
    return {'n': int(len(v)), 'rmse': float(np.sqrt(np.mean(diff ** 2))),
            'rmse_centered': float(np.sqrt(np.mean((diff - diff.mean()) ** 2))),
            'bias': float(diff.mean()), 'pearson_r': float(np.corrcoef(v, i)[0, 1])}


def compare_with_imu(result, trial, cfg):
    """回傳 (加上 imu_hip、imu_knee 的 curves, 一致性摘要 dict, 逐格 IMU dict)。"""
    n = len(result.xy)
    columns = [imu_column(j, s) for j in IMU_COLUMNS for s in ('L', 'R')]
    imu0 = vidimu.imu_on_video_frames(trial, n, columns)
    lag_per_frame, (lag_start, drift_per_min), pass_lags = estimate_lag_track(
        result, imu0, cfg['validation']['max_lag_frames'])
    imu = vidimu.imu_on_video_frames(trial, n, columns, extra_lag_frames=lag_per_frame)

    n_points = cfg['strides']['curve_points']
    curves = result.curves.copy()
    curves['imu_hip'] = np.nan
    curves['imu_knee'] = np.nan
    strides = result.strides.set_index('stride_id') if not result.strides.empty else None
    for stride_id, group in curves.groupby('stride_id'):
        s = strides.loc[stride_id]
        for joint in ('hip', 'knee'):
            seg = imu[imu_column(joint, s.side)][s.hs_start:s.hs_end + 1]
            if np.isfinite(seg).all():
                curves.loc[group.index, f'imu_{joint}'] = np.interp(
                    np.linspace(s.hs_start, s.hs_end, n_points), np.arange(s.hs_start, s.hs_end + 1), seg)

    near_ids = (result.strides.loc[(result.strides['near_far'] == 'near') & result.strides['valid'], 'stride_id']
                if strides is not None else [])
    near = curves['stride_id'].isin(near_ids)
    summary = {'trial': result.name, 'sync_offset_s': vidimu.sync_offset(trial),
               'lag_start_frames': lag_start, 'lag_drift_frames_per_min': drift_per_min,
               'pass_lags': ' '.join(map(str, pass_lags)),
               'near_strides': int(curves.loc[near, 'stride_id'].nunique())}
    for joint in ('knee', 'hip'):
        stats = curve_agreement(curves.loc[near, joint].to_numpy(), curves.loc[near, f'imu_{joint}'].to_numpy())
        summary.update({f'{joint}_{k}': v for k, v in stats.items()})
    return curves, summary, imu

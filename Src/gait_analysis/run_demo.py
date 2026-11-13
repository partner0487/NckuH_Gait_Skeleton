"""11/20 展覽用展示資料（定義書第四節）。

先執行 run_extract.py 萃取關節點，再執行：
    python Src/gait_analysis/run_demo.py                          # 分析全部，自動挑 3 位做展示
    python Src/gait_analysis/run_demo.py --demo-subjects S40 S41 S42

輸出（Db/Gait/）：
    <試驗>/<試驗>_keypoints|_events|_strides|_curves.csv    每個試驗的四個 CSV
    demo/<受試者>_knee_overlay.png                         膝角曲線疊圖（影片 vs IMU）
    demo/<受試者>_panel.png                                數字面板
    demo/<受試者>_demo.mp4                                 展示影片
    agreement_A01.csv                                      全部試驗的 IMU 一致性與面板數值
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gait_analysis import gait, imu_compare, plots, render_demo, vidimu
from gait_analysis.run_extract import keypoints_cache_path

MIN_NEAR_STRIDES = 4


def write_trial_csvs(result, curves, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, result.name)
    gait.keypoints_table(result).to_csv(f'{prefix}_keypoints.csv', index=False, float_format='%.3f')
    result.events.to_csv(f'{prefix}_events.csv', index=False, float_format='%.3f')
    result.strides.to_csv(f'{prefix}_strides.csv', index=False, float_format='%.3f')
    curves.to_csv(f'{prefix}_curves.csv', index=False, float_format='%.3f')


def pick_demo_trials(table, count=3):
    """近側有效週期足夠的試驗中，取膝角扣平均差 RMSE 最小者（每位受試者一支）。"""
    ok = table[table['near_strides'] >= MIN_NEAR_STRIDES].sort_values('knee_rmse_centered')
    return ok.drop_duplicates('subject').head(count)['trial'].tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--activity', default='A01')
    parser.add_argument('--demo-subjects', nargs='*', help='指定展示受試者；省略則自動挑選')
    parser.add_argument('--no-video', action='store_true', help='不產生展示影片')
    args = parser.parse_args()

    cfg = gait.load_config()
    demo_dir = os.path.join(vidimu.OUTPUT_ROOT, 'demo')
    os.makedirs(demo_dir, exist_ok=True)

    rows, cache = [], {}
    for trial in vidimu.find_trials(args.activity):
        cache_path = keypoints_cache_path(trial)
        if not os.path.isfile(cache_path):
            print(f'{trial.name}: 尚未萃取關節點，略過')
            continue
        data = np.load(cache_path)
        result = gait.analyze(trial.name, data['keypoints'], cfg)
        curves, agreement, imu = imu_compare.compare_with_imu(result, trial, cfg)
        write_trial_csvs(result, curves, os.path.join(vidimu.OUTPUT_ROOT, trial.name))
        values = gait.trial_summary(result)
        rows.append({'subject': trial.subject, **agreement, **values})
        cache[trial.name] = (trial, result, curves, agreement, imu, values, data['bboxes'])
        print(f'{trial.name}: {values["passes"]} 趟、近側週期 {values["near_strides"]}、'
              f'膝 r={agreement["knee_pearson_r"]:.2f} RMSE={agreement["knee_rmse"]:.1f}° '
              f'（扣平均差 {agreement["knee_rmse_centered"]:.1f}°）、髖 RMSE={agreement["hip_rmse"]:.1f}°、'
              f'IMU 位移 {agreement["lag_start_frames"]:+.0f} 格、漂移 {agreement["lag_drift_frames_per_min"]:+.1f} 格/分',
              flush=True)

    table = pd.DataFrame(rows)
    table_path = os.path.join(vidimu.OUTPUT_ROOT, f'agreement_{args.activity}.csv')
    table.to_csv(table_path, index=False, float_format='%.3f')
    print(f'彙總表：{table_path}')

    if args.demo_subjects:
        demo = [name for name in cache if name.split('_')[0] in args.demo_subjects]
    else:
        demo = pick_demo_trials(table)
    print('展示試驗：', ', '.join(demo))
    for name in demo:
        trial, result, curves, agreement, imu, values, bboxes = cache[name]
        subject = trial.subject
        plots.knee_overlay(curves, result.strides, agreement, os.path.join(demo_dir, f'{subject}_knee_overlay.png'))
        plots.number_panel(values, subject, os.path.join(demo_dir, f'{subject}_panel.png'))
        if not args.no_video:
            imu_knee = render_demo.near_series(
                result, {s: imu[imu_compare.imu_column('knee', s)] for s in ('L', 'R')})
            out = os.path.join(demo_dir, f'{subject}_demo.mp4')
            render_demo.render(result, trial.video_path, out, values, imu_knee=imu_knee, bboxes=bboxes)
            print(f'  {out}', flush=True)


if __name__ == '__main__':
    main()

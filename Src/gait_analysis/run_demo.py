"""11/20 展覽用展示資料（定義書第四節）。

先執行 run_extract.py 萃取關節點，再執行：
    python Src/gait_analysis/run_demo.py                          # 分析全部，自動挑 3 位做展示
    python Src/gait_analysis/run_demo.py --demo-subjects S40 S41 S42

輸出（Db/Record/，與 GUI 共用，一支影片一個資料夾）：
    <試驗>/<試驗>_keypoints|_events|_strides|_curves.csv    每個試驗的四個 CSV
    <試驗>/<試驗>_demo.mp4、_panel.png、_knee_overlay.png   展示試驗的展示影片、數字面板、膝角疊圖
    _summary/agreement_A01.csv                             全部試驗的 IMU 一致性與面板數值
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gait_analysis import plots, vidimu
from gait_analysis.pipeline import analyze_video
from gait_analysis.render_demo import render
from gait_analysis.style import ACTIVITY_NAMES

MIN_NEAR_STRIDES = 4


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

    rows, analyses = [], {}
    for trial in vidimu.find_trials(args.activity):
        try:
            analysis = analyze_video(trial.video_path)
        except FileNotFoundError as error:
            print(f'{trial.name}: {error}，略過')
            continue
        agreement, values = analysis.agreement, analysis.values
        rows.append({'subject': trial.subject, **agreement, **values})
        analyses[trial.name] = analysis
        print(f'{trial.name}: {values["passes"]} 趟、近側週期 {values["near_strides"]}、'
              f'膝 r={agreement["knee_pearson_r"]:.2f} RMSE={agreement["knee_rmse"]:.1f}° '
              f'（扣平均差 {agreement["knee_rmse_centered"]:.1f}°）、髖 RMSE={agreement["hip_rmse"]:.1f}°、'
              f'IMU 位移 {agreement["lag_start_frames"]:+.0f} 格、漂移 {agreement["lag_drift_frames_per_min"]:+.1f} 格/分',
              flush=True)

    table = pd.DataFrame(rows)
    os.makedirs(vidimu.SUMMARY_DIR, exist_ok=True)
    table_path = os.path.join(vidimu.SUMMARY_DIR, f'agreement_{args.activity}.csv')
    table.to_csv(table_path, index=False, float_format='%.3f')
    print(f'彙總表：{table_path}')

    if args.demo_subjects:
        demo = [name for name in analyses if name.split('_')[0] in args.demo_subjects]
    else:
        demo = pick_demo_trials(table)
    print('展示試驗：', ', '.join(demo))
    activity = ACTIVITY_NAMES.get(args.activity, args.activity)
    for name in demo:
        analysis = analyses[name]
        prefix = os.path.join(analysis.out_dir, name)
        plots.knee_overlay(analysis.curves, analysis.result.strides, analysis.agreement,
                           f'{prefix}_knee_overlay.png')
        plots.number_panel(analysis.values, f'{analysis.trial.subject} 步態數字面板（{args.activity} {activity}）',
                           f'{prefix}_panel.png')
        if not args.no_video:
            out = f'{prefix}_demo.mp4'
            render(analysis.renderer(), analysis.video_path, out)
            print(f'  {out}', flush=True)


if __name__ == '__main__':
    main()

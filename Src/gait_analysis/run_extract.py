"""批次萃取 VIDIMU 有 IMU 受試者的關節點，並寫出每支影片的輸出檔（與 GUI 載入影片時相同）。

用法（專案根目錄）：
    python Src/gait_analysis/run_extract.py                 # 所有 IMU 受試者的 A01
    python Src/gait_analysis/run_extract.py --subjects S40 S41 --activity A01
已是新格式（含多人資料）的快取會跳過萃取，加 --overwrite 重做。
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gait_analysis import vidimu                                            # noqa: E402
from gait_analysis.extract_keypoints import load_model                      # noqa: E402
from gait_analysis.pipeline import analyze_video, cache_path, has_current_cache   # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--activity', default='A01')
    parser.add_argument('--subjects', nargs='*')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    trials = vidimu.find_trials(args.activity, args.subjects)
    todo = [t for t in trials if args.overwrite or not has_current_cache(t.video_path)]
    print(f'{len(trials)} 個試驗，需萃取 {len(todo)} 個', flush=True)
    if not todo:
        return
    # torch 要在任何 scipy 計算之前載入，否則 Windows 上可能 DLL 衝突
    model = load_model()
    for index, trial in enumerate(todo, 1):
        start = time.time()
        print(f'[{index}/{len(todo)}] {trial.name}', flush=True)
        if args.overwrite and os.path.isfile(cache_path(trial.video_path)):
            os.remove(cache_path(trial.video_path))     # 讓 analyze_video 重新萃取並重寫所有輸出
        analysis = analyze_video(trial.video_path, model)
        print(f'  完成，{time.time() - start:.0f} 秒；自動目標 ID {analysis.auto_id}'
              + ''.join(f'；⚠ {w}' for w in analysis.warnings), flush=True)


if __name__ == '__main__':
    main()

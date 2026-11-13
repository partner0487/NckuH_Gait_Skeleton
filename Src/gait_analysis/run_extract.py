"""批次萃取 VIDIMU 影片的關節點。

用法（專案根目錄）：
    python Src/gait_analysis/run_extract.py                 # 所有 IMU 受試者的 A01
    python Src/gait_analysis/run_extract.py --subjects S40 S41 --activity A01
已萃取過的試驗會跳過，加 --overwrite 重做。
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gait_analysis import vidimu
from gait_analysis.extract_keypoints import extract_video, load_model


def keypoints_cache_path(trial):
    return os.path.join(vidimu.OUTPUT_ROOT, 'keypoints_cache', f'{trial.name}.npz')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--activity', default='A01')
    parser.add_argument('--subjects', nargs='*')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    trials = vidimu.find_trials(args.activity, args.subjects)
    todo = [t for t in trials if args.overwrite or not os.path.isfile(keypoints_cache_path(t))]
    print(f'{len(trials)} 個試驗，需萃取 {len(todo)} 個', flush=True)
    if not todo:
        return
    model = load_model()
    for index, trial in enumerate(todo, 1):
        start = time.time()
        print(f'[{index}/{len(todo)}] {trial.name}', flush=True)
        extract_video(model, trial.video_path, keypoints_cache_path(trial))
        print(f'  完成，{time.time() - start:.0f} 秒', flush=True)


if __name__ == '__main__':
    main()

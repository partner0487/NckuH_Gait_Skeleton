"""模型載入：YOLO26 人物偵測、ViTPose（Halpe-26）姿態估計，與 BoTSORT 追蹤參數。

路徑一律以本檔位置換算成絕對路徑，從哪個資料夾啟動都可以；設定值固定寫在這裡，不讀命令列參數。
"""
import os
import sys
from types import SimpleNamespace

from ultralytics import YOLO

UI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # Src/UI_Control
SRC_DIR = os.path.dirname(UI_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.append(os.path.join(SRC_DIR, 'tracker'))       # extract_keypoints 由此匯入 tracker.mc_bot_sort

import mmpretrain  # noqa: F401,E402 — registers mmpretrain.VisionTransformer into mmengine registry
from mmpose.apis import init_model as init_pose_estimator  # noqa: E402

YOLO_WEIGHTS = os.path.join(PROJECT_ROOT, 'Db', 'pretrain', 'yolo26m-seg.pt')
POSE_CONFIG = os.path.join(SRC_DIR, 'mmpose_main', 'configs', 'body_2d_keypoint', 'topdown_heatmap', 'haple',
                           'ViTPose_base_simple_halpe_256x192.py')
POSE_CHECKPOINT = os.path.join(PROJECT_ROOT, 'Db', 'pretrain', 'epoch_210.pth')

# BoTSORT 參數；追蹤器由 extract_keypoints 依每支影片的 fps 各自建立（track_buffer 以 30 fps 計）
TRACKER_ARGS = {
    'track_high_thresh': 0.3,
    'track_low_thresh': 0.05,
    'new_track_thresh': 0.4,
    'track_buffer': 360,
    'match_thresh': 0.8,
    'aspect_ratio_thresh': 1.6,
    'min_box_area': 10.0,
    'mot20': True,                  # fuse score
    'cmc_method': 'sparseOptFlow',
    'with_reid': False,
    'fast_reid_config': os.path.join(SRC_DIR, 'tracker', 'fast_reid', 'configs', 'MOT17', 'sbs_S50.yml'),
    'fast_reid_weights': os.path.join(SRC_DIR, 'tracker', 'fast_reid', 'mot17_sbs_S50.pth'),
    'proximity_thresh': 0.5,
    'appearance_thresh': 0.25,
    'jde': False,
    'ablation': False,
}


class Model(object):
    def __init__(self):
        self.tracker_args = SimpleNamespace(**TRACKER_ARGS)
        self.detector = YOLO(YOLO_WEIGHTS)
        self.pose_estimator = init_pose_estimator(POSE_CONFIG, POSE_CHECKPOINT)

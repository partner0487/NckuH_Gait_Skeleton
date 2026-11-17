from argparse import ArgumentParser
import argparse
import sys
import os
from ultralytics import YOLO
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../..", "tracker"))     # extract_keypoints 由此匯入 tracker.mc_bot_sort
import mmpretrain  # noqa: F401 — registers mmpretrain.VisionTransformer into mmengine registry
from mmpose.apis import init_model as init_pose_estimator


class Model(object):
    def __init__(self):
        self.detect_args_yolo = self.setDetectParserYolo()
        self.pose_args = self.setPoseParser()
        self.tracker_args = self.setTrackerParser()
        self.detector = YOLO(self.detect_args_yolo.model)
        self.pose_estimator = init_pose_estimator(
            self.pose_args.pose_config,
            self.pose_args.pose_checkpoint
        )

    def setDetectParserYolo(self) -> ArgumentParser:
        parser = argparse.ArgumentParser(description="YOLO Model Argument Parsing")
        parser.add_argument('--model', type=str, default='../../Db/pretrain/yolo26m-seg.pt')
        return parser.parse_args()

    def setPoseParser(self) -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument('--pose-config', default='../mmpose_main/configs/body_2d_keypoint/topdown_heatmap/haple/ViTPose_base_simple_halpe_256x192.py')
        parser.add_argument('--pose-checkpoint', default='../../Db/pretrain/epoch_210.pth')
        return parser.parse_args()

    def setTrackerParser(self) -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument("--track_high_thresh", type=float, default=0.3)
        parser.add_argument("--track_low_thresh", default=0.05, type=float)
        parser.add_argument("--new_track_thresh", default=0.4, type=float)
        parser.add_argument("--track_buffer", type=int, default=360)
        parser.add_argument("--match_thresh", type=float, default=0.8)
        parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6)
        parser.add_argument('--min_box_area', type=float, default=10)
        parser.add_argument("--fuse-score", dest="mot20", default=True, action='store_true')
        parser.add_argument("--cmc-method", default="sparseOptFlow", type=str)
        parser.add_argument("--with-reid", dest="with_reid", default=False)
        parser.add_argument("--fast-reid-config", dest="fast_reid_config", default='../tracker/fast_reid/configs/MOT17/sbs_S50.yml', type=str)
        parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default='../tracker/fast_reid/mot17_sbs_S50.pth', type=str)
        parser.add_argument('--proximity_thresh', type=float, default=0.5)
        parser.add_argument('--appearance_thresh', type=float, default=0.25)
        tracker_args = parser.parse_args()
        tracker_args.jde = False
        tracker_args.ablation = False
        # 追蹤器由 extract_keypoints 依每支影片的 fps 各自建立（track_buffer 以 30 fps 計）
        return tracker_args

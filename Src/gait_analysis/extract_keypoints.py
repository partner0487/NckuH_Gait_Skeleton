"""逐格萃取 Halpe-26 關節點（YOLO 人物框 + ViTPose），不經 GUI。

每支影片只取一位受試者：優先與上一格人物框重疊者，否則取面積最大者。
輸出 npz：keypoints (T, 26, 3) = x, y, conf；沒偵測到的格為 NaN。
"""
import os
import sys
from contextlib import contextmanager

import cv2
import numpy as np

UI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'UI_Control'))
NUM_KPTS = 26


@contextmanager
def _ui_context():
    # Model 以 argparse 讀 sys.argv、以相對路徑找權重，需在 UI_Control 下建立
    old_cwd, old_argv = os.getcwd(), sys.argv
    if UI_DIR not in sys.path:
        sys.path.insert(0, UI_DIR)
    os.chdir(UI_DIR)
    sys.argv = [old_argv[0]]
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv


def load_model():
    with _ui_context():
        from utils.model import Model
        return Model()


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _pick_box(boxes, prev_box):
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    if prev_box is not None:
        ious = np.array([_iou(box, prev_box) for box in boxes])
        if ious.max() > 0.3:
            return boxes[int(np.argmax(ious * areas))]
    return boxes[int(np.argmax(areas))]


def extract_video(model, video_path, out_path, conf_thres=0.3, log_every=200, progress=None, should_stop=None):
    """progress(done, total) 回報進度；should_stop() 為 True 時中止、不寫檔並回傳 None。"""
    from mmpose.apis import inference_topdown
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keypoints, bboxes = [], []
    prev_box = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        det = model.detector.predict(frame, classes=[0], conf=conf_thres, verbose=False)[0]
        boxes = det.boxes.xyxy.cpu().numpy()
        kpts = np.full((NUM_KPTS, 3), np.nan)
        box = np.full(4, np.nan)
        if len(boxes):
            box = _pick_box(boxes, prev_box)
            prev_box = box
            pred = inference_topdown(model.pose_estimator, frame, box[None])[0].pred_instances
            kpts[:, :2] = pred.keypoints[0][:NUM_KPTS]
            kpts[:, 2] = pred.keypoint_scores[0][:NUM_KPTS]
        keypoints.append(kpts)
        bboxes.append(box)
        if log_every and len(keypoints) % log_every == 0:
            print(f'  {os.path.basename(video_path)} {len(keypoints)}/{total}', flush=True)
        if progress and len(keypoints) % 10 == 0:
            progress(len(keypoints), total)
        if should_stop and should_stop():
            cap.release()
            return None
    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, keypoints=np.asarray(keypoints), bboxes=np.asarray(bboxes),
                        image_size=np.array([width, height]), video_path=video_path)
    return out_path

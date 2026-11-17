"""逐格萃取 Halpe-26 關節點（YOLO 人物框 + ViTPose），不經 GUI。

每格對畫面中所有人估計骨架，並以 BoTSORT 追蹤給每個人固定 ID（可在 GUI 點選目標）。
自動目標：優先與上一格人物框重疊者，否則取面積最大者。
輸出 npz（format 2）：
    keypoints (T, 26, 3) = x, y, conf、bboxes (T, 4)   自動目標；沒偵測到的格為 NaN
    person_frame/person_id/person_bbox/person_kpts       所有人（N 筆），沒對上追蹤的 ID 為 -1
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
    """自動目標在 boxes 中的索引。"""
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    if prev_box is not None:
        ious = np.array([_iou(box, prev_box) for box in boxes])
        if ious.max() > 0.3:
            return int(np.argmax(ious * areas))
    return int(np.argmax(areas))


def _match_ids(boxes, targets, min_iou=0.5):
    """把追蹤結果對回偵測框（IoU 由大到小配對），沒對上的為 -1。"""
    ids = np.full(len(boxes), -1, int)
    pairs = sorted(((_iou(box, target.tlbr), i, int(target.track_id))
                    for i, box in enumerate(boxes) for target in targets), reverse=True)
    used = set()
    for iou, i, track_id in pairs:
        if iou < min_iou:
            break
        if ids[i] == -1 and track_id not in used:
            ids[i] = track_id
            used.add(track_id)
    return ids


def extract_video(model, video_path, out_path, conf_thres=0.3, frame_rate=30.0, log_every=200,
                  progress=None, should_stop=None):
    """progress(done, total) 回報進度；should_stop() 為 True 時中止、不寫檔並回傳 None。"""
    from mmpose.apis import inference_topdown
    from tracker.mc_bot_sort import BoTSORT
    # 每支影片自己建一個追蹤器（依影片 fps），不沿用 model.tracker 的狀態
    tracker = BoTSORT(model.tracker_args, frame_rate=frame_rate)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keypoints, bboxes = [], []
    rows = {'frame': [], 'id': [], 'bbox': [], 'kpts': []}
    prev_box = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx = len(keypoints)
        det = model.detector.predict(frame, classes=[0], conf=conf_thres, verbose=False)[0]
        boxes = det.boxes.xyxy.cpu().numpy().reshape(-1, 4)
        scores = det.boxes.conf.cpu().numpy().reshape(-1, 1)
        targets = tracker.update(np.hstack((boxes, scores, np.zeros_like(scores))), frame)
        kpts = np.full((NUM_KPTS, 3), np.nan)
        box = np.full(4, np.nan)
        if len(boxes):
            all_kpts = np.full((len(boxes), NUM_KPTS, 3), np.nan)
            for i, sample in enumerate(inference_topdown(model.pose_estimator, frame, boxes)):
                pred = sample.pred_instances
                all_kpts[i, :, :2] = pred.keypoints[0][:NUM_KPTS]
                all_kpts[i, :, 2] = pred.keypoint_scores[0][:NUM_KPTS]
            ids = _match_ids(boxes, targets)
            pick = _pick_box(boxes, prev_box)
            box, kpts = boxes[pick], all_kpts[pick]
            prev_box = box
            rows['frame'] += [frame_idx] * len(boxes)
            rows['id'] += ids.tolist()
            rows['bbox'] += list(boxes)
            rows['kpts'] += list(all_kpts)
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
    np.savez_compressed(
        out_path, format=2, keypoints=np.asarray(keypoints), bboxes=np.asarray(bboxes),
        image_size=np.array([width, height]), video_path=video_path,
        video_bytes=os.path.getsize(video_path),
        person_frame=np.asarray(rows['frame'], int), person_id=np.asarray(rows['id'], int),
        person_bbox=np.asarray(rows['bbox'], float).reshape(-1, 4),
        person_kpts=np.asarray(rows['kpts'], float).reshape(-1, NUM_KPTS, 3))
    return out_path

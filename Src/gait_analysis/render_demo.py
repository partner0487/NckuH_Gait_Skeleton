"""展示影片（定義書第四節「展示交付物」）。

1280×720：左半為行走畫面加骨架（臉部模糊），右半同步畫出近側腿髖、膝角度曲線
（膝角疊 IMU），下方為數字面板與資料來源註記。
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gait_analysis.gait import KPT, side_index
from gait_analysis.style import FONT_PATH, VIDIMU_FOOTER, panel_items

W, H = 1280, 720
VIDEO_BOX = (0, 56, 640, 536)                      # x0, y0, x1, y1
HIP_BOX = (700, 70, 1260, 290)
KNEE_BOX = (700, 318, 1260, 528)
PANEL_Y = (548, 680)
WINDOW_S = 6.0

BG = (255, 255, 255)
INK = (40, 40, 40)
MUTED = (120, 120, 120)
GRID = (225, 225, 225)
VIDEO_BGR = (180, 119, 31)                         # #1f77b4
IMU_BGR = (40, 39, 214)                            # #d62728
NEAR_BGR = (0, 140, 255)
FAR_BGR = (170, 170, 170)
MARK_BGR = (0, 160, 0)

HEAD_KPTS = (0, 1, 2, 3, 4, 17)


class _Fonts:
    def __init__(self):
        self.cache = {}

    def __call__(self, size):
        if size not in self.cache:
            self.cache[size] = ImageFont.truetype(FONT_PATH, size)
        return self.cache[size]


FONT = _Fonts()


def near_series(result, values_by_side):
    out = np.full(len(result.xy), np.nan)
    for p in result.passes:
        out[p.start:p.end] = values_by_side[p.near_side][p.start:p.end]
    return out


def blur_face(frame, kpts, bbox):
    head = [kpts[k, :2] for k in HEAD_KPTS if np.isfinite(kpts[k, 0]) and kpts[k, 2] >= 0.3]
    height = bbox[3] - bbox[1] if np.isfinite(bbox).all() else 200
    radius = int(max(16, 0.09 * height))
    if head:
        cx, cy = np.mean(head, axis=0)
    elif np.isfinite(bbox).all():
        cx, cy = (bbox[0] + bbox[2]) / 2, bbox[1] + radius
    else:
        return frame
    x0, y0 = int(max(0, cx - radius)), int(max(0, cy - radius))
    x1, y1 = int(min(frame.shape[1], cx + radius)), int(min(frame.shape[0], cy + radius))
    if x1 <= x0 or y1 <= y0:
        return frame
    roi = frame[y0:y1, x0:x1]
    blurred = cv2.GaussianBlur(roi, (0, 0), radius / 2)
    mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.circle(mask, (int(cx) - x0, int(cy) - y0), radius, 255, -1)
    roi[mask > 0] = blurred[mask > 0]
    return frame


def draw_skeleton(frame, xy, near_side):
    def pt(k):
        return None if not np.isfinite(xy[k]).all() else (int(xy[k, 0]), int(xy[k, 1]))

    def line(a, b, color, width):
        pa, pb = pt(a), pt(b)
        if pa and pb:
            cv2.line(frame, pa, pb, color, width, cv2.LINE_AA)

    shoulders, hips = KPT['shoulder'], KPT['hip']
    neck = (xy[shoulders[0]] + xy[shoulders[1]]) / 2
    pelvis = (xy[hips[0]] + xy[hips[1]]) / 2
    if np.isfinite(neck).all() and np.isfinite(pelvis).all():
        cv2.line(frame, tuple(neck.astype(int)), tuple(pelvis.astype(int)), (90, 90, 90), 3, cv2.LINE_AA)
    sides = sorted(('L', 'R'), key=lambda s: s == near_side)       # 近側腿最後畫，疊在上層
    for side in sides:
        color = NEAR_BGR if side == near_side else FAR_BGR
        width = 4 if side == near_side else 2
        chain = [side_index(p, side) for p in ('hip', 'knee', 'ankle', 'heel', 'toe')]
        for a, b in zip(chain, chain[1:]):
            line(a, b, color, width)
        line(side_index('ankle', side), side_index('toe', side), color, width)
        for k in chain:
            p = pt(k)
            if p:
                cv2.circle(frame, p, 4 if side == near_side else 3, color, -1, cv2.LINE_AA)
    return frame


def _chart_xy(box, t, value, t0, y_range):
    x0, y0, x1, y1 = box
    x = x0 + (t - t0) / WINDOW_S * (x1 - x0)
    y = y1 - (value - y_range[0]) / (y_range[1] - y_range[0]) * (y1 - y0)
    return x, y


def draw_chart(canvas, labels, box, title, y_range, ticks, series, t_now, fps, markers=()):
    x0, y0, x1, y1 = box
    t0 = max(0.0, t_now - WINDOW_S)
    for tick in ticks:
        _, y = _chart_xy(box, t0, tick, t0, y_range)
        cv2.line(canvas, (x0, int(y)), (x1, int(y)), GRID, 1)
        labels.append((x0 - 8, y - 9, f'{tick}°', 15, MUTED, 'ra'))
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (200, 200, 200), 1)
    labels.append((x0, y0 - 26, title, 19, INK, 'la'))
    frame_now = int(round(t_now * fps))
    f_start = int(t0 * fps)
    for values, color, width in series:
        pts = []
        for f in range(f_start, frame_now + 1):
            v = values[f]
            if np.isfinite(v):
                pts.append(_chart_xy(box, f / fps, np.clip(v, *y_range), t0, y_range))
            elif pts:
                _polyline(canvas, pts, color, width)
                pts = []
        _polyline(canvas, pts, color, width)
    for frame, value in markers:
        if f_start <= frame <= frame_now:
            x, y = _chart_xy(box, frame / fps, value, t0, y_range)
            cv2.circle(canvas, (int(x), int(y)), 6, MARK_BGR, -1, cv2.LINE_AA)
            anchor = 'ra' if x > x1 - 90 else 'la' if x < x0 + 90 else 'ma'     # 靠邊時往內對齊，不壓到刻度
            labels.append((x, y + 10, f'最大伸展 {-value:.0f}°', 14, (0, 120, 0), anchor))


def _polyline(canvas, pts, color, width):
    if len(pts) >= 2:
        cv2.polylines(canvas, [np.round(pts).astype(np.int32)], False, color, width, cv2.LINE_AA)


def _put_labels(canvas, labels):
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    for x, y, text, size, color_bgr, anchor in labels:
        draw.text((x, y), text, font=FONT(size), fill=color_bgr[::-1], anchor=anchor)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)




class DemoRenderer:
    """逐格合成展示畫面；GUI 播放與匯出影片共用。"""

    def __init__(self, result, summary_values, imu_knee=None, bboxes=None,
                 header_right='', footer=VIDIMU_FOOTER):
        """result 為 gait.TrialResult；imu_knee 為逐格近側腿 IMU 膝角，bboxes 為逐格人物框（皆可省略）。"""
        self.result = result
        self.values = summary_values
        self.imu_knee = imu_knee
        self.bboxes = bboxes
        self.header_right = header_right
        self.footer = footer
        self.hip = near_series(result, {s: result.angles[s]['hip'] for s in ('L', 'R')})
        self.knee = near_series(result, {s: result.angles[s]['knee'] for s in ('L', 'R')})
        self.near_by_frame = {f: p.near_side for p in result.passes for f in range(p.start, p.end)}

        strides = result.strides
        self.hip_markers = []
        if not strides.empty:
            near = strides[(strides['near_far'] == 'near') & strides['valid']].sort_values('hs_end')
            for s in near.itertuples():
                seg = result.angles[s.side]['hip'][s.hs_start:s.hs_end + 1]
                self.hip_markers.append((s.hs_start + int(np.nanargmin(seg)), np.nanmin(seg)))
        near_events = result.events[result.events['near_far'] == 'near']
        self.event_frames = {(r.frame, r.event) for r in near_events.itertuples()}

    @property
    def n_frames(self):
        return len(self.result.xy)

    def render_frame(self, frame_idx, frame):
        """frame 為原始影像（BGR），回傳 1280×720 的展示畫面。"""
        result, fps = self.result, self.result.fps
        bbox = self.bboxes[frame_idx] if self.bboxes is not None else np.full(4, np.nan)
        frame = blur_face(frame.copy(), result.keypoints[frame_idx], bbox)
        near_side = self.near_by_frame.get(frame_idx)
        frame = draw_skeleton(frame, result.xy[frame_idx], near_side)

        canvas = np.full((H, W, 3), BG, np.uint8)
        bx0, by0, bx1, by1 = VIDEO_BOX
        # 等比例縮放置中（非 4:3 的影片不變形）
        scale = min((bx1 - bx0) / frame.shape[1], (by1 - by0) / frame.shape[0])
        fw, fh = int(round(frame.shape[1] * scale)), int(round(frame.shape[0] * scale))
        vx0, vy0 = bx0 + (bx1 - bx0 - fw) // 2, by0 + (by1 - by0 - fh) // 2
        canvas[by0:by1, bx0:bx1] = (30, 30, 30)
        canvas[vy0:vy0 + fh, vx0:vx0 + fw] = cv2.resize(frame, (fw, fh))
        sx = sy = scale
        labels = [
            (20, 14, '一台鏡頭側拍走路 → 自動算出關節角度', 26, INK, 'la'),
            (W - 20, 18, self.header_right, 20, MUTED, 'ra'),
        ]
        for lag in range(0, 8):
            for event, text in (('HS', '著地'), ('TO', '離地')):
                if (frame_idx - lag, event) in self.event_frames and near_side:
                    part = 'heel' if event == 'HS' else 'toe'
                    p = result.xy[frame_idx, side_index(part, near_side)]
                    if np.isfinite(p).all():
                        labels.append((vx0 + p[0] * sx - 18, vy0 + p[1] * sy + 10, text, 18, (0, 90, 200), 'la'))
        if near_side:
            cv2.rectangle(canvas, (vx0 + 6, vy0 + 6), (vx0 + 196, vy0 + 34), (50, 50, 50), -1)
            labels.append((vx0 + 14, vy0 + 10, f'近側腿：{"左" if near_side == "L" else "右"}腳（橘色）',
                           17, (255, 255, 255), 'la'))

        t_now = frame_idx / fps
        draw_chart(canvas, labels, HIP_BOX, '髖關節屈伸角度（正值＝屈曲，負值＝伸展）', (-30, 50),
                   (-20, 0, 20, 40), [(self.hip, VIDEO_BGR, 3)], t_now, fps, markers=self.hip_markers)
        knee_series = [(self.knee, VIDEO_BGR, 3)]
        if self.imu_knee is not None:
            knee_series.insert(0, (self.imu_knee, IMU_BGR, 2))
        draw_chart(canvas, labels, KNEE_BOX, '膝關節屈曲角度', (-10, 80), (0, 20, 40, 60),
                   knee_series, t_now, fps)
        labels.append((KNEE_BOX[0], KNEE_BOX[3] + 6, '— 影片（單一鏡頭）', 15, VIDEO_BGR, 'la'))
        if self.imu_knee is not None:
            labels.append((KNEE_BOX[0] + 170, KNEE_BOX[3] + 6, '— 穿戴式感測器 IMU', 15, IMU_BGR, 'la'))
        labels.append((KNEE_BOX[2], KNEE_BOX[3] + 6, f'最近 {WINDOW_S:.0f} 秒', 15, MUTED, 'ra'))

        _draw_panel(canvas, labels, self.values)
        if self.footer:
            labels.append((20, 692, self.footer, 16, MUTED, 'la'))
        return _put_labels(canvas, labels)


def render(renderer, video_path, out_path, progress=None, should_stop=None):
    """把整支影片合成為展示影片。progress(done, total) 回報進度；should_stop() 為 True 時中止並回傳 False。"""
    cap = cv2.VideoCapture(video_path)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), renderer.result.fps, (W, H))
    total = renderer.n_frames
    frame_idx = 0
    try:
        while frame_idx < total:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(renderer.render_frame(frame_idx, frame))
            frame_idx += 1
            if progress and frame_idx % 30 == 0:
                progress(frame_idx, total)
            if should_stop and should_stop():
                return False
    finally:
        cap.release()
        writer.release()
    return True


def _draw_panel(canvas, labels, values):
    y0, y1 = PANEL_Y
    width = (W - 40 - 3 * 16) / 4
    for i, item in enumerate(panel_items(values)):
        x0 = 20 + i * (width + 16)
        cv2.rectangle(canvas, (int(x0), y0), (int(x0 + width), y1), (245, 247, 250), -1)
        cv2.rectangle(canvas, (int(x0), y0), (int(x0 + width), y1), (222, 215, 208), 1)
        cx = x0 + width / 2
        labels.append((cx, y0 + 10, item['label'], 18, INK, 'ma'))
        labels.append((cx, y0 + 38, item['value'], 38, INK, 'ma'))
        labels.append((cx, y0 + 100, item['note'], 14, MUTED, 'ma'))

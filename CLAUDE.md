# CLAUDE.md — 專案重要資訊

## 專案說明

人體骨架分析系統，結合 YOLO 物件偵測、ViTPose 骨架姿態估計（26 關節點 HALPE 格式）、BoTSORT 多目標追蹤，以 PyQt5 GUI 呈現；用單一側拍鏡頭計算步態指標，並以 VIDIMU 的 IMU 關節角度驗證。

## 執行環境

| 項目 | 內容 |
|------|------|
| Conda 環境 | `Gait` |
| Python | 3.8.20 |
| PyTorch | 2.0.1+cu118 |
| CUDA | 11.8 |

**啟動指令：**
```bash
conda activate Gait
cd Src/UI_Control
python main.py
```

## 關鍵套件版本（Gait 環境）

| 套件 | 版本 |
|------|------|
| torch | 2.0.1+cu118 |
| mmpose | 1.3.1（editable，來自本專案 `Src/mmpose_main`） |
| mmcv | 2.0.0 |
| mmengine | 0.10.7 |
| mmdet | 3.1.0 |
| mmpretrain | 1.2.0（editable，來自本專案 `Src/mmpretrain_main`） |
| ultralytics | 8.4.203 |
| PyQt5 | 5.15.11 |
| opencv-python | 4.8.1.78 |

## 模型檔案位置

```
Db/pretrain/
├── epoch_210.pth          # ViTPose 姿態估計模型（HALPE 26 關節點）
└── yolo26m-seg.pt         # YOLO26 人體偵測模型
```

## 程式入口與架構

```
Src/UI_Control/
├── main.py                  # 主程式入口（只有步態分析一個畫面）
├── gait_widget.py           # 步態分析主畫面（載入、播放、點選目標、批量分析、匯出）
├── gait_views.py            # 顯示元件（FrameView、AnglePlot、SummaryPanel）
├── ui/main_window.py        # PyQt5 主視窗 UI
└── utils/model.py           # 模型載入（YOLO + ViTPose + BoTSORT）

Src/gait_analysis/           # 萃取（extract_keypoints）、指標（gait）、流程與輸出（pipeline）、繪製（render_demo）
```

「2D 影片」分頁（`video_widget.py`、`cv_utils/`、`skeleton/`、`ui_utils/`、`utils/` 內的繪圖／選人工具、`camera_config/`）已於 2026-11-17 移除，需要時可從 git 歷史取回。

## mmpretrain 安裝注意事項

mmpretrain 需從本專案 `Src/mmpretrain_main` 以 editable 方式安裝（ViTPose backbone 需要）：
```powershell
Remove-Item -Recurse -Force Src\mmpretrain_main\mmpretrain\.mim
cd Src\mmpretrain_main
python setup.py develop --no-deps
cd ..\..
```

`utils/model.py` 中已加入 `import mmpretrain` 確保 VisionTransformer 被註冊到 mmengine registry。

## mmpose 安裝注意事項

mmpose 需從本專案的 `Src/mmpose_main` 以 editable 方式安裝：
```powershell
Remove-Item -Recurse -Force Src\mmpose_main\mmpose\.mim
cd Src\mmpose_main
python setup.py develop --no-deps
cd ..\..
```

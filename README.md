# NckuH-Gait-Skeleton

人體骨架與步態分析系統，整合 YOLO26 物件偵測、ViTPose 骨架姿態估計（Halpe 26 關節點）與 BoTSORT 多目標追蹤：

- **GUI**（PyQt5）：對影片逐格偵測骨架、選人、匯出 JSON／CSV，支援整個資料夾批量處理。
- **步態分析**（`Src/gait_analysis`）：用單一側拍鏡頭的 2D 關節點計算步態指標，並與 VIDIMU 公開資料集的 IMU 關節角度比對。

---

## 目錄

1. [環境需求](#環境需求)
2. [安裝步驟](#安裝步驟)
3. [模型準備](#模型準備)
4. [啟動程式](#啟動程式)
5. [UI 操作說明](#ui-操作說明)
6. [輸出格式](#輸出格式)
7. [步態分析（VIDIMU 展示版）](#步態分析vidimu-展示版)
8. [攝影機支援](#攝影機支援)
9. [專案結構](#專案結構)
10. [常見問題](#常見問題)

---

## 環境需求

| 項目 | 版本 |
|------|------|
| OS | Windows 10 / 11 |
| Python | 3.8 |
| CUDA | 11.8 |
| cuDNN | 對應 CUDA 11.8 |
| Anaconda | 任意版本 |

> GPU 為必要條件，CPU-only 模式未支援。

---

## 安裝步驟

> **注意**：所有指令請在 **Anaconda Prompt** 或已初始化 conda 的 PowerShell 中執行。
> PowerShell 需先執行 `conda init powershell` 並重新開啟視窗，才能使用 `conda activate`。

---

### Step 0 — 取得原始碼

```bash
git clone https://github.com/partner0487/NckuH_Gait_Skeleton.git
cd NckuH_Gait_Skeleton
```

以下指令都在這個專案根目錄執行。

---

### Step 1 — 建立 Conda 環境

```bash
conda create -n Gait python=3.8 -y
conda activate Gait
```

---

### Step 2 — 安裝 PyTorch（CUDA 11.8）

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

---

### Step 3 — 安裝基本套件

在專案根目錄執行，包含 PyQt5 GUI 所需套件。CSV 匯出使用 Python 內建的 `csv` 模組，不需額外安裝套件。

新版 `lapx` 從原始碼編譯時預設需要 Rust，先設定環境變數略過即可（只編譯不需要 Rust 的既有演算法）：

```powershell
# PowerShell
$env:LAPX_BUILD_RUST = "0"
pip install -r requirements.txt
```

```bat
:: Anaconda Prompt（cmd）
set LAPX_BUILD_RUST=0
pip install -r requirements.txt
```

---

### Step 4 — 驗證 PyQt5

PyQt5 已包含在 `requirements.txt`，完成 Step 3 後可確認安裝版本：

```bash
python -c "from PyQt5.QtCore import PYQT_VERSION_STR; print(PYQT_VERSION_STR)"
# 應輸出: 5.15.11
```

---

### Step 5 — 安裝 OpenMMLab 套件

```bash
pip install -U openmim
mim install "mmcv==2.0.0"
mim install "mmdet==3.1.0"
mim install mmengine
```

---

### Step 6 — 安裝 mmpose（從本專案原始碼）

Windows 上 `.mim` 資料夾有權限問題，**必須先刪除再安裝**：

```powershell
# 在專案根目錄執行
# 刪除 .mim 資料夾（若不存在會顯示錯誤可忽略）
Remove-Item -Recurse -Force Src\mmpose_main\mmpose\.mim

# 安裝
cd Src\mmpose_main
python setup.py develop --no-deps
cd ..\..
```

---

### Step 7 — 安裝 mmpretrain（從本專案原始碼）

ViTPose 的 Vision Transformer backbone 需要此套件，同樣需先刪除 `.mim`：

```powershell
# 在專案根目錄執行
# 刪除 .mim 資料夾（若不存在會顯示錯誤可忽略）
Remove-Item -Recurse -Force Src\mmpretrain_main\mmpretrain\.mim

# 安裝
cd Src\mmpretrain_main
python setup.py develop --no-deps
cd ..\..
```

---

### Step 8 — 安裝 faiss 與 cython_bbox

```bash
conda install -c conda-forge faiss-gpu "mkl>=2025,<2026" -y
pip install cython_bbox
```

> **一定要固定 MKL 版本**：只寫 `faiss-gpu` 時，conda 會一併裝上 MKL 2026。這版只提供 `mkl_intel_thread.3.dll`，而 numpy 需要 `.2.dll`，結果 numpy、torch 全部無法 import（錯誤訊息：`Cannot load mkl_intel_thread.2.dll`）。若已經發生，執行 `conda install -c conda-forge "mkl>=2025,<2026" -y` 即可修復。

---

### Step 9 — 驗證安裝

執行以下指令，確認所有關鍵套件正常：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 應輸出: 2.0.1+cu118 True

python -c "import mmpose; print(mmpose.__version__)"
# 應輸出: 1.3.1

python -c "import mmpretrain; print(mmpretrain.__version__)"
# 應輸出: 1.2.0

python -c "from ultralytics import YOLO; print('YOLO OK')"
# 應輸出: YOLO OK
```

> **libiomp5md.dll 衝突**：若出現 OpenMP 相關警告，請刪除：
> `C:\Users\<你的帳號>\anaconda3\envs\Gait\Library\bin\libiomp5md.dll`

---

## 模型準備

模型權重檔不在 GitHub 上，請分別下載：

| 檔案名稱 | 用途 | 大小（約）| 下載 |
|---------|------|---------|------|
| `epoch_210.pth` | ViTPose 姿態估計模型（HALPE 26 關節點） | ~1.1 GB | [Google Drive](https://drive.google.com/drive/folders/1FJDGG1D4qhj-ivjHMgmz0nLkM_zeKDjX?usp=drive_link) |
| `yolo26m-seg.pt` | YOLO26 人體偵測模型（Ultralytics 官方 COCO 權重，需要 ultralytics 8.4 以上） | ~52 MB | [Ultralytics v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-seg.pt) |

`yolo26m-seg.pt` 的 SHA256 為 `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`。舊版的 `yolo11m-seg.pt` 已不再使用。

下載後，將兩個檔案放至專案的 `Db/pretrain/` 資料夾（若不存在請手動建立）：

```
NckuH_Gait_Skeleton/
├── Db/
│   ├── pretrain/
│   │   ├── epoch_210.pth        ← 下載後放這裡
│   │   └── yolo26m-seg.pt       ← 下載後放這裡
│   └── Record/                  ← 輸出資料夾（自動建立）
└── Src/
```

---

## 啟動程式

```bash
conda activate Gait
cd Src\UI_Control
python main.py
```

---

## UI 操作說明

### 介面總覽

```
┌─────────────────────────────────────┬──────────────────────────┐
│                                     │  2D 關節點               │
│                                     │  ┌──────────────────┐    │
│                                     │  │  檔案            │    │
│         影片顯示區                   │  │  檔名: xxx       │    │
│         (FrameView)                 │  │  FPS: 280        │    │
│                                     │  │ [載入原始影片]    │    │
│                                     │  │ [處理和儲存]      │    │
│                                     │  │ [載入處理過的影片] │    │
│                                     │  │ [批量處理資料夾]   │    │
│                                     │  └──────────────────┘    │
│                                     │                          │
│                                     │  顯示資訊                │
│                                     │  ☑ 人體骨架  □ 人物框    │
│                                     │  □ 選擇人                │
├─────────────────────────────────────┤                          │
│  [<<]  [▶]  [>>]   ────────●────   │                          │
│                      394 / 580      │                          │
└─────────────────────────────────────┴──────────────────────────┘
```

---

### 按鈕與功能說明

#### 檔案區

| 按鈕 | 功能 |
|------|------|
| **載入原始影片** | 開啟檔案選擇對話框，載入未處理的 MP4 影片。影片載入後會自動逐幀解析並顯示第 0 幀。 |
| **處理和儲存** | 從第 0 幀開始處理，抵達最後一幀後補做尚未處理的影格，再將原畫面影片、人物框＋骨架影片（`_Sk26.mp4`）、JSON 與 CSV 儲存至 `Db/Record/{影片名稱}/`。 |
| **載入處理過的影片** | 載入影片及對應的骨架 JSON，直接顯示人物框與骨架，不重新執行模型；支援原畫面影片與 `_Sk26.mp4`。 |
| **批量處理資料夾** | 選擇資料夾後，依檔名自然排序（`T2` 排在 `T10` 之前）逐支以「處理和儲存」流程處理資料夾內的 `.mp4`／`.avi`，每支存完才接下一支。處理中按鈕變成「停止批量處理」。詳見下方「操作流程：批量處理」。 |

#### 播放控制區

| 按鈕 | 功能 |
|------|------|
| **`<<`** | 退回上一幀 |
| **`▶` / `\|\|`** | 播放 / 暫停 |
| **`>>`** | 前進下一幀 |
| **進度條（Slider）** | 拖曳跳至任意幀，每移動一格就會觸發該幀的骨架偵測 |

> **鍵盤快捷鍵**：按住 `A` 退後一幀，按 `D` 前進一幀（需視窗取得焦點）。

#### 顯示資訊區（Checkbox）

| Checkbox | 功能 |
|----------|------|
| **人體骨架** | 顯示／隱藏預覽骨架，同時啟用／停用預覽時的骨架偵測；不影響儲存前補做未處理影格 |
| **人物框** | 立即顯示／隱藏預覽中的人物框與 ID，暫停時也會更新；需已有該影格的偵測資料 |
| **選擇人** | 勾選後顯示可選人物框，左鍵點選人物可指定追蹤 ID；原始影片只對該人物執行骨架估計，取消勾選則恢復全部人物。已處理影片則篩選已儲存的人物資料。 |

輸出的 `_Sk26.mp4` 固定啟用人物框與骨架，不受預覽顯示開關影響。未選人時對所有有效追蹤人物估計骨架；指定人物後，儲存前會以目前選定的追蹤 ID 補齊全片的骨架估計，影片、JSON 與 CSV 都只保留該人物的結果。

---

### 操作流程：對影片進行骨架偵測

以下為完整的操作步驟（以跑步影片為例）：

**① 載入影片**

點擊「**載入原始影片**」→ 在對話框中選擇 MP4 影片 → 等待讀取完成（上方會顯示檔案名稱與解析度）。

**② 開始偵測**

點擊「**處理和儲存**」→ 程式會自動從第 0 幀開始播放並逐幀進行骨架偵測。

> 偵測速度取決於 GPU 效能、人物數與影片解析度。FPS 顯示骨架處理計時結果，已處理影片顯示「—」；不代表原影片幀率或完整輸出速度。

**③ 選取追蹤目標（選擇人）**

1. 載入原始影片，勾選「**選擇人**」，程式會啟用骨架顯示並顯示可選人物框
2. 用滑鼠左鍵點擊想分析的人物框，該影格會立即改為只估計此人物的骨架
3. 點擊「**處理和儲存**」，從頭處理並輸出此人物的結果；可點選其他人物切換目標
4. 取消「**選擇人**」可恢復對所有有效追蹤人物進行骨架估計

> 人物 ID 使用 BoTSORT 追蹤結果，拖曳進度條時會依時間順序補做前面的偵測與追蹤，以維持 ID 對應。遮擋或離開畫面仍可能導致追蹤中斷；選定 ID 未出現時，不會改以其他人物或整張畫面估計骨架。

**④ 儲存結果**

- 點擊「**處理和儲存**」後抵達最後一幀會自動儲存；一般預覽或選人不會自行觸發儲存
- 儲存前會補做尚未處理的影格，這個階段可能需要等待
- 輸出檔案儲存至 `Db/Record/{影片名稱}/` 資料夾，同名檔案會覆寫

**⑤ 檢視輸出與重新載入**

可直接使用影片播放器開啟 `{影片名稱}_Sk26.mp4` 檢視人物框與骨架，或使用 Excel 開啟 CSV 分析座標。

使用「**載入處理過的影片**」選擇輸出資料夾中的 `{影片名稱}.mp4` 或 `{影片名稱}_Sk26.mp4`，兩者都會對應到同資料夾的 `{影片名稱}.json`，不再需要舊版球棒／揮棒 JSON。建議選擇未繪製標記的 `{影片名稱}.mp4`，以便自由切換人物框與骨架顯示；`_Sk26.mp4` 本身已包含的標記無法移除。

載入後會直接顯示儲存結果，缺少人物的影格也不會重新偵測。若 JSON 不存在或缺少必要欄位，程式會顯示錯誤訊息。此模式的「選擇人」只篩選 JSON 中已有的人物；若要重新估計或分析未儲存的人物，請以「**載入原始影片**」開啟未繪製標記的影片。CSV 目前僅供匯出分析，不作為重新載入來源。

---

### 操作流程：批量處理

1. 點擊「**批量處理資料夾**」，選擇影片所在資料夾（例如 `Db_NckuH_Gait\Db_VideoSmallSize\Db_VideoSOriginal\S01`）
2. 程式依檔名排序後逐支載入、從第 0 幀處理到最後一幀並儲存，輸出位置與單支處理相同（`Db/Record/{影片名稱}/`）
3. 檔名後方會顯示進度，例如 `S01_A03_T01 (5/26)`；全部完成後跳出「已處理並儲存 X/Y 支影片」

注意事項：

- 只處理所選資料夾本身，不會往子資料夾找；S01、S02… 需分別選擇
- 批量期間「載入原始影片」「載入處理過的影片」「處理和儲存」會停用
- 每支影片都偵測畫面中所有人；選人設定在每支開始時重置
- 播放中按「停止批量處理」會等目前這支存完再停；暫停狀態下按則立即結束
- 中途手動暫停會停在該支，按播放才會繼續
- 損壞或讀不到畫面的影片會卡在「讀取影片中」，請先移除

---

### 骨架關節點對照（HALPE 26 點）

| ID | 部位 | ID | 部位 |
|----|------|----|------|
| 0 | 鼻子 | 13 | 左膝 |
| 1 | 左眼 | 14 | 右膝 |
| 2 | 右眼 | 15 | 左踝 |
| 3 | 左耳 | 16 | 右踝 |
| 4 | 右耳 | 17 | 頭部 |
| 5 | 左肩 | 18 | 頸部 |
| 6 | 右肩 | 19 | 臀部 |
| 7 | 左肘 | 20 | 左大腳趾 |
| 8 | 右肘 | 21 | 右大腳趾 |
| 9 | 左腕 | 22 | 左小腳趾 |
| 10 | 右腕 | 23 | 右小腳趾 |
| 11 | 左髖 | 24 | 左腳跟 |
| 12 | 右髖 | 25 | 右腳跟 |

---

## 輸出格式

分析完成後，結果儲存至 `Db/Record/{影片名稱}/`：

| 檔案 | 說明 |
|------|------|
| `{影片名稱}.mp4` | 由載入的影格重新編碼的原畫面影片，未繪製骨架與人物框 |
| `{影片名稱}_Sk26.mp4` | 逐幀繪製人物框與骨架；指定人物時只輸出該人物，不受預覽顯示開關影響 |
| `{影片名稱}.json` | 逐幀骨架資料（含 Bounding Box + 26 關節點座標） |
| `{影片名稱}.csv` | 每列一個影格／人物／關節點，含人物框、座標、信心值與人工修改標記 |

CSV 與 JSON 同時儲存。CSV 採用 UTF-8 BOM 編碼，可直接以 Excel 開啟；欄位為 `frame_number`（從 0 開始）、`person_id`、`bbox_x1`、`bbox_y1`、`bbox_x2`、`bbox_y2`、`keypoint_id`（0–25）、`keypoint_name`、`x`、`y`、`confidence`、`is_modified`（0：未修改；1：人工修改）。座標以原始影片像素為單位；沒有偵測資料時僅輸出標題列。

兩份 MP4 都由 OpenCV 逐幀寫入，不保留原影片音軌。原畫面影片也不是來源檔案的逐位元複製。

「處理和儲存」會從第 0 幀開始處理，儲存前會補做尚未處理的影格。輸出的骨架影片固定啟用人物框與骨架；模型未偵測到人物的影格仍保留原畫面，不補造座標。

JSON 使用 records 陣列，每筆為一個影格中的一個人物。以下範例僅節錄前兩個關節點，完整資料為 26 點；每點依序為 `[x, y, confidence, is_modified]`：
```json
[
  {
    "person_id": 1,
    "bbox": [100.0, 50.0, 300.0, 450.0],
    "keypoints": [
      [200.0, 80.0, 0.98, 0.0],
      [195.0, 75.0, 0.95, 1.0]
    ],
    "frame_number": 0
  }
]
```

---

## 步態分析（VIDIMU 展示版）

依《步態指標定義書（VIDIMU 側面版）》第四節實作展示版：用 **VIDIMU** 公開資料集（[Zenodo 8210563](https://zenodo.org/records/8210563)，CC BY 4.0）的側拍行走影片計算步態指標，並與同一次行走的 IMU 關節角度比對。程式不經過 GUI，直接以命令列執行。

### 資料放置

VIDIMU 資料集（約 2.8 GB）不在 GitHub 上，請自行下載後放到專案根目錄的 `Db_NckuH_Gait/`：

```
Db_NckuH_Gait/
├── Database/
│   ├── VideoAndImus/S40 … S57/    # ik_*.mot（IMU 關節角度，50 Hz）
│   └── BodyMeasurements.csv
├── Analysis/VideoAndImuSync/infoToSync.csv   # 影片與 IMU 的同步裁切量
└── Db_VideoSmallSize/Db_VideoSOriginal/S01 … S57/   # 側拍影片（640×480，30 fps）
```

有 IMU 的受試者共 16 位（S40–S57，原作者已剔除 S43、S45）。

### 執行

```bash
conda activate Gait

# 1. 萃取關節點（GPU，16 支 A01 約 10 分鐘；已萃取過的會跳過）
python Src/gait_analysis/run_extract.py

# 2. 計算指標、與 IMU 比對、產生展示資料（自動挑 3 位）
python Src/gait_analysis/run_demo.py

# 指定展示受試者
python Src/gait_analysis/run_demo.py --demo-subjects S40 S42 S47
```

### 輸出（`Db/Gait/`）

| 檔案 | 內容 |
|------|------|
| `{試驗}/{試驗}_keypoints.csv` | 一列一格：26 點原始 x、y、conf 與濾波後 x、y |
| `{試驗}/{試驗}_events.csv` | 一列一個著地（HS）／離地（TO）事件：趟、側別、近側／遠側、是否為該趟首末步 |
| `{試驗}/{試驗}_strides.csv` | 一列一個步態週期：A1 髖角、A2 膝角、B6 軀幹前傾、B1 步頻、是否有效 |
| `{試驗}/{試驗}_curves.csv` | 一列一個週期百分點（0–100%）：影片 hip／knee／trunk 與 IMU imu_hip／imu_knee |
| `agreement_A01.csv` | 全部試驗的 IMU 一致性（RMSE、扣平均差 RMSE、Pearson r）與面板數值 |
| `demo/{受試者}_demo.mp4` | 展示影片：左半骨架疊在行走畫面（臉部模糊），右半髖、膝曲線同步畫出，膝角疊 IMU |
| `demo/{受試者}_knee_overlay.png` | 膝角曲線疊圖（影片 vs IMU，各週期細線與平均粗線） |
| `demo/{受試者}_panel.png` | 數字面板：步頻、膝活動範圍、最大髖伸展、軀幹前傾角 |

### 展示版指標

| 代號 | 指標 | 計算方式 |
|------|------|----------|
| A1 | 髖屈伸角度（含最大髖伸展） | 大腿角 − 軀幹角；最大髖伸展 = 每週期最小值 × −1 |
| A2 | 膝屈曲角度 | 大腿角 − 小腿角 |
| B1 | 步頻（步/分） | 60 ÷ 左右腳相鄰兩次著地的平均間隔；遠側腳偵測不到時改用 120 ÷ 近側腿週期時間 |
| B6 | 軀幹前傾角 | 頸部（雙肩中點）在骨盆（雙髖中點）前方為正；往右、往左兩個方向分別平均後再平均，抵銷相機傾斜 |

節段角一律從鉛直向下量起、遠端往前為正。A 類只用近側腿（每趟膝、踝、腳跟平均信心值較高的一側）。

### 處理流程與門檻值

所有門檻值集中在 [`Src/gait_analysis/gait_config.yaml`](Src/gait_analysis/gait_config.yaml)：

1. **前處理**：信心值 < 0.3 視為缺值；連續缺值 ≤ 5 格用三次樣條補值；4 階 Butterworth 6 Hz 零相位低通濾波
2. **切趟**：骨盆水平速度方向不變且大於該段平均速度 50% 的直線段（至少 1 秒），轉身不算
3. **近側腿**：膝、踝、腳跟平均信心值較高的一側
4. **著地／離地**（Zeni 2008）：腳跟相對骨盆最前 = HS，腳尖相對骨盆最後 = TO；同一隻腳兩次 HS 至少間隔 0.6 秒，且必須是 HS → TO → HS 交替
5. **IMU 對齊**：依 `infoToSync.csv` 的裁切量換算時間偏移，再以膝角在 ±10 格內找最佳位移

### 與定義書不同的實作細節

以下幾點是在 VIDIMU 資料上實測後補上的，都記錄在輸出檔中可供檢查：

| 項目 | 定義書 | 實作 | 原因 |
|------|--------|------|------|
| 近側腿 | 每趟各自判斷 | 同一行進方向的各趟合併判斷 | 單趟兩腳信心值常只差 0.01，逐趟判斷會偶爾判反；相機固定時同方向的近側腿必定相同 |
| IMU 對齊 | 同步結果 + ±10 格互相關 | 第一趟 ±10 格，之後每趟在前一趟位移的 ±10 格內追蹤，再以直線擬合 | S41、S54 的影片時鐘與 IMU 有線性漂移（約 +50～60 格/分），固定位移會讓膝角相關係數掉到 0.1～0.2 |
| 剔除步 | HS → TO → HS 不交替者 | 另外剔除週期時間偏離該試驗中位數 30% 以上者（`_strides.csv` 的 `invalid_reason`） | 轉身前後會重複偵測著地，產生 0.6 秒、離地點落在 11% 的假週期 |
| 步頻 | 60 ÷ 平均步時 | 步時只採 0.3～1.0 秒（週期上下限的一半），試驗層級以步時加權平均 | 只有一個步時的短趟會算出 600 步/分 |

### 已知限制

- VIDIMU 受試者是年輕健康成人，數值不能套用高齡切點；畫面與報告一律寫「與 IMU 的一致性為初步結果」，不寫「臨床驗證」
- IMU（OpenSim IK）不是金標準。膝角在著地時常有數度的固定偏差（IMU 呈現過伸），因此同時報告扣除平均差後的 RMSE
- 本定義的髖角是大腿對**軀幹**，IMU 與文獻是大腿對**骨盆**，數值不能直接比對文獻切點
- 每趟只有 2～3 步，每支影片的近側有效週期約 6～13 個
- 受試者穿深色長褲時兩腳難以區分，模型會左右互換（如 S56，膝角相關係數僅 0.58），這類試驗不適合用於展示
- 影片 metadata 的 fps 寫 29／30／31 不等，但幀數 ÷ 30 與 IMU 時長吻合，程式固定以 30 fps 計算

---

## 攝影機支援

| 攝影機類型 | 狀態 | 說明 |
|-----------|------|------|
| 影片檔（MP4 等）| ✅ 支援 | 主要使用方式 |
| FLIR / Point Grey 工業相機 | ⚠️ 僅有底層擷取程式 | 需安裝 Spinnaker SDK + `EasyPySpin`；目前主視窗未啟用相機 Tab |
| Sony AX700 / 一般相機（HDMI 擷取卡）| 🔧 規劃中 | 透過 `cv2.VideoCapture()` 讀取，需新增 `SonyCaptureThread` |

---

## 專案結構

```
NckuH_Gait_Skeleton/
├── Db/                        # 不進 git
│   ├── pretrain/              # 模型權重檔
│   ├── Record/                # GUI 分析輸出（自動建立）
│   └── Gait/                  # 步態分析輸出（自動建立）
├── Db_NckuH_Gait/             # VIDIMU 資料集（不進 git，需自行下載）
├── Src/
│   ├── gait_analysis/         # 步態指標（VIDIMU 展示版）
│   │   ├── gait_config.yaml   # 門檻值設定
│   │   ├── run_extract.py     # 批次萃取關節點
│   │   ├── run_demo.py        # 指標計算、IMU 比對、展示資料
│   │   ├── gait.py            # 前處理、切趟、著地離地、角度與指標
│   │   ├── imu_compare.py     # 影片與 IMU 角度比對
│   │   ├── vidimu.py          # VIDIMU 檔案存取與同步
│   │   ├── plots.py           # 曲線疊圖、數字面板
│   │   └── render_demo.py     # 展示影片
│   ├── UI_Control/            # 主程式
│   │   ├── main.py            # 入口點
│   │   ├── video_widget.py    # 影片分析 Tab 邏輯
│   │   ├── ui/                # PyQt5 UI 定義（.py + .ui）
│   │   ├── skeleton/          # 骨架偵測核心（ViTPose + YOLO + tracker）
│   │   ├── cv_utils/          # 影像控制與執行緒
│   │   ├── utils/             # 分析、濾波、視覺化工具
│   │   └── camera_config/     # 攝影機設定
│   ├── mmengine_main/        # OpenMMLab mmengine 原始碼
│   ├── mmpretrain_main/      # ViTPose backbone（editable 安裝）
│   ├── mmpose_main/           # OpenMMLab mmpose（本地版本，editable 安裝）
│   ├── mmyolo_main/           # OpenMMLab mmyolo
│   ├── tracker/               # BoTSORT 多目標追蹤器
│   └── cython_bbox-0.1.3/     # Bounding Box 計算工具
├── requirements.txt           # Python 套件依賴清單
├── CLAUDE.md                  # 開發環境紀錄（給 Claude Code 使用）
└── README.md                  # 本文件
```

---

## 常見問題

**Q：程式啟動時出現 `ModuleNotFoundError: No module named 'mmpose'`**

A：mmpose 需從本專案原始碼安裝。執行：
```bash
conda activate Gait
cd Src\mmpose_main
python setup.py develop --no-deps
```
若出現 `PermissionError [WinError 5]`，先刪除 `Src\mmpose_main\mmpose\.mim` 資料夾。

---

**Q：程式啟動時出現 `ModuleNotFoundError: No module named 'ultralytics'`**

A：執行：
```bash
conda activate Gait
pip install ultralytics==8.4.203
```

---

**Q：出現 `EasyPySpin` 相關錯誤**

A：`EasyPySpin` 是 FLIR 工業相機的 SDK，非必要套件，已透過 `try/except` 處理。若仍出錯，代表程式碼有未預期的路徑直接使用它，請回報 issue。

---

**Q：`pip install -r requirements.txt` 在 `lapx` 失敗，訊息提到 Rust 或 Cargo**

A：先設定 `LAPX_BUILD_RUST=0` 再重新安裝（見 Step 3）。

---

**Q：import numpy 或 torch 時出現 `Cannot load mkl_intel_thread.2.dll`**

A：conda 裝了 MKL 2026。執行 `conda install -c conda-forge "mkl>=2025,<2026" -y` 降版即可（見 Step 8）。

---

**Q：import onnx 時出現 `Descriptors cannot be created directly`**

A：protobuf 太新，`requirements.txt` 已固定為 `protobuf==3.20.3`；重新執行 `pip install protobuf==3.20.3`。注意 `onnx==1.8.1` 與 `numpy 1.24` 本身不相容（`np.object` 已移除），只影響 `Src/tracker/fast_reid/tools/deploy/` 的 ONNX 匯出工具，主程式不受影響。

---

**Q：出現 `libiomp5md.dll` 衝突警告**

A：刪除 `C:\Users\<帳號>\anaconda3\envs\Gait\Library\bin\libiomp5md.dll` 即可解決。

---

**Q：影片偵測速度很慢**

A：確認 `torch.cuda.is_available()` 回傳 `True`。若為 `False`，代表 PyTorch 使用 CPU 模式，需重新安裝 CUDA 版本的 PyTorch（見 Step 2）。

---

**Q：到另一台電腦安裝時，mmpose setup.py 失敗**

A：常見原因是 `.mim` 資料夾的 Windows 權限問題。步驟：
1. 手動刪除 `Src\mmpose_main\mmpose\.mim`
2. 再次執行 `python setup.py develop --no-deps`

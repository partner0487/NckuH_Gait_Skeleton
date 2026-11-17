# NckuH-Gait-Skeleton

人體骨架與步態分析系統，整合 YOLO26 物件偵測、ViTPose 骨架姿態估計（Halpe 26 關節點）與 BoTSORT 多目標追蹤：

- **GUI**（PyQt5）：載入側拍行走影片，偵測骨架、點選目標人物、即時顯示步態指標與角度曲線，並匯出 JSON／CSV 與展示影片；支援一次批量分析資料集內所有受試者的步態影片。
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
8. [專案結構](#專案結構)
9. [常見問題](#常見問題)

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

主視窗就是「步態分析」介面：載入側拍行走影片後，程式在背景偵測畫面中所有人的骨架並給每個人追蹤 ID，再計算目標人物的步態指標。指標定義與 VIDIMU 驗證見[步態分析](#步態分析vidimu-展示版)一節。

1. **載入影片**：選擇側拍行走影片（對話框預設開在 `Db/Record`）。程式在背景萃取關節點並計算指標，進度條顯示萃取進度；可按「取消」中止。萃取過的影片有快取（`Db/Record/_cache/keypoints/`），再次載入會直接讀取。分析後原始影片會複製一份到 `Db/Record/{影片名稱}/`，之後直接從那裡載入即可（同樣讀快取，VIDIMU 影片也照樣疊 IMU）。匯出的展示影片（`_demo.mp4`）與骨架影片（`_Sk26.mp4`）畫面已疊上圖表，程式會拒絕載入並提示原始影片
2. **播放**：左邊是影片與骨架（臉部模糊、近側腿橘色、著地／離地當下標記），右邊是近側腿的髖、膝角度曲線（有 IMU 時疊紅色虛線），下方是數字面板。畫面直接以 Qt 繪製，放大視窗也保持清晰。可用 `<<`、`▶`、`>>`、進度條，或鍵盤 `A`／`D` 逐格、空白鍵播放；**在曲線上拖曳時間游標或點擊**可跳到該時間。勾選「曲線跟隨播放」時曲線顯示目前時間前 4 秒、後 2 秒，取消勾選可用滑鼠滾輪縮放、拖曳平移整段曲線
3. **右側資訊**：目標人物、行走趟數、近側有效週期數；若有 IMU 則顯示膝角一致性（r、扣平均差 RMSE）
4. **匯出展示影片**：輸出 1280×720 MP4，並在同一位置輸出數字面板（`_panel.png`），有 IMU 時另輸出膝角疊圖（`_knee_overlay.png`）。預設位置 `Db/Record/{影片名稱}/{影片名稱}_demo.mp4`
5. **開啟輸出資料夾**：開啟 `Db/Record/{影片名稱}/`
6. **點選目標人物**：畫面中有兩人以上時，每個人都會畫出框與追蹤 ID（目標為橘色實線，其他人白色虛線）。點某個人的框就改以他為目標重算指標並重畫 `_Sk26.mp4`，選擇會記住，下次載入同一支影片沿用。按「**改回自動選擇目標**」回到自動（每格取畫面中最大、且與上一格位置連續的人；VIDIMU 這類單人影片不需要點選）
7. **批量分析步態影片**：自動讀取 `Db_NckuH_Gait/Db_VideoSmallSize/Db_VideoSOriginal/` 底下每位受試者（S01…S57）的步態影片 `Sxx_A01_Tnn.mp4`，略過 `_Npose` 檔。VIDIMU 每人的步態多半只有一趟且編號不固定（S01～S39 多為 T01，S41、S49 等為 T02，S50、S54 為 T03），所以有幾趟就跑幾趟，目前共 53 位、56 支。開始前會顯示支數、已有快取的支數與預估時間（新影片每支約 30 秒）。每支都會建立快取並寫出全部基本檔與步態 CSV；單支失敗不會中斷，結束時列出失敗的影片。可按「取消」在目前這支結束後停止。找不到資料集時會改為詢問影片根目錄（底下為 S01、S02… 資料夾）

不同影片的處理方式：

| 影片 | fps | IMU 疊線 | 資料來源註記 |
|------|-----|----------|--------------|
| VIDIMU 有 IMU 的試驗（S40–S57） | 固定 30 | ✅ | ✅ |
| VIDIMU 沒有 IMU 的試驗（S01–S39） | 固定 30 | — | ✅ |
| 其他影片（含檔名同為 `Sxx_Axx_Txx` 但不在資料集內者） | 讀影片 metadata | — | — |

非 4:3 的影片會等比例縮放置中，不會變形。

輸出檔（例如 CSV 在 Excel、`_Sk26.mp4` 在播放器）正被其他程式開著時，分析照常顯示，右側會提示哪個檔案沒更新；關閉後重新載入就會補寫。

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

輸出都放在 `Db/Record/`，**一支影片一個資料夾**。底線開頭的資料夾是共用資料，不是影片：

```
Db/Record/
├── _cache/
│   ├── keypoints/             # 關節點快取（.npz，含畫面中所有人），刪掉只會讓下次載入重新萃取
│   └── targets.json           # 手動點選的目標人物
├── _summary/                  # 跨影片彙總，如 agreement_A01.csv
└── {影片名稱}/
    ├── {影片名稱}.mp4、_Sk26.mp4、.json、.csv                 ← 基本檔（載入即輸出）
    ├── {影片名稱}_keypoints、_events、_strides、_curves.csv   ← 步態 CSV（載入即輸出）
    └── {影片名稱}_demo.mp4、_panel.png、_knee_overlay.png     ← 「匯出展示影片」／run_demo.py
```

每支影片不論有沒有 IMU 都有同一組基本檔與步態 CSV；有 IMU 時只是 `_curves.csv` 多了 IMU 欄位，匯出時多一張 `_knee_overlay.png`。

| 檔案 | 內容 |
|------|------|
| `{影片名稱}.mp4` | 原始影片的複本（逐位元複製，不重新編碼），方便之後從 `Db/Record` 直接載入 |
| `{影片名稱}.json`、`.csv` | 畫面中所有被追蹤到的人的骨架 |
| `{影片名稱}_Sk26.mp4` | 原解析度、原 fps；目標人物畫完整骨架（近側腿橘色）與橘框，其他人只畫框與 ID。只有目標或快取改變時才重畫 |
| `{影片名稱}_keypoints.csv` | 一列一格：26 點原始 x、y、conf 與濾波後 x、y |
| `{影片名稱}_events.csv` | 一列一個著地（HS）／離地（TO）事件：趟、側別、近側／遠側、是否為該趟首末步 |
| `{影片名稱}_strides.csv` | 一列一個步態週期：A1 髖角、A2 膝角、B6 軀幹前傾、B1 步頻、是否有效 |
| `{影片名稱}_curves.csv` | 一列一個週期百分點（0–100%）：影片 hip／knee／trunk 與 IMU imu_hip／imu_knee |
| `{影片名稱}_demo.mp4` | 展示影片：左半骨架疊在行走畫面（臉部模糊），右半髖、膝曲線同步畫出，膝角疊 IMU |
| `{影片名稱}_knee_overlay.png` | 膝角曲線疊圖（影片 vs IMU，各週期細線與平均粗線），只有 IMU 試驗才有 |
| `{影片名稱}_panel.png` | 數字面板：步頻、膝活動範圍、最大髖伸展、軀幹前傾角 |
| `_summary/agreement_A01.csv` | `run_demo.py` 產生：全部試驗的 IMU 一致性（RMSE、扣平均差 RMSE、Pearson r）與面板數值 |

`.csv` 採用 UTF-8 BOM 編碼，可直接以 Excel 開啟；每列一個影格／人物／關節點，欄位為 `frame_number`（從 0 開始）、`person_id`、`bbox_x1`、`bbox_y1`、`bbox_x2`、`bbox_y2`、`keypoint_id`（0–25）、`keypoint_name`、`x`、`y`、`confidence`、`is_modified`（保留欄位，固定為 0）。座標以原始影片像素為單位。

舊版「2D 影片」分頁（已移除）在 2026-11-16 以前輸出的 `.mp4`／`_Sk26.mp4` fps 被減半，播放會變成慢動作；在步態分析重新載入該影片就會以新版覆寫。

JSON 使用 records 陣列，每筆為一個影格中的一個人物。以下範例僅節錄前兩個關節點，完整資料為 26 點；每點依序為 `[x, y, confidence, is_modified]`：
```json
[
  {
    "person_id": 1,
    "bbox": [100.0, 50.0, 300.0, 450.0],
    "keypoints": [
      [200.0, 80.0, 0.98, 0.0],
      [195.0, 75.0, 0.95, 0.0]
    ],
    "frame_number": 0
  }
]
```

---

## 步態分析（VIDIMU 展示版）

依《步態指標定義書（VIDIMU 側面版）》第四節實作展示版：用 **VIDIMU** 公開資料集（[Zenodo 8210563](https://zenodo.org/records/8210563)，CC BY 4.0）的側拍行走影片計算步態指標，並與同一次行走的 IMU 關節角度比對。可以在 GUI 逐支操作（見[UI 操作說明](#ui-操作說明)），也可以用命令列批次產生。

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

### 命令列：批次執行

```bash
conda activate Gait

# 1. 萃取關節點並寫出基本檔（GPU，16 支 A01 約 20 分鐘；已是新格式快取的會跳過）
python Src/gait_analysis/run_extract.py

# 2. 計算指標、與 IMU 比對、產生展示資料（自動挑 3 位）
python Src/gait_analysis/run_demo.py

# 指定展示受試者
python Src/gait_analysis/run_demo.py --demo-subjects S40 S42 S47
```

### 輸出

檔案說明見[輸出格式](#輸出格式)。`run_demo.py` 只為挑出的 3 支展示試驗產生 `_demo.mp4`、`_panel.png`、`_knee_overlay.png`，結束時會列出路徑。

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
- 受試者穿深色長褲時兩腳難以區分，模型會左右互換（如 S56，膝角相關係數僅約 0.56），這類試驗不適合用於展示
- 影片 metadata 的 fps 寫 29／30／31 不等，但幀數 ÷ 30 與 IMU 時長吻合，程式固定以 30 fps 計算

---

## 專案結構

```
NckuH_Gait_Skeleton/
├── Db/                        # 不進 git
│   ├── pretrain/              # 模型權重檔
│   └── Record/                # 輸出，一支影片一個資料夾（自動建立）
├── Db_NckuH_Gait/             # VIDIMU 資料集（不進 git，需自行下載）
├── Src/
│   ├── gait_analysis/         # 步態指標（VIDIMU 展示版）
│   │   ├── gait_config.yaml   # 門檻值設定
│   │   ├── run_extract.py     # 批次萃取關節點
│   │   ├── run_demo.py        # 指標計算、IMU 比對、展示資料
│   │   ├── gait.py            # 前處理、切趟、著地離地、角度與指標
│   │   ├── imu_compare.py     # 影片與 IMU 角度比對
│   │   ├── vidimu.py          # VIDIMU 檔案存取與同步
│   │   ├── pipeline.py        # 單支影片完整流程（GUI 與 run_demo 共用）
│   │   ├── plots.py           # 曲線疊圖、數字面板
│   │   ├── render_demo.py     # 匯出展示影片的逐格合成
│   │   └── style.py           # 字型、面板文字、資料來源註記
│   ├── UI_Control/            # 主程式
│   │   ├── main.py            # 入口點
│   │   ├── gait_widget.py     # 主畫面邏輯（載入、播放、點選目標、批量分析、匯出）
│   │   ├── gait_views.py      # 顯示元件（影片骨架、角度曲線、數字面板）
│   │   ├── ui/                # PyQt5 主視窗定義（.py + .ui）
│   │   └── utils/model.py     # 模型載入（YOLO + ViTPose + BoTSORT）
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

"""膝角曲線疊圖與數字面板（靜態 PNG）。"""
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


def _find_cjk_font():
    # 專案字型檔不進 git（授權）；沒有時改用 Windows 內建的微軟正黑體
    bundled = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'UI_Control', 'utils',
                                           'R-PMingLiU-TW-2.ttf'))
    if os.path.isfile(bundled):
        return bundled
    return os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts', 'msjh.ttc')


FONT_PATH = _find_cjk_font()
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams['axes.unicode_minus'] = False

VIDEO_COLOR = '#1f77b4'
IMU_COLOR = '#d62728'


def knee_overlay(curves, strides, summary, out_path):
    """近側腿每個週期的膝角：細線為各週期，粗線為平均，影片與 IMU 各一色。"""
    near_ids = strides.loc[(strides['near_far'] == 'near') & strides['valid'], 'stride_id']
    near = curves[curves['stride_id'].isin(near_ids)]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for _, group in near.groupby('stride_id'):
        ax.plot(group['pct'], group['knee'], color=VIDEO_COLOR, alpha=0.18, lw=1)
        ax.plot(group['pct'], group['imu_knee'], color=IMU_COLOR, alpha=0.18, lw=1)
    mean = near.groupby('pct')[['knee', 'imu_knee']].mean()
    ax.plot(mean.index, mean['knee'], color=VIDEO_COLOR, lw=2.5, label='影片（單一鏡頭）')
    ax.plot(mean.index, mean['imu_knee'], color=IMU_COLOR, lw=2.5, ls='--', label='穿戴式感測器 IMU')
    ax.set_xlabel('步態週期（%，著地 → 下一次著地）')
    ax.set_ylabel('膝屈曲角度（度）')
    subject = summary['trial'].split('_')[0]
    ax.set_title(f'{subject} 膝關節角度：影片與 IMU 比對（{summary["near_strides"]} 個步態週期）')
    ax.text(0.01, 0.97, f'相關係數 r = {summary["knee_pearson_r"]:.2f}\n'
                        f'扣除平均差後 RMSE = {summary["knee_rmse_centered"]:.1f}°',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', ec='#cccccc'))
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 100)
    fig.text(0.01, 0.005, '資料來源：VIDIMU（CC BY 4.0）｜公開資料，年輕健康成人；與 IMU 一致性為初步結果',
             fontsize=7, color='#666666')
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path)
    plt.close(fig)


def number_panel(values, subject, out_path):
    """數字面板：步頻、膝活動範圍、最大髖伸展、軀幹前傾角。"""
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.6), dpi=150)
    for ax, item in zip(axes, panel_items(values)):
        ax.axis('off')
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc='#f5f7fa', ec='#d0d7de'))
        ax.text(0.5, 0.80, item['label'], ha='center', fontsize=12, transform=ax.transAxes)
        ax.text(0.5, 0.42, item['value'], ha='center', fontsize=26, weight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.12, item['note'], ha='center', fontsize=8.5, color='#555555', transform=ax.transAxes)
    fig.suptitle(f'{subject} 步態數字面板（A01 往前走）', fontsize=13)
    fig.text(0.01, 0.01, '資料來源：VIDIMU（CC BY 4.0）｜公開資料，年輕健康成人；與 IMU 一致性為初步結果',
             fontsize=7, color='#666666')
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(out_path)
    plt.close(fig)


def panel_items(values):
    return [
        {'label': '步頻', 'value': f'{values["cadence"]:.0f} 步/分', 'note': '高齡者中等強度參考：100 步/分'},
        {'label': '膝活動範圍', 'value': f'{values["knee_rom"]:.0f}°', 'note': '每個步態週期最大與最小角度的差'},
        {'label': '最大髖伸展', 'value': f'{values["hip_ext"]:.0f}°', 'note': '後腳蹬地時大腿往後伸的角度'},
        {'label': '軀幹前傾角', 'value': f'{values["trunk"]:.1f}°', 'note': '探索性指標'},
    ]

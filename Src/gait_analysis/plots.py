"""膝角曲線疊圖與數字面板（靜態 PNG）。

用 Figure + Agg canvas 直接繪圖、不經 pyplot，在 Qt 程式中呼叫也不會切換全域 backend。
"""
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from gait_analysis.style import FONT_PATH, IMU_COLOR, VIDEO_COLOR, VIDIMU_FOOTER, panel_items

font_manager.fontManager.addfont(FONT_PATH)
FONT_NAME = font_manager.FontProperties(fname=FONT_PATH).get_name()
RC = {'font.family': FONT_NAME, 'axes.unicode_minus': False}


def _save(fig, out_path):
    FigureCanvasAgg(fig)
    fig.savefig(out_path)


def knee_overlay(curves, strides, summary, out_path, footer=VIDIMU_FOOTER):
    """近側腿每個週期的膝角：細線為各週期，粗線為平均，影片與 IMU 各一色。"""
    import matplotlib
    with matplotlib.rc_context(RC):
        near_ids = strides.loc[(strides['near_far'] == 'near') & strides['valid'], 'stride_id']
        near = curves[curves['stride_id'].isin(near_ids)]
        fig = Figure(figsize=(7, 4.2), dpi=150)
        ax = fig.add_subplot()
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
        if footer:
            fig.text(0.01, 0.005, footer, fontsize=7, color='#666666')
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        _save(fig, out_path)


def number_panel(values, title, out_path, footer=VIDIMU_FOOTER):
    """數字面板：步頻、膝活動範圍、最大髖伸展、軀幹前傾角。"""
    import matplotlib
    with matplotlib.rc_context(RC):
        fig = Figure(figsize=(11, 2.6), dpi=150)
        axes = fig.subplots(1, 4)
        for ax, item in zip(axes, panel_items(values)):
            ax.axis('off')
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc='#f5f7fa', ec='#d0d7de'))
            ax.text(0.5, 0.80, item['label'], ha='center', fontsize=12, transform=ax.transAxes)
            ax.text(0.5, 0.42, item['value'], ha='center', fontsize=26, weight='bold', transform=ax.transAxes)
            ax.text(0.5, 0.12, item['note'], ha='center', fontsize=8.5, color='#555555', transform=ax.transAxes)
        fig.suptitle(title, fontsize=13)
        if footer:
            fig.text(0.01, 0.01, footer, fontsize=7, color='#666666')
        fig.tight_layout(rect=(0, 0.05, 1, 0.92))
        _save(fig, out_path)

"""展示資料共用的字型、文字與數字面板內容（不依賴 matplotlib，GUI 也可引用）。"""
import os

VIDIMU_FOOTER = '資料來源：VIDIMU（CC BY 4.0）｜公開資料，年輕健康成人；與 IMU 一致性為初步結果'
ACTIVITY_NAMES = {'A01': '往前走', 'A02': '往後走', 'A03': '沿線斜走'}


def find_cjk_font():
    # 專案字型檔不進 git（授權）；沒有時改用 Windows 內建的微軟正黑體
    bundled = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'UI_Control', 'utils',
                                           'R-PMingLiU-TW-2.ttf'))
    if os.path.isfile(bundled):
        return bundled
    return os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts', 'msjh.ttc')


FONT_PATH = find_cjk_font()


def _fmt(value, pattern):
    return pattern.format(value) if value == value else '—'      # NaN 顯示破折號


def panel_items(values):
    return [
        {'label': '步頻', 'value': _fmt(values['cadence'], '{:.0f} 步/分'), 'note': '高齡者中等強度參考：100 步/分'},
        {'label': '膝活動範圍', 'value': _fmt(values['knee_rom'], '{:.0f}°'), 'note': '每個步態週期最大與最小角度的差'},
        {'label': '最大髖伸展', 'value': _fmt(values['hip_ext'], '{:.0f}°'), 'note': '後腳蹬地時大腿往後伸的角度'},
        {'label': '軀幹前傾角', 'value': _fmt(values['trunk'], '{:.1f}°'), 'note': '探索性指標'},
    ]

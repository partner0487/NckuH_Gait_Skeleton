import sys
import os
# 沒有先 conda activate（例如在 IDE 直接執行）時沒有 CONDA_PREFIX，改用目前 Python 所在的環境
env_root = os.environ.get("CONDA_PREFIX") or sys.prefix
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(env_root, "Library", "plugins", "platforms")
from PyQt5.QtWidgets import QApplication, QMainWindow
from gait_widget import GaitTabControl
from ui.main_window import Ui_MainWindow
from utils.model import Model


class Main(QMainWindow):
    def __init__(self):
        super(Main, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("步態分析")
        self.model = Model()
        self.init_tabs()

    def init_tabs(self):
        self.gait_tab = GaitTabControl(self.model)
        self.ui.Two_d_Tab.addTab(self.gait_tab, "步態分析")
        self.ui.Two_d_Tab.setTabBarAutoHide(True)        # 只有一個分頁，不顯示分頁列

    def closeEvent(self, event):
        self.gait_tab.shutdown()
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = Main()
    main_window.show()
    sys.exit(app.exec_())

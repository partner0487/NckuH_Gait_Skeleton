import sys
import os
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(os.environ["CONDA_PREFIX"], "Library", "plugins", "platforms")
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

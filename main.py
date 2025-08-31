import sys
import os
import traceback
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

from modules import *
from modules import Settings
from modules.app_functions import AppFunctions
from widgets import *
from modules.utils import resource_path

widgets = None

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        # SET AS GLOBAL WIDGETS
        # ///////////////////////////////////////////////////////////////
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        global widgets
        widgets = self.ui

        self.appFunctions = AppFunctions(self)
        self.appFunctions.load_last_workspace()
        self.appFunctions.update_anchor_settings()

        self.ui.pushButton.clicked.connect(self.appFunctions.open_existing_workspace)
        self.ui.pushButton_4.clicked.connect(self.appFunctions.save_as_new_workspace)
        self.ui.g_anchorNum.valueChanged.connect(self.appFunctions.update_anchor_settings)

        # USE CUSTOM TITLE BAR | USE AS "False" FOR MAC OR LINUX
        # ///////////////////////////////////////////////////////////////
        Settings.ENABLE_CUSTOM_TITLE_BAR = True

        # APP NAME
        # ///////////////////////////////////////////////////////////////
        title = "U-Guide"
        description = "Subway Monitoring System"
        # APPLY TEXTS
        self.setWindowTitle(title)
        widgets.titleRightInfo.setText(description)

        # TOGGLE MENU
        # ///////////////////////////////////////////////////////////////
        widgets.toggleButton.clicked.connect(lambda: UIFunctions.toggleMenu(self, True))

        # SET UI DEFINITIONS
        # ///////////////////////////////////////////////////////////////
        UIFunctions.uiDefinitions(self)

        # BUTTONS CLICK
        # ///////////////////////////////////////////////////////////////

        # LEFT MENUS
        widgets.btn_home.clicked.connect(self.buttonClick)
        widgets.btn_new.clicked.connect(self.buttonClick)


        # SHOW APP
        # ///////////////////////////////////////////////////////////////
        self.show()

    def refresh_ui(self):
        self.repaint()

    # BUTTONS CLICK
    # Post here your functions for clicked buttons
    # ///////////////////////////////////////////////////////////////
    def buttonClick(self):
        # GET BUTTON CLICKED
        btn = self.sender()
        btnName = btn.objectName()

        # SHOW HOME PAGE
        if btnName == "btn_home":
            widgets.stackedWidget.setCurrentWidget(widgets.page)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))

        # SHOW NEW PAGE
        if btnName == "btn_new":
            widgets.stackedWidget.setCurrentWidget(widgets.page_2) # SET PAGE
            UIFunctions.resetStyle(self, btnName) # RESET ANOTHERS BUTTONS SELECTED
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet())) # SELECT MENU


    # RESIZE EVENTS
    # ///////////////////////////////////////////////////////////////
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'appFunctions') and hasattr(self, 'workspace_settings'):
            self.appFunctions.draw_workspace_box(
                x=0, y=0,
                workspace_width=self.workspace_settings.get("workspace_width", 0),
                workspace_height=self.workspace_settings.get("workspace_height", 0),
                anchors=self.appFunctions.anchor_data,
                vertices=self.appFunctions.vertex_data
            )

    # MOUSE CLICK EVENTS
    # ///////////////////////////////////////////////////////////////
    def mousePressEvent(self, event):
        self.dragPos = event.globalPosition().toPoint()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("icon.ico"))
    window = MainWindow()
    sys.exit(app.exec())

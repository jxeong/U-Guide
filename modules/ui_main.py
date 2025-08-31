# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QComboBox, QDoubleSpinBox,
    QFormLayout, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QSizePolicy, QSpinBox, QStackedWidget, QVBoxLayout,
    QWidget)
from modules import resources_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(940, 643)
        MainWindow.setMinimumSize(QSize(940, 560))
        self.styleSheet = QWidget(MainWindow)
        self.styleSheet.setObjectName(u"styleSheet")
        font = QFont()
        font.setFamilies([u"Segoe UI"])
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        self.styleSheet.setFont(font)
        self.styleSheet.setStyleSheet(u"QWidget {\n"
"    color: #000000;\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"Tooltip */\n"
"QToolTip {\n"
"	color: #333;\n"
"	background-color: #f8f8f2;\n"
"	border: 1px solid #CCC;\n"
"	background-image: none;\n"
"	background-position: left center;\n"
"    background-repeat: no-repeat;\n"
"	border: none;\n"
"	border-left: 2px solid rgb(255, 121, 198);\n"
"	text-align: left;\n"
"	padding-left: 8px;\n"
"	margin: 0px;\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"Bg App */\n"
"#bgApp {	\n"
"	background-color: #f8f8f2;\n"
"	border: 1px solid #CCC;\n"
"    color: #44475a;\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"Left Menu */\n"
"#leftMenuBg {	\n"
"	background-color: #6272a4;\n"
"}\n"
"#topLogo {\n"
"	background-color: #6272a4;\n"
"	background-image: url(:/images/images/images/PyDracula.pn"
                        "g);\n"
"	background-position: centered;\n"
"	background-repeat: no-repeat;\n"
"}\n"
"#titleLeftApp { font: 12pt \"Segoe UI Semibold\"; color: #f8f8f2; }\n"
"#titleLeftDescription { font: 8pt \"Segoe UI\"; color: #bd93f9; }\n"
"\n"
"/* MENUS */\n"
"#topMenu .QPushButton {	\n"
"	background-position: left center;\n"
"    background-repeat: no-repeat;\n"
"	border: none;\n"
"	border-left: 22px solid transparent;\n"
"	background-color: transparent;\n"
"	text-align: left;\n"
"	padding-left: 44px;\n"
"    color: #f8f8f2;\n"
"}\n"
"#topMenu .QPushButton:hover {\n"
"	background-color: #bd93f9;\n"
"}\n"
"#topMenu .QPushButton:pressed {	\n"
"	background-color: #ff79c6;\n"
"	color: rgb(255, 255, 255);\n"
"}\n"
"#bottomMenu .QPushButton {	\n"
"	background-position: left center;\n"
"    background-repeat: no-repeat;\n"
"	border: none;\n"
"	border-left: 20px solid transparent;\n"
"	background-color:transparent;\n"
"	text-align: left;\n"
"	padding-left: 44px;\n"
"    color: #f8f8f2;\n"
"}\n"
"#bottomMenu .QPushButton:hover {\n"
""
                        "	background-color: #bd93f9;\n"
"}\n"
"#bottomMenu .QPushButton:pressed {	\n"
"	background-color: #ff79c6;\n"
"	color: rgb(255, 255, 255);\n"
"}\n"
"#leftMenuFrame{\n"
"	border-top: 3px solid #6a7cb1;\n"
"}\n"
"\n"
"/* Toggle Button */\n"
"#toggleButton {\n"
"	background-position: left center;\n"
"    background-repeat: no-repeat;\n"
"	border: none;\n"
"	border-left: 20px solid transparent;\n"
"	background-color: #5b6996;\n"
"	text-align: left;\n"
"	padding-left: 44px;\n"
"	color: #f8f8f2;\n"
"}\n"
"#toggleButton:hover {\n"
"	background-color: #bd93f9;\n"
"}\n"
"#toggleButton:pressed {	\n"
"	background-color: #ff79c6;\n"
"	color: rgb(255, 255, 255);\n"
"}\n"
"\n"
"/* Title Menu */\n"
"#titleRightInfo { padding-left: 10px; }\n"
"\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"Extra Tab */\n"
"#extraLeftBox {	\n"
"	background-color: #495474;\n"
"    color: #f8f8f2;\n"
"}\n"
"#extraTopBg{	\n"
"	background-color: rgb(189, 147, 249)\n"
"}\n"
"\n"
"/*"
                        " Icon */\n"
"#extraIcon {\n"
"	background-position: center;\n"
"	background-repeat: no-repeat;\n"
"	background-image: url(:/icons/images/icons/icon_settings.png);\n"
"}\n"
"\n"
"/* Label */\n"
"#extraLabel { color: rgb(255, 255, 255); }\n"
"\n"
"/* Btn Close */\n"
"#extraCloseColumnBtn { background-color: rgba(255, 255, 255, 0); border: none;  border-radius: 5px; }\n"
"#extraCloseColumnBtn:hover { background-color: rgb(196, 161, 249); border-style: solid; border-radius: 4px; }\n"
"#extraCloseColumnBtn:pressed { background-color: rgb(180, 141, 238); border-style: solid; border-radius: 4px; }\n"
"\n"
"/* Extra Content */\n"
"#extraContent{\n"
"	border-top: 3px solid #6272a4;\n"
"}\n"
"\n"
"/* Extra Top Menus */\n"
"#extraTopMenu .QPushButton {\n"
"    background-position: left center;\n"
"    background-repeat: no-repeat;\n"
"	border: none;\n"
"	border-left: 22px solid transparent;\n"
"	background-color:transparent;\n"
"	text-align: left;\n"
"	padding-left: 44px;\n"
"    color: #f8f8f2;\n"
"}\n"
"#extraTopMenu ."
                        "QPushButton:hover {\n"
"	background-color: #5d6c99;\n"
"}\n"
"#extraTopMenu .QPushButton:pressed {	\n"
"	background-color: rgb(189, 147, 249);\n"
"	color: rgb(255, 255, 255);\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"Content App */\n"
"#contentTopBg{	\n"
"	background-color: #6272a4;\n"
"}\n"
"#contentBottom{\n"
"	border-top: 3px solid #bd93f9;\n"
"}\n"
"#titleRightInfo{\n"
"    color: #f8f8f2;\n"
"}\n"
"\n"
"/* Top Buttons */\n"
"#rightButtons .QPushButton { background-color: rgba(255, 255, 255, 0); border: none;  border-radius: 5px; }\n"
"#rightButtons .QPushButton:hover { background-color: #bd93f9; border-style: solid; border-radius: 4px; }\n"
"#rightButtons .QPushButton:pressed { background-color: #ff79c6; border-style: solid; border-radius: 4px; }\n"
"\n"
"/* Theme Settings */\n"
"#extraRightBox { background-color: #495474; }\n"
"#themeSettingsTopDetail { background-color: #6272a4; }\n"
"\n"
"/* Bottom Bar */\n"
"#bottomBar { bac"
                        "kground-color: #495474 }\n"
"#bottomBar QLabel { font-size: 11px; color: #f8f8f2; padding-left: 10px; padding-right: 10px; padding-bottom: 2px; }\n"
"\n"
"/* CONTENT SETTINGS */\n"
"/* MENUS */\n"
"#contentSettings .QPushButton {\n"
"    background-position: left center;\n"
"    background-repeat: no-repeat;\n"
"	border: none;\n"
"	border-left: 22px solid transparent;\n"
"	background-color:transparent;\n"
"	text-align: left;\n"
"	padding-left: 44px;\n"
"    color: #f8f8f2;\n"
"}\n"
"#contentSettings .QPushButton:hover {\n"
"	background-color: #5d6c99;\n"
"}\n"
"#contentSettings .QPushButton:pressed {	\n"
"	background-color: rgb(189, 147, 249);\n"
"	color: rgb(255, 255, 255);\n"
"}\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"QTableWidget */\n"
"QTableWidget {	\n"
"	background-color: transparent;\n"
"	padding: 10px;\n"
"	border-radius: 5px;\n"
"	gridline-color: #9faeda;\n"
"    outline: none;\n"
"}\n"
"QTableWidget::item{\n"
"	border-color: #9faeda"
                        ";\n"
"	padding-left: 5px;\n"
"	padding-right: 5px;\n"
"	gridline-color: #9faeda;\n"
"}\n"
"QTableWidget::item:selected{\n"
"	background-color: rgb(189, 147, 249);\n"
"    color: #f8f8f2;\n"
"}\n"
"QHeaderView::section{\n"
"	background-color: #6272a4;\n"
"	max-width: 30px;\n"
"	border: none;\n"
"	border-style: none;\n"
"}\n"
"QTableWidget::horizontalHeader {	\n"
"	background-color: #6272a4;\n"
"}\n"
"QHeaderView::section:horizontal\n"
"{\n"
"    border: 1px solid #6272a4;\n"
"	background-color: #6272a4;\n"
"	padding: 3px;\n"
"	border-top-left-radius: 7px;\n"
"    border-top-right-radius: 7px;\n"
"    color: #f8f8f2;\n"
"}\n"
"QHeaderView::section:vertical\n"
"{\n"
"    border: 1px solid #6272a4;\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"LineEdit */\n"
"QLineEdit {\n"
"	background-color: #6272a4;\n"
"	border-radius: 5px;\n"
"	border: 2px solid #6272a4;\n"
"	padding-left: 10px;\n"
"	selection-color: rgb(255, 255, 255);\n"
"	selection-ba"
                        "ckground-color: rgb(255, 121, 198);\n"
"    color: #f8f8f2;\n"
"}\n"
"QLineEdit:hover {\n"
"	border: 2px solid rgb(64, 71, 88);\n"
"}\n"
"QLineEdit:focus {\n"
"	border: 2px solid #ff79c6;\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"PlainTextEdit */\n"
"QPlainTextEdit {\n"
"	background-color: #6272a4;\n"
"	border-radius: 5px;\n"
"	padding: 10px;\n"
"	selection-color: rgb(255, 255, 255);\n"
"	selection-background-color: rgb(255, 121, 198);\n"
"    color: #f8f8f2;\n"
"}\n"
"QPlainTextEdit  QScrollBar:vertical {\n"
"    width: 8px;\n"
" }\n"
"QPlainTextEdit  QScrollBar:horizontal {\n"
"    height: 8px;\n"
" }\n"
"QPlainTextEdit:hover {\n"
"	border: 2px solid rgb(64, 71, 88);\n"
"}\n"
"QPlainTextEdit:focus {\n"
"	border: 2px solid #ff79c6;\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"ScrollBars */\n"
"QScrollBar:horizontal {\n"
"    border: none;\n"
"    background: "
                        "#6272a4;\n"
"    height: 8px;\n"
"    margin: 0px 21px 0 21px;\n"
"	border-radius: 0px;\n"
"}\n"
"QScrollBar::handle:horizontal {\n"
"    background: rgb(189, 147, 249);\n"
"    min-width: 25px;\n"
"	border-radius: 4px\n"
"}\n"
"QScrollBar::add-line:horizontal {\n"
"    border: none;\n"
"    background: #6272a4;\n"
"    width: 20px;\n"
"	border-top-right-radius: 4px;\n"
"    border-bottom-right-radius: 4px;\n"
"    subcontrol-position: right;\n"
"    subcontrol-origin: margin;\n"
"}\n"
"QScrollBar::sub-line:horizontal {\n"
"    border: none;\n"
"    background: #6272a4;\n"
"    width: 20px;\n"
"	border-top-left-radius: 4px;\n"
"    border-bottom-left-radius: 4px;\n"
"    subcontrol-position: left;\n"
"    subcontrol-origin: margin;\n"
"}\n"
"QScrollBar::up-arrow:horizontal, QScrollBar::down-arrow:horizontal\n"
"{\n"
"     background: none;\n"
"}\n"
"QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal\n"
"{\n"
"     background: none;\n"
"}\n"
" QScrollBar:vertical {\n"
"	border: none;\n"
"    backg"
                        "round-color: #6272a4;\n"
"    width: 8px;\n"
"    margin: 21px 0 21px 0;\n"
"	border-radius: 0px;\n"
" }\n"
" QScrollBar::handle:vertical {	\n"
"	background: rgb(189, 147, 249);\n"
"    min-height: 25px;\n"
"	border-radius: 4px\n"
" }\n"
" QScrollBar::add-line:vertical {\n"
"     border: none;\n"
"    background: #6272a4;\n"
"     height: 20px;\n"
"	border-bottom-left-radius: 4px;\n"
"    border-bottom-right-radius: 4px;\n"
"     subcontrol-position: bottom;\n"
"     subcontrol-origin: margin;\n"
" }\n"
" QScrollBar::sub-line:vertical {\n"
"	border: none;\n"
"    background: #6272a4;\n"
"     height: 20px;\n"
"	border-top-left-radius: 4px;\n"
"    border-top-right-radius: 4px;\n"
"     subcontrol-position: top;\n"
"     subcontrol-origin: margin;\n"
" }\n"
" QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {\n"
"     background: none;\n"
" }\n"
"\n"
" QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {\n"
"     background: none;\n"
" }\n"
"\n"
"/* //////////////////////////////////////"
                        "///////////////////////////////////////////////////////////\n"
"CheckBox */\n"
"QCheckBox::indicator {\n"
"    border: 3px solid #6272a4;\n"
"	width: 15px;\n"
"	height: 15px;\n"
"	border-radius: 10px;\n"
"    background: #6272a4;\n"
"}\n"
"QCheckBox::indicator:hover {\n"
"    border: 3px solid rgb(119, 136, 187);\n"
"}\n"
"QCheckBox::indicator:checked {\n"
"    background: 3px solid #bd93f9;\n"
"	border: 3px solid #bd93f9;	\n"
"	background-image: url(:/icons/images/icons/cil-check-alt.png);\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"RadioButton */\n"
"QRadioButton::indicator {\n"
"    border: 3px solid #6272a4;\n"
"	width: 15px;\n"
"	height: 15px;\n"
"	border-radius: 10px;\n"
"    background: #6272a4;\n"
"}\n"
"QRadioButton::indicator:hover {\n"
"    border: 3px solid rgb(119, 136, 187);\n"
"}\n"
"QRadioButton::indicator:checked {\n"
"    background: 3px solid #bd93f9;\n"
"	border: 3px solid #bd93f9;	\n"
"}\n"
"\n"
"/* ///////////////"
                        "//////////////////////////////////////////////////////////////////////////////////\n"
"ComboBox */\n"
"QComboBox{\n"
"	background-color: #6272a4;\n"
"	border-radius: 5px;\n"
"	border: 2px solid #6272a4;\n"
"	padding: 5px;\n"
"	padding-left: 10px;\n"
"    color: #f8f8f2;\n"
"}\n"
"QComboBox:hover{\n"
"	border: 2px solid #7284b9;\n"
"}\n"
"QComboBox::drop-down {\n"
"	subcontrol-origin: padding;\n"
"	subcontrol-position: top right;\n"
"	width: 25px; \n"
"	border-left-width: 3px;\n"
"	border-left-color: #6272a4;\n"
"	border-left-style: solid;\n"
"	border-top-right-radius: 3px;\n"
"	border-bottom-right-radius: 3px;	\n"
"	background-image: url(:/icons/images/icons/cil-arrow-bottom.png);\n"
"	background-position: center;\n"
"	background-repeat: no-reperat;\n"
" }\n"
"QComboBox QAbstractItemView {\n"
"	color: rgb(255, 121, 198);	\n"
"	background-color: #6272a4;\n"
"	padding: 10px;\n"
"	selection-background-color: #6272a4;\n"
"}\n"
"\n"
"/* ///////////////////////////////////////////////////////////////////////////////"
                        "//////////////////\n"
"Sliders */\n"
"QSlider::groove:horizontal {\n"
"    border-radius: 5px;\n"
"    height: 10px;\n"
"	margin: 0px;\n"
"	background-color: #6272a4;\n"
"}\n"
"QSlider::groove:horizontal:hover {\n"
"	background-color: #6272a4;\n"
"}\n"
"QSlider::handle:horizontal {\n"
"    background-color: rgb(189, 147, 249);\n"
"    border: none;\n"
"    height: 10px;\n"
"    width: 10px;\n"
"    margin: 0px;\n"
"	border-radius: 5px;\n"
"}\n"
"QSlider::handle:horizontal:hover {\n"
"    background-color: rgb(195, 155, 255);\n"
"}\n"
"QSlider::handle:horizontal:pressed {\n"
"    background-color: rgb(255, 121, 198);\n"
"}\n"
"\n"
"QSlider::groove:vertical {\n"
"    border-radius: 5px;\n"
"    width: 10px;\n"
"    margin: 0px;\n"
"	background-color: #6272a4;\n"
"}\n"
"QSlider::groove:vertical:hover {\n"
"	background-color: #6272a4;\n"
"}\n"
"QSlider::handle:vertical {\n"
"    background-color: rgb(189, 147, 249);\n"
"	border: none;\n"
"    height: 10px;\n"
"    width: 10px;\n"
"    margin: 0px;\n"
"	border-radi"
                        "us: 5px;\n"
"}\n"
"QSlider::handle:vertical:hover {\n"
"    background-color: rgb(195, 155, 255);\n"
"}\n"
"QSlider::handle:vertical:pressed {\n"
"    background-color: rgb(255, 121, 198);\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"CommandLinkButton */\n"
"#pagesContainer QCommandLinkButton {	\n"
"	color: rgb(255, 121, 198);\n"
"	border-radius: 5px;\n"
"	padding: 5px;\n"
"    border: 2px solid #ff79c6;\n"
"    color: #ff79c6;\n"
"}\n"
"#pagesContainer QCommandLinkButton:hover {	\n"
"	color: rgb(255, 170, 255);\n"
"	background-color: #6272a4;\n"
"}\n"
"#pagesContainer QCommandLinkButton:pressed {	\n"
"	color: rgb(189, 147, 249);\n"
"	background-color: #586796;\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"Button */\n"
"#pagesContainer QPushButton {\n"
"	border: 2px solid #6272a4;\n"
"	border-radius: 5px;	\n"
"	background-color: #6272a4;\n"
"    color: #f8f8f2;\n"
""
                        "}\n"
"#pagesContainer QPushButton:hover {\n"
"	background-color: #7082b6;\n"
"	border: 2px solid #7082b6;\n"
"}\n"
"#pagesContainer QPushButton:pressed {	\n"
"	background-color: #546391;\n"
"	border: 2px solid #ff79c6;\n"
"}\n"
"\n"
"\n"
"")
        self.verticalLayout_5 = QVBoxLayout(self.styleSheet)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.bgApp = QFrame(self.styleSheet)
        self.bgApp.setObjectName(u"bgApp")
        self.bgApp.setStyleSheet(u"")
        self.bgApp.setFrameShape(QFrame.Shape.NoFrame)
        self.bgApp.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout_7 = QGridLayout(self.bgApp)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.gridLayout_7.setContentsMargins(0, 0, 0, 0)
        self.leftMenuBg = QFrame(self.bgApp)
        self.leftMenuBg.setObjectName(u"leftMenuBg")
        self.leftMenuBg.setMinimumSize(QSize(60, 0))
        self.leftMenuBg.setMaximumSize(QSize(60, 16777215))
        self.leftMenuBg.setFrameShape(QFrame.Shape.NoFrame)
        self.leftMenuBg.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_3 = QVBoxLayout(self.leftMenuBg)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.leftMenuFrame = QFrame(self.leftMenuBg)
        self.leftMenuFrame.setObjectName(u"leftMenuFrame")
        self.leftMenuFrame.setFrameShape(QFrame.Shape.NoFrame)
        self.leftMenuFrame.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalMenuLayout = QVBoxLayout(self.leftMenuFrame)
        self.verticalMenuLayout.setSpacing(0)
        self.verticalMenuLayout.setObjectName(u"verticalMenuLayout")
        self.verticalMenuLayout.setContentsMargins(0, 0, 0, 0)
        self.toggleBox = QFrame(self.leftMenuFrame)
        self.toggleBox.setObjectName(u"toggleBox")
        self.toggleBox.setMaximumSize(QSize(16777215, 45))
        self.toggleBox.setFrameShape(QFrame.Shape.NoFrame)
        self.toggleBox.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_4 = QVBoxLayout(self.toggleBox)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.toggleButton = QPushButton(self.toggleBox)
        self.toggleButton.setObjectName(u"toggleButton")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.toggleButton.sizePolicy().hasHeightForWidth())
        self.toggleButton.setSizePolicy(sizePolicy)
        self.toggleButton.setMinimumSize(QSize(0, 45))
        self.toggleButton.setFont(font)
        self.toggleButton.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.toggleButton.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.toggleButton.setStyleSheet(u"background-image: url(:/icons/images/icons/icon_menu.png);")

        self.verticalLayout_4.addWidget(self.toggleButton)


        self.verticalMenuLayout.addWidget(self.toggleBox)

        self.topMenu = QFrame(self.leftMenuFrame)
        self.topMenu.setObjectName(u"topMenu")
        self.topMenu.setFrameShape(QFrame.Shape.NoFrame)
        self.topMenu.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_8 = QVBoxLayout(self.topMenu)
        self.verticalLayout_8.setSpacing(0)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.btn_home = QPushButton(self.topMenu)
        self.btn_home.setObjectName(u"btn_home")
        sizePolicy.setHeightForWidth(self.btn_home.sizePolicy().hasHeightForWidth())
        self.btn_home.setSizePolicy(sizePolicy)
        self.btn_home.setMinimumSize(QSize(0, 45))
        self.btn_home.setFont(font)
        self.btn_home.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_home.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.btn_home.setStyleSheet(u"background-image: url(:/icons/images/icons/cil-home.png);")

        self.verticalLayout_8.addWidget(self.btn_home)

        self.btn_new = QPushButton(self.topMenu)
        self.btn_new.setObjectName(u"btn_new")
        sizePolicy.setHeightForWidth(self.btn_new.sizePolicy().hasHeightForWidth())
        self.btn_new.setSizePolicy(sizePolicy)
        self.btn_new.setMinimumSize(QSize(0, 45))
        self.btn_new.setFont(font)
        self.btn_new.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_new.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.btn_new.setStyleSheet(u"background-image: url(:/icons/images/icons/cil-file.png);")

        self.verticalLayout_8.addWidget(self.btn_new)


        self.verticalMenuLayout.addWidget(self.topMenu, 0, Qt.AlignmentFlag.AlignTop)


        self.verticalLayout_3.addWidget(self.leftMenuFrame)


        self.gridLayout_7.addWidget(self.leftMenuBg, 0, 0, 1, 1)

        self.contentBox = QFrame(self.bgApp)
        self.contentBox.setObjectName(u"contentBox")
        self.contentBox.setFrameShape(QFrame.Shape.NoFrame)
        self.contentBox.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_2 = QVBoxLayout(self.contentBox)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.contentTopBg = QFrame(self.contentBox)
        self.contentTopBg.setObjectName(u"contentTopBg")
        self.contentTopBg.setMinimumSize(QSize(0, 50))
        self.contentTopBg.setMaximumSize(QSize(16777215, 50))
        self.contentTopBg.setFrameShape(QFrame.Shape.NoFrame)
        self.contentTopBg.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout = QHBoxLayout(self.contentTopBg)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 10, 0)
        self.leftBox = QFrame(self.contentTopBg)
        self.leftBox.setObjectName(u"leftBox")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.leftBox.sizePolicy().hasHeightForWidth())
        self.leftBox.setSizePolicy(sizePolicy1)
        self.leftBox.setFrameShape(QFrame.Shape.NoFrame)
        self.leftBox.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_3 = QHBoxLayout(self.leftBox)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.titleRightInfo = QLabel(self.leftBox)
        self.titleRightInfo.setObjectName(u"titleRightInfo")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.titleRightInfo.sizePolicy().hasHeightForWidth())
        self.titleRightInfo.setSizePolicy(sizePolicy2)
        self.titleRightInfo.setMaximumSize(QSize(16777215, 45))
        font1 = QFont()
        font1.setFamilies([u"Segoe UI"])
        font1.setBold(True)
        font1.setItalic(False)
        self.titleRightInfo.setFont(font1)
        self.titleRightInfo.setStyleSheet(u"QLabel {\n"
"    font-weight: bold;\n"
"	font-size: 20px;\n"
"}")
        self.titleRightInfo.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout_3.addWidget(self.titleRightInfo)


        self.horizontalLayout.addWidget(self.leftBox)

        self.rightButtons = QFrame(self.contentTopBg)
        self.rightButtons.setObjectName(u"rightButtons")
        self.rightButtons.setMinimumSize(QSize(0, 28))
        self.rightButtons.setFrameShape(QFrame.Shape.NoFrame)
        self.rightButtons.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.rightButtons)
        self.horizontalLayout_2.setSpacing(5)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.minimizeAppBtn = QPushButton(self.rightButtons)
        self.minimizeAppBtn.setObjectName(u"minimizeAppBtn")
        self.minimizeAppBtn.setMinimumSize(QSize(28, 28))
        self.minimizeAppBtn.setMaximumSize(QSize(28, 28))
        self.minimizeAppBtn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        icon = QIcon()
        icon.addFile(u":/icons/images/icons/icon_minimize.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.minimizeAppBtn.setIcon(icon)
        self.minimizeAppBtn.setIconSize(QSize(20, 20))

        self.horizontalLayout_2.addWidget(self.minimizeAppBtn)

        self.maximizeRestoreAppBtn = QPushButton(self.rightButtons)
        self.maximizeRestoreAppBtn.setObjectName(u"maximizeRestoreAppBtn")
        self.maximizeRestoreAppBtn.setMinimumSize(QSize(28, 28))
        self.maximizeRestoreAppBtn.setMaximumSize(QSize(28, 28))
        font2 = QFont()
        font2.setFamilies([u"Segoe UI"])
        font2.setPointSize(10)
        font2.setBold(False)
        font2.setItalic(False)
        font2.setStyleStrategy(QFont.PreferDefault)
        self.maximizeRestoreAppBtn.setFont(font2)
        self.maximizeRestoreAppBtn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        icon1 = QIcon()
        icon1.addFile(u":/icons/images/icons/icon_maximize.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.maximizeRestoreAppBtn.setIcon(icon1)
        self.maximizeRestoreAppBtn.setIconSize(QSize(20, 20))

        self.horizontalLayout_2.addWidget(self.maximizeRestoreAppBtn)

        self.closeAppBtn = QPushButton(self.rightButtons)
        self.closeAppBtn.setObjectName(u"closeAppBtn")
        self.closeAppBtn.setMinimumSize(QSize(28, 28))
        self.closeAppBtn.setMaximumSize(QSize(28, 28))
        self.closeAppBtn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        icon2 = QIcon()
        icon2.addFile(u":/icons/images/icons/icon_close.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.closeAppBtn.setIcon(icon2)
        self.closeAppBtn.setIconSize(QSize(20, 20))

        self.horizontalLayout_2.addWidget(self.closeAppBtn)


        self.horizontalLayout.addWidget(self.rightButtons, 0, Qt.AlignmentFlag.AlignRight)


        self.verticalLayout_2.addWidget(self.contentTopBg)

        self.contentBottom = QFrame(self.contentBox)
        self.contentBottom.setObjectName(u"contentBottom")
        self.contentBottom.setFrameShape(QFrame.Shape.NoFrame)
        self.contentBottom.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_6 = QVBoxLayout(self.contentBottom)
        self.verticalLayout_6.setSpacing(0)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.content = QFrame(self.contentBottom)
        self.content.setObjectName(u"content")
        self.content.setFrameShape(QFrame.Shape.NoFrame)
        self.content.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_4 = QHBoxLayout(self.content)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.pagesContainer = QFrame(self.content)
        self.pagesContainer.setObjectName(u"pagesContainer")
        self.pagesContainer.setStyleSheet(u"")
        self.pagesContainer.setFrameShape(QFrame.Shape.NoFrame)
        self.pagesContainer.setFrameShadow(QFrame.Shadow.Raised)
        self.formLayout = QFormLayout(self.pagesContainer)
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setContentsMargins(10, 10, 10, 10)
        self.stackedWidget = QStackedWidget(self.pagesContainer)
        self.stackedWidget.setObjectName(u"stackedWidget")
        self.stackedWidget.setStyleSheet(u"background: transparent;")
        self.page = QWidget()
        self.page.setObjectName(u"page")
        self.gridLayout_20 = QGridLayout(self.page)
        self.gridLayout_20.setObjectName(u"gridLayout_20")
        self.WorkspaceName = QLabel(self.page)
        self.WorkspaceName.setObjectName(u"WorkspaceName")
        self.WorkspaceName.setStyleSheet(u"font-size: 25px;\n"
"    font-weight: bold;\n"
"    qproperty-alignment: AlignCenter;")

        self.gridLayout_20.addWidget(self.WorkspaceName, 1, 0, 1, 1)

        self.gridLayout_19 = QGridLayout()
        self.gridLayout_19.setObjectName(u"gridLayout_19")
        self.workspace = QFrame(self.page)
        self.workspace.setObjectName(u"workspace")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.workspace.sizePolicy().hasHeightForWidth())
        self.workspace.setSizePolicy(sizePolicy3)
        self.workspace.setStyleSheet(u"background-color: rgb(52, 59, 72);")
        self.workspace.setFrameShape(QFrame.Shape.StyledPanel)
        self.workspace.setFrameShadow(QFrame.Shadow.Raised)

        self.gridLayout_19.addWidget(self.workspace, 0, 0, 1, 2)


        self.gridLayout_20.addLayout(self.gridLayout_19, 2, 0, 1, 1)

        self.B_tagOX = QGroupBox(self.page)
        self.B_tagOX.setObjectName(u"B_tagOX")
        self.B_tagOX.setMaximumSize(QSize(16777215, 80))
        self.B_tagOX.setStyleSheet(u"QGroupBox {\n"
"    border: none;         /* \uacbd\uacc4\uc120 \uc81c\uac70 (\uc6d0\ud558\uba74 \uc720\uc9c0 \uac00\ub2a5) */\n"
"}\n"
"\n"
"QGroupBox::title {\n"
"    subcontrol-origin: margin;\n"
"    subcontrol-position: top left;\n"
"    padding: 0;\n"
"    color: transparent;   /* \ud14d\uc2a4\ud2b8 \ud22c\uba85\ud654 */\n"
"}\n"
"")
        self.B_tagOX.setFlat(False)
        self.gridLayout_4 = QGridLayout(self.B_tagOX)
        self.gridLayout_4.setSpacing(0)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.onLabel = QLabel(self.B_tagOX)
        self.onLabel.setObjectName(u"onLabel")
        self.onLabel.setMaximumSize(QSize(16777215, 40))
        self.onLabel.setStyleSheet(u"border: 1px solid #000000;\n"
"    font-size: 16px;\n"
"    font-weight: bold;\n"
"    qproperty-alignment: AlignCenter;")

        self.gridLayout_4.addWidget(self.onLabel, 0, 1, 1, 1)

        self.onPerson = QLabel(self.B_tagOX)
        self.onPerson.setObjectName(u"onPerson")
        self.onPerson.setMaximumSize(QSize(16777215, 30))
        self.onPerson.setStyleSheet(u"border: 1px solid #000000;\n"
"    font-size: 16px;\n"
"    font-weight: bold;\n"
"    qproperty-alignment: AlignCenter;\n"
"border-top:0;")

        self.gridLayout_4.addWidget(self.onPerson, 1, 1, 1, 1)

        self.offLabel = QLabel(self.B_tagOX)
        self.offLabel.setObjectName(u"offLabel")
        self.offLabel.setMaximumSize(QSize(16777215, 40))
        self.offLabel.setStyleSheet(u"border: 1px solid #000000;\n"
"    font-size: 16px;\n"
"    font-weight: bold;\n"
"    qproperty-alignment: AlignCenter;")

        self.gridLayout_4.addWidget(self.offLabel, 0, 0, 1, 1)

        self.offPerson = QLabel(self.B_tagOX)
        self.offPerson.setObjectName(u"offPerson")
        self.offPerson.setMaximumSize(QSize(16777215, 30))
        self.offPerson.setStyleSheet(u"border: 1px solid #000000;\n"
"    font-size: 16px;\n"
"    font-weight: bold;\n"
"    qproperty-alignment: AlignCenter;\n"
"border-top:0;")

        self.gridLayout_4.addWidget(self.offPerson, 1, 0, 1, 1)

        self.helpLabel = QLabel(self.B_tagOX)
        self.helpLabel.setObjectName(u"helpLabel")
        self.helpLabel.setMaximumSize(QSize(16777215, 40))
        self.helpLabel.setStyleSheet(u"border: 1px solid #000000;\n"
"    font-size: 16px;\n"
"    font-weight: bold;\n"
"    qproperty-alignment: AlignCenter;")

        self.gridLayout_4.addWidget(self.helpLabel, 0, 2, 1, 1)

        self.helpPerson = QLabel(self.B_tagOX)
        self.helpPerson.setObjectName(u"helpPerson")
        self.helpPerson.setMaximumSize(QSize(16777215, 30))
        self.helpPerson.setStyleSheet(u"border: 1px solid #000000;\n"
"    font-size: 16px;\n"
"    font-weight: bold;\n"
"    qproperty-alignment: AlignCenter;\n"
"border-top:0;")

        self.gridLayout_4.addWidget(self.helpPerson, 1, 2, 1, 1)


        self.gridLayout_20.addWidget(self.B_tagOX, 3, 0, 1, 1)

        self.btnExportCsv = QPushButton(self.page)
        self.btnExportCsv.setObjectName(u"btnExportCsv")
        self.btnExportCsv.setStyleSheet(u"background-color: rgb(52, 59, 72);")

        self.gridLayout_20.addWidget(self.btnExportCsv, 0, 0, 1, 1)

        self.stackedWidget.addWidget(self.page)
        self.page_2 = QWidget()
        self.page_2.setObjectName(u"page_2")
        self.page_2.setStyleSheet(u"")
        self.verticalLayout = QVBoxLayout(self.page_2)
        self.verticalLayout.setSpacing(10)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.row_1 = QFrame(self.page_2)
        self.row_1.setObjectName(u"row_1")
        self.row_1.setFrameShape(QFrame.Shape.StyledPanel)
        self.row_1.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_16 = QVBoxLayout(self.row_1)
        self.verticalLayout_16.setSpacing(0)
        self.verticalLayout_16.setObjectName(u"verticalLayout_16")
        self.verticalLayout_16.setContentsMargins(0, 0, 0, 0)
        self.row_2 = QFrame(self.row_1)
        self.row_2.setObjectName(u"row_2")
        self.row_2.setMinimumSize(QSize(0, 150))
        self.row_2.setFrameShape(QFrame.Shape.StyledPanel)
        self.row_2.setFrameShadow(QFrame.Shadow.Raised)
        self.row_3 = QFrame(self.row_2)
        self.row_3.setObjectName(u"row_3")
        self.row_3.setGeometry(QRect(0, 0, 821, 411))
        self.row_3.setMinimumSize(QSize(0, 150))
        self.row_3.setStyleSheet(u"QComboBox::drop-down {\n"
"    border: none;\n"
"    background: transparent;\n"
"}\n"
"\n"
"QComboBox QAbstractItemView {\n"
"    background-color: #1e1e1e;\n"
"    color: #ff79c6;\n"
"    border: 1px solid #444444;\n"
"    selection-background-color: #44475a;\n"
"    selection-color: #ffffff;\n"
"    z-index: 1000;  /* \ub4dc\ub86d\ub2e4\uc6b4\uc774 \ud56d\uc0c1 \uc704\uc5d0 \uc624\ub3c4\ub85d \uc124\uc815 */\n"
"}")
        self.row_3.setFrameShape(QFrame.Shape.StyledPanel)
        self.row_3.setFrameShadow(QFrame.Shadow.Raised)
        self.layoutWidget = QWidget(self.row_3)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.layoutWidget.setGeometry(QRect(30, 110, 481, 109))
        self.gridLayout_2 = QGridLayout(self.layoutWidget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.vertexY = QDoubleSpinBox(self.layoutWidget)
        self.vertexY.setObjectName(u"vertexY")
        self.vertexY.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.vertexY.setMaximum(1000.000000000000000)

        self.gridLayout.addWidget(self.vertexY, 1, 7, 1, 1)

        self.labelBoxBlenderInstalation_15 = QLabel(self.layoutWidget)
        self.labelBoxBlenderInstalation_15.setObjectName(u"labelBoxBlenderInstalation_15")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.labelBoxBlenderInstalation_15.sizePolicy().hasHeightForWidth())
        self.labelBoxBlenderInstalation_15.setSizePolicy(sizePolicy4)
        self.labelBoxBlenderInstalation_15.setFont(font)
        self.labelBoxBlenderInstalation_15.setStyleSheet(u"")

        self.gridLayout.addWidget(self.labelBoxBlenderInstalation_15, 1, 4, 1, 1)

        self.vertexX = QDoubleSpinBox(self.layoutWidget)
        self.vertexX.setObjectName(u"vertexX")
        self.vertexX.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.vertexX.setMaximum(1000.000000000000000)

        self.gridLayout.addWidget(self.vertexX, 1, 5, 1, 1)

        self.labelBoxBlenderInstalation_16 = QLabel(self.layoutWidget)
        self.labelBoxBlenderInstalation_16.setObjectName(u"labelBoxBlenderInstalation_16")
        sizePolicy4.setHeightForWidth(self.labelBoxBlenderInstalation_16.sizePolicy().hasHeightForWidth())
        self.labelBoxBlenderInstalation_16.setSizePolicy(sizePolicy4)
        self.labelBoxBlenderInstalation_16.setFont(font)
        self.labelBoxBlenderInstalation_16.setStyleSheet(u"")

        self.gridLayout.addWidget(self.labelBoxBlenderInstalation_16, 1, 6, 1, 1)

        self.labelBoxBlenderInstalation_12 = QLabel(self.layoutWidget)
        self.labelBoxBlenderInstalation_12.setObjectName(u"labelBoxBlenderInstalation_12")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.labelBoxBlenderInstalation_12.sizePolicy().hasHeightForWidth())
        self.labelBoxBlenderInstalation_12.setSizePolicy(sizePolicy5)
        self.labelBoxBlenderInstalation_12.setFont(font)
        self.labelBoxBlenderInstalation_12.setStyleSheet(u"")

        self.gridLayout.addWidget(self.labelBoxBlenderInstalation_12, 0, 0, 1, 1)

        self.vertexCount = QSpinBox(self.layoutWidget)
        self.vertexCount.setObjectName(u"vertexCount")

        self.gridLayout.addWidget(self.vertexCount, 0, 1, 1, 1)

        self.labelBoxBlenderInstalation_14 = QLabel(self.layoutWidget)
        self.labelBoxBlenderInstalation_14.setObjectName(u"labelBoxBlenderInstalation_14")
        sizePolicy5.setHeightForWidth(self.labelBoxBlenderInstalation_14.sizePolicy().hasHeightForWidth())
        self.labelBoxBlenderInstalation_14.setSizePolicy(sizePolicy5)
        self.labelBoxBlenderInstalation_14.setFont(font)
        self.labelBoxBlenderInstalation_14.setStyleSheet(u"")
        self.labelBoxBlenderInstalation_14.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout.addWidget(self.labelBoxBlenderInstalation_14, 1, 0, 1, 1)

        self.vertexSelect = QComboBox(self.layoutWidget)
        self.vertexSelect.setObjectName(u"vertexSelect")
        self.vertexSelect.setStyleSheet(u"background-color: rgb(52, 59, 72);")
        self.vertexSelect.setIconSize(QSize(16, 16))

        self.gridLayout.addWidget(self.vertexSelect, 1, 1, 1, 1)


        self.gridLayout_2.addLayout(self.gridLayout, 1, 0, 1, 1)

        self.labelBoxBlenderInstalation_4 = QLabel(self.layoutWidget)
        self.labelBoxBlenderInstalation_4.setObjectName(u"labelBoxBlenderInstalation_4")
        self.labelBoxBlenderInstalation_4.setFont(font1)
        self.labelBoxBlenderInstalation_4.setStyleSheet(u"QWidget {\n"
"    font-weight: bold;\n"
"	font-size: 20px;\n"
"}\n"
"")

        self.gridLayout_2.addWidget(self.labelBoxBlenderInstalation_4, 0, 0, 1, 1)

        self.layoutWidget_2 = QWidget(self.row_3)
        self.layoutWidget_2.setObjectName(u"layoutWidget_2")
        self.layoutWidget_2.setGeometry(QRect(30, 360, 601, 41))
        self.gridLayout_6 = QGridLayout(self.layoutWidget_2)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.gridLayout_6.setContentsMargins(0, 0, 0, 0)
        self.pushButton = QPushButton(self.layoutWidget_2)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.pushButton.setStyleSheet(u"background-color: rgb(52, 59, 72);\n"
"")
        icon3 = QIcon()
        icon3.addFile(u":/icons/images/icons/cil-folder.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.pushButton.setIcon(icon3)
        self.pushButton.setIconSize(QSize(13, 13))

        self.gridLayout_6.addWidget(self.pushButton, 0, 0, 1, 1)

        self.pushButton_4 = QPushButton(self.layoutWidget_2)
        self.pushButton_4.setObjectName(u"pushButton_4")
        self.pushButton_4.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.pushButton_4.setStyleSheet(u"background-color: rgb(52, 59, 72);")
        icon4 = QIcon()
        icon4.addFile(u":/icons/images/icons/cil-save.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.pushButton_4.setIcon(icon4)
        self.pushButton_4.setIconSize(QSize(13, 13))

        self.gridLayout_6.addWidget(self.pushButton_4, 0, 1, 1, 1)

        self.pushButton_3 = QPushButton(self.layoutWidget_2)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.pushButton_3.setStyleSheet(u"background-color: rgb(52, 59, 72);")
        icon5 = QIcon()
        icon5.addFile(u":/icons/images/icons/cil-pencil.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.pushButton_3.setIcon(icon5)
        self.pushButton_3.setIconSize(QSize(13, 13))

        self.gridLayout_6.addWidget(self.pushButton_3, 0, 2, 1, 1)

        self.layoutWidget_3 = QWidget(self.row_3)
        self.layoutWidget_3.setObjectName(u"layoutWidget_3")
        self.layoutWidget_3.setGeometry(QRect(30, 230, 481, 109))
        self.gridLayout_8 = QGridLayout(self.layoutWidget_3)
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.gridLayout_8.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_9 = QGridLayout()
        self.gridLayout_9.setObjectName(u"gridLayout_9")
        self.labelBoxBlenderInstalation_26 = QLabel(self.layoutWidget_3)
        self.labelBoxBlenderInstalation_26.setObjectName(u"labelBoxBlenderInstalation_26")
        sizePolicy5.setHeightForWidth(self.labelBoxBlenderInstalation_26.sizePolicy().hasHeightForWidth())
        self.labelBoxBlenderInstalation_26.setSizePolicy(sizePolicy5)
        self.labelBoxBlenderInstalation_26.setFont(font)
        self.labelBoxBlenderInstalation_26.setStyleSheet(u"")

        self.gridLayout_9.addWidget(self.labelBoxBlenderInstalation_26, 0, 0, 1, 1)

        self.h_tagNum = QSpinBox(self.layoutWidget_3)
        self.h_tagNum.setObjectName(u"h_tagNum")

        self.gridLayout_9.addWidget(self.h_tagNum, 0, 4, 1, 2)

        self.labelBoxBlenderInstalation_25 = QLabel(self.layoutWidget_3)
        self.labelBoxBlenderInstalation_25.setObjectName(u"labelBoxBlenderInstalation_25")
        self.labelBoxBlenderInstalation_25.setFont(font)
        self.labelBoxBlenderInstalation_25.setStyleSheet(u"")

        self.gridLayout_9.addWidget(self.labelBoxBlenderInstalation_25, 1, 7, 1, 1)

        self.j_anchorX = QDoubleSpinBox(self.layoutWidget_3)
        self.j_anchorX.setObjectName(u"j_anchorX")
        self.j_anchorX.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.j_anchorX.setMaximum(1000.000000000000000)

        self.gridLayout_9.addWidget(self.j_anchorX, 1, 6, 1, 1)

        self.labelBoxBlenderInstalation_24 = QLabel(self.layoutWidget_3)
        self.labelBoxBlenderInstalation_24.setObjectName(u"labelBoxBlenderInstalation_24")
        sizePolicy5.setHeightForWidth(self.labelBoxBlenderInstalation_24.sizePolicy().hasHeightForWidth())
        self.labelBoxBlenderInstalation_24.setSizePolicy(sizePolicy5)
        self.labelBoxBlenderInstalation_24.setFont(font)
        self.labelBoxBlenderInstalation_24.setStyleSheet(u"")
        self.labelBoxBlenderInstalation_24.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_9.addWidget(self.labelBoxBlenderInstalation_24, 1, 0, 1, 2)

        self.k_anchorY = QDoubleSpinBox(self.layoutWidget_3)
        self.k_anchorY.setObjectName(u"k_anchorY")
        self.k_anchorY.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.k_anchorY.setMaximum(1000.000000000000000)

        self.gridLayout_9.addWidget(self.k_anchorY, 1, 8, 1, 1)

        self.labelBoxBlenderInstalation_22 = QLabel(self.layoutWidget_3)
        self.labelBoxBlenderInstalation_22.setObjectName(u"labelBoxBlenderInstalation_22")
        self.labelBoxBlenderInstalation_22.setFont(font)
        self.labelBoxBlenderInstalation_22.setStyleSheet(u"")

        self.gridLayout_9.addWidget(self.labelBoxBlenderInstalation_22, 0, 3, 1, 1)

        self.g_anchorNum = QSpinBox(self.layoutWidget_3)
        self.g_anchorNum.setObjectName(u"g_anchorNum")

        self.gridLayout_9.addWidget(self.g_anchorNum, 0, 1, 1, 2)

        self.labelBoxBlenderInstalation_23 = QLabel(self.layoutWidget_3)
        self.labelBoxBlenderInstalation_23.setObjectName(u"labelBoxBlenderInstalation_23")
        sizePolicy4.setHeightForWidth(self.labelBoxBlenderInstalation_23.sizePolicy().hasHeightForWidth())
        self.labelBoxBlenderInstalation_23.setSizePolicy(sizePolicy4)
        self.labelBoxBlenderInstalation_23.setFont(font)
        self.labelBoxBlenderInstalation_23.setStyleSheet(u"")

        self.gridLayout_9.addWidget(self.labelBoxBlenderInstalation_23, 1, 5, 1, 1)

        self.i_anchorSelect = QComboBox(self.layoutWidget_3)
        self.i_anchorSelect.setObjectName(u"i_anchorSelect")
        self.i_anchorSelect.setStyleSheet(u"background-color: rgb(52, 59, 72);\n"
"\n"
"QComboBox::drop-down {\n"
"    border: none;\n"
"    background: transparent;\n"
"}\n"
"\n"
"QComboBox QAbstractItemView {\n"
"    background-color: #1e1e1e;\n"
"    color: #black;\n"
"    border: 1px solid #444444;\n"
"    selection-background-color: #44475a;\n"
"    selection-color: #ffffff;\n"
"    z-index: 1000;  /* \ub4dc\ub86d\ub2e4\uc6b4\uc774 \ud56d\uc0c1 \uc704\uc5d0 \uc624\ub3c4\ub85d \uc124\uc815 */\n"
"}")
        self.i_anchorSelect.setIconSize(QSize(16, 16))

        self.gridLayout_9.addWidget(self.i_anchorSelect, 1, 2, 1, 3)


        self.gridLayout_8.addLayout(self.gridLayout_9, 1, 0, 1, 1)

        self.labelBoxBlenderInstalation_6 = QLabel(self.layoutWidget_3)
        self.labelBoxBlenderInstalation_6.setObjectName(u"labelBoxBlenderInstalation_6")
        self.labelBoxBlenderInstalation_6.setFont(font1)
        self.labelBoxBlenderInstalation_6.setStyleSheet(u"QWidget {\n"
"    font-weight: bold;\n"
"	font-size: 20px;\n"
"}\n"
"")

        self.gridLayout_8.addWidget(self.labelBoxBlenderInstalation_6, 0, 0, 1, 1)

        self.labelBoxBlenderInstalation_3 = QLabel(self.row_3)
        self.labelBoxBlenderInstalation_3.setObjectName(u"labelBoxBlenderInstalation_3")
        self.labelBoxBlenderInstalation_3.setGeometry(QRect(31, 11, 186, 27))
        self.labelBoxBlenderInstalation_3.setFont(font1)
        self.labelBoxBlenderInstalation_3.setStyleSheet(u"QWidget {\n"
"    font-weight: bold;\n"
"	font-size: 20px;\n"
"}\n"
"")
        self.layoutWidget_4 = QWidget(self.row_3)
        self.layoutWidget_4.setObjectName(u"layoutWidget_4")
        self.layoutWidget_4.setGeometry(QRect(30, 50, 481, 54))
        self.gridLayout_3 = QGridLayout(self.layoutWidget_4)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.label_4 = QLabel(self.layoutWidget_4)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_3.addWidget(self.label_4, 0, 2, 1, 1)

        self.b_workspace_height = QDoubleSpinBox(self.layoutWidget_4)
        self.b_workspace_height.setObjectName(u"b_workspace_height")
        self.b_workspace_height.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.b_workspace_height.setMaximum(1000.000000000000000)

        self.gridLayout_3.addWidget(self.b_workspace_height, 1, 1, 1, 1)

        self.label_5 = QLabel(self.layoutWidget_4)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_3.addWidget(self.label_5, 1, 2, 1, 1)

        self.a_workspace_width = QDoubleSpinBox(self.layoutWidget_4)
        self.a_workspace_width.setObjectName(u"a_workspace_width")
        self.a_workspace_width.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.a_workspace_width.setMaximum(1000.000000000000000)

        self.gridLayout_3.addWidget(self.a_workspace_width, 0, 1, 1, 1)

        self.label_2 = QLabel(self.layoutWidget_4)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout_3.addWidget(self.label_2, 0, 0, 1, 1)

        self.label_3 = QLabel(self.layoutWidget_4)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout_3.addWidget(self.label_3, 1, 0, 1, 1)


        self.verticalLayout_16.addWidget(self.row_2)


        self.verticalLayout.addWidget(self.row_1)

        self.stackedWidget.addWidget(self.page_2)

        self.formLayout.setWidget(0, QFormLayout.SpanningRole, self.stackedWidget)


        self.horizontalLayout_4.addWidget(self.pagesContainer)


        self.verticalLayout_6.addWidget(self.content)

        self.bottomBar = QFrame(self.contentBottom)
        self.bottomBar.setObjectName(u"bottomBar")
        self.bottomBar.setMinimumSize(QSize(0, 22))
        self.bottomBar.setMaximumSize(QSize(16777215, 22))
        self.bottomBar.setFrameShape(QFrame.Shape.NoFrame)
        self.bottomBar.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_5 = QHBoxLayout(self.bottomBar)
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.frame_size_grip = QFrame(self.bottomBar)
        self.frame_size_grip.setObjectName(u"frame_size_grip")
        self.frame_size_grip.setMinimumSize(QSize(20, 0))
        self.frame_size_grip.setMaximumSize(QSize(20, 16777215))
        self.frame_size_grip.setFrameShape(QFrame.Shape.NoFrame)
        self.frame_size_grip.setFrameShadow(QFrame.Shadow.Raised)

        self.horizontalLayout_5.addWidget(self.frame_size_grip)


        self.verticalLayout_6.addWidget(self.bottomBar)


        self.verticalLayout_2.addWidget(self.contentBottom)


        self.gridLayout_7.addWidget(self.contentBox, 0, 1, 1, 1)


        self.verticalLayout_5.addWidget(self.bgApp)

        MainWindow.setCentralWidget(self.styleSheet)

        self.retranslateUi(MainWindow)

        self.stackedWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.toggleButton.setText(QCoreApplication.translate("MainWindow", u"Hide", None))
        self.btn_home.setText(QCoreApplication.translate("MainWindow", u"Home", None))
        self.btn_new.setText(QCoreApplication.translate("MainWindow", u"Workspace", None))
        self.titleRightInfo.setText(QCoreApplication.translate("MainWindow", u"Subway Monitoring System", None))
#if QT_CONFIG(tooltip)
        self.minimizeAppBtn.setToolTip(QCoreApplication.translate("MainWindow", u"Minimize", None))
#endif // QT_CONFIG(tooltip)
        self.minimizeAppBtn.setText("")
#if QT_CONFIG(tooltip)
        self.maximizeRestoreAppBtn.setToolTip(QCoreApplication.translate("MainWindow", u"Maximize", None))
#endif // QT_CONFIG(tooltip)
        self.maximizeRestoreAppBtn.setText("")
#if QT_CONFIG(tooltip)
        self.closeAppBtn.setToolTip(QCoreApplication.translate("MainWindow", u"Close", None))
#endif // QT_CONFIG(tooltip)
        self.closeAppBtn.setText("")
        self.WorkspaceName.setText(QCoreApplication.translate("MainWindow", u"Subway Monitoring Status", None))
        self.B_tagOX.setTitle(QCoreApplication.translate("MainWindow", u"GroupBox", None))
        self.onLabel.setText(QCoreApplication.translate("MainWindow", u"\uc2b9\ucc28 \uc608\uc815 \uc778\uc6d0", None))
        self.onPerson.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.offLabel.setText(QCoreApplication.translate("MainWindow", u"\ud558\ucc28 \uc608\uc815 \uc778\uc6d0", None))
        self.offPerson.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.helpLabel.setText(QCoreApplication.translate("MainWindow", u"\ub3c4\uc6c0 \uc694\uccad \uc778\uc6d0", None))
        self.helpPerson.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.btnExportCsv.setText(QCoreApplication.translate("MainWindow", u"data collect button", None))
        self.vertexY.setSuffix("")
        self.labelBoxBlenderInstalation_15.setText(QCoreApplication.translate("MainWindow", u"X:", None))
        self.vertexX.setPrefix("")
        self.vertexX.setSuffix("")
        self.labelBoxBlenderInstalation_16.setText(QCoreApplication.translate("MainWindow", u"Y:", None))
        self.labelBoxBlenderInstalation_12.setText(QCoreApplication.translate("MainWindow", u"Vertex Count", None))
        self.labelBoxBlenderInstalation_14.setText(QCoreApplication.translate("MainWindow", u"Coordinates", None))
        self.labelBoxBlenderInstalation_4.setText(QCoreApplication.translate("MainWindow", u"Danger Zone Settings", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u" Open Existing Workspace", None))
        self.pushButton_4.setText(QCoreApplication.translate("MainWindow", u" Save as New Workspace", None))
        self.pushButton_3.setText(QCoreApplication.translate("MainWindow", u" Edit Workspace", None))
        self.labelBoxBlenderInstalation_26.setText(QCoreApplication.translate("MainWindow", u"Anchor", None))
        self.labelBoxBlenderInstalation_25.setText(QCoreApplication.translate("MainWindow", u"Y:", None))
        self.j_anchorX.setPrefix("")
        self.j_anchorX.setSuffix("")
        self.labelBoxBlenderInstalation_24.setText(QCoreApplication.translate("MainWindow", u"Anchor Position", None))
        self.k_anchorY.setSuffix("")
        self.labelBoxBlenderInstalation_22.setText(QCoreApplication.translate("MainWindow", u"Tag", None))
        self.labelBoxBlenderInstalation_23.setText(QCoreApplication.translate("MainWindow", u"X:", None))
        self.labelBoxBlenderInstalation_6.setText(QCoreApplication.translate("MainWindow", u"Anchor - Tag Settings", None))
        self.labelBoxBlenderInstalation_3.setText(QCoreApplication.translate("MainWindow", u"Workspace Settings", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"m", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"m", None))
        self.a_workspace_width.setSuffix("")
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Workspace Width", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Workspace Height", None))
    # retranslateUi


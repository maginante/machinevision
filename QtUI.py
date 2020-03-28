# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'QtUI.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1360, 768)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(0, 0, 1360, 768))
        self.frame.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")


        self.x1_label = QtWidgets.QLabel(self.frame)
        self.x1_label.setGeometry(QtCore.QRect(1020, 60, 330, 60))
        self.x1_label.setStyleSheet("background-color: rgb(46, 52, 54);\n"
"font: 57 13pt \"Ubuntu\";\n"
"color: rgb(255, 255, 255);")
        self.x1_label.setObjectName("x1_label")


        self.x1_dist_label = QtWidgets.QLabel(self.frame)
        self.x1_dist_label.setGeometry(QtCore.QRect(1020, 130, 330, 60))
        self.x1_dist_label.setStyleSheet("background-color: rgb(46, 52, 54);\n"
"font: 57 13pt \"Ubuntu\";\n"
"color: rgb(237, 212, 0);")
        self.x1_dist_label.setObjectName("x1_dist_label")

        self.x1_check_label = QtWidgets.QLabel(self.frame)
        self.x1_check_label.setGeometry(QtCore.QRect(1020, 130, 60, 60))
        self.x1_check_label.setStyleSheet("background-color: rgb(46, 52, 54);\n"
"font: 57 13pt \"Ubuntu\";\n"
"color: rgb(237, 212, 0);")
        self.x1_check_label.setObjectName("x1_check_label")

        self.feedbackX1_label = QtWidgets.QLabel(self.frame)
        self.feedbackX1_label.setGeometry(QtCore.QRect(1020, 200, 330, 60))
        self.feedbackX1_label.setStyleSheet("background-color: rgb(46, 52, 54);\n"
"font: 57 13pt \"Ubuntu\";\n"
"color: rgb(237, 212, 0);")
        self.feedbackX1_label.setObjectName("feedbackX1_label")


        self.y1_label = QtWidgets.QLabel(self.frame)
        self.y1_label.setGeometry(QtCore.QRect(1020, 280, 330, 60))
        self.y1_label.setStyleSheet("background-color: rgb(46, 52, 54);\n"
"font: 57 13pt \"Ubuntu\";\n"
"color: rgb(255, 255, 255);")
        self.y1_label.setObjectName("y1_label")

        self.y1_dist_label = QtWidgets.QLabel(self.frame)
        self.y1_dist_label.setGeometry(QtCore.QRect(1020, 350, 330, 60))
        self.y1_dist_label.setStyleSheet("background-color: rgb(46, 52, 54);\n"
"font: 57 13pt \"Ubuntu\";\n"
"color: rgb(115, 210, 22);")
        self.y1_dist_label.setObjectName("y1_dist_label")


        self.y1_check_label = QtWidgets.QLabel(self.frame)
        self.y1_check_label.setGeometry(QtCore.QRect(1020, 350, 60, 60))
        self.y1_check_label.setStyleSheet("background-color: rgb(46, 52, 54);\n"
"font: 57 13pt \"Ubuntu\";\n"
"color: rgb(237, 212, 0);")
        self.y1_check_label.setObjectName("y1_check_label")

        self.feedbackY1_label = QtWidgets.QLabel(self.frame)
        self.feedbackY1_label.setGeometry(QtCore.QRect(1020, 420, 330, 60))
        self.feedbackY1_label.setStyleSheet("background-color: rgb(46, 52, 54);\n"
"font: 57 13pt \"Ubuntu\";\n"
"color: rgb(237, 212, 0);")
        self.feedbackY1_label.setObjectName("feedbackY1_label")

        self.counter_label = QtWidgets.QLabel(self.frame)
        self.counter_label.setGeometry(QtCore.QRect(1020, 500, 330, 60))
        self.counter_label.setStyleSheet("background-color: rgb(46, 52, 54);\n"
"font: 57 13pt \"Ubuntu\";\n"
"color: rgb(255, 255, 255);")
        self.counter_label.setObjectName("counter_label")

        self.ranges_label = QtWidgets.QLabel(self.frame)
        self.ranges_label.setGeometry(QtCore.QRect(1020, 570, 330, 60))
        self.ranges_label.setStyleSheet("background-color: rgb(46, 52, 54);\n"
"font: 57 13pt \"Ubuntu\";\n"
"color: rgb(255, 255, 255);")
        self.ranges_label.setObjectName("ranges_label")

        self.Blank_label = QtWidgets.QLabel(self.frame)
        self.Blank_label.setGeometry(QtCore.QRect(10, 10, 1360, 41))
        self.Blank_label.setStyleSheet("background-color: rgb(46, 52, 54);\n"
"font: 57 13pt \"Ubuntu\";\n"
"color: rgb(255, 255, 255);")
        self.Blank_label.setObjectName("Blank_label")

        self.video_show_label = QtWidgets.QLabel(self.frame)
        self.video_show_label.setGeometry(QtCore.QRect(10, 60, 1000, 651))
        self.video_show_label.setStyleSheet("background-color: rgb(46, 52, 54);")
        self.video_show_label.setText("")
        self.video_show_label.setObjectName("video_show_label")
        # MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.x1_label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-weight:600; color:#ffffff;\">X1</span></p></body></html>"))
        self.x1_dist_label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-weight:600; color:#ffffff;\">Distx</span></p></body></html>"))
        self.y1_label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-weight:600; color:#ffffff;\">Y1</span></p></body></html>"))
        self.y1_dist_label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-weight:600; color:#ffffff;\">Disty1</span></p></body></html>"))
        self.Blank_label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-weight:600; color:#ffffff;\">Blank Name</span></p></body></html>"))

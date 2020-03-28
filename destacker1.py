# -*- coding: utf-8 -*-
import os
import sys
import cv2
import math
import time
import threading
import numpy as np
import settings as st
import patterns as pt

from datetime import datetime
from collections import Counter
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot

from timeit import default_timer as timer

from QtUI import *


class Thread(QThread):
    changePixmap = pyqtSignal(QImage)
    DefX1=0
    DefY1=0
    X1=0
    Y1=0
    FeedbackX1=""
    FeedbackY1=""
    BlankName="AI Ready"
    Counter=0
    FeedbackColorX1 = st.GRAY
    FeedbackColorY1 = st.GRAY
    FeedbackColorfontX1 = st.GRAY
    FeedbackColorfontY1 = st.GRAY
    status=[False, False, False, False]
    message = True
    blank_check = [-1, 0, -1]
    path_blank = st.BLANK_URL
    path_car = st.CAR_URL
    empty = True
    wait = 0
    RangeX1 = 0
    RangeY1 = 0
    mode = 1
    pile_list = []
    standby = True
    last_known_SlidecarTopCorner = [0, 0]
    last_known_SlidecarBottomCorner = [0, 0]
    last_known_SlidecarRightLeftCorner = [0, 0]
    last_known_SlidecarRightCorner = [0, 0]
    SlidecarTopCorner = [0, 0]
    SlidecarBottomCorner = [0, 0]
    SlidecarRightLeftCorner = [0, 0]
    SlidecarRightCorner = [0, 0]
    SlidecarL1Point = [0,0]
    SlidecarL2Point = [0,0]
    SlidecarB1Point = [0,0]
    SlidecarB2Point = [0,0]


    def PileMeasure(self, crop, listlong):
        sample = cv2.Canny(crop, 1, 90)
        pixel = 1
        for row in sample:
            if row[0] == 255:
                break
            pixel += 1
        self.pile_list.append(pixel)
        if len(self.pile_list) > listlong:
            self.pile_list.pop(0)
        data = Counter(self.pile_list)
        mode = data.most_common(1)[0][0]
        return mode

    def Distance(self, point1, point2):
        distance = math.sqrt(math.pow(point2[0] - point1[0], 2) + math.pow(point2[1] - point1[1], 2))/6.9429
        return distance

    def Sleep():
        print("gone")
        time.sleep(st.SLEEP)

    def TemplateMatching(self, rgbImage, template_img):
        top_left=()
        bottom_right=()
        img_gray = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2GRAY)
        w, h = template_img.shape[::-1]
        res = cv2.matchTemplate(img_gray, template_img, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        if max_val>0.8 and top_left is not None:
            bottom_right = (top_left[0] + w, top_left[1] + h)
            return (top_left, bottom_right, w, h)
        else:
            top_left=()
            bottom_right=()
            return (top_left, bottom_right, w, h)

    def Sleeping(self, rgbImage):
        img_gray = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2GRAY)
        template_img = cv2.imread(os.path.join(self.path_car,'nocar.jpg'), 0)
        w, h = template_img.shape[::-1]
        res = cv2.matchTemplate(img_gray,template_img,cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        print("Sleeping...")
        if max_val>0.8:
            self.BlankName="Processing..."
            self.standby = True
        else:
            self.DefX1=0
            self.DefY1=0
            self.X1=0
            self.Y1=0
            self.FeedbackX1=""
            self.FeedbackY1=""
            self.BlankName="AI Ready"
            self.Counter=0
            self.FeedbackColorX1 = st.GRAY
            self.FeedbackColorY1 = st.GRAY
            self.FeedbackColorfontX1 = st.GRAY
            self.FeedbackColorfontY1 = st.GRAY
            self.status=[False, False, False, False]
            self.message = True
            self.blank_check = [-1, 0, -1]
            self.path_blank = st.BLANK_URL
            self.path_car = st.CAR_URL
            self.empty = True
            self.wait = 0
            self.RangeX1 = 0
            self.RangeY1 = 0
            self.mode = 1
            self.pile_list = []
            self.standby = False
            self.last_known_SlidecarTopCorner = [0, 0]
            self.last_known_SlidecarBottomCorner = [0, 0]
            self.last_known_SlidecarRightLeftCorner = [0, 0]
            self.last_known_SlidecarRightCorner = [0, 0]
            self.SlidecarTopCorner = [0, 0]
            self.SlidecarBottomCorner = [0, 0]
            self.SlidecarRightLeftCorner = [0, 0]
            self.SlidecarRightCorner = [0, 0]

    def SlideCarDetection(self, roi_up, roi_down):
        templates_path=st.CAR_URL
        templates=os.listdir(templates_path)
        for template in templates:
            if template=="SlidecarTopTemplate.jpg":
                template_img = cv2.imread(os.path.join(templates_path,template),0)
                (top_left, bottom_right,w,h)=self.TemplateMatching(roi_up, template_img)
                if len(top_left)!=0:
                    self.SlidecarTopCorner=(int(top_left[0] + (w/2)) + st.ROICUPXMIN, int(top_left[1] + (h/2)) + st.ROICUPYMIN)
                    self.last_known_SlidecarTopCorner = self.SlidecarTopCorner
                    self.status[0] = False
                else:
                    self.SlidecarTopCorner = self.last_known_SlidecarTopCorner
                    self.status[0] = True

            elif template=="SlidecardBottomTemplate.jpg":
                template_img = cv2.imread(os.path.join(templates_path,template),0)
                (top_left, bottom_right,w,h)=self.TemplateMatching(roi_down,template_img)
                if len(top_left)!=0:
                    self.SlidecarBottomCorner=(int(top_left[0] + (w/2)) + st.ROICDOWNXMIN, int(top_left[1] + (h/2)) + st.ROICDOWNYMIN)
                    self.status[1] = False
                else:
                    self.SlidecarBottomCorner=self.last_known_SlidecarBottomCorner
                    self.status[1] = True

            elif template=="SlidecarRightLeftTemplate.jpg":
                template_img = cv2.imread(os.path.join(templates_path,template),0)
                (top_left, bottom_right,w,h)=self.TemplateMatching(roi_down, template_img)
                if len(top_left)!=0:
                    self.SlidecarRightLeftCorner=(int(top_left[0] + (w/2)) + st.ROICDOWNXMIN, int(top_left[1] + (h/2)) + st.ROICDOWNYMIN)
                    self.last_known_SlidecarRightLeftCorner = self.SlidecarRightLeftCorner
                    self.status[2] = False
                else:
                    self.SlidecarRightLeftCorner=self.last_known_SlidecarRightLeftCorner
                    self.status[2] = True

            elif template=="SlidecarRightTemplate.jpg":
                template_img = cv2.imread(os.path.join(templates_path,template),0)
                (top_left, bottom_right,w,h)=self.TemplateMatching(roi_down,template_img)
                if len(top_left)!=0:
                    self.SlidecarRightCorner=(int(top_left[0] + (w/2)) + st.ROICDOWNXMIN, int(top_left[1] + (h/2)) + st.ROICDOWNYMIN)
                    self.last_known_SlidecarRightCorner = self.SlidecarRightCorner
                    self.status[3] = False
                else:
                    self.SlidecarRightCorner=self.last_known_SlidecarRightCorner
                    self.status[3] = True

            else:
                if all(self.status) == True and self.message == True: #if all status are TRUE send message and record
                    self.message = False
                    #now = datetime.now()
                    #timestamp = now.strftime("%d-%m-%Y %H:%M:%S")
                    #print("timestamp =", timestamp)
                    #print("blank = ", Thread.BlankName)
                    #print("X1 = ", Thread.DefX1, Thread.X1)
                    #print("Y1 = ", Thread.DefY1, Thread.Y1)
                    self.last_known_SlidecarTopCorner = [0, 0]
                    self.last_known_SlidecarBottomCorner = [0, 0]
                    self.last_known_SlidecarRightLeftCorner = [0, 0]
                    self.last_known_SlidecarRightCorner = [0, 0]
                    self.SlidecarTopCorner = [0, 0]
                    self.SlidecarBottomCorner = [0, 0]
                    self.SlidecarRightLeftCorner = [0, 0]
                    self.SlidecarRightCorner = [0, 0]
                    self.standby = True

    def BlankRecognitionUnique(self, rgbImage, DetectdType):
        template = pt.PATTERNS[DetectdType]["TEMP"]
        top_left = ()
        bottom_right = ()
        img_gray = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2GRAY)
        template_img = cv2.imread(os.path.join(self.path_blank, template), 0)
        w, h = template_img.shape[::-1]
        res = cv2.matchTemplate(img_gray,template_img,cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        if max_val>0.8:
            self.BlankCheckerLoop(DetectdType)
            bottom_right = (top_left[0] + w, top_left[1] + h)
        else:
            if self.blank_check[1] > 0:
                self.blank_check[1] -= 1
            else:
                self.blank_check = [-1, 0, -1]
        return (top_left,bottom_right, DetectdType)

    def CheckEmpty(self, rgbImage):
        img_gray = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2GRAY)
        template_img = cv2.imread(os.path.join(self.path_car,'empty.jpg'), 0)
        w, h = template_img.shape[::-1]
        res = cv2.matchTemplate(img_gray,template_img,cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        if max_val>0.8:
            self.empty = True
            self.DefX1=0
            self.DefY1=0
            self.X1= 0
            self.Y1= 0
            self.BlankName = "waiting for blank..."
        else:
            self.empty = False


    def BlankCheckerLoop(self, detected):
        if self.blank_check[0] != detected:
            self.blank_check[0] = detected
            self.blank_check[1] += 1
        else:
            self.blank_check[1] += 1
            if self.blank_check[1] >= self.wait:
                self.blank_check[1] = self.wait

    def BlankRecognition(self,rgbImage):
        img_gray = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2GRAY)
        templates_path = self.path_blank
        if os.path.exists(templates_path):
            top_left = ()
            bottom_right = ()
            for template in pt.PATTERNS:
                if template["UP"] == False:
                    continue
                template_img = cv2.imread(os.path.join(templates_path, template["TEMP"]),0)
                w, h = template_img.shape[::-1]
                res = cv2.matchTemplate(img_gray,template_img,cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                top_left = max_loc
                if max_val>0.8:
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    self.DetectdType = template["ID"]
                    self.BlankCheckerLoop(self.DetectdType)
                    break
            return (top_left, bottom_right, self.DetectdType)

    def run(self):
        self.DetectdType=-1
        currentframe = 0
        IMG_SIZE = st.IMGSIZE
        cap = cv2.VideoCapture("1.avi")

        if st.DEVMODE == True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.DEVFRAME)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_SIZE[1])
        while True:
            ret, frame = cap.read()
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                roi_empty = frame[st.ROIEMPTYYMIN:st.ROIEMPTYYMAX, st.ROIEMPTYXMIN:st.ROIEMPTYXMAX]
                roi_blank = frame[st.ROIBLANKYMIN:st.ROIBLANKYMAX, st.ROIBLANKXMIN:st.ROIBLANKXMAX]
                roi_up = frame[st.ROICUPYMIN:st.ROICUPYMAX, st.ROICUPXMIN:st.ROICUPXMAX]
                roi_down = frame[st.ROICDOWNYMIN:st.ROICDOWNYMAX, st.ROICDOWNXMIN:st.ROICDOWNXMAX]

                if self.standby == True and (currentframe%120) == 0:
                    self.Sleeping(frame.copy())
                else:
                    if st.DEVMODE == True:
                        cv2.rectangle(rgbImage, (st.ROIBLANKXMIN, st.ROIBLANKYMIN), (st.ROIBLANKXMAX, st.ROIBLANKYMAX), st.BLUE, st.LTHICK_REC)
                        cv2.rectangle(rgbImage, (st.ROIEMPTYXMIN, st.ROIEMPTYYMIN), (st.ROIEMPTYXMAX, st.ROIEMPTYYMAX), st.WHITE, st.LTHICK_REC)
                        cv2.rectangle(rgbImage, (st.ROICUPXMIN, st.ROICUPYMIN), (st.ROICUPXMAX, st.ROICUPYMAX), st.PINK, 3)
                        cv2.rectangle(rgbImage, (st.ROICDOWNXMIN, st.ROICDOWNYMIN), (st.ROICDOWNXMAX, st.ROICDOWNYMAX), st.PINK, 3)
                        cv2.circle(rgbImage, (self.SlidecarTopCorner[0], self.SlidecarTopCorner[1]), st.COPEN_REC, st.PINK, 3)
                        cv2.circle(rgbImage, (self.SlidecarBottomCorner[0], self.SlidecarBottomCorner[1]), st.COPEN_REC, st.PINK, 3)

                    if (currentframe%40)==0 and self.blank_check[1] <= 0:
                        self.CheckEmpty(roi_empty.copy())

                    if (currentframe%15)==0:
                        self.SlideCarDetection(roi_up.copy(), roi_down.copy())
                    
                    if (currentframe%7)==0 and self.empty == False:
                        if self.blank_check[1] == 0:
                            (top_left,bottom_right,self.DetectdType)=self.BlankRecognition(roi_blank.copy())
                        else:
                            (top_left,bottom_right,self.DetectdType)=self.BlankRecognitionUnique(roi_blank.copy(), self.DetectdType)

                        if len(top_left)==0 or len(bottom_right)==0 or self.DetectdType==-1:
                            pass
                        else:
                            blank_data = pt.PATTERNS[self.DetectdType]
                            blank_defaults = blank_data["SETS"]
                            self.wait = blank_data["WAIT"]

                            Thread.BlankName = blank_data["NAME"]
                            Thread.DefX1 =  blank_defaults["DEFX1"]
                            Thread.DefY1 =  blank_defaults["DEFY1"]

                            calibrate_X1 =  blank_defaults["CALIB_X1"]
                            calibrate_Y1 =  blank_defaults["CALIB_Y1"]
                            slidecar_bottom_default_x = blank_defaults["CARBOT_X"]
                            slidecar_bottom_default_y = blank_defaults["CARBOT_Y"]
                            slidecar_left_default_x = blank_defaults["CARLEF_X"]
                            slidecar_left_default_y = blank_defaults["CARLEF_Y"]
                            blank_left_top_x =  blank_defaults["BLANKP1_X"]
                            blank_left_top_y =  blank_defaults["BLANKP1_Y"]
                            blank_right_bottom_x =  blank_defaults["BLANKP2_X"]
                            blank_right_bottom_y =  blank_defaults["BLANKP2_Y"]
                            variation_allowed_x1 =  blank_defaults["RANGE_X1"]
                            variation_allowed_y1 =  blank_defaults["RANGE_Y1"]

                            SlideCarRightPoint=(self.SlidecarRightLeftCorner[0]+slidecar_bottom_default_x, self.SlidecarRightLeftCorner[1]+slidecar_bottom_default_y)
                            SlideCarLeftPoint=(self.SlidecarBottomCorner[0]+slidecar_left_default_x, self.SlidecarBottomCorner[1]+slidecar_left_default_y)

                            top = top_left[0] + st.ROIBLANKXMIN, top_left[1] + st.ROIBLANKYMIN
                            bottom = bottom_right[0] + st.ROIBLANKXMIN, bottom_right[1] + st.ROIBLANKYMIN

                            crop_ymin = top[1] + blank_data["CROP"][0][0]
                            crop_ymax = top[1] + blank_data["CROP"][0][1]
                            crop_xmin = top[0] + blank_data["CROP"][1][0]
                            crop_xmax = top[0] + blank_data["CROP"][1][1]
                            dif = crop_ymax - crop_ymin
                            roi_counter = frame[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

                            self.mode = self.PileMeasure(roi_counter, blank_data["LISTLONG"])
                            dif_y = dif - self.mode

                            BlankRightLeftPoint = (top_left[0] + blank_left_top_x + st.ROIBLANKXMIN), (top_left[1]+blank_left_top_y + st.ROIBLANKYMIN - dif_y)
                            BlankBottomRightPoint = (bottom_right[0] + blank_right_bottom_x + st.ROIBLANKXMIN), (bottom_right[1] + blank_right_bottom_y + st.ROIBLANKYMIN + self.mode)


                            distanceX1 = self.Distance(BlankBottomRightPoint, SlideCarRightPoint)
                            distanceY1 = self.Distance(SlideCarLeftPoint, BlankRightLeftPoint)
                            variationX1 = Thread.DefX1 - distanceX1
                            variationY1 = Thread.DefY1 - distanceY1

                            if st.DEVMODE == True:
                                cv2.rectangle(rgbImage, (crop_xmin, crop_ymin), (crop_xmax, crop_ymax), st.PINK, 1)
                                print("mode: ", self.mode)

                            Thread.Counter = np.around(self.mode * blank_data["RATIO"][0] / blank_data["RATIO"][1], 0)

                            #Thread.X1=round(float(variationX1 + calibrate_X1), 2)
                            #Thread.Y1=round(float(variationY1 + calibrate_Y1), 2)
                            Thread.X1 = np.around(variationX1 + calibrate_X1, 1)
                            Thread.Y1 = np.around(variationY1 + calibrate_Y1, 1)

                            if -variation_allowed_x1 <= Thread.X1 <= variation_allowed_x1:
                                Thread.FeedbackX1= "[   OK   ]"
                                Thread.FeedbackColorfontX1 = st.GREEN
                                Thread.FeedbackColorX1 = st.GREEN
                            elif Thread.X1 < variation_allowed_x1:
                                Thread.FeedbackX1= "<<< MOVE LEFT"
                                Thread.FeedbackColorX1 = st.RED
                                Thread.FeedbackColorfontX1 = st.WHITE
                            else:
                                Thread.FeedbackX1= "MOVE RIGHT >>>"
                                Thread.FeedbackColorX1 = st.RED
                                Thread.FeedbackColorfontX1 = st.WHITE

                            if -variation_allowed_y1 <= Thread.Y1 <= variation_allowed_y1:
                                Thread.FeedbackY1= "[   OK   ]"
                                Thread.FeedbackColorfontY1 = st.GREEN
                                Thread.FeedbackColorY1 = st.GREEN
                            elif Thread.Y1 < variation_allowed_y1:
                                Thread.FeedbackY1= "FORWARD"
                                Thread.FeedbackColorY1 = st.RED
                                Thread.FeedbackColorfontY1 = st.WHITE
                            else:
                                Thread.FeedbackY1= "BACKWARD"
                                Thread.FeedbackColorY1 = st.RED
                                Thread.FeedbackColorfontY1 = st.WHITE

                            self.SlidecarL1Point = (self.SlidecarTopCorner[0]+st.P1_LEFTLINE_CARX,self.SlidecarTopCorner[1]+st.P1_LEFTLINE_CARY)
                            self.SlidecarL2Point = (self.SlidecarBottomCorner[0]+st.P2_LEFTLINE_CARX, self.SlidecarBottomCorner[1]+st.P2_LEFTLINE_CARY)
                            self.SlidecarB1Point = (self.SlidecarRightLeftCorner[0]+st.P1_BTMLINE_CARX, self.SlidecarRightLeftCorner[1]+st.P1_BTMLINE_CARY)
                            self.SlidecarB2Point = (self.SlidecarRightCorner[0]+st.P2_BTMLINE_CARX, self.SlidecarRightCorner[1]+st.P2_BTMLINE_CARY)
                            
                            Thread.RangeX1 = variation_allowed_x1
                            Thread.RangeY1 = variation_allowed_y1

                            cv2.rectangle(rgbImage, top, bottom, st.BLACK, st.LTHICK_REC)
                            
                            cv2.line(rgbImage, SlideCarLeftPoint, BlankRightLeftPoint, st.BLUE_L, st.LTHICK_MES)
                            cv2.line(rgbImage, BlankBottomRightPoint, SlideCarRightPoint, st.GREEN, st.LTHICK_MES)
               
                            cv2.circle(rgbImage, BlankBottomRightPoint, st.COPEN_REC, st.WHITE, st.CTHICK_MES)
                            cv2.circle(rgbImage, BlankRightLeftPoint, st.COPEN_REC, st.WHITE, st.CTHICK_MES)
                
                            cv2.circle(rgbImage, BlankBottomRightPoint, 5, st.WHITE, -1)
                            cv2.circle(rgbImage, BlankRightLeftPoint, 5, st.WHITE, -1)
                            
                cv2.line(rgbImage, (self.SlidecarL1Point[0], self.SlidecarL1Point[1]), (self.SlidecarL2Point[0],self.SlidecarL2Point[1]), st.BLUE_L, st.LTHICK_CAR)
                cv2.line(rgbImage, (self.SlidecarB1Point[0], self.SlidecarB1Point[1]), (self.SlidecarB2Point[0],self.SlidecarB2Point[1]), st.GREEN, st.LTHICK_CAR)
                
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(1200, 850, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
                currentframe += 1

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.initUI()
        path_blank = st.BLANK_URL
        path_car = st.CAR_URL

        if not os.path.exists(path_car):
            print("*** Important directory not found:: path_car***")
            exit()
        if not os.path.exists(path_blank):
            print("*** Important directory not found:: path_blank***")
            exit()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.ui.video_show_label.setPixmap(QPixmap.fromImage(image))
        self.ui.x1_label.setText("X   "+str(Thread.DefX1)+'"')
        self.ui.x1_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.x1_label.setStyleSheet("background-color: rgb(20,20,20);\n""color: rgb"+ str(st.WHITE))
        self.ui.x1_label.setFont(QtGui.QFont("Ubuntu",35,weight=QtGui.QFont.Bold))

        self.ui.y1_label.setText("Y   "+str(Thread.DefY1)+'"')
        self.ui.y1_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.y1_label.setStyleSheet("background-color: rgb(20,20,20);\n""color: rgb"+ str(st.WHITE))
        self.ui.y1_label.setFont(QtGui.QFont("Ubuntu",35,weight=QtGui.QFont.Bold))

        self.ui.x1_dist_label.setText(str(Thread.X1)+'"')
        self.ui.x1_dist_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.x1_dist_label.setStyleSheet("background-color: rgb(20,20,20);\n""color: rgb"+ str(Thread.FeedbackColorfontX1))
        self.ui.x1_dist_label.setFont(QtGui.QFont("Ubuntu",35,weight=QtGui.QFont.Bold))

        self.ui.y1_dist_label.setText(str(Thread.Y1)+'"')
        self.ui.y1_dist_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.y1_dist_label.setStyleSheet("background-color: rgb(20,20,20);\n""color: rgb"+ str(Thread.FeedbackColorfontY1))
        self.ui.y1_dist_label.setFont(QtGui.QFont("Ubuntu",35,weight=QtGui.QFont.Bold))

        self.ui.Blank_label.setText(Thread.BlankName)
        self.ui.Blank_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.Blank_label.setStyleSheet("background-color: rgb(20,20,20);\n""color: rgb(255, 255, 255)")
        self.ui.Blank_label.setFont(QtGui.QFont("Ubuntu",30,weight=QtGui.QFont.Bold))

        self.ui.feedbackX1_label.setText(Thread.FeedbackX1)
        self.ui.feedbackX1_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.feedbackX1_label.setStyleSheet("background-color: rgb(20,20,20);\n""color: rgb"+ str(Thread.FeedbackColorfontX1))
        self.ui.feedbackX1_label.setFont(QtGui.QFont("Ubuntu",25,weight=QtGui.QFont.Bold))

        self.ui.feedbackY1_label.setText(Thread.FeedbackY1)
        self.ui.feedbackY1_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.feedbackY1_label.setStyleSheet("background-color: rgb(20,20,20);\n""color: rgb"+ str(Thread.FeedbackColorfontY1))
        self.ui.feedbackY1_label.setFont(QtGui.QFont("Ubuntu",25,weight=QtGui.QFont.Bold))

        self.ui.counter_label.setText("b ≈ " + str(Thread.Counter))
        self.ui.counter_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.counter_label.setStyleSheet("background-color: rgb(20,20,20);\n""color: rgb"+ str(st.BLUE_L))
        self.ui.counter_label.setFont(QtGui.QFont("Ubuntu",18,weight=QtGui.QFont.Bold))

        self.ui.ranges_label.setText("x~" + str(Thread.RangeX1) + "  y~" + str(Thread.RangeY1))
        self.ui.ranges_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.ranges_label.setStyleSheet("background-color: rgb(20,20,20);\n""color: rgb"+ str(st.BLUE_L))
        self.ui.ranges_label.setFont(QtGui.QFont("Ubuntu",18,weight=QtGui.QFont.Bold))

        self.ui.x1_check_label.setStyleSheet("background-color: rgb"+str(Thread.FeedbackColorX1)+";\n""color: rgb(255, 255, 255)")
        self.ui.y1_check_label.setStyleSheet("background-color: rgb"+str(Thread.FeedbackColorY1)+";\n""color: rgb(255, 255, 255)")

    def initUI(self):
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.setWindowTitle("Press 1 Destacker 1")
    #mainWindow.showFullScreen()
    mainWindow.show()
    sys.exit(app.exec_())

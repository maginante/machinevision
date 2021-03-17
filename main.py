# -*- coding: utf-8 -*-
# import matplotlib.pyplot as plt
import os
import sys
import cv2
import math
# import emoji
# import gspread
import threading
import importlib
import numpy as np
import settings as st
import patterns as pt
from statistics import mean
from QtUI import *
from pylogix import PLC
from datetime import datetime

# from collections import Counter
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QWidget, QApplication
# from twilio.rest import Client
# from oauth2client.service_account import ServiceAccountCredentials


class Thread(QThread):
    blank_shot_count = 0
    empty_shot_count = 0
    up_shot_count = 0
    down_shot_count = 0
    recipe = 0
    changePixmap = pyqtSignal(QImage)
    BlankID = -1
    Empty_slide_car_blank_name = "Inteligencia Artificial OK"
    Empty_blank_name = "procesando "
    BlankName = Empty_slide_car_blank_name
    DefX1 = 0
    DefY1 = 0
    X1 = 0
    Y1 = 0
    FeedbackX1 = ""
    FeedbackY1 = ""
    Counter = 0
    mode = 0
    pile_list = []
    FeedbackColorX1 = st.GRAY
    FeedbackColorY1 = st.GRAY
    FeedbackColorfontX1 = st.GRAY
    FeedbackColorfontY1 = st.GRAY
    FeedbackColorfont = st.GRAY
    FeedbackColorNetwork = st.BLUE_L
    path_blank = st.BLANK_URL
    path_car = st.CAR_URL
    slide_car_in_status = False
    message = True
    empty = True
    standby = True
    RangeX1 = 0
    RangeY1 = 0
    last_known_SlidecarBottomCorner = [0, -20]
    last_known_SlidecarRightLeftCorner = [0, -20]
    blank_data = {}
    blank_recognition = 0
    pile_accurancy = 0
    FeedbackColorPile = st.GRAY
    SlidecarBottomCorner = [0, -20]
    SlidecarRightLeftCorner = [0, -20]
    point_intersection_Y = (0, -20)
    point_intersection_X = (0, -20)
    point_intersection_W = (0, -20)

    blank_locX_norm = []
    blank_locY_norm = []
    x_norm_list = []
    y_norm_list = []

    line_up = []
    line_down = []

    last_line_up = []
    last_line_down = []

    xmeasure = 0
    ymeasure = 0
    currentframe = 0

    car = False
    blank = False
    empty = False
    pixel_x = 0
    pixel_y = 0

    def get_data(self):
        if st.PLC == True:
            with PLC() as comm:
                comm.IPAddress = st.PLCIP
                tag_data = comm.Read(st.PLCTAG)
                if tag_data.Value != None:
                    Thread.recipe = tag_data.Value
                    self.recipe = tag_data.Value
                    self.blank_data = pt.PATTERNS.get(self.recipe)
                    Thread.FeedbackColorNetwork = st.GREEN
                    if pt.PATTERNS.get(self.recipe)['ENABLE'] == False:
                        self.recipe = 0
                        Thread.recipe = 0
                else:
                    Thread.FeedbackColorNetwork = st.RED
                    Thread.recipe = st.RECIPE
                    self.recipe = Thread.recipe
                    self.blank_data = pt.PATTERNS.get(
                        self.recipe)
        else:
            Thread.FeedbackColorNetwork = st.RED
            Thread.recipe = st.RECIPE
            self.recipe = Thread.recipe
            self.blank_data = pt.PATTERNS.get(
                self.recipe)
        timer = threading.Timer(st.TIMER, self.get_data).start()

    def AllParametersReset(self):
        self.BlankID = -1
        self.DefX1 = 0
        self.DefY1 = 0
        self.X1 = 0
        self.Y1 = 0
        self.FeedbackX1 = ""
        self.FeedbackY1 = ""
        self.BlankName = Thread.Empty_blank_name + \
            pt.PATTERNS.get(self.recipe)["NAME"]
        self.Counter = 0
        self.message = True
        self.empty = True
        self.RangeX1 = 0
        self.RangeY1 = 0
        self.standby = True
        self.pile_list = []
        self.mode = 0
        self.last_known_SlidecarBottomCorner = [0, -20]
        self.last_known_SlidecarRightLeftCorner = [0, -200]
        self.slide_car_in_status = False
        self.top = (0, -20)
        self.bottom = (0, -20)
        self.SlideCarLeftPoint = (0, -20)
        self.point_intersection_Y = (0, -20)
        self.point_intersection_X = (0, -20)
        self.SlideCarRightPoint = (0, -20)
        self.FeedbackColorX1 = st.GRAY
        self.FeedbackColorY1 = st.GRAY
        self.FeedbackColorfontX1 = st.GRAY
        self.FeedbackColorfontY1 = st.GRAY
        self.FeedbackColorfont = st.GRAY
        self.FeedbackColorPile = st.GRAY
        self.pile_accurancy = 0
        self.line_up = []
        self.line_down = []
        self.line_list = []
        self.last_line_up = []
        self.last_line_down = []
        self.pixel_x = 0
        self.pixel_y = 0

    def UI_AllParametersReset(self):
        Thread.BlankID = -1
        # Thread.DefX1 = 0
        # Thread.DefY1 = 0
        Thread.X1 = 0
        Thread.Y1 = 0
        Thread.FeedbackX1 = ""
        Thread.FeedbackY1 = ""
        Thread.BlankName = Thread.Empty_blank_name + \
            pt.PATTERNS.get(self.recipe)["NAME"]
        Thread.Counter = 0
        Thread.mode = 0
        Thread.message = True
        Thread.empty = True
        # Thread.RangeX1 = 0
        # Thread.RangeY1 = 0
        Thread.standby = True
        Thread.last_known_SlidecarBottomCorner = [0, -20]
        Thread.last_known_SlidecarRightLeftCorner = [0, -20]
        Thread.SlidecarBottomCorner = [0, -20]
        Thread.SlidecarRightLeftCorner = [0, -20]
        # COPY GRAY -20
        Thread.FeedbackColorX1 = st.GRAY
        Thread.FeedbackColorY1 = st.GRAY
        Thread.FeedbackColorfontX1 = st.GRAY
        Thread.FeedbackColorfontY1 = st.GRAY
        Thread.FeedbackColorfont = st.GRAY
        Thread.FeedbackColorPile = st.GRAY
        Thread.pile_accurancy = 0
        self.top = (0, -20)
        self.bottom = (0, -20)
        self.SlideCarLeftPoint = (0, -20)
        self.SlideCarRightPoint = (0, -20)
        self.point_intersection_X = (0, -20)
        self.point_intersection_Y = (0, -20)
        self.line_down = []
        self.line_up = []
        self.line_list = []
        self.last_line_up = []
        self.last_line_down = []
        self.pixel_x = 0
        self.pixel_y = 0

    def NoBlankParametersReset(self):
        self.BlankID = -1
        self.DefX1 = 0
        self.DefY1 = 0
        self.X1 = 0
        self.Y1 = 0
        self.FeedbackX1 = ""
        self.FeedbackY1 = ""
        self.BlankName = Thread.Empty_blank_name + \
            pt.PATTERNS.get(self.recipe)["NAME"]
        self.Counter = 0
        self.message = True
        self.empty = True
        self.RangeX1 = 0
        self.RangeY1 = 0
        self.mode = 0
        self.pile_list = []
        self.top = (0, -20)
        self.bottom = (0, -20)
        self.SlideCarLeftPoint = (0, -20)
        self.SlideCarRightPoint = (0, -20)

        self.point_intersection_X = (0, -20)
        self.point_intersection_Y = (0, -20)
        self.pile_accurancy = 0
        self.FeedbackColorX1 = st.GRAY
        self.FeedbackColorY1 = st.GRAY
        self.FeedbackColorfontX1 = st.GRAY
        self.FeedbackColorfontY1 = st.GRAY
        self.FeedbackColorfont = st.GRAY
        self.FeedbackColorPile = st.GRAY
        self.line_up = []
        self.line_down = []
        self.line_list = []
        self.last_line_up = []
        self.last_line_down = []
        self.pixel_x = 0
        self.pixel_y = 0

    def get_intersect(self, a1, a2, b1, b2):
        s = np.vstack([a1, a2, b1, b2])
        h = np.hstack((s, np.ones((4, 1))))
        l1 = np.cross(h[0], h[1])
        l2 = np.cross(h[2], h[3])
        x, y, z = np.cross(l1, l2)
        if z == 0:
            return (float('inf'), float('inf'))
        return (x/z, y/z)

    def slope(self, x1, y1, x2, y2):
        m = 0
        if (x2-x1) != 0:
            m = (y2 - y1) / (x2 - x1)
        return m

    def intercept(self, s, x, y):
        intcpt = y-(s*x)
        return intcpt

    def line_distance(self, line1, line2):
        line_dist = abs(line1[3]-line2[3]) / \
            math.sqrt(1+math.pow(line1[2], 2))
        return line_dist

    def Distance(self, point1, point2, convertion):
        distance = math.sqrt(
            math.pow(point2[0] - point1[0], 2) + math.pow(point2[1] - point1[1], 2)) / convertion
        return distance

    def DistancePX(self, point1, point2):
        distance = math.sqrt(
            math.pow(point2[0] - point1[0], 2) + math.pow(point2[1] - point1[1], 2))
        return distance

    def PileNormalize(self, value):
        self.pile_list.append(value)
        if len(self.pile_list) > st.PILEMODENORM:
            self.pile_list.pop(0)
        self.pile_accurancy = len(self.pile_list)
        Thread.pile_accurancy = self.pile_accurancy
        self.mode = int(mean(self.pile_list))
        if (self.pile_accurancy % 2) == 0:
            Thread.FeedbackColorPile = st.GREEN
        else:
            Thread.FeedbackColorPile = st.BLACK

    def TopNormalize(self, value):
        valueX = value[0]
        valueY = value[1]
        self.blank_locX_norm.append(valueX)
        self.blank_locY_norm.append(valueY)
        if len(self.blank_locX_norm) > st.MEASURENORM:
            self.blank_locX_norm.pop(0)
        valueX = round(mean(self.blank_locX_norm))
        return (valueX, valueY)

    def XNormalize(self, value):
        self.x_norm_list.append(round(value, 1))
        if len(self.x_norm_list) > st.MEASURENORM:
            self.x_norm_list.pop(0)
        z = round(mean(self.x_norm_list), 1)
        return z

    def YNormalize(self, value):
        self.y_norm_list.append(value)
        if len(self.y_norm_list) > st.MEASURENORM:
            self.y_norm_list.pop(0)
        y = round(mean(self.y_norm_list), 1)
        return y

    def equation(self, point1, point2):
        a = point2[1] - point1[1]
        b = point1[0] - point2[0]
        c = (a * (point1[0])) + (b * (point1[1]))
        return a, b, c

    def midpoint(self, x1, y1, x2, y2):
        return ((x1 + x2)/2, (y1 + y2)/2)

    def filter_lines(self, li):
        pile_lines = []
        bottom_line = []
        top_line = li[0][1]
        inside = range(0, top_line[1]+5)
        for line in li:
            if line[1][1] not in inside:
                pile_lines.append(line[1])
                self.last_line_down = line[1]

        if len(pile_lines) >= 1:
            bottom_line = min(pile_lines)
            self.last_line_down = bottom_line
        else:
            bottom_line = self.last_line_down

        lines = [top_line, bottom_line]
        print(lines)
        return lines

    def PileMeasure(self, crop, maxpile):
        lu = []
        pre_lines = []
        two_lines = []
        h, w, ch = crop.shape
        zeros = np.zeros((h, w))
        low = np.array([0, 0, 50], np.uint8)
        high = np.array([180, 255, 200], np.uint8)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, low, high)
        res = cv2.bitwise_or(crop, crop, mask=mask)
        res = cv2.GaussianBlur(res, (5, 5), 0)
        resn = cv2.normalize(res, zeros, 0, 255, cv2.NORM_MINMAX)
        cannied = cv2.Canny(resn, 100, 100)
        min_line_length = w / st.LEN_DIV
        max_line_gap = w / st.GAP_DIV

        lines = cv2.HoughLinesP(cannied, 1, np.pi/45, 20,
                                np.array([]), min_line_length, max_line_gap)
        # cv2.imwrite("im.jpg", cannied)
        # for line in lines:
        #     cv2.line(resn, (line[0][0], line[0][1]),
        #              (line[0][2], line[0][3]), (0, 0, 255), 1)
        # cv2.imwrite("resn.jpg", resn)

        if lines is not None:
            for line in lines:
                point1 = (line[0][0], line[0][1])
                point2 = (line[0][2], line[0][3])
                a, b, c = self.equation(point1, point2)
                if b != 0:
                    py1 = int((c+(-a*(1)))/b)
                    py2 = int((c+(-a*(w)))/b)
                else:
                    py1 = int((c+(-a*(1))))
                    py2 = int((c+(-a*(w))))

                fline = [1, py1, w, py2]
                if (all(i in range(1, 200) for i in fline)) == True:
                    lu.append([1, py1, w, py2])

            lines = sorted(lu, key=lambda a_entry: a_entry[1])

        if lines is not None and len(lines) > 1:
            # self.top_line = lines[0]
            pile_range = range(lines[0][1]-3, lines[0][1] + maxpile)
            # parent1 = (self.top_line[0], self.top_line[1])
            # parent2 = (self.top_line[2], self.top_line[3])
            # top_slope = self.slope(
            #     parent1[0], parent1[1], parent2[0], parent2[1])
            min_range_slope = st.SLOPEMIN
            max_range_slope = st.SLOPEMAX
            for line in lines:
                line_slope = self.slope(line[0], line[1], line[2],
                                        line[3])
                if line_slope < max_range_slope and line_slope > min_range_slope:
                    mp = self.midpoint(line[0], line[1], line[2], line[3])
                    mpc_x = int(mp[0])
                    mpc_y = int(mp[1])
                    mpu_x = int(mp[0])-5
                    mpu_y = int(mp[1])-5
                    mpd_x = int(mp[0])+5
                    mpd_y = int(mp[1])+5
                    up = crop[mpu_y:mpc_y, mpu_x:mpc_x]
                    down = crop[mpc_y:mpd_y, mpc_x:mpd_x]
                    shapes = [up.shape[0], up.shape[1],
                              down.shape[0], down.shape[1]]
                    if 0 in shapes:
                        pass
                    else:
                        up_avg = int(np.average(up))
                        down_avg = int(np.average(down))
                        difera = up_avg-down_avg
                        if difera not in range(0, 5):
                            if line[1] in pile_range:
                                pre_lines.append([difera, line])
        # print("pre ", pre_lines)
        if len(pre_lines) > 1:
            two_lines = self.filter_lines(pre_lines)
            pile_height = two_lines[1][1]-two_lines[0][1]
            if pile_height > 1:
                self.PileNormalize(pile_height)
                return two_lines

    def TemplateMatching(self, rgbImage, template_img):
        top_left = ()
        bottom_right = ()
        img_gray = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2GRAY)
        w, h = template_img.shape[::-1]
        res = cv2.matchTemplate(img_gray, template_img, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        if max_val > st.TEMPSENSE_TM and top_left is not None:
            bottom_right = (top_left[0] + w, top_left[1] + h)
            return (top_left, bottom_right, w, h)
        else:
            top_left = ()
            bottom_right = ()
            return (top_left, bottom_right, w, h)

    def Sleeping(self, rgbImage):
        img_gray = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2GRAY)
        template_img = cv2.imread(os.path.join(self.path_car, 'nocar.jpg'), 0)
        w, h = template_img.shape[::-1]
        res = cv2.matchTemplate(img_gray, template_img, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > st.TEMPSENSE_CA:
            self.standby = True
            self.AllParametersReset()
            self.UI_AllParametersReset()
        else:
            self.standby = False

    def SlideCarDetection(self, roi_up, roi_down):
        present_left = False
        present_right = False
        templates_path = st.CAR_URL
        templates = os.listdir(templates_path)
        for template in templates:
            if template == "SlidecardBottomTemplate.jpg":
                template_img = cv2.imread(
                    os.path.join(templates_path, template), 0)
                (top_left, bottom_right, w, h) = self.TemplateMatching(
                    roi_up, template_img)
                if len(top_left) != 0:
                    present_right = True
                    self.SlidecarBottomCorner = (round(
                        top_left[0] + (w/2)) + st.ROICUPXMIN, round(top_left[1] + (h/2)) + st.ROICUPYMIN)
                    self.last_known_SlidecarBottomCorner = self.SlidecarBottomCorner
                else:
                    self.SlidecarBottomCorner = self.last_known_SlidecarBottomCorner
            elif template == "SlidecarRightLeftTemplate.jpg":
                template_img = cv2.imread(
                    os.path.join(templates_path, template), 0)
                (top_left, bottom_right, w, h) = self.TemplateMatching(
                    roi_down, template_img)
                if len(top_left) != 0:
                    present_left = True
                    self.SlidecarRightLeftCorner = (round(
                        top_left[0] + (w/2)) + st.ROICDOWNXMIN, round(top_left[1] + (h/2)) + st.ROICDOWNYMIN)
                    self.last_known_SlidecarRightLeftCorner = self.SlidecarRightLeftCorner

                else:
                    self.SlidecarRightLeftCorner = self.last_known_SlidecarRightLeftCorner
        if present_left == False and present_right == False:
            self.car = False
            self.blank = False
            self.empty = True
        else:
            self.car = True

    def BlankRecognitionUnique(self, rgbImage):
        if pt.PATTERNS.get(self.recipe)['ENABLE'] == False:
            self.recipe = 0
        template = pt.PATTERNS[self.recipe]["TEMP"]
        top_left = ()
        bottom_right = ()
        img_gray = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2GRAY)
        template_img = cv2.imread(
            os.path.join(self.path_blank, template), 0)
        w, h = template_img.shape[::-1]
        res = cv2.matchTemplate(
            img_gray, template_img, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        self.blank_recognition = round(max_val*100)
        top_left = max_loc
        x_v, y_v = self.TopNormalize(top_left)
        top_left = (x_v, y_v)
        if max_val > st.TEMPSENSE_BL:
            self.blank = True
            bottom_right = (top_left[0] + w, top_left[1] + h)
        else:
            self.CheckEmpty(rgbImage)
        return (top_left, bottom_right)

    def CheckEmpty(self, rgbImage):
        img_gray = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2GRAY)
        template_img = cv2.imread(os.path.join(self.path_car, 'empty.jpg'), 0)
        w, h = template_img.shape[::-1]
        res = cv2.matchTemplate(img_gray, template_img, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        if max_val > st.TEMPSENSE_EM:
            self.blank = False
            self.empty = True
            self.UI_AllParametersReset()
        else:
            self.empty = False

    def run(self):
        self.currentframe = 0
        self.get_data()
        self.top = (0, 0)
        self.bottom = (0, 0)
        self.SlideCarLeftPoint = (0, 0)
        self.SlideCarRightPoint = (0, 0)

        IMG_SIZE = st.IMGSIZE

        def gstreamer_pipeline(
                capture_width=3264,
                capture_height=1848,
                display_width=IMG_SIZE[0],
                display_height=IMG_SIZE[1],
                framerate=st.FPS,
                flip_method=st.FLIP):
            return (
                "nvarguscamerasrc ! "
                "video/x-raw(memory:NVMM), "
                "width=(int)%d, height=(int)%d, "
                "format=(string)NV12, framerate=(fraction)%d/1 ! "
                "nvvidconv flip-method=%d ! "
                "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=(string)BGR ! appsink"
                % (
                    capture_width,
                    capture_height,
                    framerate,
                    flip_method,
                    display_width,
                    display_height,
                )
            )
        if st.GSTREAM == True:
            cap = cv2.VideoCapture(gstreamer_pipeline(
                flip_method=st.FLIP), cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture("creww.avi")

        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                roi_empty = frame[st.ROIEMPTYYMIN:st.ROIEMPTYYMAX,
                                  st.ROIEMPTYXMIN: st.ROIEMPTYXMAX]
                roi_blank = frame[st.ROIBLANKYMIN: st.ROIBLANKYMAX,
                                  st.ROIBLANKXMIN: st.ROIBLANKXMAX]
                roi_up = frame[st.ROICUPYMIN: st.ROICUPYMAX,
                               st.ROICUPXMIN:st.ROICUPXMAX]
                roi_down = frame[st.ROICDOWNYMIN:st.ROICDOWNYMAX,
                                 st.ROICDOWNXMIN:st.ROICDOWNXMAX]

                if st.BLANKSHOT > self.blank_shot_count:
                    cv2.imwrite(os.path.dirname(os.path.abspath(__file__)) +
                                '/blank_shot_'+str(self.blank_shot_count)+'.jpg', roi_blank)
                    self.blank_shot_count += 1
                    print('blankshot ', self.blank_shot_count)

                if st.EMPTYSHOT > self.empty_shot_count:
                    cv2.imwrite(os.path.dirname(os.path.abspath(__file__)) +
                                '/empty_shot.jpg', roi_empty)
                    self.empty_shot_count += 1
                    print('emptyshot ', self.empty_shot_count)

                if st.UPSHOT > self.up_shot_count:
                    cv2.imwrite(os.path.dirname(os.path.abspath(__file__)) +
                                '/up_shot.jpg', roi_up)
                    self.up_shot_count += 1
                    print('upshot ', self.up_shot_count)

                if st.DOWNSHOT > self.down_shot_count:
                    cv2.imwrite(os.path.dirname(os.path.abspath(__file__)) +
                                '/down_shot.jpg', roi_down)
                    self.down_shot_count += 1
                    print('downshot ', self.down_shot_count)

                if (self.currentframe % 48) == 0:
                    self.Sleeping(frame.copy())

                if (self.currentframe % 14) == 0:
                    if self.standby == True:
                        self.Sleeping(roi_blank.copy())
                    else:
                        self.SlideCarDetection(roi_up.copy(), roi_down.copy())
                        importlib.reload(pt)
                        importlib.reload(st)

                if (self.currentframe % 5) == 0 and self.car == True:
                    (top_left, bottom_right) = self.BlankRecognitionUnique(
                        roi_blank.copy())
                    if self.empty == False and self.blank == False:
                        self.Sleeping(roi_blank.copy())
                    else:
                        if len(top_left) == 0 or len(bottom_right) == 0:
                            self.NoBlankParametersReset()
                        else:
                            self.blank_data = pt.PATTERNS.get(self.recipe)
                            Thread.BlankName = self.blank_data["NAME"]
                            if self.recipe != 0:
                                Thread.FeedbackColorfont = st.WHITE
                                blank_defaults = self.blank_data["SETS"]
                                Thread.BlankID = self.recipe
                                Thread.DefX1 = blank_defaults["DEFX1"]
                                Thread.DefY1 = blank_defaults["DEFY1"]
                                slidecar_bottom_default_x = blank_defaults["CARBOT_X"]
                                slidecar_bottom_default_y = blank_defaults["CARBOT_Y"]
                                slidecar_left_default_x = blank_defaults["CARLEF_X"]
                                slidecar_left_default_y = blank_defaults["CARLEF_Y"]
                                variation_allowed_x1 = blank_defaults["RANGE_X1"]
                                variation_allowed_y1 = blank_defaults["RANGE_Y1"]

                                crop_ymin = self.top[1] + \
                                    self.blank_data["CROP"][0][0]
                                crop_ymax = self.top[1] + \
                                    self.blank_data["CROP"][0][1]
                                crop_xmin = self.top[0] + \
                                    self.blank_data["CROP"][1][0]
                                crop_xmax = self.top[0] + \
                                    self.blank_data["CROP"][1][1]

                                self.top = top_left[0] + \
                                    st.ROIBLANKXMIN, top_left[1] + \
                                    st.ROIBLANKYMIN
                                self.bottom = bottom_right[0] + \
                                    st.ROIBLANKXMIN, bottom_right[1] + \
                                    st.ROIBLANKYMIN

                                roi_counter = frame[crop_ymin:crop_ymax,
                                                    crop_xmin:crop_xmax]

                                pile_lines = self.PileMeasure(
                                    roi_counter, self.blank_data["PILE"])

                                if pile_lines is not None:
                                    line1 = pile_lines[0]
                                    line2 = pile_lines[1]

                                    self.line_up = (
                                        line1[0]+crop_xmin, line1[1]+crop_ymin), (line1[2]+crop_xmin, line1[3]+crop_ymin)

                                    # self.line_down = (int(
                                    #     line1[0]+crop_xmin), int(line1[1]+crop_ymin+self.mode)), (int(line1[2]+crop_xmin), int(line1[3]+(crop_ymin+self.mode)))

                                    self.line_down = (int(
                                        line2[0]+crop_xmin), int(line2[1]+crop_ymin)), (int(line2[2]+crop_xmin), int(line2[3]+(crop_ymin)))

                                Thread.mode = self.mode
                                self.SlideCarRightPoint = (
                                    self.SlidecarRightLeftCorner[0]+slidecar_bottom_default_x, self.SlidecarRightLeftCorner[1]+slidecar_bottom_default_y)
                                self.SlideCarLeftPoint = (
                                    self.SlidecarBottomCorner[0]+slidecar_left_default_x, self.SlidecarBottomCorner[1]+slidecar_left_default_y)

                                distanceX1 = self.Distance(
                                    self.point_intersection_X, self.SlideCarRightPoint, self.blank_data["CONVX"])
                                distanceY1 = self.Distance(
                                    self.SlideCarLeftPoint, self.point_intersection_Y, self.blank_data["CONVY"])

                                variationX1 = Thread.DefX1 - distanceX1
                                variationY1 = Thread.DefY1 - distanceY1

                                if st.SHOWPIXELS == True:
                                    cv2.rectangle(
                                        frame, (crop_xmin, crop_ymin), (crop_xmax, crop_ymax), st.PINK, 3)
                                    distancePX = (self.DistancePX(
                                        self.point_intersection_X, self.SlideCarRightPoint))/blank_defaults["DEFX1"]
                                    distancePY = (self.DistancePX(
                                        self.SlideCarLeftPoint, self.point_intersection_Y))/blank_defaults["DEFY1"]
                                    Thread.pixel_x = round(distancePX, 3)
                                    Thread.pixel_y = round(distancePY, 3)
                                    print(distancePX)
                                    # print(
                                    #     'x {} y {}'.format(
                                    #         round(distancePX, 2), round(distancePY, 2)))

                                Thread.Counter = int(
                                    self.mode * self.blank_data["RATIO"][0] / self.blank_data["RATIO"][1])

                                xmes = self.XNormalize(variationX1)
                                ymes = self.YNormalize(variationY1)

                                self.xmeasure = Thread.DefX1 - xmes
                                self.ymeasure = Thread.DefY1 - ymes

                                if abs(xmes) == 0.0:
                                    Thread.X1 = 0.0
                                else:
                                    Thread.X1 = xmes

                                if abs(ymes) == 0.0:
                                    Thread.Y1 = 0.0
                                else:
                                    Thread.Y1 = ymes

                                if -variation_allowed_x1 <= Thread.X1 <= variation_allowed_x1:
                                    Thread.FeedbackX1 = "   OK   "
                                    Thread.FeedbackColorfontX1 = st.GREEN
                                    Thread.FeedbackColorX1 = st.GREEN
                                elif Thread.X1 < variation_allowed_x1:
                                    Thread.FeedbackX1 = "<<< IZQUIERDA"
                                    Thread.FeedbackColorX1 = st.RED
                                    Thread.FeedbackColorfontX1 = st.WHITE
                                else:
                                    Thread.FeedbackX1 = "DERECHA >>>"
                                    Thread.FeedbackColorX1 = st.RED
                                    Thread.FeedbackColorfontX1 = st.WHITE

                                if -variation_allowed_y1 <= Thread.Y1 <= variation_allowed_y1:
                                    Thread.FeedbackY1 = "   OK   "
                                    Thread.FeedbackColorfontY1 = st.GREEN
                                    Thread.FeedbackColorY1 = st.GREEN
                                elif Thread.Y1 < variation_allowed_y1:
                                    Thread.FeedbackY1 = "ADELANTE"
                                    Thread.FeedbackColorY1 = st.RED
                                    Thread.FeedbackColorfontY1 = st.WHITE
                                else:
                                    Thread.FeedbackY1 = "ATRAS"
                                    Thread.FeedbackColorY1 = st.RED
                                    Thread.FeedbackColorfontY1 = st.WHITE

                                Thread.RangeX1 = variation_allowed_x1
                                Thread.RangeY1 = variation_allowed_y1

                if st.DEVMODE == True:
                    cv2.rectangle(frame, (st.ROIBLANKXMIN, st.ROIBLANKYMIN),
                                  (st.ROIBLANKXMAX, st.ROIBLANKYMAX), st.BLUE, st.LTHICK_REC)
                    cv2.rectangle(frame, (st.ROIEMPTYXMIN, st.ROIEMPTYYMIN),
                                  (st.ROIEMPTYXMAX, st.ROIEMPTYYMAX), st.WHITE, st.LTHICK_REC)
                    cv2.rectangle(frame, (st.ROICUPXMIN, st.ROICUPYMIN),
                                  (st.ROICUPXMAX, st.ROICUPYMAX), st.PINK, 2)
                    cv2.rectangle(frame, (st.ROICDOWNXMIN, st.ROICDOWNYMIN),
                                  (st.ROICDOWNXMAX, st.ROICDOWNYMAX), st.PINK, 2)

                    cv2.circle(
                        frame, (self.SlidecarBottomCorner[0], self.SlidecarBottomCorner[1]), st.COPEN_REC, st.BLACK, 3)

                    cv2.circle(
                        frame, (self.SlidecarRightLeftCorner[0], self.SlidecarRightLeftCorner[1]), st.COPEN_REC, st.BLACK, 3)

                    cv2.rectangle(frame, self.top, self.bottom,
                                  st.WHITE, st.LTHICK_REC)

                p1 = (int(self.top[0] + self.blank_data["P1"][0]),
                      int(self.top[1] + self.blank_data["P1"][1] + self.mode))
                p2 = (int(self.bottom[0] + self.blank_data["P2"][0]),
                      int(self.bottom[1] + self.blank_data["P2"][1] + self.mode))

                cen = self.blank_data["CENTER"]

                cv2.circle(frame, p1, 3, st.GREEN, -1)
                cv2.circle(frame, p2, 3, st.GREEN, -1)
                cv2.circle(frame, cen, 3, st.GREEN, -1)

                cv2.line(frame, p1, p2, st.GREEN, 1)

                cv2.rectangle(frame, (self.top[0], self.top[1]-30), (self.top[0]+400, self.top[1]+10),
                              st.BLACK, -1)
                cv2.putText(frame, ' '+str(self.blank_recognition) + '% ' +
                            self.blank_data["NAME"], self.top, cv2.FONT_HERSHEY_SIMPLEX, 1, st.GREEN,  2)

                if self.line_down:

                    self.point_intersection_X = self.get_intersect(
                        cen, self.SlideCarRightPoint, self.line_down[0], self.line_down[1])
                    self.point_intersection_X = (
                        int(self.point_intersection_X[0]), int(self.point_intersection_X[1]))

                    self.point_intersection_Y = self.get_intersect(
                        cen, self.SlideCarLeftPoint, p1, p2)
                    self.point_intersection_Y = (
                        int(self.point_intersection_Y[0]), int(self.point_intersection_Y[1]))

                    cv2.line(frame, self.SlideCarLeftPoint,
                             self.point_intersection_Y, st.RED, 2)
                    cv2.line(frame, (self.line_up[0][0], self.line_up[0][1]),
                             (self.line_up[1][0], self.line_up[1][1]), st.GREEN, 2)
                    cv2.line(frame, (int(self.line_down[0][0]), int(self.line_down[0][1])), (
                        int(self.line_down[1][0]), int(self.line_down[1][1])), st.GREEN, 2)
                    cv2.circle(frame, self.line_up[0], 2, (255, 0, 255), 3)
                    cv2.circle(
                        frame, (self.line_up[0][0], self.line_up[0][1]+self.mode), 2, (255, 0, 255), 3)

                    cv2.rectangle(frame, (self.point_intersection_X[0]-10, self.point_intersection_X[1] - int(self.mode) - 60), (self.point_intersection_X[0]+120, self.point_intersection_X[1] - int(self.mode)),
                                   st.BLACK, -1)

                    cv2.rectangle(frame, (self.point_intersection_Y[0]-10, self.point_intersection_Y[1] - int(self.mode) - 60), (self.point_intersection_Y[0]+120, self.point_intersection_Y[1] - int(self.mode)),
                                   st.BLACK, -1)
                    cv2.putText(frame, str(self.xmeasure), (self.point_intersection_X[0], self.point_intersection_X[1] - int(self.mode+15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, st.YELLOW,  2)
                    cv2.putText(frame, str(self.ymeasure), (self.point_intersection_Y[0], self.point_intersection_Y[1] - int(self.mode+15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, st.BLUE_L,  2)

                    cv2.line(frame, self.point_intersection_X,
                             self.SlideCarRightPoint, st.YELLOW, st.LTHICK_MES)

                    cv2.circle(frame, self.point_intersection_X,
                               int(self.mode), st.YELLOW, st.CTHICK_MES)

                    cv2.circle(frame, self.point_intersection_X,
                               5, st.RED, -1)
                    cv2.circle(frame, self.SlideCarRightPoint,
                               5, st.RED, -1)

                    cv2.line(frame, self.SlideCarLeftPoint,
                             self.point_intersection_Y, st.BLUE_L, st.LTHICK_MES)
                    cv2.circle(frame, self.point_intersection_Y,
                               int(self.mode), st.BLUE_L, st.CTHICK_MES)
                    cv2.circle(frame, self.point_intersection_Y,
                               5, st.RED, -1)
                    cv2.circle(frame, self.SlideCarLeftPoint,
                               5, st.RED, -1)

                cv2.circle(frame, (300, 300),
                           10, st.RED, -1)
                cv2.circle(frame, (700, 230),
                           10, st.RED, -1)
                cv2.circle(frame, (1030, 610),
                           10, st.RED, -1)

                h, w, ch = frame.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(
                    frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(1200, 750, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)

                self.currentframe += 1
                if self.currentframe > 10000:
                    self.currentframe = 1


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.initUI()
        self.NotifX1 = []
        self.NotifY1 = []
        self.NotifBlankID = []
        self.i = 0
        self.time_counter = 0
        self.initial_time_seconds = 0
        self.reduce_seconds = False
        self.trigger_false_zero = False
        self.time_for_calibration = 30
        path_blank = st.BLANK_URL
        path_car = st.CAR_URL

        if not os.path.exists(path_car):
            print("*** Important directory not found:: path_car***")
            exit()
        if not os.path.exists(path_blank):
            print("*** Important directory not found:: path_blank***")
            exit()

    def NotificationAndUpdate(self):
        CurrentTime = datetime.now()
        print("Whatsapp Notification sent Succesfully.....for the Blank:")
        # timestamp = CurrentTime.strftime("%d-%m-%Y %H:%M:%S")
        # BlankIDNotif = self.NotifBlankID[-1]
        # BlankDataNotif = pt.PATTERNS[BlankIDNotif]
        # BlankDataDefaultsNotif = BlankDataNotif["SETS"]
        # if st.WHATSAPP == "OFF" or st.SHEETS == "OFF":
        # 	print("*"*50)
        # 	print("Timestamp        : ", timestamp)
        # 	print("BlankName        : ", BlankDataNotif["NAME"])
        # 	print("Default X1       : ", BlankDataDefaultsNotif["DEFX1"])
        # 	print("Default Y1       : ", BlankDataDefaultsNotif["DEFY1"])
        # 	print("Range Allowed X1 : ", BlankDataDefaultsNotif["RANGE_X1"])
        # 	print("Range Allowed Y1 : ", BlankDataDefaultsNotif["RANGE_Y1"])
        # 	print("X1 Variation     :", self.NotifX1[-1])
        # 	print("Y1 Variation     :", self.NotifY1[-1])
        # 	print("*"*50)

        # if st.WHATSAPP == "ON":
        # 	BodyMessageTemplate = st.DEVICE+": "+emoji.emojize(':ballot_box_with_check:')+" "+BlankDataNotif["NAME"]+" Variation Not Allowed. "\
        # 		+ "X"+str(BlankDataDefaultsNotif["DEFX1"])+"("+str(self.NotifX1[-1])+"/"+str(BlankDataDefaultsNotif["RANGE_X1"])+"), "\
        # 		+ "Y"+str(BlankDataDefaultsNotif["DEFY1"])+"("+str(
        # 			self.NotifY1[-1])+"/"+str(BlankDataDefaultsNotif["RANGE_Y1"])+")."

        # 	# print(BodyMessageTemplate)
        # 	account_sid = 'ACb94deeac206c90b86b33d71bffe81b60'
        # 	auth_token = '0424e865934b30115436f96b5793464e'
        # 	client = Client(account_sid, auth_token)

        # 	message = client.messages.create(
        # 		from_='whatsapp:+14155238886',
        # 		body=BodyMessageTemplate,
        # 		to='whatsapp:+522227822264'
        # 	)
        # 	print("-"*50)
        # 	print("Whatsapp Notification sent Succesfully.....for the Blank:",
        # 		  BlankDataNotif["NAME"])
        # 	print("-"*50)
        # 	# print(message.sid)

        # if st.SHEETS == "ON":
        # 	print("-"*50)
        # 	scope = ['https://www.googleapis.com/auth/drive',
        # 			 'https://www.googleapis.com/auth/drive.file']
        # 	file_name = 'credentials_googleapi.json'
        # 	credentials = ServiceAccountCredentials.from_json_keyfile_name(
        # 		file_name, scope)
        # 	client = gspread.authorize(credentials)
        # 	destacker_sheet = client.open('destackers_log').sheet1
        # 	NewBlankData = [timestamp, st.DEVICE, BlankDataNotif["NAME"], BlankDataDefaultsNotif["DEFX1"], self.NotifX1[-1], BlankDataDefaultsNotif["RANGE_X1"],
        # 					BlankDataDefaultsNotif["DEFY1"], self.NotifY1[-1], BlankDataDefaultsNotif["RANGE_Y1"], self.mode]
        # 	destacker_sheet.append_row(NewBlankData)
        # 	print("Google sheet updated Succesfully.....for the Blank:",
        # 		  BlankDataNotif["NAME"])
        # 	print("-"*50)

    @ pyqtSlot(QImage)
    def setImage(self, image):
        nospace = ''
        x1_string = ('X ', str(Thread.DefX1), '"')
        y1_string = ('Y ', str(Thread.DefY1), '"')
        info_string = (st.DEVICE, "-R", str(Thread.recipe))
        pile_string = ("pila≈ ", str(Thread.Counter),
                       " u / ", str(Thread.mode))
        ranges_string = ("rangos≈ X~", str(Thread.RangeX1),
                         " / Y~", str(Thread.RangeY1))
        percent_string = (str(Thread.pile_accurancy), " %")
        pixel_string = (str(Thread.pixel_x), " @ ", str(Thread.pixel_y))

        self.ui.video_show_label.setPixmap(QPixmap.fromImage(image))
        self.ui.x1_label.setText(nospace.join(x1_string))
        self.ui.x1_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.x1_label.setStyleSheet(
            "background-color: rgb(20,20,20);\n""color: rgb" + str(st.YELLOW))
        self.ui.x1_label.setFont(QtGui.QFont(
            "Ubuntu Light", 30, weight=QtGui.QFont.Bold))

        self.ui.y1_label.setText(nospace.join(y1_string))
        self.ui.y1_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.y1_label.setStyleSheet(
            "background-color: rgb(20,20,20);\n""color: rgb" + str(st.BLUE_L))
        self.ui.y1_label.setFont(QtGui.QFont(
            "Ubuntu Light", 30, weight=QtGui.QFont.Bold))

        self.ui.x1_dist_label.setText(str(Thread.X1))
        self.ui.x1_dist_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.x1_dist_label.setStyleSheet(
            "background-color: rgb(20,20,20);\n""color: rgb" + str(Thread.FeedbackColorfontX1))
        self.ui.x1_dist_label.setFont(QtGui.QFont(
            "Ubuntu Light", 30, weight=QtGui.QFont.Bold))

        self.ui.y1_dist_label.setText(str(Thread.Y1))
        self.ui.y1_dist_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.y1_dist_label.setStyleSheet(
            "background-color: rgb(20,20,20);\n""color: rgb" + str(Thread.FeedbackColorfontY1))
        self.ui.y1_dist_label.setFont(QtGui.QFont(
            "Ubuntu Light", 30, weight=QtGui.QFont.Bold))

        self.ui.Blank_label.setText(Thread.BlankName)
        self.ui.Blank_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.Blank_label.setStyleSheet(
            "background-color: rgb(20,20,20);\n""color: rgb(255, 255, 255)")
        self.ui.Blank_label.setFont(QtGui.QFont(
            "Ubuntu Light", 30, weight=QtGui.QFont.Bold))

        self.ui.recipe_label.setText(nospace.join(info_string))
        self.ui.recipe_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.recipe_label.setStyleSheet(
            "background-color: rgb(0,0,0);\n""color: rgb" + str(Thread.FeedbackColorNetwork))
        self.ui.recipe_label.setFont(QtGui.QFont(
            "Ubuntu Light", 25, weight=QtGui.QFont.Bold))

        self.ui.feedbackX1_label.setText(Thread.FeedbackX1)
        self.ui.feedbackX1_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.feedbackX1_label.setStyleSheet(
            "background-color: rgb(20,20,20);\n""color: rgb" + str(Thread.FeedbackColorfontX1))
        self.ui.feedbackX1_label.setFont(
            QtGui.QFont("Ubuntu Light", 20, weight=QtGui.QFont.Bold))

        self.ui.feedbackY1_label.setText(Thread.FeedbackY1)
        self.ui.feedbackY1_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.feedbackY1_label.setStyleSheet(
            "background-color: rgb(20,20,20);\n""color: rgb" + str(Thread.FeedbackColorfontY1))
        self.ui.feedbackY1_label.setFont(
            QtGui.QFont("Ubuntu Light", 20, weight=QtGui.QFont.Bold))

        self.ui.counter_label.setText(nospace.join(pile_string))
        self.ui.counter_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.counter_label.setStyleSheet(
            "background-color: rgb(20,20,20);\n""color: rgb" + str(Thread.FeedbackColorfont))
        self.ui.counter_label.setFont(QtGui.QFont(
            "Ubuntu Light", 18, weight=QtGui.QFont.Bold))

        self.ui.ranges_label.setText(
            nospace.join(ranges_string))
        self.ui.ranges_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.ranges_label.setStyleSheet(
            "background-color: rgb(20,20,20);\n""color: rgb" + str(Thread.FeedbackColorfont))
        self.ui.ranges_label.setFont(QtGui.QFont(
            "Ubuntu Light", 18, weight=QtGui.QFont.Bold))

        self.ui.x1_check_label.setStyleSheet(
            "background-color: rgb"+str(Thread.FeedbackColorX1)+";\n""color: rgb(255, 255, 255)")
        self.ui.y1_check_label.setStyleSheet(
            "background-color: rgb"+str(Thread.FeedbackColorY1)+";\n""color: rgb(255, 255, 255)")
        self.ui.pile_check.setStyleSheet(
            "background-color: rgb"+str(Thread.FeedbackColorPile)+";\n""color: rgb(255, 255, 255)")

        self.ui.pile_label.setText(nospace.join(percent_string))
        self.ui.pile_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.pile_label.setStyleSheet(
            "background-color: rgb(20,20,20);\n""color: rgb" + str(Thread.FeedbackColorfont))
        self.ui.pile_label.setFont(QtGui.QFont(
            "Ubuntu Light", 30, weight=QtGui.QFont.Bold))

        self.ui.pixels_label.setText(nospace.join(pixel_string))
        self.ui.pixels_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.pixels_label.setStyleSheet(
            "background-color: rgb(0,0,0);\n""color: rgb(255, 255, 255)")
        self.ui.pixels_label.setFont(QtGui.QFont(
            "Ubuntu Light", 15))

    def initUI(self):
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.setWindowTitle("Press 1 Destacker 1")
    flags = QtCore.Qt.WindowFlags(
        QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
    mainWindow.setWindowFlags(flags)
    mainWindow.showFullScreen()
    mainWindow.show()
    sys.exit(app.exec_())
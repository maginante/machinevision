# TURN true TO SEE REGIONS
DEVMODE = False
DEVFRAME = 0
IMGSIZE = [1280, 720]

# STANDBY TIME IN SECONDS
SLEEP = 120

# TIMER TIME IN SECONDS
TIMER = 5

# BLANK URL TEMPLATES
BLANK_URL = './templates/blank'

# SLIDECAR URL TEMPLATES
CAR_URL = './templates/car'

# LINE THICKNESS FOR MEASURES
LTHICK_MES = 5
# LINE THICKNESS FOR CAR REFERENCES
LTHICK_CAR = 6
# LINE THICKNESS FOR BLANK DETECTED RECTANGLE
LTHICK_REC = 4
# CIRCLE SIZE FOR REFERENCE POINTS IN DETECTED RECTANGLE
COPEN_REC = 30
CTHICK_MES = 2

# BOTTOM LINE POINTS CALIBRATION -- POINT 1 = RIGHT, POINT 2 = LEFT
P1_BTMLINE_CARX = 0
P1_BTMLINE_CARY = 8
P2_BTMLINE_CARX = 0
P2_BTMLINE_CARY = 5

# LEFT LINE POINTS CALIBRATION -- POINT 1 = TOP, POINT 2 = BOTTOM
P1_LEFTLINE_CARX = -5
P1_LEFTLINE_CARY = 0
P2_LEFTLINE_CARX = -1
P2_LEFTLINE_CARY = 0

#COLORS
GREEN = (0, 255, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 255)
BLUE_L = (0, 255, 255)
PINK = (255, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (120, 120, 120)

# CROP POINTS FOR --BLANK-- DETECTION REGION OF INTEREST
ROIBLANKXMIN = 330
ROIBLANKXMAX = 900
ROIBLANKYMIN = 50
ROIBLANKYMAX = 500

# CROP POINTS FOR --EMPTY-- DETECTION REGION OF INTEREST
ROIEMPTYXMIN = 300
ROIEMPTYXMAX = 500
ROIEMPTYYMIN = 180
ROIEMPTYYMAX = 350

# CROP POINTS FOR --SLIDECAR UP-- DETECTION REGION OF INTEREST
ROICUPXMIN = 0
ROICUPXMAX = 320
ROICUPYMIN = 0
ROICUPYMAX = 150

# CROP POINTS FOR --SLIDECAR DOWN-- DETECTION REGION OF INTEREST
ROICDOWNXMIN = 330
ROICDOWNXMAX = 1150
ROICDOWNYMIN = 410
ROICDOWNYMAX = 720


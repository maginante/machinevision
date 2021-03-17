# -*- coding: utf-8 -*-

# 	3	:  	CAB_BACK_DJ             OK
# 	4	:  	BSA_MEGA_RH_DJ          OK
# 	5	:  	BSA_MEGA_LH_DJ          OK
# 	7	:  	FENDER_OTR
# 	8	:  	RR_DOOR_OUTER_DJ
# 	14	: 	BOX_OUTER_6_3_RH_DJ
# 	15	: 	BOX_OUTER_6_3_LH_DJ
# 	18	: 	BOX_OUTER_6_3_RH_DJ
# 	19	: 	BOX_OUTER_6_3_LH_DJ
# 	20	: 	FRT_DOOR_OUTER_DS
# 	21	: 	FRT_DOOR_OUTER_DJ
# 	23	: 	BSA_STD_LH_DS           OK
# 	24	: 	BSA_STD_RH_DS           OK
# 	27	: 	ROOF_OUTER_DS
# 	31	: 	BSA_CREW_LH_DJ          OK
# 	32	: 	BSA_CREW_RH_DJ
# 	37	: 	BOX_OTR_8_LH_DS
# 	38	: 	BOX_OTR_8_RH_DS
# 	39	: 	BOX_OTR_8_LH_DS
# 	40	: 	BOX_OTR_8_RH_DS


PATTERNS = {
    3:  {
        'ENABLE': True,
        'NAME': 'CAB_BACK_DJ',
        'TEMP': 'REC.jpg',
        'RATIO': [233, 49],
        'PILE': 30,
        'CROP': [(175, 255), (150, 180)],
        'SETS': {
                'DEFX1': 60.0,
                'DEFY1': 14.0,
                'CALIB_X1': -7.4,
                'CALIB_Y1': 4.2,
                'RANGE_X1': 0.3,
                'RANGE_Y1': 0.5,
                'CARBOT_X': 85,
                'CARBOT_Y': -35,
                'CARLEF_X': -112,
                'CARLEF_Y': -205,
                'BLANKP1_X': 25,
                'BLANKP1_Y': 227,
                'BLANKP2_X': -18,
                'BLANKP2_Y': -45,
        },
    },

    4:  {
        'ENABLE': True,
        'NAME': 'BSA_MEGA_RH_DJ',
        'TEMP': 'BSA_MEGA_RH_DJ.jpg',
        'RATIO': [233, 49],
        'PILE': 30,
        'CROP': [(120, 200), (220, 290)],
        # 'CROP': [(250, 350), (260, 330)],
        'CENTER': (550, 280),
        'P1': (-70, -20),
        'P2': (-250, 1),
        'CONVX': 7.17011,
        'CONVY': 8.35917,
        'SETS': {
            'DEFX1': 42.0,
            'DEFY1': 13.0,
            'CALIB_X1': 0,
            'CALIB_Y1': 0,
            'RANGE_X1': 0.3,
            'RANGE_Y1': 0.5,
            'CARBOT_X': 85,
            'CARBOT_Y': -35,
            'CARLEF_X': -112,
            'CARLEF_Y': -205,
            'BLANKP1_X': -25,
            'BLANKP1_Y': 135,
            'BLANKP2_X': -42,
            'BLANKP2_Y': -44,
        },
    },

    5:  {
        'ENABLE': True,
        'NAME': 'BSA_MEGA_LH_DJ',
        'TEMP': 'BSA_MEGA_LH_DJ.jpg',
        'RATIO': [233, 49],
        'PILE': 30,
        'CROP': [(140, 210), (70, 120)],
        'SETS': {
            'DEFX1': 31.0,
            'DEFY1': 13.0,
            'CALIB_X1': 0,
            'CALIB_Y1': 0,
            'RANGE_X1': 0.3,
            'RANGE_Y1': 0.5,
            'CARBOT_X': 85,
            'CARBOT_Y': -35,
            'CARLEF_X': -112,
            'CARLEF_Y': -205,
            'BLANKP1_X': -22,
            'BLANKP1_Y': 140,
            'BLANKP2_X': -67,
            'BLANKP2_Y': 2,
        },
    },

    7:  {
        'ENABLE': False,
        'NAME': 'FENDER_OTR',
        'TEMP': 'FENDER_OTR.jpg',
        'RATIO': [233, 49],
        'PILE': 30,
        'CROP': [(283, 325), (270, 280)],
        'SETS': {
            'DEFX1': 31.5,
            'DEFY1': 9.8,
            'CALIB_X1': 0.3,
            'CALIB_Y1': 4.2,
            'RANGE_X1': 0.1,
            'RANGE_Y1': 0.4,
            'CARBOT_X': 85,
            'CARBOT_Y': -35,
            'CARLEF_X': -112,
            'CARLEF_Y': -205,
            'BLANKP1_X': -22,
            'BLANKP1_Y': 207,
            'BLANKP2_X': 0,
            'BLANKP2_Y': 0,
        },
    },

    8:  {
        'ENABLE': False,
        'NAME': 'RR_DOOR_OUTER_DJ',
        'TEMP': 'REC.jpg',
        'RATIO': [233, 49],
        'PILE': 30,
        'CROP': [(283, 325), (270, 280)],
        'SETS': {
            'DEFX1': 34.5,
            'DEFY1': 23.0,
            'CALIB_X1': 0.3,
            'CALIB_Y1': 1.3,
            'RANGE_X1': 0.3,
            'RANGE_Y1': 0.3,
            'CARBOT_X': 85,
            'CARBOT_Y': -35,
            'CARLEF_X': -112,
            'CARLEF_Y': -191,
            'BLANKP1_X': -35,
            'BLANKP1_Y': 180,
            'BLANKP2_X': -50,
            'BLANKP2_Y': -35,
        },
    },

    14: {
        'ENABLE': False,
        'NAME': 'BOX_OUTER_6_3_RH_DJ',
        'TEMP': 'REC.jpg',
        'RATIO': [233, 49],
        'PILE': 30,
        'CROP': [(283, 325), (270, 280)],
        'SETS': {
            'DEFX1': 40.5,
            'DEFY1': 20.0,
            'CALIB_X1': 0.3,
            'CALIB_Y1': 4.2,
            'RANGE_X1': 0.2,
            'RANGE_Y1': 0.5,
            'CARBOT_X': 85,
            'CARBOT_Y': -35,
            'CARLEF_X': -112,
            'CARLEF_Y': -205,
            'BLANKP1_X': -22,
            'BLANKP1_Y': 207,
            'BLANKP2_X': 0,
            'BLANKP2_Y': 0,
        },
    },
    15: {
        'ENABLE': False,
        'NAME': 'BOX_OUTER_6_3_LH_DJ',
        'TEMP': 'REC.jpg',
        'RATIO': [233, 49],
        'PILE': 30,
        'CROP': [(283, 325), (270, 280)],
        'SETS': {
            'DEFX1': 51.0,
            'DEFY1': 20.0,
            'CALIB_X1': 0.3,
            'CALIB_Y1': 4.2,
            'RANGE_X1': 0.2,
            'RANGE_Y1': 0.5,
            'CARBOT_X': 85,
            'CARBOT_Y': -35,
            'CARLEF_X': -112,
            'CARLEF_Y': -205,
            'BLANKP1_X': -22,
            'BLANKP1_Y': 207,
            'BLANKP2_X': 0,
            'BLANKP2_Y': 0,
        },
    },

    18: {
        'ENABLE': False,
        'NAME': 'BOX_OUTER_6_3_RH_DJ',
        'TEMP': 'REC.jpg',
        'RATIO': [233, 49],
        'PILE': 30,
        'CROP': [(283, 325), (270, 280)],
        'SETS': {
            'DEFX1': 40.5,
            'DEFY1': 20.0,
            'CALIB_X1': 0.3,
            'CALIB_Y1': 4.2,
            'RANGE_X1': 0.2,
            'RANGE_Y1': 0.5,
            'CARBOT_X': 85,
            'CARBOT_Y': -35,
            'CARLEF_X': -112,
            'CARLEF_Y': -205,
            'BLANKP1_X': -22,
            'BLANKP1_Y': 207,
            'BLANKP2_X': 0,
            'BLANKP2_Y': 0,
        },
    },

    19: {
        'ENABLE': False,
        'NAME': 'BOX_OUTER_6_3_LH_DJ',
        'TEMP': 'REC.jpg',
        'RATIO': [233, 49],
        'PILE': 30,
        'CROP': [(283, 325), (270, 280)],
        'SETS': {
            'DEFX1': 51.0,
            'DEFY1': 20.0,
            'CALIB_X1': 0.3,
            'CALIB_Y1': 4.2,
            'RANGE_X1': 0.2,
            'RANGE_Y1': 0.5,
            'CARBOT_X': 85,
            'CARBOT_Y': -35,
            'CARLEF_X': -112,
            'CARLEF_Y': -205,
            'BLANKP1_X': -22,
            'BLANKP1_Y': 207,
            'BLANKP2_X': 0,
            'BLANKP2_Y': 0,
        },
    },

    20: {
        'ENABLE': False,
        'NAME': 'FRT_DOOR_OUTER_DS',
        'TEMP': 'REC.jpg',
        'RATIO': [233, 49],
        'PILE': 30,
        'CROP': [(283, 325), (270, 280)],
        'SETS': {
            'DEFX1': 34.5,
            'DEFY1': 14.0,
            'CALIB_X1': 0.3,
            'CALIB_Y1': 4.2,
            'RANGE_X1': 0.2,
            'RANGE_Y1': 0.5,
            'CARBOT_X': 85,
            'CARBOT_Y': -35,
            'CARLEF_X': -112,
            'CARLEF_Y': -205,
            'BLANKP1_X': -22,
            'BLANKP1_Y': 207,
            'BLANKP2_X': 0,
            'BLANKP2_Y': 0,
        },
    },

    21: {
        'ENABLE': False,
        'NAME': 'FRT_DOOR_OUTER_DJ',
        'TEMP': 'REC.jpg',
        'RATIO': [233, 49],
        'PILE': 30,
        'CROP': [(283, 325), (270, 280)],
        'SETS': {
            'DEFX1': 36.0,
            'DEFY1': 17.8,
            'CALIB_X1': 0.3,
            'CALIB_Y1': 4.2,
            'RANGE_X1': 0.2,
            'RANGE_Y1': 0.5,
            'CARBOT_X': 85,
            'CARBOT_Y': -35,
            'CARLEF_X': -112,
            'CARLEF_Y': -205,
            'BLANKP1_X': -22,
            'BLANKP1_Y': 207,
            'BLANKP2_X': 0,
            'BLANKP2_Y': 0,
        },
    },

    23: {
        'ENABLE': True,
        'NAME': 'BSA_STD_LH_DS_',
        'TEMP': 'BSA_STD_LH_DS.jpg',
        'RATIO': [300, 49],
        'PILE': 30,
        'CROP': [(160, 240), (250, 300)],
        'SETS': {
            'DEFX1': 55.0,
            'DEFY1': 13.0,
            'CALIB_X1': -5.6,
            'CALIB_Y1': 2.5,
            'RANGE_X1': 0.3,
            'RANGE_Y1': 0.5,
            'CARBOT_X': 87,
            'CARBOT_Y': -36,
            'CARLEF_X': -140,
            'CARLEF_Y': -255,
            'BLANKP1_X': 0,
            'BLANKP1_Y': 205,
            'BLANKP2_X': -150,
            'BLANKP2_Y': -13,
        },
    },

    24: {
        'ENABLE': True,
        'NAME': 'BSA_STD_RH_DS',
        'TEMP': 'BSA_STD_RH_DS.jpg',
        'RATIO': [233, 49],
        'PILE': 30,
        'CROP': [(160, 240), (250, 300)],
        'SETS': {
            'DEFX1': 59.0,
            'DEFY1': 13.0,
            'CALIB_X1': 0.3,
            'CALIB_Y1': 4.2,
            'RANGE_X1': 0.3,
            'RANGE_Y1': 0.5,
            'CARBOT_X': 87,
            'CARBOT_Y': -36,
            'CARLEF_X': -140,
            'CARLEF_Y': -255,
            'BLANKP1_X': 0,
            'BLANKP1_Y': 205,
            'BLANKP2_X': -150,
            'BLANKP2_Y': -13,
        },
    },

    27:  {
        'ENABLE': False,
        'NAME': 'ROOF_OUTER_DS',
        'TEMP': 'REC.jpg',
        'RATIO': [233, 49],
        'PILE': 30,
        'CROP': [(283, 325), (270, 280)],
        'SETS': {
            'DEFX1': 64.5,
            'DEFY1': 19.9,
            'CALIB_X1': 0.3,
            'CALIB_Y1': 4.2,
            'RANGE_X1': 0.2,
            'RANGE_Y1': 0.5,
            'CARBOT_X': 85,
            'CARBOT_Y': -35,
            'CARLEF_X': -112,
            'CARLEF_Y': -205,
            'BLANKP1_X': -22,
            'BLANKP1_Y': 207,
            'BLANKP2_X': 0,
            'BLANKP2_Y': 0,
        },
    },

    31:  {
        'ENABLE': True,
        'NAME': 'BSA_CREW_LH_DJ',
        'TEMP': 'BSA_CREW_LH_DJ.jpg',
        'RATIO': [250, 49],
        'PILE': 60,
        'PILE_DISTANCE': [10, 70],
        'CROP': [(240, 340), (270, 350)],
        'CENTER': (550, 280),
        'P1': (-70, 75),
        'P2': (-250, -15),
        'CONVX': 7.15,
        'CONVY': 8.36,
        'SETS': {
            'DEFX1': 31.5,
            'DEFY1': 13.0,
            'RANGE_X1': 0.2,
            'RANGE_Y1': 1.5,
            'CARBOT_X': 85,
            'CARBOT_Y': -35,
            'CARLEF_X': -112,
            'CARLEF_Y': -205,
        },
    },

    32: {
        'ENABLE': True,
        'NAME': 'BSA_CREW_RH_DJ',
        'TEMP': 'BSA_CREW_RH_DJ.jpg',
        'RATIO': [233, 49],
        'PILE': 30,
        'CROP': [(283, 325), (270, 280)],
        'SETS': {
            'DEFX1': 54.0,
            'DEFY1': 13.0,
            'CALIB_X1': 0.3,
            'CALIB_Y1': 4.2,
            'RANGE_X1': 0.1,
            'RANGE_Y1': 0.4,
            'CARBOT_X': 110,
            'CARBOT_Y': -50,
            'CARLEF_X': -112,
            'CARLEF_Y': -210,
            'BLANKP1_X': -22,
            'BLANKP1_Y': 207,
            'BLANKP2_X': 0,
            'BLANKP2_Y': 0,
        },
    },

    37:  {
        'ENABLE': False,
        'NAME': 'BOX_OTR_8_LH_DS',
        'TEMP': 'REC.jpg',
        'RATIO': [233, 49],
        'PILE': 30,
        'CROP': [(283, 325), (270, 280)],
        'SETS': {
            'DEFX1': 33.0,
            'DEFY1': 19.3,
            'CALIB_X1': 0.3,
            'CALIB_Y1': 4.2,
            'RANGE_X1': 0.2,
            'RANGE_Y1': 0.5,
            'CARBOT_X': 85,
            'CARBOT_Y': -35,
            'CARLEF_X': -112,
            'CARLEF_Y': -205,
            'BLANKP1_X': -22,
            'BLANKP1_Y': 207,
            'BLANKP2_X': 0,
            'BLANKP2_Y': 0,
        },
    },

    38: {
        'ENABLE': False,
        'NAME': 'BOX_OTR_8_RH_DS',
        'TEMP': 'BOX_OTR_8_RH_DS.jpg',
        'RATIO': [233, 49],
        'PILE': 30,
        'CROP': [(283, 325), (270, 280)],
        'SETS': {
            'DEFX1': 34.5,
            'DEFY1': 19.3,
            'CALIB_X1': 0.3,
            'CALIB_Y1': 4.2,
            'RANGE_X1': 0.2,
            'RANGE_Y1': 0.5,
            'CARBOT_X': 85,
            'CARBOT_Y': -35,
            'CARLEF_X': -112,
            'CARLEF_Y': -205,
            'BLANKP1_X': -22,
            'BLANKP1_Y': 207,
            'BLANKP2_X': 0,
            'BLANKP2_Y': 0,
        },
    },

    39: {
        'ENABLE': True,
        'NAME': 'BOX_OTR_8_LH_DS',
        'TEMP': 'REC.jpg',
        'RATIO': [233, 49],
        'PILE': 100,
        'CROP': [(150, 280), (140, 230)],
        'CENTER': (550, 280),
        'P1': (-70, 75),
        'P2': (-250, -15),
        'CONVX': 7.68,
        'CONVY': 8.36,
        'SETS': {
            'DEFX1': 33.0,
            'DEFY1': 19.3,
            'CALIB_X1': 0.3,
            'CALIB_Y1': 4.2,
            'RANGE_X1': 0.2,
            'RANGE_Y1': 0.5,
            'CARBOT_X': 85,
            'CARBOT_Y': -35,
            'CARLEF_X': -112,
            'CARLEF_Y': -205,
            'BLANKP1_X': -22,
            'BLANKP1_Y': 207,
            'BLANKP2_X': 0,
            'BLANKP2_Y': 0,
        },
    },

    40: {
        'ENABLE': False,
        'NAME': 'BOX_OTR_8_RH_DS',
        'TEMP': 'BOX_OTR_8_RH_DS.jpg',
        'RATIO': [233, 49],
        'PILE': 30,
        'CROP': [(283, 325), (270, 280)],
        'SETS': {
            'DEFX1': 34.5,
            'DEFY1': 19.3,
            'CALIB_X1': 0.3,
            'CALIB_Y1': 4.2,
            'RANGE_X1': 0.2,
            'RANGE_Y1': 0.5,
            'CARBOT_X': 85,
            'CARBOT_Y': -35,
            'CARLEF_X': -112,
            'CARLEF_Y': -205,
            'BLANKP1_X': -22,
            'BLANKP1_Y': 207,
            'BLANKP2_X': 0,
            'BLANKP2_Y': 0,
        },
    },

    0:  {
        'ENABLE': True,
        'NAME': '- receta desconocida o desactivada -',
        'TEMP': 'none.jpg',
        'RATIO': [1, 1],
        'PILE': 30,
        'CROP': [(1, 2), (1, 2)],
        'SETS': {
                'DEFX1': 0,
                'DEFY1': 0,
                'CALIB_X1': 0,
                'CALIB_Y1': 0,
                'RANGE_X1': 0,
                'RANGE_Y1': 0,
                'CARBOT_X': 0,
                'CARBOT_Y': 0,
                'CARLEF_X': 0,
                'CARLEF_Y': 0,
                'BLANKP1_X': 0,
                'BLANKP1_Y': 0,
                'BLANKP2_X': 0,
                'BLANKP2_Y': 0,
        },
    },
}
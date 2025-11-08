import cv2

class ArucoConfig:
    def __init__(self, config_name):
        self.config_name = config_name
        # Use integer constant instead of string
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
        self.marker_size = 0.05  # 5 cm
        self.placement_name = config_name
        self.marker_positions = {
            9:  (0,   0,   0),
            14: (20,  0,   0),
            19: (0,   20,  0),
            25: (20,  20,  0),
        }
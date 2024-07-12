import cv2
import numpy as np
tennis_diam = 6.7
def detection(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    blurred = cv2.GaussianBlur(mask, (11, 11), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    distance = []
    coordinates = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * np.pi * area ) / ( perimeter ** 2)
            if circularity > 0.5:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                aspect_ratio = 1.0
                if 0.7 < aspect_ratio < 1.3:
                    cv2.circle(frame, center, radius, (0, 255, 0), 2)
                    apparent_diameter_px = radius * 2
                    distance_cm = ( tennis_diam * frame.shape[1] / ( 2 * apparent_diameter_px * np.tan(np.deg2rad(62.2/2))))
                    distance.append(distance_cm)
                    coordinates.append(center)
    return frame, distance, coordinates

class DetectionResult: 
    def __init__(self):
        self.distance = None
        self.x = None
        self.y = None
        self.degree = None
        self.status = " "
    
    def set_distance(self, distance):
        self.distance = distance
    
    def set_x(self, x):
        self.x = x
        
    def set_y(self, y):
        self.y = y
        
    def set_degree(self, degree):
        self.degree = degree
        
    def set_status(self, status):
        self.status = status
    
    def get_text(self):
        return f"{self.distance:.2f} # {self.x:.2f} # {self.y:.2f} # {self.status} # {self.degree:.2f}"
    
    
def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to access camera")
            break
        detected_camera, distance, coordinate = detection(frame)
        
        if distance:
            min_distance_index = np.argmin(distance)
            min_distance = distance[min_distance_index]
            closet_object_center = coordinate[min_distance_index]
            
            y = min_distance
            x = (closet_object_center[0] - frame.shape[1] // 2 ) * ( min_distance / frame.shape[1])
            angle_radians = np.arctan2(x,y)
            angle_degree = np.degrees(angle_radians)
            
            result = DetectionResult()
            if -20 <= x <= 20 and 0 <= y <= 20:
                result.set_status("1")
            result.set_distance(min_distance)
            result.set_x(x)
            result.set_y(y)
            result.set_degree(angle_degree)
            result = result.get_text()
            print(result)
            cv2.imshow('frame',frame)
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break
    cap.release()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
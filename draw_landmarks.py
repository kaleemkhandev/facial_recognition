from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2 
import mediapipe as mp
import cv2
import urllib
import urllib.parse
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]
# Define the thresholds for gesture detection

FINGERS_OPEN_THRESHOLD = 30  # Adjust as needed
FINGERS_CLOSED_THRESHOLD = 20  # Adjust as needed
THUMB_UP_THRESHOLD = 23  # Adjust as needed
THUMB_DOWN_THRESHOLD = -20  # Adjust as needed
FOUR_FINGERS_OPEN_THRESHOLD = 80  # Adjust as needed
THUMB_CLOSED_THRESHOLD = 50  # Adjust as needed
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green


def get_landmark_coordinates(normalised_hand_landmarks_list:list,width,height):
    new_landmarks_list = []
    for point in normalised_hand_landmarks_list:
        x_cord = int((point.x)*width)
        y_cord = int((point.y)*height)
        z_cord =int((point.z)*100)
        new_landmarks_list.append([x_cord,y_cord])
    return new_landmarks_list

# Function to calculate the Euclidean distance between two landmarks
def calculate_distance(landmark1, landmark2):
    x1, y1 = landmark1
    x2, y2 = landmark2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def check_close_fingers(hand_landmarks):
    p0x,p0y =hand_landmarks[0]
    p7x,p7y = hand_landmarks[7]
    d07 = calculate_distance([p0x, p0y], [p7x, p7y])

    p6x,p6y = hand_landmarks[6]
    d06 = calculate_distance([p0x, p0y], [p6x, p6y])

    p11x,p11y = hand_landmarks[11]
    d011 = calculate_distance([p0x, p0y], [p11x, p11y])

    p10x,p10y = hand_landmarks[10]
    d010 = calculate_distance([p0x, p0y], [p10x, p10y])

    p15x,p15y = hand_landmarks[15]
    d015 = calculate_distance([p0x, p0y], [p15x, p15y])

    p14x, p14y = hand_landmarks[14]
    d014 = calculate_distance([p0x, p0y], [p14x, p14y])

    p19x,p19y = hand_landmarks[19]
    d019 = calculate_distance([p0x, p0y], [p19x, p19y])

    p18x,p18y = hand_landmarks[18] 
    d018 = calculate_distance([p0x, p0y], [p18x, p18y])
    close = []         
    if d06>d07:
        close.append(1)
    if d010>d011:
        close.append(2)
    if d014>d015:
        close.append(3)
    if d018>d019:
        close.append(4)

    return close
# Function to detect the gestures based on hand landmarks
def detect_gesture(hand_landmarks):
    # thmbs_up = False
    # thmbs_dwn = False
    # palm_flg = False

    thumb_landmark = hand_landmarks[4]  # Thumb landmark
    index_finger_landmark = hand_landmarks[8]  # Index finger landmark
    middle_finger_landmark = hand_landmarks[12]  # Middle finger landmark
    ring_finger_landmark = hand_landmarks[16]  # Ring finger landmark
    little_finger_landmark = hand_landmarks[20]  # Little finger landmark

    close_finger_check = check_close_fingers(hand_landmarks)
    ind_m,mid_m,ring_m,lit_m = hand_landmarks[6][1],hand_landmarks[10][1],hand_landmarks[14][1],hand_landmarks[18][1]
    thmb_y, ind_y,mid_y, ring_y,lit_y = thumb_landmark[1],index_finger_landmark[1], middle_finger_landmark[1], ring_finger_landmark[1],little_finger_landmark[1]
    thmb_x, ind_x,mid_x, ring_x,lit_x = thumb_landmark[0],index_finger_landmark[0], middle_finger_landmark[0], ring_finger_landmark[0],little_finger_landmark[0]

    smallest_y = 0
    for y in [ind_m,mid_m,ring_m,lit_m]: #if thumb (up) is above all, its y will be smallest
        if thmb_y<y:
            smallest_y+=1
    
    largest_y = 0
    for y in [ind_y,mid_y, ring_y,lit_y]:  #if thumb (down) is blelow all, its y will be largest
        if thmb_y>y:
            largest_y+=1

    largest_x = 0
    for x in [thmb_x, ind_x,mid_x, ring_x,lit_x]:
        if thmb_x>x:
            largest_x+=1

    if smallest_y==4:
        return "Thumbs_up"
    
    elif largest_y==4 and len(close_finger_check)==4 :
        return "Thumbs_down"

    elif largest_y ==4 and thumb_landmark[0]<index_finger_landmark[0] and len(close_finger_check)<2:
        return "four_finger"
    
    elif largest_y==4 and largest_x==4 and len(close_finger_check)==0:
        return "Palm"

    elif thumb_landmark[0]<index_finger_landmark[0] and len(close_finger_check)==4:
        return "Fist"
    
    else:
        return " "




if __name__ =="__main__":
    url_list = ["https://i.postimg.cc/MKMYg2gk/face-side-1.jpg",
                "https://i.postimg.cc/gJd1s75X/face-side-2.jpg",
                "https://i.postimg.cc/4dJFV97p/face-side-3.jpg",
                "https://i.postimg.cc/cLwDzmfQ/fist.jpg",
                "https://i.postimg.cc/pdBsWLGr/fist-1.jpg",
                "https://i.postimg.cc/qqWLbhRb/four-fingers.jpg",
                "https://i.postimg.cc/MHf0Q6Dy/palm.jpg",
                "https://i.postimg.cc/CMR9Nbr4/palm-1.jpg",
                "https://i.postimg.cc/Z5njKK7j/thumbs-up.jpg",
                "https://i.postimg.cc/XNggfytQ/thumbs-up-1.jpg",
                "https://i.postimg.cc/C56T496H/fist-1.jpg",
                "https://i.postimg.cc/NFFZLpfy/fist-2.jpg",
                "https://i.postimg.cc/vZZXycTB/fourfingers-2.jpg",
                "https://i.postimg.cc/MKqy37NN/test2.jpg",
                "https://i.postimg.cc/GpdVc77R/IMG-6789.jpg",
                "https://i.postimg.cc/GpdVc77R/IMG-6789.jpg",
                "https://i.postimg.cc/C56T496H/fist-1.jpg",
                "https://i.postimg.cc/43CK5SDg/palm-2.jpg",
                "https://i.postimg.cc/7hgDqKBg/thumbsup-1.jpg",
                "https://i.postimg.cc/fLrHFtZP/thd.jpg",
                "https://i.postimg.cc/g20YngSV/thd.jpg"
                ]
    # url_list = ["https://api.onevalidator.com/temp-images/167e3820-20ee-11ee-99cb-7599d383ef76.jpg",
    #     "https://api.onevalidator.com/temp-images/167e5f31-20ee-11ee-99cb-7599d383ef76.jpg",        
    #     "https://api.onevalidator.com/temp-images/167e5f33-20ee-11ee-99cb-7599d383ef76.jpg",
    #     "https://api.onevalidator.com/temp-images/167e5f30-20ee-11ee-99cb-7599d383ef76.jpg",
    #     "https://api.onevalidator.com/temp-images/167e5f32-20ee-11ee-99cb-7599d383ef76.jpg",
    #     "https://api.onevalidator.com/temp-images/167e5f34-20ee-11ee-99cb-7599d383ef76.jpg" ]
    # Load the Mediapipe HandLandmark model
    base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                        num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    identifier = "haider"
    urllib.parse.quote(':')
    url_response = urllib.request.urlopen(url_list[9])
    img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
    # Read and preprocess the input image
    image = cv2.imdecode(img_array, -1)
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height,width = image.shape[0],image.shape[1]
    to_show = image.copy() 
    cv2.imshow("image",to_show)
    cv2.waitKey(0)
    #creating mp image object
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    # STEP 4: Detect hand landmarks from the input image.
    detection_result = detector.detect(image)
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    points = get_landmark_coordinates(hand_landmarks_list[0],width,height)
    gesture = detect_gesture(points)

    for connection in CONNECTIONS:
        x0, y0= points[connection[0]]
        x1, y1 = points[connection[1]]
        # print(f'first execution:{end_1-start_1}')
        cv2.line(to_show, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)

    to_show = cv2.putText(to_show, gesture, (50, 50),  cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Result",to_show)
    cv2.waitKey(0)
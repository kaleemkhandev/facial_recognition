import cv2
import os
import pickle
import argparse
import urllib
import urllib.parse
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# from src.hand_tracker import HandTracker
from face_recognition_videos import facial_recog,closed_hand_check,thumbs_up_check,thumbs_down_check
from draw_landmarks import get_landmark_coordinates,check_close_fingers
from utils import get_image_paths,face_encodings
WINDOW="Hand Tracking"
FONT = cv2.FONT_HERSHEY_SIMPLEX
# PALM_MODEL_PATH = "models/palm_detection_without_custom_op.tflite"
# LANDMARK_MODEL_PATH = "models/hand_landmark.tflite"
ANCHORS_PATH = "models/anchors.csv"
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]


def processing(img_obj,mp_Dectector,hand_check,width,height):

    detection_result = mp_Dectector.detect(img_obj)
    hand_landmarks_list = detection_result.hand_landmarks
    if len(hand_landmarks_list)==0:
        return hand_check

    points = get_landmark_coordinates(hand_landmarks_list[0],width,height)
    if points is not None:
        if (points[2][0] > points[0][0]) and (points[17][1] > points[20][1]):
            hand_check = True
            pass
            # cv2.putText(frame ,'Right Hand' ,(110,50) ,FONT ,1, (204,229,255), 3)
            # if (points[5][0] > points[2][0]) or (points[17][0] >= points[16][0]):
            #     print("straighten pls")
            #     # cv2.putText(frame,'Please Straigten Your Hand',(150,150) ,FONT , 1, (87,58,25), 3) 
            # else:
            
        else:
            print("use right hand")
            # cv2.putText(frame,'Please use Your Right Hand',(110,50),FONT , 1, (87,58,25), 3)
    
    return hand_check

# def facial_recognition(frame,name_encodings):
#     frame = facial_recog(frame ,name_encodings)
#     return frame
 

# def model_init():
#     hand_3d = "True"
#     detector = HandTracker(
#         hand_3d,
#         PALM_MODEL_PATH,
#         LANDMARK_MODEL_PATH,
#         ANCHORS_PATH,
#         box_shift=0.2,
#         box_enlarge=1
#     )
#     eye_detector = blink_detection.f_detector.eye_blink_detector()
#     # load the encodings + names dictionary
#     # with open(f"Encodings/{identifier}.pickle", "rb") as f:
#     #     name_encodings_dict = pickle.load(f)
    
#     return detector,eye_detector

# def run(vid_cap):

#     counter = 0
#     total = 0
#     hand_check = False
#     face_recog_flag = False
#     detector,eye_detector = model_init()
#     with open("encodings.pickle", "rb") as f:
#         name_encodings_dict = pickle.load(f)
#     if vid_cap.isOpened():
#         hasFrame, frame = vid_cap.read()
#         frame = cv2.flip(frame, 1)
#     else:
#         hasFrame = False
    
#     while hasFrame:
#         frame,hand_check = processing(frame,detector,hand_check)
#         if hand_check:
#             frame = cv2.flip(frame, 1)
#             frame,counter,total = blink_detect(frame,eye_detector,counter,total)
#             if total==3:
#                 face_recog_flag = True

#             #after facial recognition, hand check flag will become false again, counter & total blinks will be reset to zero
#         if face_recog_flag:
#             hand_check = False
#             # counter = 0
#             total = 0
#             frame,names = facial_recog(frame,name_encodings_dict)

#             if len(names)<2:
#                 face_recog_flag = False
#                 return names

#         hasFrame, frame = vid_cap.read()
#         frame = cv2.flip(frame, 1)
#         if hasFrame==False:
#             return ["No Detections"]


# def run_image(frame,identifier):
#     hand_check = False
#     detector,_ = model_init()
#     frame = cv2.resize(frame,(600,500),interpolation = cv2.INTER_LINEAR)
#     frame,hand_check = processing(frame,detector,hand_check)
#     # check for the encodings first
#     # if not os.path.isfile(f"Encodings/{identifier}.pickle"):
#         return {"Warning":"User isn't registered. Please signup first."}
#     with open(f"Encodings/{identifier}.pickle", "rb") as f:
#         name_encodings_dict = pickle.load(f)

#     if hand_check:
#         frame,names = facial_recog(frame,name_encodings_dict)
#         if 0<len(names)<2:
#             return {"Names":names}
#         else:
#             return {"Warning": "Either no or more than two faces detetcted"}
#     else:
#         return {"Alert": "Show your Right hand and straighten it please!"}

def run_on_url_list(url_list,identifier):
    #initialize the media pipe model
    base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                        num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)
    if not os.path.isfile(f"Encodings/{identifier}.pickle"):
        return {"Warning":"User isn't registered. Please signup first."}

    with open(f"Encodings/{identifier}.pickle", "rb") as f:
        name_encodings_dict = pickle.load(f)

    hand_check = False
    for img_url in url_list:
        urllib.parse.quote(':')
        url_response = urllib.request.urlopen(img_url)
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        image = cv2.imdecode(img_array, -1)
        image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
        frame = cv2.resize(image,(600,500),interpolation = cv2.INTER_LINEAR)
        # frame = cv2.flip(frame,1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        height,width = frame.shape[0],frame.shape[1]
        hand_check = processing(image,detector,hand_check,width,height)
        # check for the encodings first
        if hand_check:
            frame,names = facial_recog(frame,name_encodings_dict)
            if 0<len(names)<2:
                if names[0] !='Unknown':
                    return {"Names":names}
            else:
                continue
    
    return {"Alert": "Show your face and Right hand and straighten it please!"}

def Gestures_Check(image,identifier):

    base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                        num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)
    height,width = image.shape[0],image.shape[1]
    to_draw = image.copy()
    #creating mp image object
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    # STEP 4: Detect hand landmarks from the input image.
    detection_result = detector.detect(image)
    hand_landmarks_list = detection_result.hand_landmarks
    # handedness_list = detection_result.handedness
    if hand_landmarks_list is None: 
        return {"Alert":"No hands detected"}
    # if handedness_list[0][0].display_name != "Right": 
    #     return {"Alert":"No Right hands detected"}

    hand_landmarks = get_landmark_coordinates(hand_landmarks_list[0],width,height)

    if not os.path.isfile(f"Encodings/{identifier}.pickle"):
        return {"Warning":"User isn't registered. Please signup first."}

    with open(f"Encodings/{identifier}.pickle", "rb") as f:
        name_encodings_dict = pickle.load(f)

    # STEP 5: Process the classification result. In this case, visualize it.
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
        print('Gesture: Thumbs Up')
        tu_check=True
        frame,names = facial_recog(to_draw,name_encodings_dict)
        if 0<len(names)<2:
            if names[0] !='Unknown':
                return {"user_name":names,
                        "four_finger_flag":False,
                        "thumbs_up_flag": True,
                        "thumbs_down_flag":False,
                        "fist_flag":False,
                        "palm_flag":False}
            else:
                return {"Alert":"Unknown user trying to invade."}
        else:
            return {"Warning": "Either no or more than two faces detetcted"}
    
    elif largest_y==4 and len(close_finger_check)==4 :
        print('Gesture: Thumbs Down')
        td_check=True
        frame,names = facial_recog(to_draw,name_encodings_dict)
        if 0<len(names)<2:
            if names[0] !='Unknown':
                return {"user_name":names,
                        "four_finger_flag":False,
                        "thumbs_up_flag": False,
                        "thumbs_down_flag":True,
                        "fist_flag":False,
                        "palm_flag":False}
            else:
                return {"Alert":"Unknown user trying to invade."}
        else:
            return {"Warning": "Either no or more than two faces detetcted"}
    
    elif  largest_y ==4 and thumb_landmark[0]<index_finger_landmark[0] and len(close_finger_check)==0:
        print('Gesture: Four fingers')
        ff_check=True
        frame,names = facial_recog(to_draw,name_encodings_dict)
        if 0<len(names)<2:
            if names[0] !='Unknown':
                return {"user_name":names,
                        "four_finger_flag":True,
                        "thumbs_up_flag": False,
                        "thumbs_down_flag":False,
                        "fist_flag":False,
                        "palm_flag":False}
            else:
                return {"Alert":"Unknown user trying to invade."}
        else:
            return {"Warning": "Either no or more than two faces detetcted"}
    

    elif largest_y==4 and largest_x==4 and len(close_finger_check)==0:
        print("palm")
        palm_check = True
        frame,names = facial_recog(to_draw,name_encodings_dict)
        if 0<len(names)<2:
            if names[0] !='Unknown':
                return {"user_name":names,
                        "four_finger_flag":False,
                        "thumbs_up_flag": False,
                        "thumbs_down_flag":False,
                        "fist_flag":False,
                        "palm_flag":True}
            else:
                return {"Alert":"Unknown user trying to invade."}
        else:
            return {"Warning": "Either no or more than two faces detetcted"}

    elif thumb_landmark[0]<index_finger_landmark[0] and len(close_finger_check)==4:
        print("Fist")
        fist_check = True
        frame,names = facial_recog(to_draw,name_encodings_dict)
        if 0<len(names)<2:
            if names[0] !='Unknown':
                return {"user_name":names,
                        "four_finger_flag":False,
                        "thumbs_up_flag": False,
                        "thumbs_down_flag":False,
                        "fist_flag":True,
                        "palm_flag":False}
            else:
                return {"Alert":"Unknown user trying to invade."}
        else:
            return {"Warning": "Either no or more than two faces detetcted"}

    else:
        return {" "}



def face_encoding(root_dir):
    class_names = os.listdir(root_dir)
    image_paths = get_image_paths(root_dir, class_names)
    # initialize a dictionary to store the name of each person and the corresponding encodings
    name_encondings_dict = {}
    # initialize the number of images processed
    nb_current_image = 1
    # now we can loop over the image paths, locate the faces, and encode them
    for image_path in image_paths:
        print(f"Image processed {nb_current_image}/{len(image_paths)}")
        # load the image
        image = cv2.imread(image_path)
        # get the face embeddings
        encodings = face_encodings(image)
        # get the name from the image path
        name = image_path.split(os.path.sep)[-2]
        # get the encodings for the current name
        e = name_encondings_dict.get(name, [])
        # update the list of encodings for the current name
        e.extend(encodings)
        # update the list of encodings for the current name
        name_encondings_dict[name] = e
        nb_current_image += 1

    # save the name encodings dictionary to disk
    with open("encodings.pickle", "wb") as f:
        pickle.dump(name_encondings_dict, f)


    
def get_facial_encodings(identifier:str,urls_list:list):
    # initialize a dictionary to store the name of each person and the corresponding encodings
    name_encondings_dict = {}

    # initialize the number of images processed
    nb_current_image = 1
    name = identifier
    # now we can loop over the image paths, locate the faces, and encode them
    for url in urls_list:
        # load the image
        urllib.parse.quote(':')
        url_response = urllib.request.urlopen(url)
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        image = cv2.imdecode(img_array, -1)
        image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
        # get the face embeddings
        encodings = face_encodings(image)
        # get the name from the image url
        # name = url.split('/')[-1].split('-')[0] #name of the user
        # get the encodings for the current name
        e = name_encondings_dict.get(name, [])
        # update the list of encodings for the current name
        e.extend(encodings)
        # update the list of encodings for the current name
        name_encondings_dict[name] = e
        nb_current_image += 1

    # save the name encodings dictionary to disk
    if not os.path.isdir('Encodings'):
        os.mkdir("Encodings") 
    with open(f"Encodings/{identifier}.pickle", "wb") as f:
        pickle.dump(name_encondings_dict, f)

if __name__ == "__main__": # testing the signup api
    # url_list = ["https://prod-images.tcm.com/Master-Profile-Images/LeonardoDiCaprio.jpg",
    #         "https://media.gettyimages.com/id/513145516/photo/actor-leonardo-dicaprio-attends-the-88th-annual-academy-awards-at-hollywood-highland-center-on.jpg?s=612x612&w=0&k=20&c=qdbpBg3hT2efIn7wnAKmkqlN3mqO_ICSSa8jd0-p2E0=",
    #         "https://media.gettyimages.com/id/1177758661/photo/leonardo-dicaprio-speaks-onstage-during-the-2019-global-citizen-festival-power-the-movement.jpg?s=612x612&w=0&k=20&c=mT14ucF4BmAD7y1XbRvR47iJlPbJdm5GpNvWC_8C48Y=",
    #         "https://media.gettyimages.com/id/510254326/photo/leonardo-dicaprio-attends-the-ee-british-academy-film-awards-at-the-royal-opera-house-on.jpg?s=612x612&w=0&k=20&c=SDuRZsr2eC5tBTaGU-WqYdNw58g3Qc92CMkc6FDlVf4=",
    #         "https://media.gettyimages.com/id/1357377703/photo/leonardo-dicaprio-attends-the-world-premierof-netflixs-dont-look-up-at-jazz-at-lincoln-center.jpg?s=612x612&w=0&k=20&c=2J3xv0vTopwcZ76PluXr9y-N1rNz-XrSnu0bisPdPAo=",
    #         "https://media.gettyimages.com/id/1357369478/photo/leonardo-dicaprio-attends-the-world-premiere-of-netflixs-dont-look-up-on-december-05-2021-in.jpg?s=612x612&w=0&k=20&c=r5mPh8xt7GcZ3oA2qdYcGKYSBde4kRvxiwFPsmDL8_8=",
    #         "https://media.gettyimages.com/id/1205143953/photo/leonardo-dicaprio-attends-the-92nd-annual-academy-awards-at-hollywood-and-highland-on.jpg?s=612x612&w=0&k=20&c=MvJqsfQzDqDdrXOh8CDPGJhCjm5C5dA9czVEAlfTIus=",
    #         "https://media.gettyimages.com/id/1200624256/photo/leonardo-dicaprio-speaks-onstage-during-the-26th-annual-screen-actors%C2%A0guild-awards-at-the.jpg?s=612x612&w=0&k=20&c=vDMgOI-8JywRaRHBAV-iYjk3MH-SYDOAJ_Do8zmCv9k="
    #         ]
    url_list = ["https://i.postimg.cc/MKMYg2gk/face-side-1.jpg",
                 "https://i.postimg.cc/gJd1s75X/face-side-2.jpg",
                 "https://i.postimg.cc/4dJFV97p/face-side-3.jpg",
                  "https://i.postimg.cc/cLwDzmfQ/fist.jpg",
              "https://i.postimg.cc/pdBsWLGr/fist-1.jpg",
                 "https://i.postimg.cc/qqWLbhRb/four-fingers.jpg",
              "https://i.postimg.cc/MHf0Q6Dy/palm.jpg",
                  "https://i.postimg.cc/CMR9Nbr4/palm-1.jpg",
                  "https://i.postimg.cc/MKqy37NN/test2.jpg",
               "https://i.postimg.cc/Z5njKK7j/thumbs-up.jpg",
              "https://i.postimg.cc/XNggfytQ/thumbs-up-1.jpg",
              "https://i.postimg.cc/C56T496H/fist-1.jpg",
              "https://i.postimg.cc/NFFZLpfy/fist-2.jpg",
              "https://i.postimg.cc/26cKphFC/fist-3.jpg",
              "https://i.postimg.cc/nr9T65h1/fist-4.jpg",
              "https://i.postimg.cc/Pq6gHgdG/fist-5.jpg",
              "https://i.postimg.cc/GpdVc77R/IMG-6789.jpg"
                  ]
    url_list = ["https://api.onevalidator.com/temp-images/65ca66a0-20ef-11ee-99cb-7599d383ef76.jpg",
        "https://api.onevalidator.com/temp-images/65ca66a2-20ef-11ee-99cb-7599d383ef76.jpg",        
        "https://api.onevalidator.com/temp-images/65ca66a4-20ef-11ee-99cb-7599d383ef76.jpg",
        "https://api.onevalidator.com/temp-images/65ca66a1-20ef-11ee-99cb-7599d383ef76.jpg",
        "https://api.onevalidator.com/temp-images/65ca66a3-20ef-11ee-99cb-7599d383ef76.jpg",
        "https://api.onevalidator.com/temp-images/65ca8db0-20ef-11ee-99cb-7599d383ef76.jpg" ]
    # data= {"leonardo":["https://s41230.pcdn.co/wp-content/uploads/2019/03/leonardo-dicaprio-environment-birthday-update-HEADER-1024x576.jpg"]}
    # data= {"leonardo":["https://static-koimoi.akamaized.net/wp-content/new-galleries/2023/06/actor-leonardo-dicaprio-once-posed-with-a-deformed-fetus-kept-in-a-glass-jar-01.jpg"]}
    # data= {"haider":["https://i.postimg.cc/MKMYg2gk/face-side-1.jpg"]}

    # identifier = list(data.keys())[0]
    identifier = "haider"
    # url_list = data[identifier]
    response = run_on_url_list(url_list,"hamza")
    print(response)
    # urllib.parse.quote(':')
    # url_response = urllib.request.urlopen(url_list[-1])
    # img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
    # image = cv2.imdecode(img_array, -1)
    # image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # response = Gestures_Check(image_rgb,identifier)
    # print("Done")
    # print(response)




# if __name__ =="__main__": # for testing the proces_mage api
#     hand_check = False
#     face_recog_flag = False
#     detector,_, name_encodings_dict = model_init()
#     frame = cv2.imread('test_images/4.jpg')
#     frame = cv2.resize(frame,(600,500),interpolation = cv2.INTER_LINEAR)
#     cv2.imshow('frame',frame)
#     cv2.waitKey(0)
#     frame,hand_check = processing(frame,detector,hand_check)
#     # cv2.imshow('predicted',frame)
#     # cv2.waitKey(0)
#     if hand_check:
#         frame,names = facial_recog(frame,name_encodings_dict)
#         cv2.imshow('predicted',frame)
#         cv2.waitKey(0)


           

# if __name__ =="__main__": #for complete face, eyes and hand landmarks detections
#     counter = 0
#     total = 0
#     hand_check = False
#     face_recog_flag = False
#     cv2.namedWindow(WINDOW)
#     vid_cap = cv2.VideoCapture(0)
#     detector,eye_detector, name_encodings_dict = model_init()
    
#     if vid_cap.isOpened():
#         hasFrame, frame = vid_cap.read()
#         frame = cv2.flip(frame, 1)
#     else:
#         hasFrame = False
    
#     while hasFrame:
        
#         frame,hand_check = processing(frame,detector,hand_check)
#         if hand_check:
#             frame = cv2.flip(frame, 1)
#             frame,counter,total = blink_detect(frame,eye_detector,counter,total)
#             if total==3:
#                 face_recog_flag = True

#             #after facial recognition, hand check flag will become false again, counter & total blinks will be reset to zero
#         if face_recog_flag:
#             hand_check = False
#             # counter = 0
#             total = 0
#             frame,names = facial_recog(frame,name_encodings_dict)
#             cv2.imshow(WINDOW, frame) 
#             cv2.waitKey(5000)
#             face_recog_flag = False
#         else:
#             cv2.imshow(WINDOW,frame)

#         hasFrame, frame = vid_cap.read()
#         frame = cv2.flip(frame, 1)
#         key = cv2.waitKey(1)
#         if key == 27:
#             break

#     vid_cap.release()
#     cv2.destroyAllWindows()


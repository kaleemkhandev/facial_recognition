import pickle
import cv2

from utils import face_rects
from utils import face_encodings
from utils import nb_of_matches



    

def facial_recog(frame,name_encodings_dict):
    encodings = face_encodings(frame)
    # this list will contain the names of each face detected in the frame
    names = []

    # loop over the encodings
    for encoding in encodings:
        # initialize a dictionary to store the name of the 
        # person and the number of times it was matched
        counts = {}
        # loop over the known encodings
        for (name, encodings) in name_encodings_dict.items():
            # compute the number of matches between the current encoding and the encodings 
            # of the known faces and store the number of matches in the dictionary
            counts[name] = nb_of_matches(encodings, encoding)
        # check if all the number of matches are equal to 0
        # if there is no match for any name, then we set the name to "Unknown"
        if all(count == 0 for count in counts.values()):
            name = "Unknown"
        # otherwise, we get the name with the highest number of matches
        else:
            name = max(counts, key=counts.get)

        # add the name to the list of names
        names.append(name)
        
    # loop over the `rectangles` of the faces in the 
    # input frame using the `face_rects` function
    for rect, name in zip(face_rects(frame), names):
        # get the bounding box for each face using the `rect` variable
        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
        # draw the bounding box of the face along with the name of the person
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    return frame,names

def closed_hand_check(frame,Hand_Lm_Dectector,hand_check):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points, bbox = Hand_Lm_Dectector(image)
    if points is not None:
        check_one= abs(points[8][0] - points[5][0])<15
        check_four=abs(points[20][0] - points[17][0])<15
        check_thumb=abs(points[4][1] - points[5][1])<15
        if (check_one)  and (check_four) and (check_thumb):
            hand_check = True
    
    return frame,hand_check

def thumbs_up_check(frame,Hand_Lm_Dectector,hand_check):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points, bbox = Hand_Lm_Dectector(image)
    if points is not None:
        check_thumbs_up= points[4][1] < (points[8][1] and points[12][1]) and points[4][1]<(points[8][1] - 90)
        check_thumbs_down= points[4][1] > (points[8][1] and points[12][1]) and points[4][1]>(points[8][1] -70)
        finger_thumbs = (abs(points[8][0]-points[5][0]) < 50) and abs(points[12][0]-points[9][0]) < 50 and (abs(points[16][0]-points[13][0]) < 50)

        if check_thumbs_up and finger_thumbs:
            hand_check = True
    
    return frame,hand_check
def thumbs_down_check(frame,Hand_Lm_Dectector,hand_check):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points, bbox = Hand_Lm_Dectector(image)
    if points is not None:
        check_thumbs_up= points[4][1] < (points[8][1] and points[12][1]) and points[4][1]<(points[8][1] - 90)
        check_thumbs_down= points[4][1] > (points[8][1] and points[12][1])  and  abs(points[4][1]- points[17][1])>150
        finger_thumbs = (abs(points[8][0]-points[5][0]) < 50) and abs(points[12][0]-points[9][0]) < 50 and (abs(points[16][0]-points[13][0]) < 50)

        if check_thumbs_down and finger_thumbs:
            hand_check = True
    
    return frame,hand_check
if __name__=="__main__":
    print('hello')
    # load the encodings + names dictionary
    with open("encodings.pickle", "rb") as f:
        name_encodings_dict = pickle.load(f)
    video_cap = cv2.VideoCapture(0)

    while True:
        _, frame = video_cap.read()
        frame = facial_recog(frame,name_encodings_dict)
        # get the 128-d face embeddings for each face in the input frame
    # show the output frame
        cv2.imshow("frame", frame)
        # wait for 1 milliseconde and if the q key is pressed, we break the loop
        if cv2.waitKey(1) == ord('q'):
            break
        
        # release the video capture and close all windows
    video_cap.release()
    cv2.destroyAllWindows()
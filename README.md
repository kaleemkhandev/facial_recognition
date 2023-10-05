## Hand tracking

### 1. File Description
- `palm_detection_without_custom_op.tflite` (Palm detection) Model file: Downloaded from the [*mediapipe-models*] repository.
- `hand_landmark.tflite` (Landmark detection) Model file: Downloaded from the [*mediapipe*] repository.    
- `anchors.csv`,`hand_tracker.py` files：Downloaded from the [*hand_tracking*] repository.

### 2. Setup
```sh
pip install -r requirements.txt
```

### 3. Implementation
- ```python main.py``` for running end to end pipeline. 
- ```python testing.py --vid_path PATH_TO_TEST_VIDEO``` To test the fast api function without connecting with the server.
- ```uvicorn run:app --reload``` to test the fast api on local machine.
- ```python face_recognition_videos.py``` for inference of the facial recog and hand landmarks detections.

### 4. Results
#### main.py:
It accesses the camera of local machine and do facial recognition after detecting the right hand landmarks. Do blink atleast three times to pass the liveness test.
#### testing.py:
It returns a json object which contains the name of the detected person in the video clip.
#### face_recognition_videos.py:
The script access the camera on local machine and do facial recognition once the right hand is visible in the camera.  

### 5. Acknowledgements
- Thanks to @metalwhale for the python implementation of the mediapipe models.
- Thanks to mediapipe for opensourcing these models.

[*mediapipe-models*]: https://github.com/junhwanjang/mediapipe-models/tree/master/palm_detection/mediapipe_models
[*mediapipe*]: https://github.com/google/mediapipe/tree/master/mediapipe/models
[*hand_tracking*]: https://github.com/wolterlw/hand_tracking

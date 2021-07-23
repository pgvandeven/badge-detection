import os
import cv2
import time
from models.faceDetection.vision.ssd.config.fd_config import define_img_size
import torch
import torchvision
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from sort.sort import *
from person import Person
import multiprocessing as mp
from utils import *

PATH_TO_1PERSON_TEST_VIDEO = 'data/1-person-test-video.mp4'
PATH_TO_2PERSON_TEST_VIDEO = 'data/2-people-test-video.mp4'
PATH_TO_MULTI_PERSON_TEST_VIDEO = 'data/multiple_person_test.mp4'

# Increasing any of these values results in a better accuracy, however slower speeds

BUFFER = 4                      # max image buffer capacity
OBJECT_LIFETIME = 6             # How long should the tracker still try to find a lost tracked person (measured in frames)
MAX_BADGE_CHECK_COUNT = 3       # How many times a full BUFFER should be checked before a person is declared to be an imposter

'''
------------------------------------------
Loading the first model (face detection)
------------------------------------------
'''
define_img_size(1280)
from models.faceDetection.vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

label_path = "models/faceDetection/voc-model-labels.txt"
model_path = "models/faceDetection/version-RFB-320.pth"
net_type = "RFB"
test_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
candidate_size = 1000
threshold = 0.75
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
face_predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)
net.load(model_path)
print('Person detection model loaded')

'''
------------------------------------------
Loading the second model (badge detection)
------------------------------------------
'''
# v1 is a resnet50      - inference time: 3.5s  - accuracy: very high
# v2 is a mobilenet_v3  - inference time: 0.2s  - accuracy: moderate
MODEL_ARCH = 'v2'

if MODEL_ARCH == 'v2':
    badge_detection_model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
elif MODEL_ARCH == 'v1':
    badge_detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

in_features = badge_detection_model.roi_heads.box_predictor.cls_score.in_features
badge_detection_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
badge_detection_model.load_state_dict(torch.load(os.path.join("models", "badgeDetection", MODEL_ARCH), map_location=torch.device(test_device)))
badge_detection_model.eval()
print('Badge detection model loaded')

# Create an instance of SORT - this is the tracker
mot_tracker = Sort(max_age=OBJECT_LIFETIME) 

# Create a multiprocessing pool
mp_pool = mp.Pool(mp.cpu_count())

cap = cv2.VideoCapture(os.path.join(PATH_TO_MULTI_PERSON_TEST_VIDEO)) 
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.mp4', fourcc, 24.0, (1920,1080))

tracked_person_list = []
frame_id = 0


while True:

    frame_id+=1
    print(frame_id)
    ret, orig_image = cap.read()

    if ret is False:
        print("Stream  ended. Exiting")
        break
    
    # Skip frames, check 1/3 frames
    if frame_id % 2 == 0 or frame_id % 3 == 0:
        continue
        

    # Basic image prep
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image_dimensions = orig_image.shape     # (h,w,c)

    # Person Detection
    faces, _, face_scores = face_predictor.predict(image, candidate_size / 2, threshold)

    # If any persons were detected, track them, and add cutout images to their BUFFER

    if len(faces) != 0:
        # Formating the arrays for (deep)SORT into a numpy array that contains lists of (x1,y1,x2,y2,score)
        face_data = []
        for i in range(len(faces)):
            # Changing the coordinates to bound the body instead of the face but also ensuring it doesn't go outside of the image bounds
            # Head to body ratio is ~ 1:4 - 1:8. That can be used to mark the required body size knowing the head measurements
            ratioW = faces[i][2]-faces[i][0]
            ratioH = (faces[i][3]-faces[i][1])*5
            faces[i][0] = int(faces[i][0])-ratioW
            faces[i][1] = int(faces[i][1])
            faces[i][2] = int(faces[i][2])+ratioW
            faces[i][3] = faces[i][1] + ratioH
            faces[i] = normaliseBBox(faces[i], image_dimensions)
            temp = np.append(faces[i], face_scores[i])
            face_data.append(temp)
        face_data = np.array(face_data)
    
        # Calling the person tracker
        track_bbs_ids = mot_tracker.update(face_data) #returns numpy array with bbox and id
        
        if len(track_bbs_ids) != 0:
            # display persons & save cutout's of tracked persons into BUFFER
            tracked_person_list, orig_image = [mp_pool.apply(getTrackedPerson, args=(tracked_person, track_bbs_ids, tracked_person_list, face_scores, orig_image, image, BUFFER, OBJECT_LIFETIME)) for tracked_person in range(len(track_bbs_ids))]
        
            mp_pool.close()
            mp_pool.join()

    # Check the buffer size, if available, check the badges and evaluate whether the person has a badge
    tracked_person_list, orig_image = [mp_pool.apply(update, args=(person, tracked_person_list, badge_detection_model, orig_image, BUFFER, MAX_BADGE_CHECK_COUNT)) for person in tracked_person_list]

    mp_pool.close()
    mp_pool.join()
    cv2.imshow('Badge Detection', orig_image)
    print("Currently storing data for {} tracked persons".format(Person.count))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


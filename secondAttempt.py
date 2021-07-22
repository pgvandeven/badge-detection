import os
import cv2
import time
from models.faceDetection.vision.ssd.config.fd_config import define_img_size
import torch
import torchvision
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
from sort.sort import *
from person import Person

PATH_TO_1PERSON_TEST_VIDEO = 'data/1-person-test-video.mp4'
PATH_TO_2PERSON_TEST_VIDEO = 'data/2-people-test-video.mp4'
define_img_size(1280)

'''
------------------------------------------
Loading the first model (face detection)
------------------------------------------
'''
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

#loads an image and returns a tensor
#(automatically scales to required input size, therefore any image can be passed forward to the model)
loader = transforms.Compose([transforms.Scale(300), transforms.ToTensor()])
def image_loader(image):
    #image = Image.open(image_name)
    if type(image) != 'PIL':
        image = Image.fromarray(image)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    return image

'''
------------------------------------------
Loading SORT
------------------------------------------
'''
#create instance of SORT
mot_tracker = Sort() 

print('Tracker initialized')
# DeepSORT is for future implementation, as it is powerful enough to distinguish between two or more similarly dressed people
#deepsort = deepsort_rbc('/Users/nkybartas/Desktop/Ultra-Light-Fast-Generic-Face-Detector-1MB-master/deep_sort/ckpts/mars-small128.ckpt-68577')

#Evaluate Badge
def badgeDetected(badge_list):
    badge = False
    if len(badge_list) > 0:
        confidence = sum(badge_list)/len(badge_list)
        if confidence >= 0.20:
            # implement badge classification here
            badge = True
            print("Person {} is wearing a badge. Confidence: {}".format(index, np.round(confidence, decimals=2)))
        elif confidence >=0.60:
            print("Can't distinguish whether person {} is wearing a badge. Checking again".format(index))
        else:
            # How to jump to the next else statement (the one below) ?
            print("Person {} is not wearing a SBP badge".format(index))
    else:
        print("Person {} is not wearing a SBP badge".format(index))

    return badge

# Initialize a couple variables once before the loop
buffer = 10
badge_list = 0
frame_id = 0
tracked_person_list = []

cap = cv2.VideoCapture(os.path.join(PATH_TO_2PERSON_TEST_VIDEO)) 

while True:

    frame_id+=1

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

    # If any persons were detected, track them, and add cutout images to their buffer

    if len(faces) != 0:
        # Formating the arrays for (deep)SORT into a numpy array that contains lists of (x1,y1,x2,y2,score)
        face_data = []
        for i in range(len(faces)):
            # Changing the coordinates to bound the body instead of the face
            faces[i][0] = int(faces[i][0])-50
            faces[i][1] = int(faces[i][1])
            faces[i][2] = int(faces[i][2])+50
            faces[i][3] = int(faces[i][3])+500
            temp = np.append(faces[i], face_scores[i])
            face_data.append(temp)
        face_data = np.array(face_data)
    
        # Calling the person tracker
        track_bbs_ids = mot_tracker.update(face_data) #returns numpy array with bbox and id
        
        if len(track_bbs_ids) != 0:
            # display persons & save cutout's of tracked persons into buffer
            for tracked_person in range(len(track_bbs_ids)):

                # Check whether the person already exists (i.e. has been detected before), and either return the old one, or create a new instance of Person
                person_id = int(track_bbs_ids[tracked_person][4])
                if len(tracked_person_list) != 0: 
                    for index in range(len(tracked_person_list)):
                        tracked_person_id = tracked_person_list[index].getID()
                        if tracked_person_id == person_id:
                            #print("match found. ID: {}".format(person_id))
                            for index in range(len(tracked_person_list)):
                                if tracked_person_list[index].getID() == person_id:
                                    person = tracked_person_list[index]
                                    break
                            break
                        elif index == len(tracked_person_list)-1:
                            #print("Creating new instance with ID: {}".format(person_id))
                            person = Person(person_id, buffer)
                            tracked_person_list.append(person)
                            break
                else:
                    #print("Creating new instance with ID: {}".format(person_id))
                    person = Person(person_id, buffer)
                    tracked_person_list.append(person)
                #time_taken = time.time() - start_time
                #print('Time taken to find match: {}'.format(time_taken))
                
                
                bbox = track_bbs_ids[tracked_person][:4]
                person_score = np.round(face_scores[tracked_person], decimals=4)
                xP = int(bbox[0])#-50
                yP = int(bbox[1])
                x1P = int(bbox[2])#+50
                y1P = int(bbox[3])#+300
                frame = image[yP:y1P, xP:x1P]
                frame = image_loader(frame)

                # If a person was ever detected with a badge, draw a green box, else - red and save image to buffer
                if person.hasBadge():
                    color = (0, 255, 0)
                else:
                    person.addImageToBuffer([frame])
                    color = (0, 0, 255)

                cv2.rectangle(orig_image, (xP, yP), (x1P, y1P), color, 2)
                cv2.putText(orig_image, ('person {} - {}'.format(person_id, person_score)), (xP, yP), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


    # Check the buffer size, if available, check the badges
    for person in tracked_person_list:

        if person.hasBadge() == 1:
            #print('We already know that person {} has a badge'.format(person.getID()))
            continue

        if person.getBufferOppacity() == buffer:
            print('Person {} has reached the max buffer size. Checking for a badge'.format(person.getID()))
            
            start_time = time.time()
            
            image_batch = person.getBuffer()
            
            badge_list = []

            for image_id in range(buffer):
                # Badge Detection
                with torch.no_grad():
                    badge_prediction = badge_detection_model(image_batch[image_id])

                cutout_image = person.getImage(image_id)
                for element in range(len(badge_prediction[0]["boxes"])):
                    badge_score = np.round(badge_prediction[0]["scores"][element].cpu().numpy(), decimals= 2)
                    if badge_score > 0.3:  
                        badges = badge_prediction[0]["boxes"][element].cpu().numpy()
                        badges = badges/(orig_image.shape[0]/cutout_image.shape[0])
                        xB = int(badges[0])# + xP - 10
                        yB = int(badges[1])# + yP - 10
                        x1B = int(badges[2])# + xP + 10
                        y1B = int(badges[3]) #+ yP + 10
                        badge_list.append(badge_score)
                        cv2.rectangle(cutout_image, (xB, yB), (x1B, y1B), (0,0,255), 2)
                        cv2.putText(cutout_image, ('badge: ' + str(badge_score)), (xB, yB), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                #Demo-ing the buffer
                windowName = 'person {}'.format(person.getID())
                cv2.imshow(windowName, cutout_image)
                cv2.waitKey(1)

            # Saving to memory that for this person the badge was found if it was indeed found (and wasn't found before)
            if not person.hasBadge():
                if badgeDetected(badge_list):
                    person.setBadge()
                    person.clearBuffer() #This person won't be checked again - free-ing up memory

            time_taken = time.time() - start_time
            print('Badge Detection Inference time: {}s for a batch of {} images'.format(np.round(time_taken, decimals=3), buffer))
            cv2.destroyWindow(windowName)

        # TODO: implement a self-destruction of Person objects (if they're not being used)

        
    cv2.imshow('Badge Detection Test', orig_image)
    #print("Currently storing data for {} tracked persons".format(len(tracked_person_list)))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
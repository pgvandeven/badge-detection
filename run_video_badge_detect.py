import argparse
import sys
import os
import cv2
from datetime import datetime
from time import sleep
from models.faceDetection.vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
import torch
import torchvision
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from torch.autograd import Variable

'''
------------------------------------------
Loading the first model (face detection)
------------------------------------------
'''

label_path = "models/voc-model-labels.txt"
model_path = "models/pretrained/version-RFB-320.pth" # model_path = "models/pretrained/version-RFB-640.pth"
net_type = "RFB"
test_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
candidate_size = 1000
threshold = 0.95
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)
net.load(model_path)

'''
------------------------------------------
Loading the second model (badge detection)
------------------------------------------
'''

badge_detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = badge_detection_model.roi_heads.box_predictor.cls_score.in_features
badge_detection_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
badge_detection_model.load_state_dict(torch.load("badgeDetectionModel/v1"))
badge_detection_model.eval()

#loads an image and returns a tensor
#(automatically scales to required input size, therefore any image can be passed forward to the model)
loader = transforms.Compose([transforms.Scale(300), transforms.ToTensor()])
def image_loader(image_name):
    #image = Image.open(image_name)
    image = loader(image_name).float()
    image = Variable(image, requires_grad=True)
    return image


cap = cv2.VideoCapture("rtsp://kplr:5hVUlm3S7o92@10.10.12.241/Streaming/channels/101") # capture from video
#cap = cv2.VideoCapture('/Users/nkybartas/Desktop/trainingPrep/streetview.mp4')         <---- use for testing

counter = 0
while True:

    ret, orig_image = cap.read()
    if orig_image is None:
        print("end")
        break

    if counter % 5 == 0:

        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

        boxes, labels, probs = predictor.predict(image, candidate_size / 2, threshold)
        #print('Time: {:.6f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
        
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            #label = f" {probs[i]:.2f}"

            x=int(box[0])-75 
            y=int(box[3])
            x1=x+124
            y1=y+300

            cropped = orig_image[y:y1, x:x1]

            if cropped.shape:
                personImg = image_loader(cropped)
                with torch.no_grad():
                    badge_prediction = badge_detection_model([personImg])

                for element in range(len(badge_prediction[0]["boxes"])):
                    boxes = badge_prediction[0]["boxes"][element].cpu().numpy()
                    score = np.round(badge_prediction[0]["scores"][element].cpu().numpy(), decimals= 4)
                    if score > 0.8:
                        cv2.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])], outline ="red", width =3)
                    cv2.text((boxes[0], boxes[1]), text = str(score))

            cv2.rectangle(orig_image, (x,y), (x1,y1), (0, 255, 0), 4)
            cv2.text((x,y), text = str(probs[i]))
    counter +=1
    
    cv2.imshow('Person Detector', orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

import os
import cv2
from datetime import datetime
from time import sleep
from models.faceDetection.vision.ssd.config.fd_config import define_img_size
import torch
import torchvision
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from torch.autograd import Variable
import pandas as pd
from PIL import Image
from matplotlib import cm


PATH_TO_CSV = '/Users/nkybartas/Desktop/Ultra-Light-Fast-Generic-Face-Detector-1MB-master/test_dataset.csv'
PATH_TO_DATASET = '/Users/nkybartas/Desktop/Ultra-Light-Fast-Generic-Face-Detector-1MB-master/data/test_dataset'
PATH_TO_TEST_IMAGES = '/Users/nkybartas/Desktop/Ultra-Light-Fast-Generic-Face-Detector-1MB-master/data/test_dataset'
define_img_size(1280)

'''
------------------------------------------
Loading the first model (face detection)
------------------------------------------
'''
from models.faceDetection.vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

label_path = "models/voc-model-labels.txt"
model_path = "models/pretrained/version-RFB-320.pth" # model_path = "models/pretrained/version-RFB-640.pth"
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

badge_detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = badge_detection_model.roi_heads.box_predictor.cls_score.in_features
badge_detection_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
badge_detection_model.load_state_dict(torch.load("badgeDetectionModel/v1", map_location=torch.device(test_device)))

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

#test_dataframe = pd.read_csv(PATH_TO_CSV)
listdir = os.listdir(PATH_TO_TEST_IMAGES)
num_faces_found, num_faces = 0, 0
num_badges_found, num_badges = 0, 0

#for index, row in test_dataframe.iterrows():
for image_name in listdir:

    orig_image = cv2.imread(os.path.join(PATH_TO_DATASET, image_name))
    if orig_image is not None:
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image_dimensions = orig_image.shape     # (h,w,c)
        faces, _, face_score = face_predictor.predict(image, candidate_size / 2, threshold)
        face_score = np.round(face_score[0].numpy(), decimals = 2)
        num_faces_found += len(faces)
        num_faces += 1
        num_badges = num_faces
        print('Accuracy of person detector: {}/{}'.format(num_faces_found, num_faces))
        for i in range(faces.size(0)):
            box = faces[i, :]
            xP = int(box[0])-50
            yP = int(box[1])
            x1P = int(box[2])+50
            y1P = int(box[3])+300

            cropped_person_image = orig_image[yP:y1P, xP:x1P]
            if cropped_person_image.shape is not None:
                personImg = image_loader(cropped_person_image)
                badge_detection_model.eval()
                with torch.no_grad():
                    badge_prediction = badge_detection_model([personImg])
                    print(badge_prediction)
                for element in range(len(badge_prediction[0]["boxes"])):
                    badge_score = np.round(badge_prediction[0]["scores"][element].cpu().numpy(), decimals= 2)
                    if badge_score > 0.45:
                        num_badges_found += 1
                        badges = badge_prediction[0]["boxes"][element].cpu().numpy()
                        badges = badges/(orig_image.shape[0]/cropped_person_image.shape[0])
                        xB = int(badges[0]) + xP - 10
                        yB = int(badges[1]) + yP - 10
                        x1B = int(badges[2]) + xP + 10
                        y1B = int(badges[3]) + yP + 10
                        cv2.rectangle(orig_image, (xB, yB), (x1B, y1B), (0, 0, 255), 2)
                        cv2.putText(orig_image, ('badge: ' + str(badge_score)), (xB, yB), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.rectangle(orig_image, (xP, yP), (x1P, y1P), (0, 0, 255), 2)
            cv2.putText(orig_image, ('person: ' + str(face_score)), (xP, yP), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Badge Detection Test', orig_image)
        print('Accuracy of badge detector: {}/{}'.format(num_badges_found, num_badges))
        cv2.waitKey()
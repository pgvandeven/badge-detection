"""
This code uses the pytorch model to detect faces from live video or camera.
"""
import argparse
import sys
import os
import cv2
from datetime import datetime
from time import sleep

from models.faceDetection.vision.ssd.config.fd_config import define_img_size



parser = argparse.ArgumentParser(
    description='detect_video')

parser.add_argument('--net_type', default="RFB", type=str,
                    help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
parser.add_argument('--input_size', default=480, type=int,
                    help='define network input size,default optional value 128/160/320/480/640/1280')
parser.add_argument('--threshold', default=0.95, type=float,
                    help='score threshold')
parser.add_argument('--candidate_size', default=1000, type=int,
                    help='nms candidate size')
parser.add_argument('--path', default="imgs", type=str,
                    help='imgs dir')
parser.add_argument('--test_device', default="cpu", type=str,
                    help='cuda:0 or cpu')
parser.add_argument('--video_path', default="/home/linzai/Videos/video/16_1.MP4", type=str,
                    help='path of video')
args = parser.parse_args()

input_img_size = args.input_size
define_img_size(input_img_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.utils.misc import Timer

label_path = "models/voc-model-labels.txt"

net_type = args.net_type

cap = cv2.VideoCapture("rtsp://kplr:5hVUlm3S7o92@10.10.12.241/Streaming/channels/101") # capture from video
#cap = cv2.VideoCapture('/Users/nkybartas/Desktop/trainingPrep/streetview.mp4')         <---- use for testing

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
test_device = args.test_device

candidate_size = args.candidate_size
threshold = args.threshold

if net_type == 'slim':
    model_path = "models/pretrained/version-slim-320.pth"
    # model_path = "models/pretrained/version-slim-640.pth"
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=test_device)
elif net_type == 'RFB':
    model_path = "models/pretrained/version-RFB-320.pth"
    # model_path = "models/pretrained/version-RFB-640.pth"
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)
else:
    print("The net type is wrong!")
    sys.exit(1)
net.load(model_path)

#timer = Timer()
imageCount = 1
counter = 0
while True:

    ret, orig_image = cap.read()
    if orig_image is None:
        print("end")
        break

    if counter % 5 == 0:

        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        #timer.start()
        boxes, labels, probs = predictor.predict(image, candidate_size / 2, threshold)
        #interval = timer.end()
        #print('Time: {:.6f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
        
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            #label = f" {probs[i]:.2f}"
            
            #x=int(box[0])-50 
            #y=int(box[1])
            #x1=int(box[2])+50
            #y1=int(box[3])+200

            x=int(box[0])-75 
            y=int(box[3])
            x1=x+124
            y1=y+300

            


            cropped = orig_image[y:y1, x:x1]
            if cropped.shape:
                now = datetime.now()
                current_time = now.strftime(r"%d-%m-%Y_%H-%M-%S")
                print(cropped.shape)
                print((x1-x)*(y1-y))
                if cv2.imwrite(os.path.join(r'/Users/nkybartas/Desktop/person_cuts', str(current_time) + '.jpg'), cropped):
                    #cv2.imwrite(os.path.join(r'/Users/nkybartas/Desktop/person_cuts', str(current_time) + 'full_.jpg'), orig_image)
                    print('saved image nr:' + str(imageCount))
                    imageCount += 1
                else:
                    raise Exception("Could not write image")
                

            cv2.rectangle(orig_image, (x,y), (x1,y1), (0, 255, 0), 4)
    counter +=1
    
    #orig_image = cv2.resize(orig_image, None, None, fx=0.8, fy=0.8)
    cv2.imshow('Person Detector', orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

from camera import SurveillanceCamera
from models import PersonDetector, BadgeDetector

PATH_TO_1PERSON_TEST_VIDEO = 'data/1-person-test-video.mp4'
PATH_TO_2PERSON_TEST_VIDEO = 'data/2-people-test-video.mp4'
PATH_TO_MULTI_PERSON_TEST_VIDEO = 'data/multiple_person_test.mp4'
PATH_TO_RTSP_HIKVSION_DOME_CAMERA = 'rtsp://kplr:5hVUlm3S7o92@10.10.12.241/Streaming/channels/101'

# Increasing any of these values results in a better accuracy, however slower speeds

BUFFER = 4  # max image buffer capacity
OBJECT_LIFETIME = 10  # How long should the tracker still try to find a lost tracked person (measured in frames)
MAX_BADGE_CHECK_COUNT = 3  # How many times a full BUFFER should be checked before a person is declared to be an imposter

person_detection_model = PersonDetector().model
badge_detection_model = BadgeDetector().model

# camera1 = SurveillanceCamera(1, person_detection_model, badge_detection_model, PATH_TO_1PERSON_TEST_VIDEO, 25, 25, BUFFER, OBJECT_LIFETIME, MAX_BADGE_CHECK_COUNT)
# camera2 = SurveillanceCamera(2, person_detection_model, badge_detection_model, PATH_TO_2PERSON_TEST_VIDEO, 25, 25, BUFFER, OBJECT_LIFETIME, MAX_BADGE_CHECK_COUNT)
# camera3 = SurveillanceCamera(3, person_detection_model, badge_detection_model, PATH_TO_MULTI_PERSON_TEST_VIDEO, 25, 3, BUFFER, OBJECT_LIFETIME, MAX_BADGE_CHECK_COUNT, record='output')
# goproCam = SurveillanceCamera('GoPro', person_detection_model, badge_detection_model, 'udp://127.0.0.1:10000', 30, 10, BUFFER, OBJECT_LIFETIME, MAX_BADGE_CHECK_COUNT, interface = True, record='output')
hikvision = SurveillanceCamera(3, person_detection_model, badge_detection_model, PATH_TO_RTSP_HIKVSION_DOME_CAMERA, 10,
                               1, BUFFER, OBJECT_LIFETIME, MAX_BADGE_CHECK_COUNT, record='output')

camera_list = [hikvision]

while True:
    hikvision.update()

    if 0xFF == ord('q'):
        break

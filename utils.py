import torch, cv2
import numpy as np
from person import Person
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image

#loads an image and returns a tensor
#(automatically scales to required input size, therefore any image can be passed forward to the model)
loader = transforms.Compose([transforms.Resize(300), transforms.ToTensor()])
def image_loader(image):
    #image = Image.open(image_name)
    if type(image) != 'PIL':
        image = Image.fromarray(image)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    return image

# Make sure all bbox coordinates are inside the image
def normaliseBBox(bbox, image_dimensions):
    if bbox[0] < 0:
        bbox[0] = 0
    if bbox[1] < 0:
        bbox[0] = 0
    if bbox[2] > image_dimensions[1]:
        bbox[2] = image_dimensions[1]
    if bbox[3] > image_dimensions[0]:
        bbox[3] = image_dimensions[0]
    return bbox

def findBadges(person, badge_detection_model, orig_image):
    image_batch = person.getBuffer()
    badge_list = []
    for image_id in range(len(image_batch)):
        
        with torch.no_grad():
            badge_prediction = badge_detection_model(image_batch[image_id])

        cutout_image = person.getImage(image_id)
        for element in range(len(badge_prediction[0]["boxes"])):
            badge_score = np.round(badge_prediction[0]["scores"][element].cpu().numpy(), decimals= 2)
            if badge_score > 0.9:  
                badges = badge_prediction[0]["boxes"][element].cpu().numpy()
                badges = badges/(orig_image.shape[0]/cutout_image.shape[0])
                xB = int(badges[0])# + xP - 10
                yB = int(badges[1])# + yP - 10
                x1B = int(badges[2])# + xP + 10
                y1B = int(badges[3]) #+ yP + 10
                badge_list.append(badge_score)
                cv2.rectangle(cutout_image, (xB, yB), (x1B, y1B), (0,0,255), 2)
                cv2.putText(cutout_image, ('badge: ' + str(badge_score)), (xB, yB), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        windowName = 'person {}'.format(person.getID())
        cv2.imshow(windowName, cutout_image)
        cv2.waitKey(1)
    cv2.destroyWindow(windowName)
    return badge_list

def update(person, tracked_person_list, badge_detection_model, orig_image, buffer_size, max_badge_check_count):
    # Self-destruction of Person objects (if they're not being used)
    if not person.isAlive():
        for index in range(len(tracked_person_list)):
            if tracked_person_list[index] == person:
                #print('Person with id: {} is in the list at index {}'.format(person.getID(),index))
                tracked_person_list.pop(index)
                Person.count -= 1

    if person.hasBadge():
        # Exit the loop
        #print('We already know that person {} has a badge'.format(person.getID()))
        pass

    if person.hasBadge() == None and person.getBufferOppacity() == buffer_size:
        image_batch = person.getBuffer()
        badge_list = []

        # Badge Detection
        badge_list = findBadges(person, badge_detection_model, orig_image)
        
        # Badge Evaluation
        if len(badge_list) > 0:
            confidence = sum(badge_list)/len(badge_list)
            if confidence >= 0.90:
                # 
                # implement badge classification here
                #
                value = True
                person.clearBuffer() #This person won't be checked again - free-ing up memory
                #print("Person {} is wearing a SBP badge. Confidence: {}".format(person.getID(), np.round(confidence, decimals=2)))
            elif confidence >=0.60:
                value = None
                #print("Can't distinguish whether person {} is wearing a badge. Checking again".format(person.getID()))
            else:
                value = None
                #print("Person {} is not wearing a SBP badge".format(person.getID()))
            person.setBadge(value)
        else:
            person.setBadge(None)
            #print("Person {} is not wearing a SBP badge".format(person.getID()))
        
        # if the badge has been checked enough times and not found, report that badge was not found.
        if person.getBadgeCheckCount() == max_badge_check_count:
            print("ALERT")
            print("Person {} does not have a badge".format(person.getID()))
            person.setBadge(False)

    return tracked_person_list, badge_detection_model

# Check whether the person already exists (i.e. has been detected before), and either return the old one, or create a new instance of Person
def getTrackedPerson(tracked_person, track_bbs_ids, tracked_person_list, face_scores, orig_image, image, buffer, object_lifetime):
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
                person = Person(person_id, buffer, object_lifetime)
                tracked_person_list.append(person)
                break
    else:
        #print("Creating new instance with ID: {}".format(person_id))
        person = Person(person_id, buffer, object_lifetime)
        tracked_person_list.append(person)
    #time_taken = time.time() - start_time
    #print('Time taken to find match: {}'.format(time_taken))
    
    
    bbox = normaliseBBox(track_bbs_ids[tracked_person][:4], orig_image.shape)
    person_score = np.round(face_scores[tracked_person], decimals=4)
    xP = int(bbox[0])#-50
    yP = int(bbox[1])
    x1P = int(bbox[2])#+50
    y1P = int(bbox[3])#+300

    frame = image[yP:y1P, xP:x1P]
    frame = image_loader(frame)

    # Reseting the age of each tracked Person Object
    person.age = 0

    # If a person was detected with a badge, draw a green box, if was detected to not have a badge - red, if it's unknown - yellow and save image to BUFFER for further checks
    if person.hasBadge() is None:
        person.addImageToBuffer([frame])
        color = (255, 255, 0)
    elif person.hasBadge() is True:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    cv2.rectangle(orig_image, (xP, yP), (x1P, y1P), color, 2)
    cv2.putText(orig_image, ('person {} - {}'.format(person_id, person_score)), (xP, yP), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return tracked_person_list, orig_image
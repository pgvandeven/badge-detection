from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

# loads an image and returns a tensor
# (automatically scales to required input size, therefore any image can be passed forward to the model)
loader = transforms.Compose([transforms.Resize(300), transforms.ToTensor()])


def image_loader(image):
    # image = Image.open(image_name)
    if type(image) != 'PIL':
        image = Image.fromarray(image)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    return image


# Make sure all bbox coordinates are inside the image
def normalise_bbox(bbox, image_dimensions):
    if bbox[0] < 0:
        bbox[0] = 0
    if bbox[1] < 0:
        bbox[0] = 0
    if bbox[2] > image_dimensions[1]:
        bbox[2] = image_dimensions[1]
    if bbox[3] > image_dimensions[0]:
        bbox[3] = image_dimensions[0]
    return bbox

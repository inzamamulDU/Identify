import argparse
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import model_zoo
from PIL import Image
import cv2  # OpenCV library for video processing

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
my_models = sorted(name for name in model_zoo.__dict__
                   if name.islower() and not name.startswith("__")
                   and callable(model_zoo.__dict__[name]))

model_names.extend(my_models)

parser = argparse.ArgumentParser(description='PyTorch Universal Deepfake Training')
parser.add_argument('-data', metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('--ratio', type=float, default=0.5)
parser.add_argument('--lfu', default=False, action="store_true")
parser.add_argument('--use_se', default=False, action="store_true")
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--root_model', type=str, default='checkpoint')
parser.add_argument('--store_name', type=str, default="")
parser.add_argument('--video_path', type=str, required=True, help='path to video file')
parser.add_argument('--fake_threshold', type=float, default=0.75, help='threshold for classifying video as fake')
parser.add_argument('--frame_interval', type=int, default=5, help='interval of frames to process')
args = parser.parse_args()

# Load the model
if args.arch in my_models:
    if args.arch.startswith('ffc_'):
        model = model_zoo.__dict__[args.arch](
            ratio=args.ratio, lfu=args.lfu, use_se=args.use_se)
    else:
        model = model_zoo.__dict__[args.arch]()
    if args.pretrained:
        print("=> WARNING: Missing pretrained model.")
elif args.pretrained:
    model = models.__dict__[args.arch](pretrained=True)
else:
    model = models.__dict__[args.arch]()

# Load the checkpoint
if args.resume:
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
else:
    print("Please provide a path to the trained model checkpoint.")
    exit()

# Define image transformations
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

# Set the model to evaluation mode
model.eval()

# Function to predict a single frame
def predict_frame(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension
    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.softmax(output, dim=1)[0]
    return probabilities

# Process the video
video_path = args.video_path
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_interval = args.frame_interval
fake_frame_count = 0
processed_frame_count = 0

for i in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    # Process every `frame_interval` frames
    if i % frame_interval == 0:
        processed_frame_count += 1
        # Convert frame to PIL image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        # Predict the frame
        probabilities = predict_frame(frame)
        fake_prob = probabilities[1].item()
        print (fake_prob)
        #print(f"Frame {i} is fake with probability {fake_prob:.2f}.")
        if fake_prob > 0.5:
            fake_frame_count += 1

cap.release()

# Determine if the video is fake or real based on the threshold
fake_ratio = fake_frame_count / processed_frame_count
is_fake = fake_ratio >= args.fake_threshold

print(f"Fake frame ratio: {fake_ratio:.2f}")
print(f"Video classified as {'fake' if is_fake else 'real'}")
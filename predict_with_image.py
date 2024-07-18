import argparse
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import model_zoo
from PIL import Image

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

# Load and preprocess the single image
image_path = "data/Nightcafe_88.jpg"  # Update with your image path
image = Image.open(image_path)

# Convert image to RGB if it has more than 3 channels
if image.mode != 'RGB':
    image = image.convert('RGB')

input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

# Set the model to evaluation mode
model.eval()

# Run the model forward pass
with torch.no_grad():
    output = model(input_batch)

# Get the probabilities
probabilities = torch.softmax(output, dim=1)[0]

# Print the predicted probabilities
print("Predicted probabilities:")
for i, prob in enumerate(probabilities):
    print(f"Class {i}: {prob.item()}")
    if i == 1:
        break

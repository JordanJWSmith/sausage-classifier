import torch
import cv2
import torchvision.transforms as transforms
import argparse
import os
from model import CNNModel, ViTModel

model_files = [file for file in os.listdir('outputs/') if file.endswith('.pth')]

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='input/valid/sausage/sausage_16.jpg', help='path to the input image')
parser.add_argument('-d', '--display', default=True, help='display each image')
parser.add_argument('-m', '--model', default=model_files[0], type=str, help='choose the model file')
args = vars(parser.parse_args())

device = ('cuda' if torch.cuda.is_available() else 'cpu')
labels = ['sausage', 'not sausage']

model_arg = args['model']
model_type = model_arg.split('_')[0]

if model_type == 'CNNModel':
    model = CNNModel().to(device)
elif model_type == 'ViTModel':
    model = ViTModel().to(device)
else:
    model = CNNModel().to(device)

checkpoint = torch.load(f'outputs/{model_arg}', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

image = cv2.imread(args['input'])
gt_class = args['input'].split('/')[2]
orig_image = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = transform(image)
image = torch.unsqueeze(image, 0)

with torch.no_grad():
    outputs = model(image.to(device))
output_label = torch.topk(outputs, 1)
pred_class = labels[int(output_label.indices)]

cv2.putText(
    orig_image,
    f'GT: {gt_class}',
    (10, 25),
    cv2.FONT_HERSHEY_PLAIN,
    0.6, (0, 255, 0), 1,
    cv2.LINE_AA
)
cv2.putText(
    orig_image,
    f'Pred: {pred_class}',
    (10, 55),
    cv2.FONT_HERSHEY_PLAIN,
    0.6, (0, 0, 255), 1,
    cv2.LINE_AA
)

print(f'Model: {model_arg}')
print(f'Image: {args["input"]} GT: {gt_class}, pred: {pred_class}')


display = args['display']

if display != 'False':
    cv2.imshow('Result', orig_image)
    cv2.waitKey(0)

# cv2.imwrite(f"outputs/{gt_class}{args['input'].split('/')[-1].split('.')[0]}.png", orig_image)

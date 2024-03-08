import torch
import os
import cv2
from PIL import Image
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from albumentations.core.transforms_interface import ImageOnlyTransform,DualTransform
from tqdm.auto import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(17)
import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
def color_palette():
    """Color palette that maps each class to RGB values.

    This one is actually taken from ADE20k.
    """
    return [[255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255],[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            ]

palette = color_palette()

def checking(batch, predicted_segmentation_maps):
    plt.figure(figsize=(30, 20))
    plt.subplot(3, 1, 1)
    image = batch['original_images'][0]
    # image = batch['pixel_values'][0].permute(1,2,0)
    plt.imshow(image)
    segmentation_map = predicted_segmentation_maps[0].cpu().numpy()
    print("PREDICT : ", np.unique(segmentation_map))
    # print(list(id2label_check[x] for x in np.unique(segmentation_map)))
    color_segmentation_map = np.zeros(
        (segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8)  # height, width, 3
   
    for label, color in enumerate(palette):
        color_segmentation_map[segmentation_map == label, :] = color
    
    ground_truth_color_seg = color_segmentation_map[..., ::-1]

    img = image * 0.5 + ground_truth_color_seg * 0.5
    img = img.astype(np.uint8)

    plt.subplot(3, 1, 2)
    plt.imshow(img)

    segmentation_map = batch["original_segmentation_maps"][0]

    print("Ground Truth : ", np.unique(segmentation_map))
    # print(list(id2label_check[x] for x in np.unique(segmentation_map)))
    color_segmentation_map = np.zeros(
        (segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8)  # height, width, 3
    for label, color in enumerate(palette):
        color_segmentation_map[segmentation_map == label, :] = color

    ground_truth_color_seg = color_segmentation_map[..., ::-1]
    img = image * 0.5 + ground_truth_color_seg * 0.5
    img = img.astype(np.uint8)
    plt.subplot(3, 1, 3)
    plt.imshow(img)
    plt.show()

id2label = {
    0:'background',
    1:'Load',
    2:'Sidewalk',
    3:'Construction',
    4:'Fence',
    5:'Pole',
    6:'Traffic Light',
    7:'Traffic Sign',
    8:'Nature',
    9:'Sky',
    10:'Person',
    11:'Rider',
    12:'Car',
    13:'Background',
}
id2label_check = {
    0:'background',
    1:'Load',
    2:'Sidewalk',
    3:'Construction',
    4:'Fence',
    5:'Pole',
    6:'Traffic Light',
    7:'Traffic Sign',
    8:'Nature',
    9:'Sky',
    10:'Person',
    11:'Rider',
    12:'Car',
    13:'background_car'
}
label2id = {id2label[x]:x for x in id2label}
# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

class CustomDataset(Dataset):
    def __init__(self, image_path:list,mask_path:list, transform=None, infer=False):
        self.transform = transform
        self.infer = infer
        self.img_path=image_path
        self.mask_path=mask_path
    def __len__(self):
        # return len(self.data)
        return len(self.img_path)
    
    def __getitem__(self, idx):
        # choice = np.random.randint(10)
        img_path = self.img_path[idx]
        image = cv2.imread(f"{img_path}")
        ori_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.infer:
            if self.transform:
                image = self.transform(image=ori_image)['image']
            return image,ori_image
        mask_path = self.mask_path[idx]
        ori_mask = cv2.imread(f"{mask_path}", cv2.IMREAD_GRAYSCALE)
        ori_mask[ori_mask==255]=0
        if self.transform:
            augmented = self.transform(image=ori_image, mask=ori_mask)
            image = augmented['image']
            mask = augmented['mask']
            if mask.size()==0 or image.size()==0:
                print("ERROR READ IMAGE FAILE")
        return image, mask ,ori_image,ori_mask

img_path  = glob("./final_ifish_image/*.png") 
mask_path = glob("./final_ifish_mask/*.png")
img_path2  = glob("./test_image/*.png") 
img_path2.sort()
mask_path2 = glob("./pesudo/*.png")
mask_path2.sort()
print(len(mask_path2))
print(len(img_path2))
img_path += img_path2
mask_path += mask_path2
print(len(mask_path))
print(len(img_path))
correct = 0
ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255
jitter = A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.1, hue=0.05,always_apply=False,p=0.2)

train_transform = A.Compose([
    A.Resize(width=1024,height=512),
    A.HorizontalFlip(p=0.5),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Resize(width=960,height=540),
    A.RandomToneCurve(p=1),
    ToTensorV2()
])
# print(len(img_path),len(mask_path))
from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(2)
valid_img_path = np.array(img_path)
valid_mask_path = np.array(mask_path)
size_path = int(len(valid_img_path)*0.8)
s1 = np.arange(len(valid_img_path))
np.random.shuffle(s1)
train, valid = s1[:size_path], s1[size_path:]
train_img_path = valid_img_path[train]
train_mask_path = valid_mask_path[train]
#  
valid2_img_path = valid_img_path[valid]
valid2_mask_path = valid_mask_path[valid]
# valid2_mask_path = traivalid2_mask_pathn_test_split(valid_img_path,valid_mask_path,test_size=0.1,random_state=64,shuffle=True)
np.random.seed(12)
# random.seed(26)
s = np.arange(len(valid2_img_path))
np.random.shuffle(s)
valid2_img_path = valid2_img_path[s][:300]
valid2_mask_path = valid2_mask_path[s][:300]
train_dataset = CustomDataset(train_img_path,train_mask_path,transform=train_transform)
valid_dataset = CustomDataset(valid2_img_path,valid2_mask_path,transform=test_transform)


from transformers import Mask2FormerImageProcessor
from transformers import Mask2FormerForUniversalSegmentation
from transformers import AutoImageProcessor

model_checkpoint = "facebook/mask2former-swin-large-cityscapes-semantic"

model = Mask2FormerForUniversalSegmentation.from_pretrained(model_checkpoint,ignore_mismatched_sizes=True,id2label=id2label,label2id=label2id)

# model = Mask2FormerForUniversalSegmentation.from_pretrained(model_checkpoint,ignore_mismatched_sizes=True)
preprocessor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
preprocessor2 = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")

preprocessor.size = {
    "height": 512,
    "width":1024 
}
preprocessor2.size = {
    "height": 540,
    "width": 960
}
preprocessor.ignore_index = 0


from torch.utils.data import DataLoader

def collate_fn(batch):
    inputs = list(zip(*batch))
    images = inputs[0]
    segmentation_maps = inputs[1]
    
    batch = preprocessor.preprocess(
        images,
        segmentation_maps=segmentation_maps,
        ignore_index=0,
        return_tensors="pt",
    )
    batch["original_images"] = inputs[2]
    batch["original_segmentation_maps"] = inputs[3]

    return batch
    
def collate_fn_test(batch):
    inputs = list(zip(*batch))
    images = inputs[0]
    segmentation_maps = inputs[1]
    batch = preprocessor2.preprocess(
        images,
        segmentation_maps=segmentation_maps,
        ignore_index=255,
        return_tensors="pt",
    )
    batch["original_images"] = inputs[2]
    batch["original_segmentation_maps"] = inputs[3]

    return batch

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn_test)
import evaluate
metric = evaluate.load("mean_iou")

train_transform = A.Compose([
    A.Resize(width=1024, height=512),
    jitter,
    A.OneOf([
         A.CropNonEmptyMaskIfExists(ignore_values=[0,1,8,9,12,13],width=512,height=512,p=0.4),
         A.RandomResizedCrop(width=1024,height=512,scale=(0.08,1.0),p=0.3),
    ], p=1),
    A.HorizontalFlip(p=0.3),
    ToTensorV2(),
])
train_dataset = CustomDataset(train_img_path,train_mask_path,transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

preprocessor2.size = {
    "height": 512,
    "width": 1024
}
before_miou = None

model.to(device)

# # 논문에서는 0.0001 ,wd =  0.05 사용
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001,weight_decay=0.05)
# model_start_idx = int(model_name.split('_')[2])+1
running_loss = 0.0
num_samples = 0
accumulation_steps = 10 # 16
total_batch=  len(train_dataloader)
error_count= 0
for epoch in range(30): # 20->50->5
  print("Epoch:", epoch)
  model.train()
  avg_cost = 0
  # model.zero_grad()   
  for idx, batch in enumerate(tqdm(train_dataloader)):
    optimizer.zero_grad()
    # Forward pass

    outputs = model(
        pixel_values=batch["pixel_values"].to(device),
        mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
        class_labels=[labels.to(device) for labels in batch["class_labels"]]
    )
    # Backward propagation
    loss = outputs.loss
    loss = loss/accumulation_steps
    loss.backward()
    batch_size = batch["pixel_values"].size(0)
    running_loss += loss.item()
    num_samples += batch_size
    if (idx+1)%accumulation_steps==0:
      optimizer.step()
      model.zero_grad()
      avg_cost += loss / total_batch
    if idx % 50 == 0:
      print("Loss:", running_loss/num_samples)
  print(f'[Epoch:{epoch}] cost = {avg_cost}')
      # Optimization 
  model.save_pretrained('new')
  model.eval()
  # model.zero_grad() 
  for idx, batch in enumerate(tqdm(test_dataloader)):
    pixel_values = batch["pixel_values"]

    # Forward pass
    with torch.no_grad():
      outputs = model(pixel_values=pixel_values.to(device))

    # get original images
    original_images = batch["original_images"]
    target_sizes = [(image.shape[0], image.shape[1]) for image in original_images]
    # predict segmentation maps
    predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs,
                                                                                  target_sizes=target_sizes)
    # get ground truth segmentation maps
    ground_truth_segmentation_maps = batch["original_segmentation_maps"]
    
    metric.add_batch(references=ground_truth_segmentation_maps, predictions=predicted_segmentation_maps)
    if idx > 50:
      break
  current = metric.compute(num_labels = 20,ignore_index=0,reduce_labels=False)
  if before_miou==None: 
    before_miou = current['mean_iou']
  else:
    if current['mean_iou'] >= before_miou:
      model.save_pretrained(f'best_model_base4_{epoch}_{before_miou}_pesudo')
      before_miou = current['mean_iou']
    else:
      model.save_pretrained('last_epoch4_pesudo')
  print("Mean IoU :", current['mean_iou'])
  print('Mean Acc :',current['mean_accuracy'])
  print('Per Categoric Iou : ',current['per_category_iou'])
  try:
    checking(batch,predicted_segmentation_maps)
  except Exception as e:
    # print(e)
    pass


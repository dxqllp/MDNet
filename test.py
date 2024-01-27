import argparse
import os

import numpy as np
import torch
from PIL import Image

from torch.utils.data import DataLoader
from torchvision.ops import nms
from tqdm import tqdm
from Mydata import MyDataset
from MDNet import MDNet
from draw_objs import draw_objs
# Some constants
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WSDDN model")
    parser.add_argument('--istrain', help='if train WSDDN', default=0, type=int)
    parser.add_argument(
        "--base_net", type=str, default="vgg", help="Base network to use"
    )
    args, unknown = parser.parse_known_args()
    test_ds = MyDataset('./DetData/ClinicDB','test')
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)
    classes = {
        0: "Flat",
        1: "Pedicle",
        2: "Edge"
    }
    # Create the network
    net = MDNet()
    net.load_state_dict(torch.load('params/clinicdb.pth',map_location='cpu'))
    net.to(DEVICE)
    net.eval()
    for (
            image,
            ssw,
            label,
            name
    ) in tqdm(test_dl, f"Test {len(test_ds)}"):
        all_scores = []
        all_dets = []
        GT = classes[torch.argmax(label).item()]
        detection_scores,classification_scores = net(image, ssw, args.istrain)
        detection_scores, classification_scores = detection_scores.detach().numpy(), classification_scores.detach().numpy()
        ssw = ssw.squeeze(0)
        for j in range(classification_scores.shape[1]):
            inds = np.where((classification_scores[:, j] >= 0.8))[0] #过滤概率大于80%
            cls_scores = classification_scores[inds, j]#保留框的类别得分
            det_scores=detection_scores[inds,j]#保留框的区域得分
            cls_boxes = ssw[inds, :]
            # np.newaxis增加一个维度   4-->(4,1)  (x1,y1,x2,y2,class_score,region_score)
            cls_dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis], det_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(torch.tensor(cls_boxes), torch.tensor(cls_scores), 0.5)
            cls_dets = cls_dets[keep, :]
            num = int(cls_dets.size / 6)
            if num == 1:
                all_scores.append(cls_dets[-1])
                cls_dets = cls_dets[np.newaxis, :]
            else:
                for i in range(num):
                    all_scores.append(cls_dets[i, -1])
            all_dets.append(cls_dets)
        # Limit to max_per_image detections *over all classes*
        image_thresh = 1 / len(ssw)
        for j in range(3):
            keep = np.where(all_dets[j][:, -1] >= image_thresh)[0]
            all_dets[j] = all_dets[j][keep, :]
        for index, elements in enumerate(all_dets):
            if len(elements) > 0:
                for element in elements:
                    box = tuple(map(int, element[:-2]))
                    pre_cls = classes[index]#'./DetData/Kavsir/test/images'
                    pic = Image.open(os.path.join('DetData/ClinicDB/test/images', name[0]))
                    plot_img = draw_objs(pic, element, pre_cls, index)
                    plot_img.save(os.path.join('./result/clinicdb', name[0]))



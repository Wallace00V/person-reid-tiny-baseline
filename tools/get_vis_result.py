import os
import sys
from config import cfg
import argparse
import torch
from torch.backends import cudnn
import torchvision.transforms as T
from PIL import Image
sys.path.append('.')
from utils.logger import setup_logger
from model import make_model

import numpy as np
import cv2
from utils.metrics import cosine_similarity


def visualizer(test_img, camid, top_k = 10, img_size=[128,128]):
    figure = np.asarray(query_img.resize((img_size[1],img_size[0])))
    for k in range(top_k):
        name = str(indices[0][k]).zfill(6)
        img = np.asarray(Image.open(img_path[indices[0][k]]).resize((img_size[1],img_size[0])))
        figure = np.hstack((figure, img))
        title=name
    figure = cv2.cvtColor(figure,cv2.COLOR_BGR2RGB)
    if not os.path.exists(cfg.OUTPUT_DIR+ "/results/"):
        print('create a new folder named results in {}'.format(cfg.OUTPUT_DIR))
        os.makedirs(cfg.OUTPUT_DIR+ "/results")
    cv2.imwrite(cfg.OUTPUT_DIR+ "/results/{}-cam{}.png".format(test_img,camid),figure)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="./configs/Market1501.yaml", help="path to config file", type=str
    )
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.freeze()

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    model = make_model(cfg, 255)
    model.load_param(cfg.TEST.TEST_WEIGHT)

    device = 'cuda'
    model = model.to(device)
    transform = T.Compose([
        T.Resize(cfg.DATA.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])



    logger = setup_logger('{}.test'.format(cfg.PROJECT_NAME), cfg.OUTPUT_DIR, if_train=False)
    model.eval()
    for test_img in os.listdir(cfg.TEST.QUERY_DIR):
        logger.info('Finding ID {} ...'.format(test_img))

        gallery_feats = torch.load(cfg.OUTPUT_DIR + '/gfeats.pth')
        img_path = np.load(cfg.OUTPUT_DIR +'/imgpath.npy')
        print(gallery_feats.shape, len(img_path))
        query_img = Image.open(cfg.TEST.QUERY_DIR + test_img)
        input = torch.unsqueeze(transform(query_img), 0)
        input = input.to(device)
        with torch.no_grad():
            query_feat = model(input)

        dist_mat = cosine_similarity(query_feat, gallery_feats)
        indices = np.argsort(dist_mat, axis=1)
        visualizer(test_img, camid='mixed', top_k=10, img_size=cfg.DATA.INPUT_SIZE)
import os
from torch.backends import cudnn

from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="./configs/Market1501.yaml", help="path to config file", type=str
    )
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.freeze()

    logger = setup_logger('{}.test'.format(cfg.PROJECT_NAME), cfg.OUTPUT_DIR, if_train=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes = make_dataloader(cfg)
    model = make_model(cfg, num_classes)
    model.load_param(cfg.TEST.TEST_WEIGHT)

    do_inference(cfg,
                 model,
                 val_loader,
                 num_query)

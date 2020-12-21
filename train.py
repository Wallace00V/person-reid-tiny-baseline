import os
from torch.backends import cudnn

from config import cfg
import argparse

from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer, WarmupMultiStepLR
from loss import make_loss
from processor import do_train

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="./configs/Market1501.yaml", help="path to config file", type=str
    )
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.freeze()

    if not os.path.exists(cfg.OUTPUT_DIR):
        os.mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger('{}'.format(cfg.PROJECT_NAME), cfg.OUTPUT_DIR, if_train=True)
    logger.info("Running with config:\n{}".format(cfg.CFG_NAME))
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    cudnn.benchmark = True
    # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.

    train_loader, val_loader, num_query, num_classes = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                  cfg.SOLVER.WARMUP_FACTOR,
                                  cfg.SOLVER.WARMUP_EPOCHS, cfg.SOLVER.WARMUP_METHOD)

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,  # modify for using self trained model
        loss_func,
        num_query
    )

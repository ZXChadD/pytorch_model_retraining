import argparse
import os
import logging
import sys
import itertools
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import glob
import re
import eval_ssd
import ts_predict
from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.datasets.ts_voc_dataset import VOCDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from shutil import copyfile

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')

parser.add_argument('--datasets', nargs='+', help='Dataset directory path')
parser.add_argument('--validation_dataset', help='Dataset directory path')
parser.add_argument('--balance_data', action='store_true',
                    help="Balance training data by down-sampling more frequent labels.")

parser.add_argument('--net', default="vgg16-ssd",
                    help="The network architecture, it can be mb1-ssd, mb1-lite-ssd, mb2-ssd-lite or vgg16-ssd.")

parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')

# Params for SGD
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--base_net_lr', default=None, type=float,
                    help='initial learning rate for base net.')
parser.add_argument('--extra_layers_lr', default=None, type=float,
                    help='initial learning rate for the layers not in base net and prediction heads.')

# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--base_net',
                    help='Pretrained base model')
parser.add_argument('--pretrained_ssd', help='Pre-trained base model')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')

# Train params
parser.add_argument('--use_cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')

# Config parser
import configparser
config_file = configparser.ConfigParser()
config_file.read("config.ini")
training = config_file["TRAININGINFO"]
sgd = config_file["SGD"]
scheduler = config_file["SCHEDULER"]
train_loop = config_file["TRAINLOOP"]
min_val_loss = train_loop.getint("min_val_loss")
epoch_limit = train_loop.getint("epoch_limit")
count = train_loop.getint("count")

# Teacher Student Retraining Loop
current_FAR = 100
iteration_count = 1
pretrained_model_path = "../models/Epoch-120-Loss-1.422243544929906.pth"

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Use Cuda.")


def train(loader, net, criterion, optimizer, device, debug_steps, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0


def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


if __name__ == '__main__':

    while True:

        all_files = []

        timer = Timer()
        create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=args.mb2_width_mult)
        config = mobilenetv1_ssd_config

        train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
        target_transform = MatchPrior(config.priors, config.center_variance,
                                      config.size_variance, 0.5)

        test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

        ####### Training dataset #######
        logging.info("Prepare training datasets.")
        datasets = []
        for dataset_path in args.datasets:
            dataset = VOCDataset(iteration_count, dataset_path, transform=train_transform,
                                 target_transform=target_transform)
            label_file = os.path.join("../models", "voc-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)

            datasets.append(dataset)
        logging.info(f"Stored labels into file {label_file}.")
        train_dataset = ConcatDataset(datasets)
        logging.info("Train dataset size: {}".format(len(train_dataset)))
        train_loader = DataLoader(train_dataset, training.getint("batch_size"),
                                  num_workers=training.getint("num_workers"),
                                  shuffle=True)

        ####### Validation dataset #######
        logging.info("Prepare Validation datasets.")
        val_dataset = VOCDataset(iteration_count, args.validation_dataset, transform=test_transform,
                                 target_transform=target_transform, is_test=True)
        logging.info(val_dataset)
        logging.info("validation dataset size: {}".format(len(val_dataset)))

        val_loader = DataLoader(val_dataset, training.getint("batch_size"),
                                num_workers=training.getint("num_workers"),
                                shuffle=False)
        logging.info("Build network.")
        net = create_net(num_classes)
        min_loss = -10000.0
        last_epoch = -1

        base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
        extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr

        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]

        logging.info(f"Init from pretrained ssd {pretrained_model_path}")
        net.init_from_pretrained_ssd(args.pretrained_ssd)
        logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

        net.to(DEVICE)

        criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                                 center_variance=0.1, size_variance=0.2, device=DEVICE)
        optimizer = torch.optim.Adam(params, lr=args.lr)

        logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
                     + f"Extra Layers learning rate: {extra_layers_lr}.")

        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, scheduler.getfloat("t_max"), last_epoch=last_epoch)
        # print(sgd.getfloat("t_max"))

        ####### Training Process #######
        logging.info(f"Start training from epoch {last_epoch + 1}.")
        for epoch in range(last_epoch + 1, train_loop.getint("num_epochs")):
            print("")
            train(train_loader, net, criterion, optimizer,
                  device=DEVICE, debug_steps=train_loop.getint("debug_steps"), epoch=epoch)
            scheduler.step()

            if epoch % train_loop.getint("validation_epochs") == 0 or epoch == train_loop.getint("num_epochs") - 1:
                val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion,
                                                                              DEVICE)
                logging.info(
                    f"Epoch: {epoch}, " +
                    f"Validation Loss: {val_loss:.4f}, " +
                    f"Validation Regression Loss {val_regression_loss:.4f}, " +
                    f"Validation Classification Loss: {val_classification_loss:.4f}"
                )
                model_path = os.path.join("../saved_models", f"{args.net}-Epoch-{epoch}-Loss-{val_loss}.pth")
                net.save(model_path)
                logging.info(f"Saved model {model_path}")

                if val_loss < min_val_loss:
                    # Saving the model
                    if min_val_loss > val_loss:
                        min_val_loss = val_loss
                        logging.info('Min loss %0.2f' % min_val_loss)
                        count = 0
                else:
                    if count == epoch_limit:
                        # Check early stopping condition
                        logging.info('Early stopping!')
                        break
                    else:
                        count += 1

        ####### Evaluating Process #######
        files = glob.glob("../saved_models" + '/*.pth')
        for file in files:
            current_file = re.findall('[^-]+(?=.pth)', file)
            all_files.append(float(current_file[0]))

        min_file_name = min(all_files)

        for x in files:
            if str(min_file_name) in x:
                min_file = x

        logging.info(f"Evaluating model {min_file}.")
        new_FAR = eval_ssd.evaluate_ssd(min_file)
        logging.info(f"Copying file {min_file}.")
        copyfile(min_file, "../models")

        if new_FAR < current_FAR:
            current_FAR = new_FAR
            iteration_count += 1
            pretrained_model_path = "../models/" + min_file_name + ".pth"
            ts_predict.predict(pretrained_model_path, iteration_count)
            logging.info(f"Iteration number: {iteration_count}.")

        elif current_FAR < new_FAR:
            logging.info(f"Training ends")
            break

import torch
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.utils import box_utils, measurements
from vision.utils.misc import str2bool, Timer
import argparse
import pathlib
import numpy as np
import logging
import sys
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor

# parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
# parser.add_argument("--use_2007_metric", type=str2bool, default=True)
# parser.add_argument("--nms_method", type=str, default="hard")
# parser.add_argument("--iou_threshold", type=float, default=0.5, help="The threshold of Intersection over Union.")
# parser.add_argument("--eval_dir", default="eval_results", type=str, help="The directory to store evaluation results.")
# parser.add_argument('--mb2_width_mult', default=1.0, type=float,
#                     help='Width Multiplifier for MobilenetV2')
# args = parser.parse_args()


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def group_annotation_by_class(dataset):
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        gt_boxes = torch.from_numpy(gt_boxes)
        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)
            if class_index not in all_difficult_cases:
                all_difficult_cases[class_index] = {}
            if image_id not in all_difficult_cases[class_index]:
                all_difficult_cases[class_index][image_id] = []
            all_difficult_cases[class_index][image_id].append(difficult)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])
    for class_index in all_difficult_cases:
        for image_id in all_difficult_cases[class_index]:
            all_gt_boxes[class_index][image_id] = torch.tensor(all_gt_boxes[class_index][image_id])
    return true_case_stat, all_gt_boxes, all_difficult_cases


def compute_average_precision_per_class(num_true_cases, gt_boxes, difficult_cases,
                                        prediction_file, iou_threshold, use_2007_metric):
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases

    # print(precision)
    # print(recall)
    if use_2007_metric:
        return measurements.compute_voc2007_average_precision(precision, recall)
    else:
        return measurements.compute_average_precision(precision, recall)


def evaluate_ssd(trained_model):
    model_path = "../saved_models/" + trained_model
    eval_path = pathlib.Path("eval_results")
    eval_path.mkdir(exist_ok=True)
    timer = Timer()
    class_names = [name.strip() for name in open("../models/voc-model-labels.txt").readlines()]

    dataset = VOCDataset("data/test", is_test=True)

    true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)

    net = create_mobilenetv2_ssd_lite(len(class_names), width_mult=1.0, is_test=True)

    timer.start("Load Model")
    net.load(model_path)
    net = net.to(DEVICE)
    print(f'It took {timer.end("Load Model")} seconds to load the model.')

    predictor = create_mobilenetv2_ssd_lite_predictor(net, nms_method="hard", device=DEVICE)

    results = []
    for i in range(len(dataset)):
        print("process image", i)
        timer.start("Load Image")
        image = dataset.get_image(i)
        print("Load Image: {:4f} seconds.".format(timer.end("Load Image")))
        timer.start("Predict")
        boxes, labels, probs = predictor.predict(image)
        print("Prediction: {:4f} seconds.".format(timer.end("Predict")))
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        results.append(torch.cat([
            indexes.reshape(-1, 1),
            labels.reshape(-1, 1).float(),
            probs.reshape(-1, 1),
            boxes + 1.0  # matlab's indexes start from 1
        ], dim=1))
    results = torch.cat(results)
    for class_index, class_name in enumerate(class_names):
        if class_index == 0: continue  # ignore background
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        with open(prediction_path, "w") as f:
            sub = results[results[:, 1] == class_index, :]
            for i in range(sub.size(0)):
                prob_box = sub[i, 2:].numpy()
                image_id = dataset.ids[int(sub[i, 0])]
                print(
                    image_id + " " + " ".join([str(v) for v in prob_box]),
                    file=f
                )
    aps = []
    print("\n\nAverage Precision Per-class:")
    for class_index, class_name in enumerate(class_names):
        if class_index == 0:
            continue
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        ap = compute_average_precision_per_class(
            true_case_stat[class_index],
            all_gb_boxes[class_index],
            all_difficult_cases[class_index],
            prediction_path,
            0.5,
            True
        )
        aps.append(ap)
        print(f"{class_name}: {ap}")

    print(f"\nAverage Precision Across All Classes:{sum(aps) / len(aps)}")

    # def GetPDFA(self, boundingboxes, IOUThreshold=0.5):
    #
    #     # Actual predictions
    #     class_actual = pandas.Series([], name='Actual')
    #
    #     # Prediction
    #     class_pred = pandas.Series([], name='Predicted')
    #
    #     # List with all ground truths (Ex: [imageName,class,confidence=1, (bb coordinates XYX2Y2)])
    #     groundTruths = []
    #     # List with all detections (Ex: [imageName,class,confidence,(bb coordinates XYX2Y2)])
    #     detections = []
    #     # Get all images
    #     images = []
    #     pd = 0
    #     low = 0
    #     high = 1
    #     count = 0
    #
    #     # Loop through all bounding boxes and separate them into GTs and detections
    #     for bb in boundingboxes.getBoundingBoxes():
    #         # [imageName, class, confidence, (bb coordinates XYX2Y2)]
    #         if bb.getBBType() == BBType.GroundTruth:
    #             groundTruths.append([
    #                 bb.getImageName(),
    #                 bb.getClassId(), 1,
    #                 bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
    #             ])
    #         else:
    #             detections.append([
    #                 bb.getImageName(),
    #                 bb.getClassId(),
    #                 bb.getConfidence(),
    #                 bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
    #             ])
    #
    #         # get class
    #         if bb.getImageName() not in images:
    #             images.append(bb.getImageName())
    #
    #     images = sorted(images)
    #
    #     while (pd != 0.90):
    #         if count > 20:
    #             break
    #         else:
    #             count+=1
    #         # Get count
    #         total_fp = 0
    #         total_tp = 0
    #
    #         if (pd == 0):
    #             confidence_level = 0.75
    #         elif (pd < 0.90):
    #             high = confidence_level
    #             confidence_level = (high + low) / 2
    #         else:
    #             low = confidence_level
    #             confidence_level = (high + low) / 2
    #
    #         for i in images:
    #             # Get only detection of class c
    #             dects = []
    #             [dects.append(d) for d in detections if d[0] == i and d[2] >= confidence_level]
    #             # Get only ground truths of class c
    #             gts = []
    #             [gts.append(g) for g in groundTruths if g[0] == i]
    #
    #             # sort detections by decreasing confidence
    #             dects = sorted(dects, key=lambda conf: conf[2], reverse=True)
    #             TP = np.zeros(len(dects))
    #             FP = np.zeros(len(dects))
    #
    #             # create dictionary with amount of gts for each image
    #             det = Counter([cc[0] for cc in gts])
    #
    #             for key, val in det.items():
    #                 det[key] = np.zeros(val)
    #
    #             # Loop through detections
    #             for d in range(len(dects)):
    #
    #                 # Find ground truth image
    #                 iouMax = sys.float_info.min
    #
    #                 # Loop through ground truths
    #                 for j in range(len(gts)):
    #                     # print('Ground truth gt => %s' % (gts[j][3],))
    #                     iou = Evaluator.iou(dects[d][3], gts[j][3])
    #                     if iou > iouMax:
    #                         iouMax = iou
    #                         jmax = j
    #
    #                 # Assign detection as true positive/don't care/false positive
    #                 if iouMax >= IOUThreshold:
    #                     if det[dects[d][0]][jmax] == 0:
    #                         TP[d] = 1  # count as true positive
    #                         det[dects[d][0]][jmax] = 1  # flag as already 'seen'
    #                         # print("TP")
    #
    #                     else:
    #                         FP[d] = 1  # count as false positive
    #                         # print("FP")
    #
    #                 # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
    #                 else:
    #                     FP[d] = 1  # count as false positive
    #                     # print("FP")
    #
    #             total_tp += np.count_nonzero(TP == 1)
    #             total_fp += np.count_nonzero(FP == 1)
    #
    #         pd = total_tp / len(groundTruths)
    #         print(confidence_level)
    #         print("PD: %.2f" % pd)
    #         print("FAR: %.2f" % (total_fp / len(images)))
    #         print(total_fp)


# evaluate_ssd("mb2-ssd-lite-Epoch-140-Loss-2.1616919381277904.pth")

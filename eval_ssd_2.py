import torch
from vision.datasets.voc_dataset import VOCDataset
from vision.utils import box_utils, measurements
from vision.utils.misc import str2bool, Timer
import pathlib
import numpy as np
import logging
import sys
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor

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


def get_groundtruths(dataset):
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        print(gt_boxes)


def evaluate_ssd(trained_model):
    PD = 0
    low = 0
    high = 1
    count = 0

    model_path = "../saved_models/" + trained_model
    eval_path = pathlib.Path("eval_results")
    eval_path.mkdir(exist_ok=True)
    timer = Timer()
    class_names = [name.strip() for name in open("../models/voc-model-labels.txt").readlines()]

    dataset = VOCDataset("../data/test", is_test=True)

    # get_groundtruths(dataset)

    # true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)

    net = create_mobilenetv2_ssd_lite(len(class_names), width_mult=1.0, is_test=True)

    timer.start("Load Model")
    net.load(model_path)
    net = net.to(DEVICE)
    print(f'It took {timer.end("Load Model")} seconds to load the model.')

    predictor = create_mobilenetv2_ssd_lite_predictor(net, nms_method="hard", device=DEVICE)

    while int(PD) != 0.90:
        ground_truths = 0
        all_positives = 0
        all_false_positives = 0

        if count > 20:
            break
        else:
            count += 1

        if PD == 0:
            confidence_level = 0
        elif PD < 0.90:
            high = confidence_level
            confidence_level = (high + low) / 2
        else:
            low = confidence_level
            confidence_level = (high + low) / 2

        for i in range(len(dataset)):
            # print("process image", i)
            timer.start("Load Image")
            image = dataset.get_image(i)
            # print("Load Image: {:4f} seconds.".format(timer.end("Load Image")))
            timer.start("Predict")
            boxes, labels, probs = predictor.predict(image)
            boxes_specific, labels_specific, probs_specific = retrieve_specific_box(boxes, labels, probs)

            image_id, annotation = dataset.get_annotation(i)
            gt_boxes, classes, is_difficult = annotation

            ground_truths += len(gt_boxes)

            for det_object in range(len(boxes_specific)):
                flag = 0
                for gt_object in range(len(gt_boxes)):
                    if iou(gt_boxes[gt_object], boxes_specific[det_object]) > 0.5 and classes[gt_object] == \
                            labels_specific[det_object]:
                        all_positives += 1
                        flag = 1
                    elif iou(gt_boxes[gt_object], boxes_specific[det_object]) > 0.5 and classes[gt_object] != \
                            labels_specific[det_object]:
                        all_false_positives += 1
                        flag = 1
                if flag == 0:
                    all_false_positives += 1

            print(all_positives)
            print(all_false_positives)
            print(all_positives / ground_truths)

        PD = all_positives / ground_truths
        FAR = all_false_positives / len(dataset)
        print(confidence_level)
        print("PD: %.2f" % PD)
        print("FAR: %.2f" % FAR)

    #     for x in range(len(probs_specific)):
    #         if probs_specific[x] > confidence_level:
    #             new_probs = np.append(new_probs, probs_specific[x])
    #             new_labels = np.append(new_labels, labels_specific[x])
    #             new_boxes = np.append(new_boxes, boxes_specific[x])
    #
    #     new_probs = torch.from_numpy(new_probs)
    #     new_labels = torch.from_numpy(new_labels)
    #     new_boxes = torch.from_numpy(new_boxes)
    #
    #     new_probs = new_probs.to(dtype=torch.float32)
    #     new_labels = new_labels.to(dtype=torch.float32)
    #     new_boxes = torch.reshape(new_boxes, (-1, 4))
    #
    #     # print("Prediction: {:4f} seconds.".format(timer.end("Predict")))
    #     indexes = torch.ones(new_labels.size(0), 1, dtype=torch.float32) * i
    #     results.append(torch.cat([
    #         indexes.reshape(-1, 1),
    #         new_labels.reshape(-1, 1).float(),
    #         new_probs.reshape(-1, 1),
    #         new_boxes + 1.0  # matlab's indexes start from 1
    #     ], dim=1))
    #
    # results = torch.cat(results)
    # print(results)
    # for class_index, class_name in enumerate(class_names):
    #     if class_index == 0: continue  # ignore background
    #     prediction_path = eval_path / f"det_test_{class_name}.txt"
    #     with open(prediction_path, "w") as f:
    #         sub = results[results[:, 1] == class_index, :]
    #         for i in range(sub.size(0)):
    #             prob_box = sub[i, 2:].numpy()
    #             image_id = dataset.ids[int(sub[i, 0])]
    #             print(
    #                 image_id + " " + " ".join([str(v) for v in prob_box]),
    #                 file=f
    #             )
    # aps = []
    # print("\n\nAverage Precision Per-class:")
    # for class_index, class_name in enumerate(class_names):
    #     if class_index == 0:
    #         continue
    #     prediction_path = eval_path / f"det_test_{class_name}.txt"
    #     ap, true_positive, false_positive = compute_average_precision_per_class(
    #         true_case_stat[class_index],
    #         all_gb_boxes[class_index],
    #         all_difficult_cases[class_index],
    #         prediction_path,
    #         0.5,
    #         True
    #     )
    #     aps.append(ap)
    #     print(f"{class_name}: {ap}")
    #     ground_truths += true_case_stat[class_index]
    #     all_positives += int(true_positive)
    #     all_false_positives += int(false_positive)
    #
    #     PD = all_positives / ground_truths
    #     FAR = all_false_positives / len(dataset)
    #
    #     print(f"\nAverage Precision Across All Classes:{sum(aps) / len(aps)}")



####### Retrieve specific boxes #######
def retrieve_specific_box(boxes, labels, probs):
    all_objects = []
    new_boxes = []
    new_labels = []
    new_probs = []
    probs = probs.numpy()
    labels = labels.numpy()
    boxes = boxes.numpy()

    for x in range(len(probs)):
        add_flag = 1

        if not all_objects:
            all_objects.append([boxes[x], labels[x], probs[x]])
            add_flag = 0
        else:
            for y in all_objects:
                # print(all_boxes)
                if iou(boxes[x], y[0]) > 0.5 and probs[x] < y[2]:
                    add_flag = 0
                elif iou(boxes[x], y[0]) > 0.5 and probs[x] > y[2]:
                    y[0] = boxes[x]
                    y[1] = labels[x]
                    y[2] = probs[x]
                    add_flag = 0

        if add_flag == 1:
            all_objects.append([boxes[x], labels[x], probs[x]])

    for objects in all_objects:
        new_boxes.append(objects[0].tolist())
        new_labels.append(objects[1])
        new_probs.append(objects[2])

    return new_boxes, new_labels, new_probs


def iou(boxA, boxB):
    # if boxes dont intersect
    if boxesIntersect(boxA, boxB) is False:
        return 0
    interArea = getIntersectionArea(boxA, boxB)
    union = getUnionAreas(boxA, boxB, interArea=interArea)
    # intersection over union
    iou = interArea / union
    assert iou >= 0
    return iou


def boxesIntersect(boxA, boxB):
    if boxA[0] > boxB[2]:
        return False  # boxA is right of boxB
    if boxB[0] > boxA[2]:
        return False  # boxA is left of boxB
    if boxA[3] < boxB[1]:
        return False  # boxA is above boxB
    if boxA[1] > boxB[3]:
        return False  # boxA is below boxB
    return True


def getIntersectionArea(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # intersection area
    return (xB - xA + 1) * (yB - yA + 1)


def getUnionAreas(boxA, boxB, interArea=None):
    area_A = getArea(boxA)
    area_B = getArea(boxB)
    if interArea is None:
        interArea = getIntersectionArea(boxA, boxB)
    return float(area_A + area_B - interArea)


def getArea(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


# def compute_average_precision_per_class(num_true_cases, gt_boxes, difficult_cases,
#                                         prediction_file, iou_threshold, use_2007_metric):
#     with open(prediction_file) as f:
#         image_ids = []
#         boxes = []
#         scores = []
#         for line in f:
#             t = line.rstrip().split(" ")
#             image_ids.append(t[0])
#             scores.append(float(t[1]))
#             box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
#             box -= 1.0  # convert to python format where indexes start from 0
#             boxes.append(box)
#         scores = np.array(scores)
#         sorted_indexes = np.argsort(-scores)
#         boxes = [boxes[i] for i in sorted_indexes]
#         image_ids = [image_ids[i] for i in sorted_indexes]
#         true_positive = np.zeros(len(image_ids))
#         false_positive = np.zeros(len(image_ids))
#         matched = set()
#         for i, image_id in enumerate(image_ids):
#             box = boxes[i]
#             if image_id not in gt_boxes:
#                 false_positive[i] = 1
#                 print("x")
#                 continue
#
#             gt_box = gt_boxes[image_id]
#             ious = box_utils.iou_of(box, gt_box)
#             max_iou = torch.max(ious).item()
#             max_arg = torch.argmax(ious).item()
#             if max_iou > iou_threshold:
#                 print(difficult_cases[image_id][max_arg])
#                 if difficult_cases[image_id][max_arg] == 0:
#                     if (image_id, max_arg) not in matched:
#                         true_positive[i] = 1
#                         matched.add((image_id, max_arg))
#                     else:
#                         false_positive[i] = 1
#             else:
#                 false_positive[i] = 1
#
#     true_positive = true_positive.cumsum()
#     false_positive = false_positive.cumsum()
#     precision = true_positive / (true_positive + false_positive)
#     recall = true_positive / num_true_cases
#
#     if use_2007_metric:
#         return measurements.compute_voc2007_average_precision(precision, recall), true_positive[-1], false_positive[-1]
#     else:
#         return measurements.compute_average_precision(precision, recall)

evaluate_ssd("0-Epoch-112-Loss-1.6827886732001054.pth")

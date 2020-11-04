import torch
from vision.datasets.voc_dataset import VOCDataset
from vision.utils.misc import str2bool, Timer
import pathlib
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

    net = create_mobilenetv2_ssd_lite(len(class_names), width_mult=1.0, is_test=True)

    timer.start("Load Model")
    net.load(model_path)
    net = net.to(DEVICE)
    print(f'It took {timer.end("Load Model")} seconds to load the model.')

    predictor = create_mobilenetv2_ssd_lite_predictor(net, nms_method="hard", device=DEVICE)

    while int(PD) != 0.6:
        ground_truths = 0
        all_positives = 0
        all_false_positives = 0

        if count > 20:
            break
        else:
            count += 1

        if PD == 0:
            confidence_level = 0
        elif PD < 0.60:
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

            # print(boxes)

            image_id, annotation = dataset.get_annotation(i)
            gt_boxes, classes, is_difficult = annotation

            ground_truths += len(gt_boxes)

            # print(boxes_specific)
            # print(gt_boxes)
            for det_object in range(len(boxes_specific)):
                flag = 0
                if probs_specific[det_object] > confidence_level:
                    for gt_object in range(len(gt_boxes)):
                        if iou(gt_boxes[gt_object], boxes_specific[det_object]) > 0.3 and classes[gt_object] == \
                                labels_specific[det_object]:
                            all_positives += 1
                            flag = 1
                        elif iou(gt_boxes[gt_object], boxes_specific[det_object]) > 0.3 and classes[gt_object] != \
                                labels_specific[det_object]:
                            all_false_positives += 1
                            flag = 1
                    if flag == 0:
                        all_false_positives += 1
            if i % 50 == 0:
                print(i)

        PD = all_positives / ground_truths
        FAR = all_false_positives / len(dataset)
        print(confidence_level)
        print("PD: %.2f" % PD)
        print("FAR: %.2f" % FAR)


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


# evaluate_ssd("0-Epoch-131-Loss-0.9376572600582189.pth")

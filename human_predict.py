import torch
from vision.datasets.voc_dataset import VOCDataset
import pathlib
from vision.utils.misc import str2bool, Timer
from vision.utils.pascal_voc_writer import Writer
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
import xml.etree.ElementTree as ET
import random
import pandas as pd

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def predict(iteration, human_accuracy):
    # conf_matrix = confusion_matrix()
    model_path = "../models/Epoch-110-Loss-1.7577346393040247.pth"
    save_path = pathlib.Path('../data/train/' + str(iteration))
    save_path.mkdir(exist_ok=True)
    timer = Timer()
    class_names = [name.strip() for name in open("../models/voc-model-labels.txt").readlines()]

    dataset = VOCDataset("../data/train", is_test=False)

    net = create_mobilenetv2_ssd_lite(len(class_names), width_mult=1.0, is_test=True)

    timer.start("Load Model")
    net.load(model_path)
    net = net.to(DEVICE)
    print(f'It took {timer.end("Load Model")} seconds to load the model.')

    predictor = create_mobilenetv2_ssd_lite_predictor(net, nms_method="hard", device=DEVICE)

    label_names = load_label_names()

    for i in range(len(dataset)):
        ####### initialise a writer to create pascal voc file #######
        writer = Writer(
            '../data/train/PNGImages/' + str(i) + '.png', 256, 256)
        image = dataset.get_image(i)
        boxes, labels, probs = predictor.predict(image)

        probs = probs.numpy()
        labels = labels.numpy()
        boxes = boxes.numpy()

        all_labels = []

        for x in range(len(probs)):
            add_flag = 1
            label_number = labels[x]
            name_of_object = label_names[label_number]
            x1 = boxes[x][0]
            y1 = boxes[x][1]
            x2 = boxes[x][2]
            y2 = boxes[x][3]
            if 0.4 <= probs[x] <= 0.5:
                writer.addObject(name_of_object, x1, y1, x2, y2, 1)
            else:
                writer.addObject(name_of_object, x1, y1, x2, y2, 0)

    # for i in range(len(dataset)):
    #     ####### initialise a writer to create pascal voc file #######
    #     writer = Writer(
    #         '../data/train/PNGImages/' + str(i) + '.png', 256, 256)
    #     image = dataset.get_image(i)
    #     boxes, labels, probs = predictor.predict(image)
    #
    #     probs = probs.numpy()
    #     labels = labels.numpy()
    #     boxes = boxes.numpy()
    #
    #     all_labels = []
    #
    #     for x in range(len(probs)):
    #         add_flag = 1
    #         label_number = labels[x]
    #         name_of_object = label_names[label_number]
    #         x1 = boxes[x][0]
    #         y1 = boxes[x][1]
    #         x2 = boxes[x][2]
    #         y2 = boxes[x][3]
    #         current_label = [x1, y1, x2, y2]
    #         detected_box = [name_of_object, probs[x], current_label]
    #
    #         if not all_labels:
    #             all_labels.append(detected_box)
    #             continue
    #
    #         for y in range(len(all_labels)):
    #             if iou(all_labels[y][2], detected_box[2]) > 0.7 and all_labels[y][1] > detected_box[1]:
    #                 add_flag = 0
    #             elif iou(all_labels[y][2], detected_box[2]) > 0.7 and all_labels[y][1] < detected_box[1]:
    #                 add_flag = 0
    #                 all_labels[y][0] = detected_box[0]
    #                 all_labels[y][1] = detected_box[1]
    #                 all_labels[y][2] = detected_box[2]
    #
    #         if add_flag == 1:
    #             all_labels.append(detected_box)
    #
    #     for each_object in all_labels:
    #         probability = random.random()
    #         if probability < human_accuracy:
    #             ####### human annotation #######
    #             objects = ET.parse('../data/train/Annotations/' + str(i) + '.xml').findall("object")
    #             for object in objects:
    #                 name_of_object_gt = object.find('name').text.lower().strip()
    #                 bbox = object.find('bndbox')
    #                 x1 = float(bbox.find('xmin').text)
    #                 y1 = float(bbox.find('ymin').text)
    #                 x2 = float(bbox.find('xmax').text)
    #                 y2 = float(bbox.find('ymax').text)
    #                 actual_box = [x1, y1, x2, y2]
    #
    #                 if iou(actual_box, each_object[2]) > 0.5:
    #                     each_object[0] = name_of_object_gt
    #
    #         writer.addObject(each_object[0], each_object[2][0], each_object[2][1], each_object[2][2], each_object[2][3])

        ####### save pascal voc file #######
        writer.save(
            '../data/train/' + str(iteration) + '/' + str(i) + '.xml')

        if i % 100 == 0:
            print("Finshed: " + str(i))


def load_label_names():
    return ['BACKGROUND', 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'ship', ]


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


# boxA = (Ax1,Ay1,Ax2,Ay2)
# boxB = (Bx1,By1,Bx2,By2)
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


# predict(0, 0.5)


def confusion_matrix():
    actual_gt = []
    predictions = []

    for i in range(1, 9):
        for j in range(0, 10):
            actual_gt.append(i)

    ### Airplane ###
    for y in range(0, 10):
        prob = random.random()
        if prob < 0.1:
            predictions.append(2)
        else:
            predictions.append(1)

    ### Automobile ###
    for y in range(0, 10):
        prob = random.random()
        if prob < 0.1:
            predictions.append(8)
        else:
            predictions.append(2)

    ### Bird ###
    for y in range(0, 10):
        prob = random.random()
        if prob < 0.3:
            predictions.append(7)
        else:
            predictions.append(3)

    ### Cat ###
    for y in range(0, 10):
        prob = random.random()
        if prob < 0.1:
            predictions.append(5)
        elif 0.1 < prob < 0.4:
            predictions.append(6)
        else:
            predictions.append(4)

    ### Deer ###
    for y in range(0, 10):
        prob = random.random()
        if prob < 0.1:
            predictions.append(4)
        elif 0.1 < prob < 0.2:
            predictions.append(6)
        else:
            predictions.append(5)

    ### Dog ###
    for y in range(0, 10):
        prob = random.random()
        if prob < 0.3:
            predictions.append(4)
        else:
            predictions.append(6)

    ### Frog ###
    for y in range(0, 10):
        prob = random.random()
        if prob < 0.2:
            predictions.append(3)
        else:
            predictions.append(7)

    ### Sheep ###
    for y in range(0, 10):
        prob = random.random()
        if prob < 0.1:
            predictions.append(1)
        else:
            predictions.append(8)

    print(actual_gt)
    print(predictions)
    actual_gt = pd.Series(actual_gt)
    predictions = pd.Series(predictions)

    df_confusion = pd.crosstab(actual_gt, predictions, rownames=['Actual'], colnames=['Predicted'], margins=True)
    df_conf_norm = df_confusion / 10

    return df_conf_norm
    # print(df_confusion)

# if __name__ == "__main__":
#     confusion_matrix()

import torch
from vision.datasets.voc_dataset import VOCDataset
import pathlib
from vision.utils.misc import str2bool, Timer
from vision.utils.pascal_voc_writer import Writer
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def predict(trained_model, iteration):
    # conf_matrix = confusion_matrix()
    model_path = "../saved_models/" + trained_model
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

    total_count_mask = 0

    file1 = open("MaskFile.txt", "a")

    margin = 0.05

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
            if 0.5 - margin <= probs[x] <= 0.5 + margin:
                writer.addObject(name_of_object, x1, y1, x2, y2, masked=1)
                total_count_mask += 1
            else:
                writer.addObject(name_of_object, x1, y1, x2, y2, masked=0)

        # for x in range(len(probs)):
        #     add_flag = 1
        #     label_number = labels[x]
        #     name_of_object = label_names[label_number]
        #     x1 = boxes[x][0]
        #     y1 = boxes[x][1]
        #     x2 = boxes[x][2]
        #     y2 = boxes[x][3]
        #     current_label = [x1, y1, x2, y2]
        #     detected_box = [name_of_object, probs[x], current_label]
        #
        #     if not all_labels:
        #         all_labels.append(detected_box)
        #         continue
        #
        #     for y in range(len(all_labels)):
        #         if iou(all_labels[y][2], detected_box[2]) > 0.7 and all_labels[y][1] > detected_box[1]:
        #             add_flag = 0
        #         elif iou(all_labels[y][2], detected_box[2]) > 0.7 and all_labels[y][1] < detected_box[1]:
        #             add_flag = 0
        #             all_labels[y][0] = detected_box[0]
        #             all_labels[y][1] = detected_box[1]
        #             all_labels[y][2] = detected_box[2]
        #
        #     if add_flag == 1:
        #         all_labels.append(detected_box)
        #
        # for each_object in all_labels:
        #     writer.addObject(each_object[0], each_object[2][0], each_object[2][1], each_object[2][2], each_object[2][3])

        ####### save pascal voc file #######


        writer.save(
            '../data/train/' + str(iteration) + '/' + str(i) + '.xml')

        if i % 100 == 0:
            print("Finshed: " + str(i))

    file1.write("Iteration: " + str(iteration) + "\n")
    file1.write("Number of masks: " + str(total_count_mask) + "\n")
    file1.write("Margin: " + str(margin) + "\n")
    file1.write("\n")
    file1.close()



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


predict("Epoch-110-Loss-1.7577346393040247.pth", 0)

# if __name__ == "__main__":
#     confusion_matrix()

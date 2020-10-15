import torch
from vision.datasets.voc_dataset import VOCDataset
import pathlib
from vision.utils.misc import str2bool, Timer
from pascal_voc_writer import Writer
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def predict(trained_model, iteration):
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

    for i in range(len(dataset)):
        ####### initialise a writer to create pascal voc file #######
        writer = Writer(
            '../data/train/PNGImages/' + str(i) + '.png', 256, 256)
        image = dataset.get_image(i)
        boxes, labels, probs = predictor.predict(image)

        probs = probs.numpy()
        labels = labels.numpy()
        boxes = boxes.numpy()

        for x in range(len(probs)):
            if probs[x] > 0.3:
                name_of_object = label_names[labels[x]]
                x1 = boxes[x][0]
                y1 = boxes[x][1]
                x2 = boxes[x][2]
                y2 = boxes[x][3]
                writer.addObject(name_of_object, x1, y1, x2, y2)

        ####### save pascal voc file #######
        writer.save(
            '../data/train/' + str(iteration) + '/' + str(i) + '.xml')

        print("Finshed: " + str(i))


def load_label_names():
    return ['BACKGROUND', 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'ship', ]


predict("Epoch-120-Loss-1.422243544929906.pth", 1)


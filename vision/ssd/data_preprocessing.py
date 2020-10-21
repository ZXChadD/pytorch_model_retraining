from ..transforms.transforms import *


class TrainAugmentation:
    def __init__(self, size, mean=0, std=1.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            # RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean),
            lambda img, boxes=None, labels=None, is_masked=None: (img / std, boxes, labels, is_masked),
            ToTensor(),
        ])

    def __call__(self, img, boxes, labels, is_masked):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: bounding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
            label_mask: boolean for whether each label is to be masked
        """
        return self.augment(img, boxes, labels, is_masked)


class TestTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            ToPercentCoords(),
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)


class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None, is_masked=None: (img / std, boxes, labels, is_masked),
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _, _ = self.transform(image)
        return image
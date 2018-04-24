"""
Mask R-CNN
Train on the toy Boat dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 boat.py train --dataset=/path/to/boat/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 boat.py train --dataset=/path/to/boat/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 boat.py train --dataset=/path/to/boat/dataset --weights=imagenet

    # Apply color splash to an image
    python3 boat.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 boat.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import cv2
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# set the gpu to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cv2.namedWindow('test', 0)

cv2.resizeWindow('test', 192, 108)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# sys.path.append("/home/gcx/repositories/cocoapi/PythonAPI")  # To find local version
from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class BoatConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "boat"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    # NUM_CLASSES = 1  # Background + baloon
    NUM_CLASSES = 1 + 1  # Background + baloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.85


############################################################
#  Dataset
############################################################

class BoatDataset(utils.Dataset):

    # def load_boat(self, dataset_dir, subset):
    #     """Load a subset of the Boat dataset.
    #     dataset_dir: Root directory of the dataset.
    #     subset: Subset to load: train or val
    #     """
    #     # Add classes. We have only one class to add.
    #     self.add_class("boat", 1, "boat")
    #
    #     # Train or validation dataset?
    #     assert subset in ["train", "val"]
    #     dataset_dir = os.path.join(dataset_dir, subset)
    #
    #     # Load annotations
    #     # VGG Image Annotator saves each image in the form:
    #     # { 'filename': '28503151_5b5b7ec140_b.jpg',
    #     #   'regions': {
    #     #       '0': {
    #     #           'region_attributes': {},
    #     #           'shape_attributes': {
    #     #               'all_points_x': [...],
    #     #               'all_points_y': [...],
    #     #               'name': 'polygon'}},
    #     #       ... more regions ...
    #     #   },
    #     #   'size': 100202
    #     # }
    #     # We mostly care about the x and y coordinates of each region
    #     # annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))   # CRUZ
    #     annotations = json.load(open(os.path.join(dataset_dir, "instances_boat.json")))
    #     annotations = list(annotations.values())  # don't need the dict keys
    #
    #     # The VIA tool saves images in the JSON even if they don't have any
    #     # annotations. Skip unannotated images.
    #     # annotations = [a for a in annotations if a['regions']]    # CRUZ
    #
    #     # Add images
    #     for a in annotations:
    #         # Get the x, y coordinaets of points of the polygons that make up
    #         # the outline of each object instance. There are stores in the
    #         # shape_attributes (see json format above)
    #         polygons = [r['shape_attributes'] for r in a['regions'].values()]
    #
    #         # load_mask() needs the image size to convert polygons to masks.
    #         # Unfortunately, VIA doesn't include it in JSON, so we must read
    #         # the image. This is only managable since the dataset is tiny.
    #         image_path = os.path.join(dataset_dir, a['filename'])
    #         image = skimage.io.imread(image_path)
    #         height, width = image.shape[:2]
    #
    #         self.add_image(
    #             "boat",
    #             image_id=a['filename'],  # use file name as a unique image id
    #             path=image_path,
    #             width=width, height=height,
    #             polygons=polygons)

    # def load_coco_boat
    def load_coco_boat(self, dataset_dir, subset, class_ids=None,
                  class_map=None, return_coco=False, auto_download=False ):

        # coco_boat = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, "instances_boat.json"))
        coco_boat = COCO("{}/annotations/instances_boat.json".format(dataset_dir))

        if subset == 'train': # TODO implement the rest of subset selection
            print('loading train subset')
        else:
            print('loading val subset')

        # Add classes. We have only one class to add.
        # self.add_class("boat", 1, "boat")

        # image_dir = "{}/{}{}".format(dataset_dir, subset, year)   # CRUZ
        image_dir = dataset_dir

        if not class_ids:
            # All classes
            class_ids = sorted(coco_boat.getCatIds())

            # All classes
        # Load all classes or a subset?
        # if not class_ids:
        self.add_class("boat", 1, "boat")
        # class_ids = False

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco_boat.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco_boat.imgs.keys())

        # Add classes
        # for i in class_ids:
        #     self.add_class("coco", i, coco_boat.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco_boat.imgs[i]['file_name']),
                width=coco_boat.imgs[i]["width"],
                height=coco_boat.imgs[i]["height"],
                annotations=coco_boat.loadAnns(coco_boat.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))

        # if return_coco:
        #     return coco

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # # If not a boat dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        # if image_info["source"] != "coco":
        #     return super(self.__class__, self).load_mask(image_id)
        #
        # # Convert polygons to a bitmap mask of shape
        # # [height, width, instance_count]
        # info = self.image_info[image_id]
        # # mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
        # #                 dtype=np.uint8)
        # mask = np.zeros([info["height"], info["width"], len(info["annotations"])],
        #                 dtype=np.uint8)
        # for i, p in enumerate(info["annotations"]):
        # # for i, p in enumerate(info["polygons"]):
        #     # Get indexes of pixels inside the polygon and set them to 1
        #     rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        #     mask[rr, cc, i] = 1
        #
        # # Return mask, and array of class IDs of each instance. Since we have
        # # one class ID only, we return an array of 1s
        # return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

        ###########################
        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            # class_id = self.map_source_class_id(   ## CRUZ
            #     "coco.{}".format(annotation['category_id']))   ## CRUZ
            class_id = 1
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)
                # class_ids.append(0)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(BoatDataset, self).load_mask(image_id)

    # The following two functions are from pycocotools with a few changes.
    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, int(height), int(width))
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, int(height), int(width))
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = BoatDataset()
    # dataset_train.load_boat(args.dataset, "train")
    dataset_train.load_coco_boat(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BoatDataset()
    dataset_val.load_coco_boat(args.dataset, "val")
    # dataset_val.load_boat(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


def splash_color(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def place_bb(image, bbs, scores):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    # gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image))

    bbs_array = np.empty((0,5))

    # Copy color pixels from the original color image where mask is set
    if len(bbs) != 0:
        for bb_idx, indv_bb in enumerate(bbs):
            print(bb_idx)
            print(indv_bb)
            top_left = (indv_bb[0], indv_bb[1])
            # cv2.rectangle(gray, (indv_bb[1], indv_bb[0]), (indv_bb[3], indv_bb[2]), [0, 0, 255], 2)
            # cv2.imshow('test', gray)
            # cv2.waitKey(1)
            bb_on_img = gray
            bbs_array = np.append(bbs_array,
                                  np.array([[indv_bb[0], indv_bb[1], indv_bb[2]- indv_bb[0], indv_bb[3] - indv_bb[1], scores[bb_idx]]]),
                                  axis=0)

    else:
        bb_on_img = gray


    return bb_on_img, bbs_array



def detect_and_place_bb(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = place_bb(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        gt_files = open(video_path[:-4] + 'mask.txt', 'w+')

        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)
        # vcapture.set(cv2.CAP_PROP_POS_FRAMES, 500)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = splash_color(image, r['masks'])

                # place bb
                splash, bbs= place_bb(splash, r['rois'], r['scores'])

                for gt_idx in range(bbs.shape[0]):
                    gt_files.write(str(count) + ' ' +
                                   str(bbs[gt_idx, 0]) + ' ' + str(bbs[gt_idx, 1]) + ' '
                                   + str(bbs[gt_idx, 2]) + ' ' + str(bbs[gt_idx, 3]) +
                                   ' 1 ' + str(bbs[gt_idx, 4]) + '\n')

                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)
    gt_files.close()


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect boats.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'deploy'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/boat/dataset/",
                        help='Directory of the boat dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "deploy":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = BoatConfig()
    else:
        class InferenceConfig(BoatConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    # if args.weights.lower() == "coco":
    # Exclude the last layers because they require a matching
    # number of classes
    model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    # else:
    #     model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "deploy":
        detect_and_place_bb(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))

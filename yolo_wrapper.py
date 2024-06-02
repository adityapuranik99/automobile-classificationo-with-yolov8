import warnings
from shutil import copy, rmtree
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import matplotlib.pyplot as plt


class YoloWrapper:
    def __init__(self, model_weights: str) -> None:
        """
        Initialize YOLOv8 model with weights.
        Args:
            model_weights (str): model weight can be one of the follows:
                - 'nano' for YOLOv8 nano model
                - 'small' for YOLOv8 small model
                - a path to a .pt file contains the weights from a previous training.
        """
        if model_weights == 'nano':
            model_weights = 'yolov8n.pt'
        elif model_weights == 'small':
            model_weights = 'yolov8s.pt'
        elif model_weights == 'medium':
            model_weights = 'yolov8m.pt'
        elif (not Path(model_weights).exists()) or (Path(model_weights).suffix != '.pt'):
            raise ValueError('The parameter model_weight should be "nano", "small" or a'
                             'path to a .pt file with saved weights')

        # initialize YOLO model
        self.model = YOLO(model_weights)

    def train(self, config: str, epochs: int = 100, name: str = None) -> None:
        """
        Train the model. After running a 'runs/detect/<name>' folder will be created and stores information
        about the training and the saved weights.
        Args:
            config (str): a path to a configuration yaml file for training.
                Such a file contains:
                    path -  absolute path to the folder contains the images and labels folders with the data
                    train - relative path to 'path' of the train images folder (images/train)
                    val -  relative path to 'path' of the validation images folder (images/val), if exists
                    nc - the number of classes
                    names - a list of the classes names
                Can be created with the create_config_file method.
            epochs (int): number of epochs for training
            name (str): the name of the results' folder. If None (default) a default name 'train #' will
                be created.

        Returns:

        """
        if Path(config).suffix != '.yaml':
            raise ValueError('Config file should be a yaml file')
        self.model.train(data=config, epochs=epochs, name=name, freeze=10)

    def predict(self, image: str | Path | np.ndarray | list[str] | list[Path] | list[np.ndarray], threshold: float = 0.25,
                ) -> list[np.ndarray]:
        """
        Predict bounding box for images.
        Args:
            image (str|Path|np.ndarray|list[str]|list[Path]|list[np.ndarray]): image data. Can be a string path
                to an image, a BGR image as numpy ndarray, a list with string paths to images or a list
                with BGR images as numpy ndarray.
            threshold (float): a number between 0 and 1 for the confidence a bounding box should have to
                consider as a detection. Default is 0.25.

        Returns:
            (list[np.ndarray]): a list with numpy ndarrays for each detection
        """
        yolo_results = self.model(image, conf=threshold)
        bounding_boxes = [torch.concatenate([x.boxes.xyxy[:, :2], x.boxes.xyxy[:, 2:] - x.boxes.xyxy[:, :2]], dim=1).cpu().numpy()
                          for x in yolo_results]
        return bounding_boxes

    def predict_and_save_to_csv(self, images: list[str] | list[Path] | list[np.ndarray], image_ids: list[str] = None,
                                path_to_save_csv: str | Path = '', threshold: float = 0.25, minimum_size: int = 100,
                                only_most_conf=True) -> None:
        """
        Predict a batch of images and return the bounding boxs prediction in a csv file with the columns:
        image_id, x_top_left, y_top_left, width, height. If there is no any prediction, a csv will not be created.
        Args:
            images (list[str] | list[Path] | list[np.ndarray]): a list with string paths to images or a list
                with BGR images as numpy ndarray.
            image_ids (list[str]): the ids of the images
            path_to_save_csv (Optional, str|Path): a path where to save the csv file. If the path is not for
                a specific csv file, the file name will be bounding_box.csv by default.
            threshold (float):  a number between 0 and 1 for the confidence a bounding box should have to
                consider as a detection. Default is 0.25.
            minimum_size (int): the minimum width and height in pixels for a bounding box to saved in the csv file.
                If the bounding box founded is smaller than the minimum size, the width and the height will update
                 to the minimum size, and the top left point will be updated accordingly to keep the object in
                 the center
            only_most_conf (bool): True to keep only the bounding box with the highest confidence for each image.
                The bounding boxes are sorted so the first one is the one with the highest confidence

        Returns:

        """
        if image_ids is None:
            if isinstance(images[0], np.ndarray):
                raise ValueError('image_ids can not be None if images is a list of numpy arrays')
            else:
                # get the name of the images as image id
                image_ids = [Path(image_path).stem for image_path in images]

        if isinstance(images[0], (str, Path)):
            h, w, _ = cv2.imread(str(images[0])).shape
        else:  # numpy array
            h, w, _ = images[0].shape

        bbox_list = self.predict(images, threshold)
        if only_most_conf:  # keep only the bounding box with the highest confidence
            bbox_list = [bboxes[[0], :] if bboxes.shape[0] > 0 else bboxes for bboxes in bbox_list]

        # if there are more than one bounding box for an image, we need to duplicate the image id
        image_ids_with_duplicates = [image_id
                                     for bbox, image_id in zip(bbox_list, image_ids)
                                     for _ in range(bbox.shape[0])]
        bbox_matrix = np.vstack(bbox_list)

        if bbox_matrix.shape[0] == 0:
            warnings.warn('A bounding boxes were not found for any of the images.'
                          'A csv file will not be created')
            return

        # set the width to minimum value
        less_than_min = bbox_matrix[:, 2] < minimum_size
        missing_width = minimum_size - bbox_matrix[less_than_min, 2]
        bbox_matrix[less_than_min, 2] = minimum_size
        bbox_matrix[less_than_min, 0] = np.minimum(
            np.maximum(bbox_matrix[less_than_min, 0] - missing_width / 2, 0),
            w - 1 - minimum_size
        )

        # set the height to minimum value
        less_than_min = bbox_matrix[:, 3] < minimum_size
        missing_height = minimum_size - bbox_matrix[less_than_min, 3]
        bbox_matrix[less_than_min, 3] = minimum_size
        bbox_matrix[less_than_min, 1] = np.minimum(
            np.maximum(bbox_matrix[less_than_min, 1] - missing_height / 2, 0),
            h - 1 - minimum_size
        )

        dict_for_csv = {
            'image_id': image_ids_with_duplicates,
            'x_top_left': bbox_matrix[:, 0],
            'y_top_left': bbox_matrix[:, 1],
            'width': bbox_matrix[:, 2],
            'height': bbox_matrix[:, 3]
        }

        bbox_dataframe = pd.DataFrame(dict_for_csv)

        path_to_save_csv = Path(path_to_save_csv)
        if path_to_save_csv.suffix == '':
            path_to_save_csv = path_to_save_csv / 'bounding_boxes.csv'
        if path_to_save_csv.suffix != '.csv':
            raise ValueError('A non-csv file is given')
        path_to_save_csv.parent.mkdir(parents=True, exist_ok=True)
        bbox_dataframe.to_csv(str(path_to_save_csv), index=False)

    def predict_and_show(self, image: str | np.ndarray, threshold: float = 0.25) -> None:
        """
        Predict bounding box for a single image and show the bounding box with its confidence.
        Args:
            image (str | np.ndarray): a path to an image or a BGR np.ndarray image to predict
                bounding box for
            threshold (float): a number between 0 and 1 for the confidence a bounding box should have to
                consider as a detection. Default is 0.25.
        Returns:

        """
        yolo_results = self.model(image, threshold=threshold)
        labeled_image = yolo_results[0].plot()
        plt.figure()
        plt.imshow(labeled_image[..., ::-1])
        plt.show()
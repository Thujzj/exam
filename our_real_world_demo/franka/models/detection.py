import numpy as np
import torch
import torch.nn as nn
import detectron2
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import os, json, cv2, random
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from groundingdino.util.inference import load_model, load_image, predict, annotate
import groundingdino.datasets.transforms as T


class Detection(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)
    def predict(self, x):
        return self.predictor(x)

    def display(self, im, prediction):
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(prediction["instances"].to("cpu"))
        cv2.imshow("1",out.get_image()[:, :, ::-1])
        cv2.waitKey(1)

class DINO():
    def __init__(self,weight_path,model_path,BOX_TRESHOLD,TEXT_TRESHOLD) -> None:
        self.model = load_model(model_path,weight_path)
        self.BOX_TRESHOLD = BOX_TRESHOLD 
        self.TEXT_TRESHOLD = TEXT_TRESHOLD

        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    def predict(self, raw_image, prompt):
        image, _ = self.transform(raw_image, None)
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=prompt,
            box_threshold=self.BOX_TRESHOLD,
            text_threshold=self.TEXT_TRESHOLD
        )
        
        return boxes, logits, phrases

    def annotate(self, image_source, boxes, logits, phrases):
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        return annotated_frame


if __name__ == "__main__":
    im = cv2.imread("./color_0.png", cv2.IMREAD_COLOR)
    
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)
    
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("1",out.get_image()[:, :, ::-1])
    # cv2.imshow("1",im)
    cv2.waitKey()

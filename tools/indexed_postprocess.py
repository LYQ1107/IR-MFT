# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
"""YOLOX postprocess with unchanged selection and original-anchor provenance."""
import torch
import torchvision


def corner_boxes(prediction):
    result = prediction.new_empty((*prediction.shape[:2], 4))
    result[:, :, 0] = prediction[:, :, 0]-prediction[:, :, 2]/2
    result[:, :, 1] = prediction[:, :, 1]-prediction[:, :, 3]/2
    result[:, :, 2] = prediction[:, :, 0]+prediction[:, :, 2]/2
    result[:, :, 3] = prediction[:, :, 1]+prediction[:, :, 3]/2
    return result


def indexed_postprocess(prediction, num_classes, conf_thre=.1, nms_thre=.7):
    """Same arithmetic/NMS as yolox.utils.boxes.postprocess; keep input private."""
    prediction = prediction.clone()
    prediction[:, :, :4] = corner_boxes(prediction)
    outputs = [None for _ in prediction]; anchors = [None for _ in prediction]
    for i, image_pred in enumerate(prediction):
        if not image_pred.size(0):
            continue
        confidence, category = torch.max(image_pred[:, 5:5+num_classes], 1, keepdim=True)
        mask = (image_pred[:, 4]*confidence.squeeze() >= conf_thre).reshape(-1)
        detections = torch.cat((image_pred[:, :5], confidence, category.float()), 1)[mask]
        indices = torch.arange(len(image_pred), device=image_pred.device)[mask]
        if not len(detections):
            continue
        selected = torchvision.ops.batched_nms(detections[:, :4], detections[:, 4]*detections[:, 5], detections[:, 6], nms_thre)
        outputs[i] = detections[selected]; anchors[i] = indices[selected]
    return outputs, anchors

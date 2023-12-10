#!/usr/bin/env python3
"""
The code is edited from docs (https://docs.luxonis.com/projects/api/en/latest/samples/Yolo/tiny_yolo/)
We add parsing from JSON files that contain configuration
"""

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import argparse
import json
import blobconverter
import requests
from time import monotonic

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Provide model name or model path for inference",
                    default='yolov7tiny_waldo_416x768', type=str)
parser.add_argument("-v", "--video", help="Path to video file", default='pexels_street_1.mp4', type=str)
args = parser.parse_args()

config_path = f"https://raw.githubusercontent.com/luxonis/depthai-model-zoo/main/models/{args.model}/config.json"
# load config from github

config = requests.get(config_path).text
config = json.loads(config)

config["nn_config"]["NN_specific_metadata"]["confidence_threshold"] = 0.3
config["nn_config"]["NN_specific_metadata"]["iou_threshold"] = 0.2
if "input_size" not in config["nn_config"]:
    config["nn_config"]["input_size"] = args.model.split("_")[-1]
labelMap = config["mappings"]["labels"]

# print config in a readable way
print(json.dumps(config, indent=4, sort_keys=True))

nnConfig = config.get("nn_config", {})

# parse input shape
if "input_size" in nnConfig:
    W, H = tuple(map(int, nnConfig.get("input_size").split('x')))

# extract metadata
metadata = nnConfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", {})
confidenceThreshold = metadata.get("confidence_threshold", {})

print(metadata)

# parse labels
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", {})

# get model path
nnPath = args.model
if not Path(nnPath).exists():
    print("No blob found at {}. Looking into DepthAI model zoo.".format(nnPath))
    nnPath = str(blobconverter.from_zoo(args.model, shaves = 7, zoo_type = "depthai", use_cache=True))
# sync outputs
syncNN = True

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
inputVideo = pipeline.create(dai.node.XLinkIn)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
nnOut = pipeline.create(dai.node.XLinkOut)

inputVideo.setStreamName("inFrame")
nnOut.setStreamName("nn")

# Network specific settings
detectionNetwork.setConfidenceThreshold(confidenceThreshold)
detectionNetwork.setNumClasses(classes)
detectionNetwork.setCoordinateSize(coordinates)
detectionNetwork.setAnchors(anchors)
detectionNetwork.setAnchorMasks(anchorMasks)
detectionNetwork.setIouThreshold(iouThreshold)
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

# Linking
inputVideo.out.link(detectionNetwork.input)
detectionNetwork.out.link(nnOut.input)

# with open('pipeline.json', 'w') as outfile:
#     json.dump(pipeline.serializeToJson(), outfile, indent=4)

with dai.Device(pipeline) as device:

    # Input queue will be used to send video frames to the device.
    qIn = device.getInputQueue(name="inFrame")
    # Output queue will be used to get nn data from the video frames.
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    frame = None
    detections = []

    startTime = time.monotonic()
    counter = 0
    fps = 0
    color_palette_16 = [
        (255, 105, 97),  # Salmon
        (255, 179, 71),  # Orange
        (255, 209, 102), # Yellow
        (144, 238, 144), # Light Green
        (102, 205, 170), # Medium Aquamarine
        (100, 149, 237), # Cornflower Blue
        (70, 130, 180),  # Steel Blue
        (135, 206, 235), # Sky Blue
        (106, 90, 205),  # Slate Blue
        (123, 104, 238), # Medium Purple
        (147, 112, 219), # Medium Purple
        (219, 112, 147), # Pale Violet Red
        (255, 20, 147),  # Deep Pink
        (255, 105, 180), # Hot Pink
        (255, 182, 193), # Light Pink
        (255, 192, 203)  # Pink
    ]

    def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
        return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
    
    def addTransparentRectangle(color, x1, y1, x2, y2, trackerFrame, alpha):
        tracker_slice = trackerFrame[y1:y2, x1:x2]
        colored_rectangle_numpy = np.zeros_like(tracker_slice, dtype=np.uint8)
        colored_rectangle_numpy[:] = color
        res = cv2.addWeighted(tracker_slice, 1-alpha, colored_rectangle_numpy, alpha, 0, tracker_slice)
        trackerFrame[y1:y2, x1:x2] = res
        
    def displayFrame(name, frame):
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        # Show the frame
        cv2.imshow(name, frame)

    cap = cv2.VideoCapture(args.video)
    frame_counter = 0
    target_fps = 30
    target_frame_time = 1.0 / target_fps
    while cap.isOpened():
        start = time.time()
        read_correctly, frame = cap.read()
        if not read_correctly:
            break
        
        frame_counter += 1
        #If the last frame is reached, reset the capture and the frame_counter
        if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            frame_counter = 0 #Or whatever as long as it is the same as next line
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
        img = dai.ImgFrame()
        img.setData(to_planar(frame, (768, 416)))
        img.setTimestamp(monotonic())
        img.setWidth(768)
        img.setHeight(416)
        qIn.send(img)

        inDet = qDet.tryGet()

        if inDet is not None:
            detections = inDet.detections

        if frame is not None:
            frame = cv2.resize(frame, (1200, 700))
            displayFrame("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break
        
        end = time.time()
        frame_time = end - start
        if frame_time < target_frame_time:
            time.sleep(target_frame_time - frame_time)

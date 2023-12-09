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
manip = pipeline.create(dai.node.ImageManip)
objectTracker = pipeline.create(dai.node.ObjectTracker)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)

manipOut = pipeline.create(dai.node.XLinkOut)
xinFrame = pipeline.create(dai.node.XLinkIn)
trackerOut = pipeline.create(dai.node.XLinkOut)
xlinkOut = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

manipOut.setStreamName("manip")
xinFrame.setStreamName("inFrame")
xlinkOut.setStreamName("trackerFrame")
trackerOut.setStreamName("tracklets")
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

manip.initialConfig.setResizeThumbnail(W, H)
manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
manip.inputImage.setBlocking(True)

objectTracker.inputTrackerFrame.setBlocking(True)
objectTracker.inputDetectionFrame.setBlocking(True)
objectTracker.inputDetections.setBlocking(True)
# objectTracker.setDetectionLabelsToTrack([0])  # track only car
# possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_IMAGELESS)
# take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

# Linking
manip.out.link(manipOut.input)
manip.out.link(detectionNetwork.input)
xinFrame.out.link(manip.inputImage)
xinFrame.out.link(objectTracker.inputTrackerFrame)
detectionNetwork.out.link(nnOut.input)
detectionNetwork.out.link(objectTracker.inputDetections)
detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
objectTracker.out.link(trackerOut.input)
objectTracker.passthroughTrackerFrame.link(xlinkOut.input)

# with open('pipeline.json', 'w') as outfile:
#     json.dump(pipeline.serializeToJson(), outfile, indent=4)

with dai.Device(pipeline) as device:

    qIn = device.getInputQueue(name="inFrame")
    trackerFrameQ = device.getOutputQueue(name="trackerFrame", maxSize=4)
    tracklets = device.getOutputQueue(name="tracklets", maxSize=4)
    qManip = device.getOutputQueue(name="manip", maxSize=4)
    qDet = device.getOutputQueue(name="nn", maxSize=4)

    startTime = time.monotonic()
    counter = 0
    fps = 0
    detections = []
    frame = None

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

    cap = cv2.VideoCapture(args.video)
    baseTs = time.monotonic()
    simulatedFps = 30
    inputFrameShape = (W, H)

    while cap.isOpened():
        read_correctly, frame = cap.read()
        if not read_correctly:
            break

        img = dai.ImgFrame()
        img.setType(dai.ImgFrame.Type.BGR888p)
        img.setData(to_planar(frame, inputFrameShape))
        img.setTimestamp(baseTs)
        baseTs += 1/simulatedFps

        img.setWidth(inputFrameShape[0])
        img.setHeight(inputFrameShape[1])
        qIn.send(img)

        trackFrame = trackerFrameQ.tryGet()
        if trackFrame is None:
            continue

        track = tracklets.get()
        manip = qManip.get()
        inDet = qDet.get()

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        detections = inDet.detections

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
        trackerFrame = trackFrame.getCvFrame()
        tracker_frame_mask = np.zeros_like(trackerFrame, dtype=np.uint8)
        trackletsData = track.tracklets
        tracklets_to_draw = trackletsData
        # tracklets_to_draw = []
        
        # for tracklet in trackletsData:
        #     roi = tracklet.roi
        #     x1 = roi.topLeft().x
        #     y1 = roi.topLeft().y
        #     x2 = roi.bottomRight().x
        #     y2 = roi.bottomRight().y
            
        #     for i in range(len(tracklets_to_draw)):
        #         other_tracklet = tracklets_to_draw[i]
        #         other_roi = other_tracklet.roi
        #         other_x1 = other_roi.topLeft().x
        #         other_y1 = other_roi.topLeft().y
        #         other_x2 = other_roi.bottomRight().x
        #         other_y2 = other_roi.bottomRight().y
                
        #         if (x1 >= other_x1 and x1 <= other_x2) or (x2 >= other_x1 and x2 <= other_x2) or (other_x1 >= x1 and other_x1 <= x2) or (other_x2 >= x1 and other_x2 <= x2):
        #             if (y1 >= other_y1 and y1 <= other_y2) or (y2 >= other_y1 and y2 <= other_y2) or (other_y1 >= y1 and other_y1 <= y2) or (other_y2 >= y1 and other_y2 <= y2):
        #                 if tracklet.roi.area() > other_tracklet.roi.area():
        #                     tracklets_to_draw[i] = tracklet
        #                 break
                
        #     else:
        #         tracklets_to_draw.append(tracklet)
        
        for tracklet in tracklets_to_draw:
            roi = tracklet.roi.denormalize(trackerFrame.shape[1], trackerFrame.shape[0])
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)

            try:
                label = labelMap[tracklet.label]
            except:
                label = tracklet.label

            # cv2.putText(trackerFrame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            # cv2.putText(trackerFrame, t.status.name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            
            color = color_palette_16[tracklet.label]
            addTransparentRectangle(color, x1, y1, x2, y2, trackerFrame, 0.3)
            tracker_frame_mask[y1:y2, x1:x2] = 1
            # addTransparentRectangle(color, x1, y1 - 10, x1 + 40, y1, trackerFrame, 0.3)
            cv2.putText(trackerFrame, str(label), (x1, y1), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (255, 255, 255))

        cv2.imshow("tracker", trackerFrame)

        if cv2.waitKey(1) == ord('q'):
            break
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
from pose import getKeypoints, getValidPairs, getPersonwiseKeypoints
import threading
from tqdm import tqdm
from utils import KeypointSmoother, KeypointGraph

W, H = (456, 256)
running = True
pose = None
detected_keypoints = None

keypoint_smoother = KeypointSmoother()
keypoint_graph = KeypointGraph()


def draw_keypoints(frame):
    global detected_keypoints

    if detected_keypoints is not None:
        scale_factor = frame.shape[0] / H
        offset_w = int(frame.shape[1] - W * scale_factor) // 2

        def scale(point):
            return int(point[0] * scale_factor) + offset_w, int(point[1] * scale_factor)
        
        detected_keypoints = keypoint_smoother.smooth_keypoints(detected_keypoints)
        vertices, edges = keypoint_graph.generate_continuous_graph(detected_keypoints)
        
        for vertex, color in vertices:
            kp = detected_keypoints[vertex]
            cv2.circle(frame, scale(kp[0:2]), 5, color, -1, cv2.LINE_AA)
            
        for edge_a, edge_b, color in edges:
            kp_a = detected_keypoints[edge_a]
            kp_b = detected_keypoints[edge_b]
            cv2.line(frame, scale(kp_a[0:2]), scale(kp_b[0:2]), color, 3, cv2.LINE_AA)


def decode_thread(in_queue):
    global detected_keypoints

    while running:
        try:
            raw_in = in_queue.tryGet()
        except RuntimeError:
            return
        if raw_in is None:
            continue
        
        # print("stuff coming in")
        
        heatmaps = np.array(raw_in.getLayerFp16('Mconv7_stage2_L2')).reshape((1, 19, 32, 57))
        pafs = np.array(raw_in.getLayerFp16('Mconv7_stage2_L1')).reshape((1, 38, 32, 57))
        heatmaps = heatmaps.astype('float32')
        pafs = pafs.astype('float32')
        outputs = np.concatenate((heatmaps, pafs), axis=1)

        new_keypoints = []

        for row in range(18):
            probMap = outputs[0, row, :, :]
            probMap = cv2.resize(probMap, (W, H))  # (456, 256)
            keypoints = getKeypoints(probMap, 0.3)

            if len(keypoints) > 0:
                new_keypoints.append(keypoints[0])
            else:
                new_keypoints.append(None)

        detected_keypoints = new_keypoints

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Provide model name or model path for inference",
                    default='human-pose-estimation-0001', type=str)
parser.add_argument("-v", "--video", help="Path to video file", default='keypoint_detection/pexels_walk_4.mp4', type=str)
parser.add_argument("-r", "--render", help="Render output video", action='store_true')

args = parser.parse_args()

nnpath = str(blobconverter.from_zoo(args.model, shaves = 8, use_cache=True))

pipeline = dai.Pipeline()
inputVideo = pipeline.create(dai.node.XLinkIn)
keypoint_network = pipeline.create(dai.node.NeuralNetwork)
nnOut = pipeline.create(dai.node.XLinkOut)

inputVideo.setStreamName("inFrame")
nnOut.setStreamName("nn")

# Network specific settings
keypoint_network.setBlobPath(nnpath)
keypoint_network.setNumInferenceThreads(2)
keypoint_network.input.setBlocking(False)

# Linking
inputVideo.out.link(keypoint_network.input)
keypoint_network.out.link(nnOut.input)

# with open('pipeline.json', 'w') as outfile:
#     json.dump(pipeline.serializeToJson(), outfile, indent=4)

with dai.Device(pipeline) as device:
    def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
        return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()
    
    def to_planar_no_reshape(arr: np.ndarray) -> np.ndarray:
        return arr.transpose(2, 0, 1).flatten()
    
    # Input queue will be used to send video frames to the device.
    qIn = device.getInputQueue(name="inFrame")
    # Output queue will be used to get nn data from the video frames.
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    
    t = threading.Thread(target=decode_thread, args=(qDet,))
    t.start()
    
    target_fps = 30
    target_frame_time = 1 / target_fps
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video file: {args.video} from directory: {Path.cwd()}")

    cap_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out_video_shape = (int(cap_w), int(cap_h))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(f'{args.video}_det.mp4', fourcc, 30.0, out_video_shape)
    frame_counter = 0
    
    frame_delay = 5
    frame_buffer = [None] * frame_delay
    Wr = int(cap_w * (H / cap_h))
    if Wr < W:
        Wr = W
    left_pad = (Wr - W) // 2
    right_pad = Wr - W - left_pad
    preview_height = 700
    preview_width = int(cap_w * (preview_height / cap_h))
        
    pbar = tqdm(total=cap.get(cv2.CAP_PROP_FRAME_COUNT), desc=f"Rendering {args.video}")
    while cap.isOpened():
        start = time.time()
        read_correctly, og_frame = cap.read()
        if not read_correctly:
            break
        frame_counter += 1
        
        frame_buffer.append(og_frame)
        del frame_buffer[0]
        
        # resize frame
        frame = cv2.resize(og_frame, (Wr, H))
        
        # pad frame left and right area to match target aspect ratio
        frame = cv2.copyMakeBorder(frame, 0, 0, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        img = dai.ImgFrame()    
        img.setData(to_planar_no_reshape(frame))
        img.setTimestamp(monotonic())
        img.setWidth(W)
        img.setHeight(H)
        qIn.send(img)
        
        frame = frame_buffer[0]
                
        if frame is not None:
            draw_keypoints(frame)
            # remove padding from frame
            # frame = frame[:, pad_W:-pad_W]
            
            out_video.write(frame)
            if args.render:
                preview_frame = cv2.resize(frame, (preview_width, preview_height))
                cv2.imshow("frame", preview_frame)
        
        if cv2.waitKey(1) == ord('q'):
            running = False
            t.join()
            break
        
        end = time.time()
        if end - start < target_frame_time:
            time.sleep(target_frame_time - (end - start))
            
        pbar.update(1)
    
out_video.release()
t.join()

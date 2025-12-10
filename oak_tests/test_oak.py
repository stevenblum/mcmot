#!/usr/bin/env python3
'''
Online Hub AI: https://hub.luxonis.com/ai/models
Uploaded yolo11 .pt file
Converted it to RCV2, entered classes, in advanced options selected create .blob
In depthai API image sizes are WxH, while in the converter it asks for HxW
The "blob" option put it into "legacy" mode, unsure if this is necessary
Downloaded the .tar.xz, this is the "achive" file that contains the model and metadata
'''

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time

nn_archive_path = "/home/scblum/Projects/mcmot/oak_tests/yolo/best.rvc2.tar.xz"

RES = (640, 416) # Width, Height
FPS = 5

class SpatialVisualizer(dai.node.HostNode):
    def __init__(self):
        dai.node.HostNode.__init__(self)
        self.sendProcessingToPipeline(True)
    def build(self, depth:dai.Node.Output, detections: dai.Node.Output, rgb: dai.Node.Output):
        self.link_args(depth, detections, rgb) # Must match the inputs to the process method

    def process(self, depthPreview, detections, rgbPreview):
        depthPreview = depthPreview.getCvFrame()
        rgbPreview = rgbPreview.getCvFrame()
        depthFrameColor = self.processDepthFrame(depthPreview)
        self.displayResults(rgbPreview, depthFrameColor, detections.detections)

    def processDepthFrame(self, depthFrame):
        depth_downscaled = depthFrame[::4]
        if np.all(depth_downscaled == 0):
            min_depth = 0
        else:
            min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
        max_depth = np.percentile(depth_downscaled, 99)
        depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
        return cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

    def displayResults(self, rgbFrame, depthFrameColor, detections):
        height, width, _ = rgbFrame.shape
        for detection in detections:
            self.drawBoundingBoxes(depthFrameColor, detection)
            self.drawDetections(rgbFrame, detection, width, height)

        cv2.imshow("depth", depthFrameColor)
        cv2.imshow("rgb", rgbFrame)
        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()

    def drawBoundingBoxes(self, depthFrameColor, detection):
        roiData = detection.boundingBoxMapping
        roi = roiData.roi
        roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
        topLeft = roi.topLeft()
        bottomRight = roi.bottomRight()
        cv2.rectangle(depthFrameColor, (int(topLeft.x), int(topLeft.y)), (int(bottomRight.x), int(bottomRight.y)), (255, 255, 255), 1)

    def drawDetections(self, frame, detection, frameWidth, frameHeight):
        x1 = int(detection.xmin * frameWidth)
        x2 = int(detection.xmax * frameWidth)
        y1 = int(detection.ymin * frameHeight)
        y2 = int(detection.ymax * frameHeight)
        label = detection.labelName
        color = (255, 255, 255)
        cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, "{:.2f}".format(detection.confidence * 100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

# Creates the pipeline and a default device implicitly
with dai.Pipeline() as p:
    # Define sources and outputs
    camRgb = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    monoLeft = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoRight = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    stereo = p.create(dai.node.StereoDepth)

    nn_archive = dai.NNArchive(str(Path(nn_archive_path).resolve()))
    print("NN Archive loaded:", nn_archive)
    
    # Configure RGB camera to output the correct resolution for NN and visualization
    #rgb_out = camRgb.requestOutput(RES, dai.ImgFrame.Type.BGR888p)
    
    spatialDetectionNetwork = p.create(dai.node.SpatialDetectionNetwork).build(camRgb, stereo, nn_archive) 
    
    #.build(camRgb, stereo, modelDescription, fps=FPS)
    #spatialDetectionNetwork.setBlobPath(blob_path)
    #spatialDetectionNetwork.setModelPath("/home/scblum/Projects/mcmot/oak_tests/yolo/best.xml")
    visualizer = p.create(SpatialVisualizer)

    # setting node configs
    stereo.setExtendedDisparity(True)
    platform = p.getDefaultDevice().getPlatform()
    if platform == dai.Platform.RVC2:
        # For RVC2, width must be divisible by 16
        stereo.setOutputSize(RES[0], RES[1])

    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)


    # Linking
    monoLeft.requestOutput((RES[0], RES[1])).link(stereo.left)
    monoRight.requestOutput((RES[0], RES[1])).link(stereo.right)

    visualizer.build(spatialDetectionNetwork.passthroughDepth, spatialDetectionNetwork.out, spatialDetectionNetwork.passthrough)

    p.run()

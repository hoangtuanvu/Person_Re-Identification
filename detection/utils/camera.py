"""camera.py

This code implements the Camera class, which encapsulates code to
handle IP CAM, USB webcam or the Jetson onboard camera.  The Camera
class is further extend to take either a video or an image file as
input.
"""

import time
import logging
import threading
import numpy as np
import cv2


def open_cam_rtsp(uri, width, height, latency):
    """Open an RTSP URI (IP CAM)."""
    gst_str = ('rtspsrc location={} latency={} ! '
               'rtph264depay ! h264parse ! omxh264dec ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! videoconvert ! '
               'appsink').format(uri, latency, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_usb(dev, width, height):
    """Open a USB webcam.

    We want to set width and height here, otherwise we could just do:
        return cv2.VideoCapture(dev)
    """
    gst_str = ('v4l2src device=/dev/video{} ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)RGB ! videoconvert ! '
               'appsink').format(dev, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_onboard(width, height):
    """Open the Jetson onboard camera.

    On versions of L4T prior to 28.1, you might need to add
    'flip-method=2' into gst_str.
    """
    gst_str = ('nvcamerasrc ! '
               'video/x-raw(memory:NVMM), '
               'width=(int)2592, height=(int)1458, '
               'format=(string)I420, framerate=(fraction)30/1 ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! videoconvert ! '
               'appsink').format(width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_default_cam():
    return cv2.VideoCapture(0)


def grab_img(cam):
    """This 'grab_img' function is designed to be run in the sub-thread.
    Once started, this thread continues to grab a new image and put it
    into the global 'img_handle', until 'thread_running' is set to False.
    """
    while cam.thread_running:
        if cam.args.use_image:
            assert cam.img_handle is not None, 'img_handle is empty in use_image case!'
            # keep using the same img, no need to update it
            time.sleep(0.01)  # yield CPU to other threads
        else:
            _, cam.img_handle = cam.cap.read()
            if cam.img_handle is None:
                logging.warning('grab_img(): cap.read() returns None...')
                break
    cam.thread_running = False


class Camera:
    """Camera class which supports reading images from theses video sources:

    1. Video file
    2. Image (jpg, png, etc.) file, repeating indefinitely
    3. RTSP (IP CAM)
    4. USB webcam
    5. Jetson onboard camera
    """

    def __init__(self, args):
        self.args = args
        self.is_opened = False
        self.thread_running = False
        self.img_handle = None
        self.img_width = 0
        self.img_height = 0
        self.cap = None
        self.thread = None

    def open(self):
        """Open camera based on command line arguments."""
        assert self.cap is None, 'Camera is already opened!'
        args = self.args
        if args.use_file:
            self.cap = cv2.VideoCapture(args.filename)
            # ignore image width/height settings here
        elif args.use_image:
            self.cap = 'OK'
            self.img_handle = cv2.imread(args.filename)
            # ignore image width/height settings here
            if self.img_handle is not None:
                self.is_opened = True
                self.img_height, self.img_width, _ = self.img_handle.shape
        elif args.use_rtsp:
            self.cap = open_cam_rtsp(
                args.rtsp_uri,
                args.image_width,
                args.image_height,
                args.rtsp_latency
            )
        elif args.use_usb:
            self.cap = open_cam_usb(
                args.video_dev,
                args.image_width,
                args.image_height
            )
        elif args.use_onboard:
            self.cap = open_cam_onboard(
                args.image_width,
                args.image_height
            )
        else:
            # by default
            self.cap = open_default_cam()

        if self.cap != 'OK':
            if self.cap.isOpened():
                # Try to grab the 1st image and determine width and height
                _, img = self.cap.read()
                if img is not None:
                    self.img_height, self.img_width, _ = img.shape
                    self.is_opened = True

    def start(self):
        assert not self.thread_running
        self.thread_running = True
        self.thread = threading.Thread(target=grab_img, args=(self,))
        self.thread.start()

    def stop(self):
        self.thread_running = False
        self.thread.join()

    def read(self):
        if self.args.use_image:
            return np.copy(self.img_handle)
        else:
            return self.img_handle

    def release(self):
        assert not self.thread_running
        if self.cap != 'OK':
            self.cap.release()

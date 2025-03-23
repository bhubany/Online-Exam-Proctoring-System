import cv2, math
import numpy as np
from mtcnn.mtcnn import MTCNN
from numpy import asarray

detector = MTCNN()  

def detect_face(frame):
    pixels = asarray(frame) 
    faces=detector.detect_faces(pixels)
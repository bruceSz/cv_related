#!/usr/bin/env python

import cv2 as cv
from matplotlib import  pyplot as plt
import os
print (os.listdir("."))
print(os.path.exists("./data/test.png"))

src = cv.imread("./data/test.png")
cv.imshow("yuantu",src)

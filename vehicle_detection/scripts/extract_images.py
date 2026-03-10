#!/usr/bin/env python3

import rosbag
import cv2
import numpy as np

bag = rosbag.Bag('/home/ubuntu/rosbag.bag')

count = 0

for topic, msg, t in bag.read_messages(topics=['/hikcamera/image_2/compressed']):

    np_arr = np.frombuffer(msg.data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    filename = f"/home/ubuntu/extracted/frame_{count:05d}.png"
    cv2.imwrite(filename, image)

    count += 1

print("Total images extracted:", count)

bag.close()

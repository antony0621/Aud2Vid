import numpy as np
import cv2
import os
import json

img_h, img_w = 128, 128
results = dict()

frames_root = 'MUSIC_frame/'
frames_class_list = os.listdir(frames_root)

for frame_class in frames_class_list:
    img_class_path = os.path.join(frames_root, frame_class)
    mp4_name_list = os.listdir(img_class_path)
    len_ = len(mp4_name_list)
    for mp4_name in mp4_name_list:
        mp4_path = os.path.join(img_class_path, mp4_name)
        mp4_frames_list = os.listdir(mp4_path)

        means, stddevs = [], []
        frame_list = []
        results[mp4_name] = dict()

        i = 1
        for frame_name in mp4_frames_list:
            frame_path = os.path.join(mp4_path, frame_name)
            img = cv2.imread(frame_path)  # 128 x 128
            img = img[:, :, :, np.newaxis]
            frame_list.append(img)
            i += 1

        total_num = i
        frames = np.concatenate(frame_list, axis=3)
        frames = frames.astype(np.float32) / 255.

        for i in range(3):
            pixels = frames[:, :, i, :].ravel()
            means.append(np.mean(pixels))
            stddevs.append(np.std(pixels))

        means.reverse()
        stddevs.reverse()

        results[mp4_name]["number"] = total_num
        results[mp4_name]["mean"] = means
        results[mp4_name]["stddev"] = stddevs

        print(frame_class, mp4_name, total_num, means, stddevs)

total_mean = np.array([0., 0., 0.])
total_std = np.array([0., 0., 0.])
total_num = 0

# save the dict data
json_tag = json.dumps(results, indent=4)
with open("results.json", "w") as json_file:
    json_file.write(json_tag)

for result in results.values():
    total_num += result["number"]
    total_mean += result["mean"] * result["number"]
    total_std += result["mean"] * result["number"]

f_means = total_mean / total_num
f_stddevs = total_std / total_num

print("normMean = {}".format(f_means))
print("normStd = {}".format(f_stddevs))


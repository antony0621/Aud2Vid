"""
1. Trim the video into 3-sec clips;
2. Separate the frames and audio;
3. Clip the  frames to 128 x 128, and ensure the players are included the images.
"""

import os
import math
import cv2 as cv
import subprocess
import argparse


def trim_video(video_path, trimmed_video_path):
    """
    1. Change the format of the raw videos to intra-frame coding, and save under the same dir with raw video;
    2. Trim and save under a new dir.
    """
    if not os.path.exists(trimmed_video_path):
        os.mkdir(trimmed_video_path)

    video_class_list = os.listdir(video_path)
    for i, video_class in enumerate(video_class_list):
        print('video class {}: {}'.format(i + 1, video_class))
        video_class_path = os.path.join(video_path, video_class)
        trimmed_class_path = os.path.join(trimmed_video_path, video_class)
        if not os.path.exists(trimmed_class_path):
            os.mkdir(trimmed_class_path)

        video_name_list = os.listdir(video_class_path)
        for video_name in video_name_list:
            video_name_path = os.path.join(video_class_path, video_name)

            # convert the raw to intra
            intra_video_name = video_name.split(".")[0] + "_intra.mp4"
            intra_video_name_path = os.path.join(video_class_path, intra_video_name)
            cmd1 = ["ffmpeg", "-i", video_name_path, "-strict", "-2", "-qscale", "0", "-intra", intra_video_name_path]

            try:
                subprocess.check_call(cmd1)
                print("=" * 100)
                print("Successfully converted the {}".format(video_name))
                print("=" * 100)
            except subprocess.CalledProcessError:
                print("failed to convert the {}!".format(video_name))

            # calculate the duration
            mini_video_duratiion = 3.0
            cap = cv.VideoCapture(intra_video_name_path)
            fps = cap.get(cv.CAP_PROP_FPS)
            num_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
            duration = num_frames / fps
            num_clips = int(duration / mini_video_duratiion)
            # trim the video
            i = 1
            while i <= num_clips:
                trimmed_video_name = intra_video_name.split("_intra.mp4")[0] + "0"*(6-count(i)) + "{}.mp4".format(i)
                trimmed_name_path = os.path.join(trimmed_class_path, trimmed_video_name)
                minute = (i - 1) // 20
                second = (i - 1) % 20 * 3
                start_time = "00:" + "0" * (1-count(minute)) + "{}:".format(minute) + \
                             "0" * (1-count(second)) + "{}".format(second)
                cmd2 = ["ffmpeg", "-ss", start_time, "-t", "00:00:03", "-i",
                        intra_video_name_path, "-vcodec", "copy", "-acodec", "copy", trimmed_name_path]

                subprocess.check_call(cmd2)
                print("=" * 100)
                print("Successfully trim the {}".format(trimmed_video_name))
                print("=" * 100)
                i += 1

            pass


def video2frames(video_path, frames_path):
    """
    Convert the videos into frames according to their ids, and save the correspondent frames.
    """
    if not os.path.exists(frames_path):
        os.mkdir(frames_path)
    video_class_list = os.listdir(video_path)
    for i, video_class in enumerate(video_class_list):
        print('video class {}: {}'.format(i+1, video_class))
        video_class_path = os.path.join(video_path, video_class)
        video_name_list = os.listdir(video_class_path)

        # create class path
        save_class_path = os.path.join(frames_path, video_class)
        if not os.path.exists(save_class_path):
            os.mkdir(save_class_path)

        for video_name in video_name_list:
            video_name_path = os.path.join(video_class_path, video_name)
            capture = cv.VideoCapture(video_name_path)
            fps = capture.get(cv.CAP_PROP_FPS)
            num_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
            width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
            height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
            print('video name:', video_name, '\n', 'fps:', fps, '; number of frames:', num_frames)
            print('width:', width, 'height:', height)

            # create name path
            save_name_path = os.path.join(save_class_path, video_name)
            if not os.path.exists(save_name_path):
                os.mkdir(save_name_path)

            _count = 0
            for j in range(num_frames):
                if _count == num_frames:
                    break

                _, frame = capture.read()  # [height, width, channel]
                # resize and center crop to 128 x 128
                ref_side = min(width, height)
                ratio = ref_side / 128
                resized_frame = cv.resize(frame, (math.ceil(width / ratio), math.ceil(height / ratio)))
                if ref_side == width:
                    long_side = math.ceil(height / ratio)
                    cropped_frame = resized_frame[long_side // 2 - 64: long_side // 2 + 64, :, :]
                else:
                    long_side = math.ceil(width / ratio)
                    cropped_frame = resized_frame[:, long_side // 2 - 64: long_side // 2 + 64, :]

                image_name = '0'*(6-count(j+1)) + '{}'.format(j+1) + '.jpg'

                save_frame_path = os.path.join(save_name_path, image_name)
                if (not os.path.isfile(save_frame_path)) or (os.stat(save_frame_path).st_size == 0):
                    cv.imwrite(save_frame_path, cropped_frame)
                    print("Successfully save frame at {} !".format(save_frame_path))
                else:
                    print("There has been content in the {} !".format(save_frame_path))
                    pass
                _count += 1

            assert len(os.listdir(save_name_path)) == num_frames, print(len(os.listdir(save_name_path)))


def video2sound(video_root_path, sound_root_path):
    """
    Extract the raw sound from the video, and the sampling will be done in data pre-process.
    """
    if not os.path.exists(sound_root_path):
        os.mkdir(sound_root_path)

    video_class_list = os.listdir(video_root_path)
    for i, video_class in enumerate(video_class_list):
        print('video class {}: {}'.format(i + 1, video_class))
        video_class_path = os.path.join(video_root_path, video_class)
        sound_class_path = os.path.join(sound_root_path, video_class)
        if not os.path.exists(sound_class_path):
            os.mkdir(sound_class_path)

        video_name_list = os.listdir(video_class_path)
        for video_name in video_name_list:
            video_name_path = os.path.join(video_class_path, video_name)

            sound_name = video_name.split(".")[0] + ".mp3"
            sound_name_path = os.path.join(sound_class_path, sound_name)

            cmd = ["ffmpeg", "-i", video_name_path, sound_name_path]
            try:
                subprocess.check_call(cmd)
                print("successfully converted the {}".format(sound_name_path))
            except subprocess.CalledProcessError:
                print("failed to convert the {}!".format(sound_name))


def count(x):
    """
    Calculate the digit capacity.
    E.g. 99 --> 2; 103 --> 3; 432431 --> 6
    """
    res = 0
    while x > 0:
        x = x // 10
        res += 1
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser("MUSIC dataset preprocess")
    parser.add_argument('--video_path', type=str, default='MUSIC_dataset/MUSIC21_solo_videos')
    parser.add_argument('--frame_path', type=str, default='MUSIC_frame')
    parser.add_argument('--sound_path', type=str, default='MUSIC_sound')
    parser.add_argument('--trimmed_video_path', type=str, default='MUSIC_trimmed')
    args = parser.parse_args()
    print(str(args), '\n')

    video_path = args.video_path
    frame_path = args.frame_path
    sound_path = args.sound_path
    trimmed_video_path = args.trimmed_video_path

    print("#.Converting to intra-coding ...")
    trim_video(video_path, trimmed_video_path)

    print("#. Video to 128 x 128 Frames")
    video2frames(trimmed_video_path, frame_path)

    print("#. Video to Sound")
    video2sound(trimmed_video_path, sound_path)



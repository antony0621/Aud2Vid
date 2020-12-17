import os
import os.path as osp
import argparse
import json
from multiprocessing import Pool
from tqdm import tqdm
import pytube


global video_output_dir, error_file, video_name
errors = 0


def download_video(youtube_id):
    """Download video from YouTube using PyTube.

    Args:
        youtube_id (str): Youtube id (https://www.youtube.com/watch?v=<youtube_id>)

    """
    global error_file, errors, video_name
    try:
        youtube = pytube.YouTube('https://www.youtube.com/watch?v=' + youtube_id)
        video = youtube.streams.first()
        video_filename = osp.join(video_output_dir, youtube_id + '.' + video.subtype)
        if (not osp.isfile(video_filename)) or (os.stat(video_filename).st_size == 0):
            try:
                video.download(output_path=video_output_dir, filename=youtube_id)
            except:
                with open(error_file, "a") as f:
                    f.write("{}/{}\n".format(video_name, youtube_id))
                    errors += 1
    except:
        with open(error_file, "a") as f:
            f.write("{}/{}\n".format(video_name, youtube_id))
            errors += 1


def main():
    parser = argparse.ArgumentParser("MUSIC dataset downloader")
    parser.add_argument('-v', '--version', type=str, default='21_solo', choices=('21_solo', '11_solo', '11_duet'))
    parser.add_argument('-w', '--workers', type=int, default=None, help="Set number of multiprocessing workers")
    parser.add_argument('--classes', nargs="+", help="Classes to download")
    parser.add_argument('--video_id_dir', type=str, default='MUSIC_id')
    parser.add_argument('--root_dir', type=str, default='MUSIC_dataset', help="Root directory of all the dataset")

    args = parser.parse_args()
    print(str(args), '\n')

    print("#.Download MUSIC{}_videos dataset...".format(args.version))

    # Define dataset root directory name
    root_dir = args.root_dir
    dataset_name = 'MUSIC{}_videos'.format(args.version)
    dataset_dir = osp.join(root_dir, dataset_name)
    if not osp.exists(dataset_dir):
        os.mkdir(dataset_dir)

    # Find the json file that contain the video ids
    video_json_file = osp.join(args.video_id_dir, 'MUSIC{}_videos.json'.format(args.version))

    # Define failed log file
    global video_output_dir, error_file, video_name
    error_file = 'MUSIC{}_failed.log'.format(args.version)
    if osp.exists(error_file):
        os.remove(error_file)

    # Parse URLs json file and get a list of YouTube IDs to download
    print("Parsing URLs json file...")
    json_file = json.load(open(video_json_file, 'r'))
    video_dict = json_file['videos']
    video_names_list = list(video_dict.keys())

    # Define the specific classes to downloaded
    classes_to_download = args.classes  # a list of names

    for video_name in classes_to_download:

        assert video_name in video_names_list, "Invalid class name !"

        video_ids = video_dict[video_name]

        video_output_dir = osp.join(dataset_dir, video_name)

        # Download YouTube videos according to given IDs
        print("\nDownloading videos of {}...".format(video_name))
        pool = Pool(args.workers)
        for _ in tqdm(pool.imap_unordered(download_video, video_ids), total=len(video_ids)):
            pass
        pool.close()


if __name__ == '__main__':
    main()

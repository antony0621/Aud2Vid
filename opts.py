import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root',
        default='MUSIC_dataset',
        type=str,
        help='root dir')
    parser.add_argument(
        '--train_tag_json_path',
        default='json_data/trainset_tag.json',
        type=str,
        help='')
    parser.add_argument(
        '--test_tag_json_path',
        default='json_data/testset_tag.json',
        type=str,
        help='')
    parser.add_argument(
        '--batch_size',
        default=16,
        type=int,
        help='batch size')
    parser.add_argument(
        '--input_channel',
        default=3,
        type=int,
        help='input image channel (3 for RGB, 1 for Grayscale)')
    parser.add_argument(
        '--alpha_recon_image',
        default=0.85,
        type=float,
        help='weight of reconstruction loss.')
    parser.add_argument(
        '--input_size',
        default=(128, 128),
        type=tuple,
        help='input image size')
    parser.add_argument(
        '--num_frames',
        default=5,
        type=int,
        help='number of frames for each video clip')
    parser.add_argument(
        '--num_predicted_frames',
        default=4,
        type=int,
        help='number of frames to predict')
    parser.add_argument(
        '--num_epochs',
        default=1000,
        type=int,
        help=
        'Max. number of epochs to train.'
    )
    parser.add_argument(
        '--lr_rate',
        default=0.001,
        type=float,
        help='learning rate used for training.'
    )
    parser.add_argument(
        '--lamda',
        default=0.1,
        type=float,
        help='weight use to penalize the generated occlusion mask.'
    )
    parser.add_argument(
        '--workers',
        default=8,
        type=int,
        help='number of workers used for data loading.'
    )
    parser.add_argument(
        '--dataset',
        default='cityscapes',
        type=str,
        help=
        'Used dataset (cityscpes | cityscapes_two_path | kth | ucf101).'
    )
    parser.add_argument(
        '--iter_to_load',
        default=1,
        type=int,
        help='iteration to load'
    )
    parser.add_argument(
        '--mask_channel',
        default=20,
        type=int,
        help='channel of the input semantic lable map'
    )
    parser.add_argument(
        '--category',
        default='walking',
        type=str,
        help='class category of the video to train (only apply to KTH and UCF101)'
    )
    parser.add_argument(
        '--seed',
        default=31415,
        type=int,
        help='Manually set random seed'
    )
    parser.add_argument(
        '--suffix',
        default='',
        type=str,
        help='model suffix'
    )
    # visualization config
    parser.add_argument('--name', type=str, default="Aud2Vid")
    parser.add_argument('--visualized', type=bool, default=False)
    parser.add_argument('--display_id', type=int, default=1)
    parser.add_argument('--display_ncols', type=int, default=16,
                        help='if positive, display all images in a single visdom web panel with certain number of '
                             'images per row.')
    parser.add_argument('--display_port', type=int, default=8097, help='visdom display port')
    parser.add_argument('--display_env', type=str, default='main',
                        help='visdom display environment name (default is "main")')
    parser.add_argument('--display_server', type=str, default="http://10.198.8.31",
                        help='visdom server of the web display')
    parser.add_argument('--display_winsize', type=int, default=128, help='display window size')

    args = parser.parse_args()

    return args
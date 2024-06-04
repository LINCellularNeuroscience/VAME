import vame
from vame.util.auxiliary import read_config, write_config
import os
import ast


def run_pipeline(
    working_directory: str,
    project: str,
    videos: list,
    poses_estimations: list
):

    config = vame.init_new_project(
        project=project,
        videos=videos,
        poses_estimations=poses_estimations,
        working_directory=working_directory,
        videotype='.mp4'
    )

    config_values = read_config(config)
    config_values['egocentric_data'] = os.environ.get('EGOCENTRIC_DATA', False)
    config_values['max_epochs'] = os.environ.get('MAX_EPOCHS', 10)
    config_values['batch_size'] = os.environ.get('BATCH_SIZE', 10)
    write_config(config, config_values)

    vame.egocentric_alignment(config, pose_ref_index=[0, 5])
    vame.create_trainset(config, check_parameter=False, pose_ref_index=[0,5])
    vame.train_model(config)
    vame.evaluate_model(config)
    vame.pose_segmentation(config)

    print('Pipeline finished!')


if __name__ == '__main__':
    """
    To run this pipeline using docker use the following commands from the root of the repository:

    docker build -t vame .
    docker run -v ./path/to/your/data/:/vame/sample_data -e VIDEOS="['./sample_data/NAME_OF_YOUR_VIDEO.mp4']" -e POSES="['./sample_data/NAME_OF_YOUR_POSE_ESTIMATION.csv']" --gpus all vame

    The flag --gpus all is optional and is used to enable GPU support. If you don't have a GPU, you can remove this flag.
    """
    working_directory = './'
    project = os.environ.get('PROJECT', 'my-vame-project')
    videos = os.environ.get('VIDEOS', ['./sample_data/cropped_video.mp4'])
    if isinstance(videos, str):
        videos = ast.literal_eval(videos)
    poses_estimations = os.environ.get('POSES', ['./sample_data/cropped_video.csv'])
    if isinstance(poses_estimations, str):
        poses_estimations = ast.literal_eval(poses_estimations)

    run_pipeline(
        working_directory=working_directory,
        project=project,
        videos=videos,
        poses_estimations=poses_estimations
    )
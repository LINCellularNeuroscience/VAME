from pytest import fixture
from vame import init_new_project
from vame.util.auxiliary import read_config, write_config
from pathlib import Path
import shutil


@fixture(scope='session')
def setup_project():
    project = 'test_project'
    videos = ['./tests/initialize_project/tests_project_sample_data/cropped_video.mp4']
    poses_estimations = ['./tests/initialize_project/tests_project_sample_data/cropped_video.csv']
    working_directory = './tests'

    # Initialize project
    config = init_new_project(project=project, videos=videos, poses_estimations=poses_estimations, working_directory=working_directory, videotype='.mp4')

    # Override config values with test values to speed up tests
    config_values = read_config(config)
    config_values['egocentric_data'] = True
    config_values['max_epochs'] = 10
    config_values['batch_size'] = 10
    write_config(config, config_values)

    project_data = {
        "project_name": project,
        "videos": videos,
        "config_path": config
    }

    yield project_data

    # Clean up
    shutil.rmtree(Path(config).parent)
    

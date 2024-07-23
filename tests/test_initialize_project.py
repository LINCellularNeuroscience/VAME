from pathlib import Path
from vame.util.auxiliary import read_config
from vame import init_new_project
import shutil


def test_project_config_exists(setup_project_not_aligned_data):
    """
    Test if the project config file exists.
    """
    assert Path(setup_project_not_aligned_data['config_path']).exists()

def test_project_name_config(setup_project_not_aligned_data):
    """
    Test if the project name is correctly set in the config file.
    """
    config = Path(setup_project_not_aligned_data['config_path'])
    config_values = read_config(config)
    assert config_values['Project'] == setup_project_not_aligned_data['project_name']


def test_existing_project():
    project = 'test_existing_project'
    videos = ['./tests/tests_project_sample_data/cropped_video.mp4']
    poses_estimations = ['./tests/tests_project_sample_data/cropped_video.csv']
    working_directory = './tests'

    config_path_creation = init_new_project(
        project=project,
        videos=videos,
        poses_estimations=poses_estimations,
        working_directory=working_directory
    )
    config_path_duplicated = init_new_project(
        project=project,
        videos=videos,
        poses_estimations=poses_estimations,
        working_directory=working_directory
    )
    assert config_path_creation == config_path_duplicated
    shutil.rmtree(Path(config_path_creation).parent)


def test_existing_project_from_folder(setup_project_from_folder):
    config = Path(setup_project_from_folder['config_path'])
    config_values = read_config(config)
    assert config_values['Project'] == setup_project_from_folder['project_name']
    assert Path(setup_project_from_folder['config_path']).exists()
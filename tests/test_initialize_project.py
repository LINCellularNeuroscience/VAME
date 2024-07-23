from pathlib import Path
from vame.util.auxiliary import read_config
from vame import init_new_project


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


def test_existing_project(setup_project_not_aligned_data):
    init_new_project(
        project=setup_project_not_aligned_data['project_name'],
        videos=setup_project_not_aligned_data['videos'],
        poses_estimations=setup_project_not_aligned_data['videos'],
        working_directory='./tests'
    )


def test_existing_project_from_folder(setup_project_from_folder):
    config = Path(setup_project_from_folder['config_path'])
    config_values = read_config(config)
    assert config_values['Project'] == setup_project_from_folder['project_name']
    assert Path(setup_project_from_folder['config_path']).exists()
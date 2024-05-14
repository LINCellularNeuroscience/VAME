from vame import init_new_project
from unittest.mock import patch
import pytest
from pathlib import Path
from datetime import datetime as dt
import shutil


def mock_shutil_copy(src, dst):
    print(f"Mock copying {src} to {dst}")


pytest_plugins = [
    "initialize_project.fixtures"
]
@pytest.mark.usefixtures("working_dir", "project_name")
def test_initialize_project_output(monkeypatch, working_dir, project_name):
    working_directory = str(working_dir)
    project = project_name
    videos = ['./dummy-video-name.mp4']
    poses_estimations = ['./dummy-pose-name.csv']

    # Mock shutil.copy to avoid copying videos and poses
    monkeypatch.setattr("vame.initialize_project.new.shutil.copy", mock_shutil_copy)

    config_path = init_new_project(
        project=project, 
        videos=videos, 
        poses_estimations=poses_estimations, 
        working_directory=working_directory, 
        videotype='.mp4'
    )

    date = dt.today()
    month = date.strftime("%B")
    project_date = f'{month[0:3]}{date.day}-{date.year}'
    expected_project_name = f'{project}-{project_date}'
    
    expected_config_path = Path(working_directory) / expected_project_name / 'config.yaml'
    assert config_path == str(expected_config_path)
    
    # TODO make this as setup/teardown fixture probably
    shutil.rmtree(str(expected_config_path.parent)) 


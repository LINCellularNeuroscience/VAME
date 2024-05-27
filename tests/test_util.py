import vame
import pytest
import os

@pytest.mark.parametrize('setup_project', [{"egocentric_aligned": True}], indirect=True)
def test_csv_to_numpy_file_exists(setup_project):
    """
    Test if the pose-estimation file was converted to a numpy array file.
    """
    config_path = setup_project['config_path']
    vame.csv_to_numpy(config_path)
    project_path = setup_project['config_data']['project_path']
    file_name = setup_project['config_data']['video_sets'][0]
    file_path = os.path.join(project_path,'data', file_name, f'{file_name}-PE-seq.npy')
    assert os.path.exists(file_path)


def test_egocentric_alignment_file_is_created(setup_project):
    """
    Test if the egocentric alignment function creates the expected file.
    """
    config_path = setup_project['config_path']
    vame.egocentric_alignment(config_path, pose_ref_index=setup_project["pose_ref_index"])
    project_path = setup_project['config_data']['project_path']
    file_name = setup_project['config_data']['video_sets'][0]
    file_path = os.path.join(project_path,'data', file_name, f'{file_name}-PE-seq.npy')
    assert os.path.exists(file_path)






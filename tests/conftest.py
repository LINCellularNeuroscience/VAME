from pytest import fixture
import vame
from vame.util.auxiliary import read_config, write_config
from pathlib import Path
import shutil


def init_project(
    project: str,
    videos: list,
    poses_estimations: list,
    working_directory: str,
    egocentric_data: bool = False,
    parametrization: str = 'hmm'
):
    config = vame.init_new_project(project=project, videos=videos, poses_estimations=poses_estimations, working_directory=working_directory, videotype='.mp4')

    # Override config values with test values to speed up tests
    config_values = read_config(config)
    config_values['egocentric_data'] = egocentric_data
    config_values['max_epochs'] = 10
    config_values['batch_size'] = 10
    config_values['parametrization'] = parametrization
    write_config(config, config_values)

    project_data = {
        "project_name": project,
        "videos": videos,
        "config_path": config,
        "config_data": config_values,
        "pose_ref_index": [0, 5]
    }

    return config, project_data



@fixture(scope='session')
def setup_project_not_aligned_data():
    project = 'test_project_align'
    videos = ['./tests/tests_project_sample_data/cropped_video.mp4']
    poses_estimations = ['./tests/tests_project_sample_data/cropped_video.csv']
    working_directory = './tests'

    # Initialize project
    config, project_data = init_project(project, videos, poses_estimations, working_directory, egocentric_data=False)

    yield project_data

    # Clean up
    shutil.rmtree(Path(config).parent)

@fixture(scope='session')
def setup_project_fixed_data():
    project = 'test_project_fixed'
    videos = ['./tests/tests_project_sample_data/cropped_video.mp4'] # TODO change to test fixed data when have it
    poses_estimations = ['./tests/tests_project_sample_data/cropped_video.csv']
    working_directory = './tests'

    # Initialize project
    config, project_data = init_project(project, videos, poses_estimations, working_directory, egocentric_data=True)

    yield project_data

    # Clean up
    shutil.rmtree(Path(config).parent)


@fixture(scope='session')
def setup_project_and_convert_csv_to_numpy(setup_project_fixed_data):
    config_path = setup_project_fixed_data['config_path']
    vame.csv_to_numpy(config_path, save_logs=True)
    return setup_project_fixed_data

@fixture(scope='session')
def setup_project_and_align_egocentric(setup_project_not_aligned_data):
    config_path = setup_project_not_aligned_data['config_path']
    vame.egocentric_alignment(
        config_path,
        pose_ref_index=setup_project_not_aligned_data["pose_ref_index"],
        save_logs=True,
    )
    return setup_project_not_aligned_data


@fixture(scope='function')
def setup_project_and_check_param_aligned_dataset(setup_project_and_align_egocentric):
    config = setup_project_and_align_egocentric['config_path']
    vame.create_trainset(
        config,
        check_parameter=True,
        pose_ref_index=setup_project_and_align_egocentric["pose_ref_index"],
        save_logs=True,
    )
    return setup_project_and_align_egocentric

@fixture(scope='function')
def setup_project_and_check_param_fixed_dataset(setup_project_and_convert_csv_to_numpy):
    # use setup_project_and_align_egocentric fixture or setup_project_and_convert_csv_to_numpy based on value of egocentric_aligned
    config = setup_project_and_convert_csv_to_numpy['config_path']
    vame.create_trainset(
        config,
        check_parameter=True,
        pose_ref_index=setup_project_and_convert_csv_to_numpy["pose_ref_index"],
        save_logs=True,
    )
    return setup_project_and_convert_csv_to_numpy


@fixture(scope='session')
def setup_project_and_create_train_aligned_dataset(setup_project_and_align_egocentric):
    config = setup_project_and_align_egocentric['config_path']
    vame.create_trainset(
        config,
        check_parameter=False,
        pose_ref_index=setup_project_and_align_egocentric["pose_ref_index"],
        save_logs=True,
    )
    return setup_project_and_align_egocentric


@fixture(scope='session')
def setup_project_and_create_train_fixed_dataset(setup_project_and_convert_csv_to_numpy):
    # use setup_project_and_align_egocentric fixture or setup_project_and_convert_csv_to_numpy based on value of egocentric_aligned
    config = setup_project_and_convert_csv_to_numpy['config_path']
    vame.create_trainset(
        config,
        check_parameter=False,
        pose_ref_index=setup_project_and_convert_csv_to_numpy["pose_ref_index"],
        save_logs=True,
    )
    return setup_project_and_convert_csv_to_numpy



@fixture(scope='session')
def setup_project_and_train_model(setup_project_and_create_train_aligned_dataset):
    config = setup_project_and_create_train_aligned_dataset['config_path']
    vame.train_model(config, save_logs=True)
    return setup_project_and_create_train_aligned_dataset

@fixture(scope='session')
def setup_project_and_evaluate_model(setup_project_and_train_model):
    config = setup_project_and_train_model['config_path']
    vame.evaluate_model(config, save_logs=True)
    return setup_project_and_train_model


@fixture(scope='session')
def setup_kmeans_project():
    project = 'test_kmeans_project'
    videos = ['./tests/tests_project_sample_data/cropped_video.mp4']
    poses_estimations = ['./tests/tests_project_sample_data/cropped_video.csv']
    working_directory = './tests'

    # Initialize project
    config, project_data = init_project(project, videos, poses_estimations, working_directory, egocentric_data=False, parametrization='kmeans')

    yield project_data

    # Clean up
    shutil.rmtree(Path(config).parent)


@fixture(scope='session')
def setup_kmeans_project_and_pose_segmentation(setup_kmeans_project):
    config_path = setup_kmeans_project["config_path"]
    vame.egocentric_alignment(
        config_path,
        pose_ref_index=setup_kmeans_project["pose_ref_index"],
        save_logs=True,
    )
    vame.create_trainset(
        config_path,
        check_parameter=False,
        pose_ref_index=setup_kmeans_project["pose_ref_index"],
        save_logs=True,
    )
    vame.train_model(config_path, save_logs=True)
    vame.evaluate_model(config_path, save_logs=True)
    vame.pose_segmentation(config_path, save_logs=True)
    return setup_kmeans_project


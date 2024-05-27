from pytest import fixture
import vame
from vame.util.auxiliary import read_config, write_config
from pathlib import Path
import shutil

def pytest_collection_modifyitems(items):
    """Modifies test items in place to ensure test modules run in a given order.
    We are using this because these are integration tests and we need to run them in a specific order to avoid errors.
    """
    MODULE_ORDER = [
        "test_initialize_project",
        "test_util",
        "test_model",
        "test_analysis"
    ]
    module_mapping = {item: item.module.__name__ for item in items}
    sorted_items = items.copy()
    # Iteratively move tests of each module to the end of the test queue
    for module in MODULE_ORDER:
        sorted_items = [it for it in sorted_items if module_mapping[it] != module] + [
            it for it in sorted_items if module_mapping[it] == module
        ]
    items[:] = sorted_items

@fixture(scope='session')
def setup_project(request):
    project = 'test_project'
    videos = ['./tests/tests_project_sample_data/cropped_video.mp4']
    poses_estimations = ['./tests/tests_project_sample_data/cropped_video.csv']
    working_directory = './tests'

    # Initialize project
    config = vame.init_new_project(project=project, videos=videos, poses_estimations=poses_estimations, working_directory=working_directory, videotype='.mp4')

    egocentric_aligned = False
    if hasattr(request, 'param'):
        egocentric_aligned = request.param.get('egocentric_aligned', False)

    # Override config values with test values to speed up tests
    config_values = read_config(config)
    config_values['egocentric_data'] = egocentric_aligned
    config_values['max_epochs'] = 10
    config_values['batch_size'] = 10
    write_config(config, config_values)

    project_data = {
        "project_name": project,
        "videos": videos,
        "config_path": config,
        "config_data": config_values,
        "pose_ref_index": [0, 5]
    }

    yield project_data

    # Clean up
    shutil.rmtree(Path(config).parent)


@fixture(scope='session')
def setup_project_and_create_train_dataset(setup_project):
    config = setup_project['config_path']
    vame.create_trainset(config, check_parameter=False, pose_ref_index=setup_project["pose_ref_index"])
    return setup_project

@fixture(scope='session')
def setup_project_and_train_model(setup_project_and_create_train_dataset):
    config = setup_project_and_create_train_dataset['config_path']
    vame.train_model(config)
    return setup_project_and_create_train_dataset

@fixture(scope='session')
def setup_project_and_evaluate_model(setup_project_and_train_model):
    config = setup_project_and_train_model['config_path']
    vame.evaluate_model(config)
    return setup_project_and_train_model
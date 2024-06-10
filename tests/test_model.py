import vame
from pathlib import Path
import os
import pytest


@pytest.mark.parametrize(
    'fixture',
    [
        'setup_project_and_create_train_aligned_dataset',
        'setup_project_and_create_train_fixed_dataset'
    ]
)
def test_create_train_dataset_output_files_exists(request, fixture):
    """
    Test if the create_trainset function creates the correct output files.
    """
    fixture_value = request.getfixturevalue(fixture)
    project_path = fixture_value['config_data']['project_path']
    train_data_path = os.path.join(project_path, "data", "train", "test_seq.npy")
    test_data_path = os.path.join(project_path, "data", "train", "test_seq.npy")

    assert os.path.exists(train_data_path)
    assert os.path.exists(test_data_path)


@pytest.mark.parametrize(
    'fixture',
    [
        'setup_project_and_check_param_aligned_dataset',
        'setup_project_and_check_param_fixed_dataset'
    ]
)
def test_create_check_param_train_dataset(request, fixture):
    """
    Test if the create_trainset executes without errors when check_parameter is set to True.
    """
    request.getfixturevalue(fixture)
    assert True


def test_train_model_losses_files_exists(setup_project_and_train_model):
    """
    Test if the train_model function creates the correct losses files.
    """
    project_path = setup_project_and_train_model['config_data']['project_path']
    model_name = setup_project_and_train_model['config_data']['model_name']

    # save logged losses
    train_losses_path = Path(project_path) / "model" / "model_losses" / f"train_losses_{model_name}.npy"
    test_losses_path = Path(project_path) / "model" / "model_losses" / f"test_losses_{model_name}.npy"
    kmeans_losses_path = Path(project_path) / "model" / "model_losses" / f"kmeans_losses_{model_name}.npy"
    kl_losses_path = Path(project_path) / "model" / "model_losses" / f"kl_losses_{model_name}.npy"
    weight_values_path = Path(project_path) / "model" / "model_losses" / f"weight_values_{model_name}.npy"
    mse_train_losses_path = Path(project_path) / "model" / "model_losses" / f"mse_train_losses_{model_name}.npy"
    mse_test_losses_path = Path(project_path) / "model" / "model_losses" / f"mse_test_losses_{model_name}.npy"
    fut_losses_path = Path(project_path) / "model" / "model_losses" / f"fut_losses_{model_name}.npy"

    assert train_losses_path.exists()
    assert test_losses_path.exists()
    assert kmeans_losses_path.exists()
    assert kl_losses_path.exists()
    assert weight_values_path.exists()
    assert mse_train_losses_path.exists()
    assert mse_test_losses_path.exists()
    assert fut_losses_path.exists()


def test_train_model_best_model_file_exists(setup_project_and_train_model):
    project_path = setup_project_and_train_model['config_data']['project_path']
    model_name = setup_project_and_train_model['config_data']['model_name']
    project_name = setup_project_and_train_model['config_data']['Project']
    best_model_path = Path(project_path) / "model" / "best_model" / f"{model_name}_{project_name}.pkl"

    assert best_model_path.exists()


def test_evaluate_model_images_exists(setup_project_and_evaluate_model):
    project_path = setup_project_and_evaluate_model['config_data']['project_path']
    model_name = setup_project_and_evaluate_model['config_data']['model_name']
    reconstruction_image_path = Path(project_path) / "model" / "evaluate" / "Future_Reconstruction.png"
    loss_image_path = Path(project_path) / "model" / "evaluate" / f'MSE-and-KL-Loss{model_name}.png'

    assert reconstruction_image_path.exists()
    assert loss_image_path.exists()


def test_train_legacy_model_losses_exists(setup_legacy_project_and_train_model):
    """
    Test if the train_model function creates the correct losses files.
    """
    project_path = setup_legacy_project_and_train_model['config_data']['project_path']
    model_name = setup_legacy_project_and_train_model['config_data']['model_name']

    # save logged losses
    train_losses_path = Path(project_path) / "model" / "model_losses" / f"train_losses_{model_name}.npy"
    test_losses_path = Path(project_path) / "model" / "model_losses" / f"test_losses_{model_name}.npy"
    kmeans_losses_path = Path(project_path) / "model" / "model_losses" / f"kmeans_losses_{model_name}.npy"
    kl_losses_path = Path(project_path) / "model" / "model_losses" / f"kl_losses_{model_name}.npy"
    weight_values_path = Path(project_path) / "model" / "model_losses" / f"weight_values_{model_name}.npy"
    mse_train_losses_path = Path(project_path) / "model" / "model_losses" / f"mse_train_losses_{model_name}.npy"
    mse_test_losses_path = Path(project_path) / "model" / "model_losses" / f"mse_test_losses_{model_name}.npy"
    fut_losses_path = Path(project_path) / "model" / "model_losses" / f"fut_losses_{model_name}.npy"

    assert train_losses_path.exists()
    assert test_losses_path.exists()
    assert kmeans_losses_path.exists()
    assert kl_losses_path.exists()
    assert weight_values_path.exists()
    assert mse_train_losses_path.exists()
    assert mse_test_losses_path.exists()
    assert fut_losses_path.exists()


def test_train_model_best_model_file_exists(setup_legacy_project_and_train_model):
    project_path = setup_legacy_project_and_train_model['config_data']['project_path']
    model_name = setup_legacy_project_and_train_model['config_data']['model_name']
    project_name = setup_legacy_project_and_train_model['config_data']['Project']
    best_model_path = Path(project_path) / "model" / "best_model" / f"{model_name}_{project_name}.pkl"

    assert best_model_path.exists()
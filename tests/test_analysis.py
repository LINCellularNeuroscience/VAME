from pathlib import Path
import vame
import pytest
from matplotlib.figure import Figure
from unittest.mock import patch
from vame.util.gif_pose_helper import background


def test_pose_segmentation_files_exists(setup_project_and_train_model):
    # Check if the files are created
    vame.pose_segmentation(setup_project_and_train_model['config_path'])
    project_path = setup_project_and_train_model['config_data']['project_path']
    file = setup_project_and_train_model['config_data']['video_sets'][0]
    model_name = setup_project_and_train_model['config_data']['model_name']
    n_cluster = setup_project_and_train_model['config_data']['n_cluster']
    parametrization = setup_project_and_train_model['config_data']['parametrization']

    save_base_path = Path(project_path) / "results" / file / model_name / f"{parametrization}-{n_cluster}"
    latent_vector_path = save_base_path / f"latent_vector_{file}.npy"
    motif_usage_path = save_base_path / f"motif_usage_{file}.npy"

    assert latent_vector_path.exists()
    assert motif_usage_path.exists()

def test_motif_videos_files_exists(setup_project_and_train_model):
    # Check if the files are created
    vame.motif_videos(setup_project_and_train_model['config_path'])
    project_path = setup_project_and_train_model['config_data']['project_path']
    file = setup_project_and_train_model['config_data']['video_sets'][0]
    model_name = setup_project_and_train_model['config_data']['model_name']
    n_cluster = setup_project_and_train_model['config_data']['n_cluster']
    parametrization = setup_project_and_train_model['config_data']['parametrization']

    save_base_path = Path(project_path) / "results" / file / model_name / f"{parametrization}-{n_cluster}" / "cluster_videos"

    assert len(list(save_base_path.glob("*.mp4"))) > 0
    assert len(list(save_base_path.glob("*.mp4"))) <= n_cluster

def test_community_files_exists(setup_project_and_train_model):
    # Check if the files are created
    vame.community(
        setup_project_and_train_model['config_path'],
        show_umap=False,
        cut_tree=2,
        cohort=False
    )
    project_path = setup_project_and_train_model['config_data']['project_path']
    file = setup_project_and_train_model['config_data']['video_sets'][0]
    model_name = setup_project_and_train_model['config_data']['model_name']
    n_cluster = setup_project_and_train_model['config_data']['n_cluster']
    parametrization = setup_project_and_train_model['config_data']['parametrization']

    save_base_path = Path(project_path) / "results" / file / model_name / f"{parametrization}-{n_cluster}" / 'community'

    transition_matrix_path = save_base_path / f"transition_matrix_{file}.npy"
    community_label_path = save_base_path / f"community_label_{file}.npy"
    hierarchy_path = save_base_path / f"hierarchy{file}.pkl"

    assert transition_matrix_path.exists()
    assert community_label_path.exists()
    assert hierarchy_path.exists()


def test_cohort_community_files_exists(setup_project_and_train_model):
    # Check if the files are created
    vame.community(
        setup_project_and_train_model['config_path'],
        show_umap=False,
        cut_tree=2,
        cohort=True
    )
    project_path = setup_project_and_train_model['config_data']['project_path']
    parametrization = setup_project_and_train_model['config_data']['parametrization']

    cohort_path = Path(project_path) /  "cohort_transition_matrix.npy"
    community_path = Path(project_path) /  "cohort_community_label.npy"
    cohort_parametrization_path = Path(project_path) /  f"cohort_{parametrization}_label.npy"
    cohort_community_bag_path = Path(project_path) /  "cohort_community_bag.npy"

    assert cohort_path.exists()
    assert community_path.exists()
    assert cohort_parametrization_path.exists()
    assert cohort_community_bag_path.exists()


def test_community_videos_files_exists(setup_project_and_train_model):

    vame.community_videos(
        config=setup_project_and_train_model['config_path']
    )
    file = setup_project_and_train_model['config_data']['video_sets'][0]
    model_name = setup_project_and_train_model['config_data']['model_name']
    n_cluster = setup_project_and_train_model['config_data']['n_cluster']
    parametrization = setup_project_and_train_model['config_data']['parametrization']
    project_path = setup_project_and_train_model['config_data']['project_path']

    save_base_path = Path(project_path) / "results" / file / model_name / f"{parametrization}-{n_cluster}" / "community_videos"

    assert len(list(save_base_path.glob("*.mp4"))) > 0
    assert len(list(save_base_path.glob("*.mp4"))) <= n_cluster


def test_visualization_output_type(setup_project_and_train_model):
    fig = vame.visualization(setup_project_and_train_model['config_path'], label=None)
    assert isinstance(fig, Figure)


@pytest.mark.parametrize("mode", ["sampling", "reconstruction", "motifs"])
def test_generative_model(setup_project_and_train_model, mode):
    # skipping mode=centers by now because it requires a cluster_center.npy file which is created only with parametrization = "kmeans"
    generative_figure = vame.generative_model(
        config=setup_project_and_train_model['config_path'],
        mode=mode
    )
    assert isinstance(generative_figure, Figure)


def test_gif(setup_project_and_train_model):
    def mock_background(path_to_file=None, filename=None, file_format=None, num_frames=None):
        num_frames = 100
        return background(path_to_file, filename, file_format, num_frames)

    with patch("vame.util.gif_pose_helper.background", side_effect=mock_background):
        vame.gif(
            config=setup_project_and_train_model['config_path'],
            pose_ref_index=[0,5],
            subtract_background=True,
            start=None,
            length=500,
            max_lag=30,
            label="motif",
            file_format='.mp4',
            crop_size=(300,300)
        )
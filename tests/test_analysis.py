from pathlib import Path
import vame
import pytest


def test_pose_segmentation_files_exists(setup_project_and_train_model):
    # Check if the files are created
    vame.pose_segmentation(setup_project_and_train_model['config_path'])
    project_path = setup_project_and_train_model['config_data']['project_path']
    file = setup_project_and_train_model['config_data']['video_sets'][0]
    model_name = setup_project_and_train_model['config_data']['model_name']
    n_cluster = setup_project_and_train_model['config_data']['n_cluster']
    parametrization = setup_project_and_train_model['config_data']['parameterization']

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
    parametrization = setup_project_and_train_model['config_data']['parameterization']

    save_base_path = Path(project_path) / "results" / file / model_name / f"{parametrization}-{n_cluster}" / "cluster_videos"
    # all_paths = []
    # for i in range(n_cluster):
    #     cluster_video_path = save_base_path / f"{file}-motif_{i}.avi"
    #     all_paths.append(cluster_video_path)

    assert len(list(save_base_path.glob("*.avi"))) == n_cluster

@pytest.mark.skip(reason="The method is not working yet.")
def test_community_files_exists(setup_project_and_train_model):
    # Check if the files are created
    vame.community(
        setup_project_and_train_model['config_path'],
        show_umap=False,
        cut_tree=2
    )
    project_path = setup_project_and_train_model['config_data']['project_path']
    parametrization = setup_project_and_train_model['config_data']['parameterization']

    cohort_path = Path(project_path) /  "cohort_transition_matrix.npy"
    community_path = Path(project_path) /  "cohort_community_label.npy"
    cohort_parametrization_path = Path(project_path) /  f"cohort_{parametrization}_label.npy"
    cohort_community_bag_path = Path(project_path) /  "cohort_community_bag.npy"

    assert cohort_path.exists()
    assert community_path.exists()
    assert cohort_parametrization_path.exists()
    assert cohort_community_bag_path.exists()



# def test_community_videos_files_exists(setup_project_and_train_model):
#     vame.community_videos(setup_project_and_train_model['config_path'])
#     project_path = setup_project_and_train_model['config_data']['project_path']
#     file = setup_project_and_train_model['config_data']['video_sets'][0]
#     model_name = setup_project_and_train_model['config_data']['model_name']
#     n_cluster = setup_project_and_train_model['config_data']['n_cluster']
#     parametrization = setup_project_and_train_model['config_data']['parameterization']

#     save_base_path = Path(project_path) / "results" / file / model_name / f"{parametrization}-{n_cluster}" / "community_videos"
#     assert len(list(save_base_path.glob("*.avi"))) == n_cluster
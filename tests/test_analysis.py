from pathlib import Path
import vame
import pytest
from matplotlib.figure import Figure
from unittest.mock import patch
from vame.util.gif_pose_helper import background


@pytest.mark.parametrize(
    'individual_parametrization,parametrization',
    [(True, 'hmm'), (False, 'hmm'), (True, 'kmeans'), (False, 'kmeans')])
def test_pose_segmentation_hmm_files_exists(setup_project_and_train_model, individual_parametrization, parametrization):
    mock_config = {**setup_project_and_train_model['config_data'], 'individual_parametrization': individual_parametrization}
    with patch("vame.util.auxiliary.read_config", return_value=mock_config):
        with patch('builtins.input', return_value='yes'):
            vame.pose_segmentation(setup_project_and_train_model['config_path'])

    project_path = setup_project_and_train_model['config_data']['project_path']
    file = setup_project_and_train_model['config_data']['video_sets'][0]
    model_name = setup_project_and_train_model['config_data']['model_name']
    n_cluster = setup_project_and_train_model['config_data']['n_cluster']
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

    hmm_save_base_path = Path(project_path) / "results" / file / model_name / f"hmm-{n_cluster}" / "cluster_videos"
    hmm_save_base_path = Path(project_path) / "results" / file / model_name / f"kmeans-{n_cluster}" / "cluster_videos"

    assert len(list(hmm_save_base_path.glob("*.mp4"))) > 0
    assert len(list(hmm_save_base_path.glob("*.mp4"))) <= n_cluster

    assert len(list(hmm_save_base_path.glob("*.mp4"))) > 0
    assert len(list(hmm_save_base_path.glob("*.mp4"))) <= n_cluster

@pytest.mark.parametrize("show_umap", [True, False])
def test_community_files_exists(setup_project_and_train_model, show_umap):
    # Check if the files are created
    vame.community(
        setup_project_and_train_model['config_path'],
        show_umap=show_umap,
        save_umap_figure=show_umap,
        cut_tree=2,
        cohort=False
    )
    project_path = setup_project_and_train_model['config_data']['project_path']
    file = setup_project_and_train_model['config_data']['video_sets'][0]
    model_name = setup_project_and_train_model['config_data']['model_name']
    n_cluster = setup_project_and_train_model['config_data']['n_cluster']

    hmm_save_base_path = Path(project_path) / "results" / file / model_name / f"hmm-{n_cluster}" / 'community'
    kmeans_save_base_path = Path(project_path) / "results" / file / model_name / f"kmeans-{n_cluster}" / 'community'

    hmm_transition_matrix_path = hmm_save_base_path / f"transition_matrix_{file}.npy"
    hmm_community_label_path = hmm_save_base_path / f"community_label_{file}.npy"
    hmm_hierarchy_path = hmm_save_base_path / f"hierarchy{file}.pkl"
    kmeans_transition_matrix_path = kmeans_save_base_path / f"transition_matrix_{file}.npy"
    kmeans_community_label_path = kmeans_save_base_path / f"community_label_{file}.npy"
    kmeans_hierarchy_path = kmeans_save_base_path / f"hierarchy{file}.pkl"

    assert hmm_transition_matrix_path.exists()
    assert hmm_community_label_path.exists()
    assert hmm_hierarchy_path.exists()
    assert kmeans_transition_matrix_path.exists()
    assert kmeans_community_label_path.exists()
    assert kmeans_hierarchy_path.exists()

    if show_umap:
        hmm_umap_save_path = hmm_save_base_path / f'{file}_umap.png'
        kmean_umap_save_path = kmeans_save_base_path / f'{file}_umap.png'
        assert hmm_umap_save_path.exists()
        assert kmean_umap_save_path.exists()


def test_cohort_community_files_exists(setup_project_and_train_model):
    # Check if the files are created
    vame.community(
        setup_project_and_train_model['config_path'],
        show_umap=False,
        cut_tree=2,
        cohort=True,
        save_logs=True
    )
    project_path = setup_project_and_train_model['config_data']['project_path']

    cohort_path = Path(project_path) /  "cohort_transition_matrix.npy"
    community_path = Path(project_path) /  "cohort_community_label.npy"
    hmm_cohort_parametrization_path = Path(project_path) /  f"cohort_hmm_label.npy"
    kmean_cohort_parametrization_path = Path(project_path) /  f"cohort_kmeans_label.npy"
    cohort_community_bag_path = Path(project_path) /  "cohort_community_bag.npy"

    assert cohort_path.exists()
    assert community_path.exists()
    assert hmm_cohort_parametrization_path.exists()
    assert kmean_cohort_parametrization_path.exists()
    assert cohort_community_bag_path.exists()


def test_community_videos_files_exists(setup_project_and_train_model):

    vame.community_videos(
        config=setup_project_and_train_model['config_path'],
        save_logs=True
    )
    file = setup_project_and_train_model['config_data']['video_sets'][0]
    model_name = setup_project_and_train_model['config_data']['model_name']
    n_cluster = setup_project_and_train_model['config_data']['n_cluster']
    project_path = setup_project_and_train_model['config_data']['project_path']

    hmm_save_base_path = Path(project_path) / "results" / file / model_name / f"hmm-{n_cluster}" / "community_videos"
    kmeans_save_base_path = Path(project_path) / "results" / file / model_name / f"kmeans-{n_cluster}" / "community_videos"

    assert len(list(hmm_save_base_path.glob("*.mp4"))) > 0
    assert len(list(hmm_save_base_path.glob("*.mp4"))) <= n_cluster

    assert len(list(kmeans_save_base_path.glob("*.mp4"))) > 0
    assert len(list(kmeans_save_base_path.glob("*.mp4"))) <= n_cluster


@pytest.mark.parametrize("label", [None, "motif", "community"])
def test_visualization_output_type(setup_project_and_train_model, label):
    figures = vame.visualization(setup_project_and_train_model['config_path'], label=label, save_logs=True)
    assert isinstance(figures, dict)
    assert isinstance(figures['hmm'], Figure)
    assert isinstance(figures['kmeans'], Figure)


@pytest.mark.parametrize(
    "mode,parametrization",
    [
        ("sampling", 'hmm'), ("reconstruction", "hmm"), ("motifs", "hmm"),
        ("sampling", 'kmeans'), ("reconstruction", "kmeans"), ("motifs", "kmeans"), ("centers", "kmeans")
    ]
)
def test_generative_model_figures(setup_project_and_train_model, mode, parametrization):
    generative_figure = vame.generative_model(
        config=setup_project_and_train_model['config_path'],
        parametrization=parametrization,
        mode=mode,
        save_logs=True
    )
    assert isinstance(generative_figure, Figure)


@pytest.mark.parametrize("label", [None, 'community', 'motif'])
def test_gif_frames_files_exists(setup_project_and_evaluate_model, label):

    with patch("builtins.input", return_value="yes"):
        vame.pose_segmentation(setup_project_and_evaluate_model["config_path"])

    def mock_background(path_to_file=None, filename=None, file_format=None, num_frames=None, save_background=True):
        num_frames = 100
        return background(path_to_file, filename, file_format, num_frames, save_background)

    vame.community(
        setup_project_and_evaluate_model["config_path"],
        show_umap=False,
        cut_tree=2,
        cohort=False,
        save_logs=False
    )
    vame.visualization(
        setup_project_and_evaluate_model["config_path"], label=label, save_logs=False
    )
    VIDEO_LEN = 30
    PARAMETRIZATION = 'hmm'
    with patch("vame.util.gif_pose_helper.background", side_effect=mock_background):
        vame.gif(
            config=setup_project_and_evaluate_model["config_path"],
            param=PARAMETRIZATION,
            pose_ref_index=[0, 5],
            subtract_background=True,
            start=None,
            length=VIDEO_LEN,
            max_lag=30,
            label=label,
            file_format=".mp4",
            crop_size=(300, 300),
        )

    # path_to_file=os.path.join(cfg['project_path'],"results",file,model_name,param+'-'+str(n_cluster),"")
    video = setup_project_and_evaluate_model["config_data"]["video_sets"][0]
    model_name = setup_project_and_evaluate_model["config_data"]["model_name"]
    n_cluster = setup_project_and_evaluate_model["config_data"]["n_cluster"]

    save_base_path = Path(setup_project_and_evaluate_model["config_data"]["project_path"]) / "results" / video / model_name / f'{PARAMETRIZATION}-{n_cluster}'

    gif_frames_path = save_base_path / "gif_frames"
    assert len(list(gif_frames_path.glob("*.png"))) == VIDEO_LEN
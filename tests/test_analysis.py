from pathlib import Path
import vame
import pytest
from matplotlib.figure import Figure
from unittest.mock import patch
from vame.util.gif_pose_helper import background


@pytest.mark.parametrize(
    'individual_parametrization,parametrization,hmm_trained',
    [(True, 'hmm', False), (False, 'hmm', False), (False, 'hmm', True), (True, 'kmeans', False), (False, 'kmeans', False)])
def test_pose_segmentation_hmm_files_exists(setup_project_and_train_model, individual_parametrization, parametrization, hmm_trained):
    mock_config = {**setup_project_and_train_model['config_data'], 'individual_parametrization': individual_parametrization}
    mock_config['hmm_trained'] = hmm_trained
    with patch("vame.analysis.pose_segmentation.read_config", return_value=mock_config) as mock_read_config:
        with patch('builtins.input', return_value='yes'):
            vame.pose_segmentation(setup_project_and_train_model['config_path'], save_logs=True)
    project_path = setup_project_and_train_model['config_data']['project_path']
    file = setup_project_and_train_model['config_data']['video_sets'][0]
    model_name = setup_project_and_train_model['config_data']['model_name']
    n_cluster = setup_project_and_train_model['config_data']['n_cluster']
    save_base_path = Path(project_path) / "results" / file / model_name / f"{parametrization}-{n_cluster}"
    latent_vector_path = save_base_path / f"latent_vector_{file}.npy"
    motif_usage_path = save_base_path / f"motif_usage_{file}.npy"

    assert latent_vector_path.exists()
    assert motif_usage_path.exists()


@pytest.mark.parametrize('parametrization', ['hmm', 'kmeans'])
def test_motif_videos_mp4_files_exists(setup_project_and_train_model, parametrization):
    vame.motif_videos(setup_project_and_train_model['config_path'], parametrization=parametrization, output_video_type='.mp4', save_logs=True)
    project_path = setup_project_and_train_model['config_data']['project_path']
    file = setup_project_and_train_model['config_data']['video_sets'][0]
    model_name = setup_project_and_train_model['config_data']['model_name']
    n_cluster = setup_project_and_train_model['config_data']['n_cluster']

    save_base_path = Path(project_path) / "results" / file / model_name / f"{parametrization}-{n_cluster}" / "cluster_videos"

    assert len(list(save_base_path.glob("*.mp4"))) > 0
    assert len(list(save_base_path.glob("*.mp4"))) <= n_cluster

@pytest.mark.parametrize('parametrization', ['hmm', 'kmeans'])
def test_motif_videos_avi_files_exists(setup_project_and_train_model, parametrization):
    # Check if the files are created
    vame.motif_videos(setup_project_and_train_model['config_path'], parametrization=parametrization, output_video_type='.avi', save_logs=True)
    project_path = setup_project_and_train_model['config_data']['project_path']
    file = setup_project_and_train_model['config_data']['video_sets'][0]
    model_name = setup_project_and_train_model['config_data']['model_name']
    n_cluster = setup_project_and_train_model['config_data']['n_cluster']

    save_base_path = Path(project_path) / "results" / file / model_name / f"{parametrization}-{n_cluster}" / "cluster_videos"

    assert len(list(save_base_path.glob("*.avi"))) > 0
    assert len(list(save_base_path.glob("*.avi"))) <= n_cluster

@pytest.mark.parametrize('parametrization', ['hmm', 'kmeans'])
def test_community_files_exists(setup_project_and_train_model, parametrization):
    # Check if the files are created
    vame.community(
        setup_project_and_train_model['config_path'],
        cut_tree=2,
        cohort=False,
        parametrization=parametrization,
        save_logs=True
    )
    project_path = setup_project_and_train_model['config_data']['project_path']
    file = setup_project_and_train_model['config_data']['video_sets'][0]
    model_name = setup_project_and_train_model['config_data']['model_name']
    n_cluster = setup_project_and_train_model['config_data']['n_cluster']

    save_base_path = Path(project_path) / "results" / file / model_name / f"{parametrization}-{n_cluster}" / 'community'

    transition_matrix_path = save_base_path / f"transition_matrix_{file}.npy"
    community_label_path = save_base_path / f"community_label_{file}.npy"
    hierarchy_path = save_base_path / f"hierarchy{file}.pkl"

    assert transition_matrix_path.exists()
    assert community_label_path.exists()
    assert hierarchy_path.exists()


@pytest.mark.parametrize('parametrization', ['hmm', 'kmeans'])
def test_cohort_community_files_exists(setup_project_and_train_model, parametrization):
    # Check if the files are created
    vame.community(
        setup_project_and_train_model['config_path'],
        cut_tree=2,
        cohort=True,
        save_logs=True,
        parametrization=parametrization
    )
    project_path = setup_project_and_train_model['config_data']['project_path']
    n_cluster = setup_project_and_train_model['config_data']['n_cluster']

    base_path = Path(project_path) / "results" / 'community_cohort' / f'{parametrization}-{n_cluster}'
    cohort_path = base_path /  "cohort_transition_matrix.npy"
    community_path = base_path /  "cohort_community_label.npy"
    cohort_parametrization_path = base_path /  f"cohort_{parametrization}_label.npy"
    cohort_community_bag_path = base_path /  "cohort_community_bag.npy"

    assert cohort_path.exists()
    assert community_path.exists()
    assert cohort_parametrization_path.exists()
    assert cohort_community_bag_path.exists()


@pytest.mark.parametrize('parametrization', ['hmm', 'kmeans'])
def test_community_videos_mp4_files_exists(setup_project_and_train_model, parametrization):

    vame.community_videos(
        config=setup_project_and_train_model['config_path'],
        parametrization=parametrization,
        save_logs=True,
        output_video_type='.mp4'
    )
    file = setup_project_and_train_model['config_data']['video_sets'][0]
    model_name = setup_project_and_train_model['config_data']['model_name']
    n_cluster = setup_project_and_train_model['config_data']['n_cluster']
    project_path = setup_project_and_train_model['config_data']['project_path']

    save_base_path = Path(project_path) / "results" / file / model_name / f"{parametrization}-{n_cluster}" / "community_videos"

    assert len(list(save_base_path.glob("*.mp4"))) > 0
    assert len(list(save_base_path.glob("*.mp4"))) <= n_cluster

@pytest.mark.parametrize('parametrization', ['hmm', 'kmeans'])
def test_community_videos_avi_files_exists(setup_project_and_train_model, parametrization):

    vame.community_videos(
        config=setup_project_and_train_model['config_path'],
        parametrization=parametrization,
        save_logs=True,
        output_video_type='.avi'
    )
    file = setup_project_and_train_model['config_data']['video_sets'][0]
    model_name = setup_project_and_train_model['config_data']['model_name']
    n_cluster = setup_project_and_train_model['config_data']['n_cluster']
    project_path = setup_project_and_train_model['config_data']['project_path']

    save_base_path = Path(project_path) / "results" / file / model_name / f"{parametrization}-{n_cluster}" / "community_videos"

    assert len(list(save_base_path.glob("*.avi"))) > 0
    assert len(list(save_base_path.glob("*.avi"))) <= n_cluster

@pytest.mark.parametrize('label,parametrization', [
    (None, 'hmm'), ('motif', 'hmm'), ('community', 'hmm'),
    (None, 'kmeans'), ('motif', 'kmeans'), ('community', 'kmeans')
])
def test_visualization_output_files(setup_project_and_train_model, label, parametrization):
    vame.visualization(setup_project_and_train_model['config_path'], parametrization=parametrization, label=label, save_logs=True)

    project_path = setup_project_and_train_model['config_data']['project_path']
    file = setup_project_and_train_model['config_data']['video_sets'][0]
    model_name = setup_project_and_train_model['config_data']['model_name']
    n_cluster = setup_project_and_train_model['config_data']['n_cluster']

    project_path = setup_project_and_train_model['config_data']['project_path']

    save_base_path = Path(project_path) / 'results' / file / model_name / f"{parametrization}-{n_cluster}" / 'community'
    assert len(list(save_base_path.glob(f"umap_vis*{file}.png"))) > 0



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


def test_generative_kmeans_wrong_mode(setup_project_and_train_model):
    with pytest.raises(ValueError):
        vame.generative_model(
            config=setup_project_and_train_model['config_path'],
            parametrization='hmm',
            mode='centers',
            save_logs=True
        )

@pytest.mark.parametrize("label", [None, 'community', 'motif'])
def test_gif_frames_files_exists(setup_project_and_evaluate_model, label):

    with patch("builtins.input", return_value="yes"):
        vame.pose_segmentation(setup_project_and_evaluate_model["config_path"])

    def mock_background(path_to_file=None, filename=None, file_format=None, num_frames=None, save_background=True):
        num_frames = 100
        return background(path_to_file, filename, file_format, num_frames, save_background)

    PARAMETRIZATION = 'hmm'
    VIDEO_LEN = 30
    vame.community(
        setup_project_and_evaluate_model["config_path"],
        cut_tree=2,
        cohort=False,
        save_logs=False,
        parametrization=PARAMETRIZATION
    )
    vame.visualization(
        setup_project_and_evaluate_model["config_path"], parametrization=PARAMETRIZATION, label=label, save_logs=False
    )
    with patch("vame.util.gif_pose_helper.background", side_effect=mock_background):
        vame.gif(
            config=setup_project_and_evaluate_model["config_path"],
            parametrization=PARAMETRIZATION,
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
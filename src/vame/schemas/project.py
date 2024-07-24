from pydantic import BaseModel, Field, ConfigDict
from typing import List
from enum import Enum

class Parametrizations(str, Enum):
    hmm = 'hmm'
    kmeans = 'kmeans'

    class Config:
        use_enum_values = True

class PoseEstimationFiletype(str, Enum):
    csv = 'csv'
    nwb = 'nwb'

    class Config:
        use_enum_values = True

class ProjectSchema(BaseModel):
    # Project attributes
    Project: str = Field(..., title='Project name')
    model_name: str = Field(default='VAME', title='Model name')
    n_cluster: int = Field(default=15, title='Number of clusters')
    pose_confidence: float = Field(default=0.99, title='Pose confidence')

    # Project path and videos
    project_path: str = Field(..., title='Project path')
    video_sets: List[str] = Field(..., title='Video sets')
    pose_estimation_filetype: PoseEstimationFiletype = Field(title='Pose estimation filetype')
    path_to_pose_nwb_series_data: str = Field(title='Path to pose series data in nwb file', default=None)

    # Data
    all_data: str = Field(default='yes', title='All data')
    egocentric_data: bool = Field(default=False, title='Egocentric data')
    robust: bool = Field(default=True, title='Robust data')
    iqr_factor: int = Field(default=4, title='IQR factor')
    axis: str = Field(default='None', title='Axis')
    savgol_filter: bool = Field(default=True, title='Savgol filter')
    savgol_length: int = Field(default=5, title='Savgol length')
    savgol_order: int = Field(default=2, title='Savgol order')
    test_fraction: float = Field(default=0.1, title='Test fraction')

    # RNN model general hyperparameter:
    pretrained_model: str = Field(default='None', title='Pretrained model')
    pretrained_weights: bool = Field(default=False, title='Pretrained weights')
    num_features: int = Field(default=12, title='Number of features')
    batch_size: int = Field(default=256, title='Batch size')
    max_epochs: int = Field(default=500, title='Max epochs')
    model_snapshot: int = Field(default=50, title='Model snapshot')
    model_convergence: int = Field(default=50, title='Model convergence')
    transition_function: str = Field(default='GRU', title='Transition function')
    beta: float = Field(default=1, title='Beta')
    beta_norm: bool = Field(default=False, title='Beta normalization')
    zdims: int = Field(default=30, title='Zdims')
    learning_rate: float = Field(default=5e-4, title='Learning rate')
    time_window: int = Field(default=30, title='Time window')
    prediction_decoder: int = Field(default=1, title='Prediction decoder')
    prediction_steps: int = Field(default=15, title='Prediction steps')
    noise: bool = Field(default=False, title='Noise')
    scheduler: int = Field(default=1, title='Scheduler')
    scheduler_step_size: int = Field(default=100, title='Scheduler step size')
    scheduler_gamma: float = Field(default=0.2, title='Scheduler gamma')
    scheduler_threshold: float = Field(default=None, title='Scheduler threshold')
    softplus: bool = Field(default=False, title='Softplus')


    # Segmentation:
    parametrizations: List[Parametrizations] = Field(
        title='Parametrizations',
        default_factory=lambda: [Parametrizations.hmm.value, Parametrizations.kmeans.value]
    )
    hmm_trained: bool = Field(default=False, title='HMM trained')
    load_data: str = Field(default='-PE-seq-clean', title='Load data')
    individual_parametrization: bool = Field(default=False, title='Individual parametrization')
    random_state_kmeans: int = Field(default=42, title='Random state kmeans')
    n_init_kmeans: int = Field(default=15, title='N init kmeans')

    # Video writer:
    length_of_motif_video: int = Field(default=1000, title='Length of motif video')

    # UMAP parameter:
    min_dist: float = Field(default=0.1, title='Min dist')
    n_neighbors: int = Field(default=200, title='N neighbors')
    random_state: int = Field(default=42, title='Random state')
    num_points: int = Field(default=30000, title='Num points')

    # RNN encoder hyperparameter:
    hidden_size_layer_1: int = Field(default=256, title='Hidden size layer 1')
    hidden_size_layer_2: int = Field(default=256, title='Hidden size layer 2')
    dropout_encoder: float = Field(default=0, title='Dropout encoder')

    # RNN reconstruction hyperparameter:
    hidden_size_rec: int = Field(default=256, title='Hidden size rec')
    dropout_rec: float = Field(default=0, title='Dropout rec')
    n_layers: int = Field(default=1, title='N layers')

    # RNN prediction hyperparameter:
    hidden_size_pred: int = Field(default=256, title='Hidden size pred')
    dropout_pred: float = Field(default=0, title='Dropout pred')

    # RNN loss hyperparameter:
    mse_reconstruction_reduction: str = Field(default='sum', title='MSE reconstruction reduction')
    mse_prediction_reduction: str = Field(default='sum', title='MSE prediction reduction')
    kmeans_loss: int = Field(default=30, title='Kmeans loss')
    kmeans_lambda: float = Field(default=0.1, title='Kmeans lambda')
    anneal_function: str = Field(default='linear', title='Anneal function')
    kl_start: int = Field(default=2, title='KL start')
    annealtime: int = Field(default=4, title='Annealtime')

    model_config: ConfigDict = ConfigDict(protected_namespaces=(), use_enum_values=True)
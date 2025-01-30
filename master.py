from eq_forecast.data.class_eq_dataset import EQ_Dataset
from eq_forecast.data.create_final_data import create_datasets
from eq_forecast.models.TGNN import TemporalGNN
from eq_forecast.utils.train_and_eval import train_and_eval
import torch


"""HYPERPARAMS"""

#Data
START_TIME="2000-01-01"
END_TIME="2015-01-01"
MIN_LAT=24.39630
MAX_LAT=49.3547868
MIN_LON=-124.7844079
MAX_LON=-66.93457
MIN_MAG=3.5
GRID_X=10
GRID_Y=10
WINDOW_DURATION_MS=86400000   #24 hrs
WINDOW_OVERLAP_DURATION_MS=43200000 #12 hrs
DELTA_TIME_DECAY=0.01

#Model
BATCH_SIZE=16
HIDDEN_DIM=32

#Training
EPOCHS=100
LR=1e-4
LR_FACTOR=0.1
LR_PATIENCE=7
EARLY_STOP_PATIENCE=10


dataset = EQ_Dataset(start_time=START_TIME, 
                 end_time=END_TIME, 
                 min_lat=MIN_LAT, 
                 max_lat=MAX_LAT, 
                 min_lon=MIN_LON, 
                 max_lon=MAX_LON, 
                 min_magnitude=MIN_MAG, 
                 batch_size=BATCH_SIZE,
                 n_lat=GRID_Y, 
                 n_lon=GRID_X, 
                 window_time_length=WINDOW_DURATION_MS, 
                 overlap_time_length=WINDOW_OVERLAP_DURATION_MS,
                 time_sensitivity=DELTA_TIME_DECAY,
                 save_data=True, 
                 agg_new_data=True)

train_dataset, val_dataset, test_dataset = create_datasets(dataset)


tgnn = TemporalGNN(node_features=5, 
                   periods=123,
                   hidden_dim=HIDDEN_DIM,
                   out_features=3,
                   batch_size=BATCH_SIZE)

train_and_eval(
    train_set=train_dataset,
    test_set=test_dataset,
    val_set=val_dataset,
    model=tgnn,
    num_epochs=EPOCHS,
    lr=LR,
    lr_factor=LR_FACTOR,
    lr_patience=LR_PATIENCE,
    model_name="TGNN",
    batch_size=BATCH_SIZE,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    checkpoint_dir="checkpoints",
    log_dir="runs",
    save_interval=10,
    early_stop_patience=EARLY_STOP_PATIENCE
)
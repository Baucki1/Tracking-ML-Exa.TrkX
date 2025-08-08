import sys, os
sys.path.append("../../")
import torch

import pkg_resources
pkg_resources.require("faiss-gpu==1.7.2")
pkg_resources.require("cupy-cuda11x==11.0.0")
pkg_resources.require("pytorch_lightning==1.9.0")

import matplotlib
from Scripts import train_metric_learning, run_metric_learning_inference, train_gnn, run_gnn_inference, build_track_candidates, evaluate_candidates
from Scripts.utils.convenience_utils import get_example_data, plot_true_graph, get_training_metrics, plot_training_metrics, plot_neighbor_performance, plot_predicted_graph, plot_track_lengths, plot_edge_performance, plot_graph_sizes
import yaml

import warnings
warnings.filterwarnings("ignore")
CONFIG = 'pipeline_config_multimuon.yaml'

with open(CONFIG, 'r') as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)

output_dir = configs['common_configs']['plot_output_directory'] if 'plot_output_directory' in configs['common_configs'] else None

# metric training
#metric_learning_trainer, metric_learning_model = train_metric_learning(CONFIG)
#embedding_metrics = get_training_metrics(metric_learning_trainer)
#embedding_metrics.to_csv(os.path.join(output_dir, 'embedding_metrics.csv'), index=False)
#print(embedding_metrics)

# gnn training
gnn_trainer, gnn_model = train_gnn(CONFIG)
gnn_metrics = get_training_metrics(gnn_trainer)
gnn_metrics.to_csv(os.path.join(output_dir, 'gnn_metrics.csv'), index=False)
print(gnn_metrics)
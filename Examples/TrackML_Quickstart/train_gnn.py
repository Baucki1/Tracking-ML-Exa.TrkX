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
embedding_training = False
embedding_eval = False
gnn_training = False
gnn_eval = False

with open(CONFIG, 'r') as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)

output_dir = os.path.join(configs['common_configs']['plot_output_directory'], configs['common_configs']['experiment_name']) if 'plot_output_directory' in configs['common_configs'] else None
if output_dir is not None and not os.path.exists(output_dir):
    os.makedirs(output_dir)

if embedding_training:
    # metric training
    metric_learning_trainer, metric_learning_model = train_metric_learning(CONFIG)
    embedding_metrics = get_training_metrics(metric_learning_trainer)
    embedding_metrics.to_csv(os.path.join(output_dir, 'embedding_metrics.csv'), index=False)
    print(embedding_metrics)
    
    if embedding_eval:
        plot_training_metrics(embedding_metrics, output_dir=output_dir)
        plot_neighbor_performance(metric_learning_model, output_dir=output_dir)
        plot_predicted_graph(metric_learning_model, output_dir=output_dir)
        plot_track_lengths(metric_learning_model, output_dir=output_dir)
        plot_graph_sizes(metric_learning_model, output_dir=output_dir)
    
    graph_builder = run_metric_learning_inference(CONFIG)
if gnn_training:
    # gnn training
    gnn_trainer, gnn_model = train_gnn(CONFIG)
    gnn_metrics = get_training_metrics(gnn_trainer)
    gnn_metrics.to_csv(os.path.join(output_dir, 'gnn_metrics.csv'), index=False)
    print(gnn_metrics)

    if gnn_eval:
        plot_training_metrics(gnn_metrics, output_dir=output_dir, savename='gnn_training_metrics.html')
        plot_edge_performance(gnn_model, output_dir=output_dir)
    
    run_gnn_inference(CONFIG)
    build_track_candidates(CONFIG)
evaluated_events, reconstructed_particles, particles, matched_tracks, tracks = evaluate_candidates(CONFIG)
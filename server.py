import flwr as fl
import torch
from torch import nn
from data_util import get_fl_dataset, dataset_cfg
from src.model import build_fp_model
import argparse
import numpy as np
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional, Union
import sys
import json
import time
from loguru import logger
import os
from datetime import datetime

# --- Parsers ---
parser = argparse.ArgumentParser()
parser.add_argument('--num_clients', type=int, default=10)
parser.add_argument('--local_data', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--log_interval', type=int, default=5)
parser.add_argument('--init', help='init ckpt')
parser.add_argument('--seeds', type=int, default=[0,1,2,3,4], nargs='+')
parser.add_argument('--asynch', action='store_true', default=False)
parser.add_argument('--niid', action='store_true', default=False)
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--model', default=4, type=int)
parser.add_argument('--total_steps', type=int, default=10, help="number of rounds of training")
parser.add_argument('--local_ep', type=int, default=6)
parser.add_argument('--algorithm', choices=['FedAVG', 'FedQNN', 'FedQT', 'FedQT-BA', 'FedPAQ', 'FedPAQ-BA', 'Q-FedUpdate', 'Q-FedUpdate-BA'], default='FedAVG', type=str)
parser.add_argument('--qmode', default=1, type=int)
parser.add_argument('--quantize_comm', action='store_true', default=False)
parser.add_argument('--adaptive_bitwidth', action='store_true', default=False)
parser.add_argument('--update_mode', default=0, type=int)
parser.add_argument('--initialization', default='uniform', choices=['uniform', 'normal'], type=str)
parser.add_argument('--m', default=5, type=int)
parser.add_argument('--Wbitwidth', default=4, type=int)
parser.add_argument('--Abitwidth', default=8, type=int)
parser.add_argument('--Ebitwidth', default=8, type=int)
parser.add_argument('--stochastic', action='store_true', default=True)
parser.add_argument('--use_bn', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--momentum', type=float, default=0)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--server_address', type=str, default="0.0.0.0:8080")
parser.add_argument('--mqtt_broker_address', type=str, default='localhost')
parser.add_argument('--mqtt_topic', type=str, default='fedqt_logs')

args = parser.parse_args()
b0 = 4

class FedQTStrategy(fl.server.strategy.FedAvg):
    def __init__(self, initial_parameters, *fl_args, **kwargs):
        super().__init__(*fl_args, **kwargs)
        self.val_loss = None
        self.f0 = None
        self.C = None
        self.comm_times = []
        self.train_times = []
        self.current_parameters = initial_parameters  # Store current global parameters
        
        # Initialize Loguru logger
        
        # Create logs directory if it doesn't exist
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create log file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"fedqt_{timestamp}.log")
        
        # Configure logger
        logger.remove()  # Remove default handler
        logger.add(log_file, rotation="100 MB", level="INFO")
        logger.add(sys.stderr, level="INFO")  # Also log to console
        
        logger.info(f"Starting {args.algorithm} server with {args.num_clients} clients")
        logger.info(f"Logs will be saved to {log_file}")
        
        self.logger = logger

    def aggregate_fit(self, server_round, results, failures):
        print(f"\n=== Server Round {server_round} - Aggregating Fit Results ===")
        print(f"Received results from {len(results)} clients")
        print(f"Failures: {len(failures)}")
        
        if not results:
            print("No results to aggregate")
            return None, {}

        # Store metrics from clients
        for _, res in results:
            self.comm_times.append(res.metrics.get("comm_time", 0))
            self.train_times.append(res.metrics.get("train_time", 0))

        if args.update_mode == 1 and args.quantize_comm:
            print("Using quantized update aggregation")
            aggregated_updates = self.aggregate_quantized_updates(results)
            current_weights = fl.common.parameters_to_ndarrays(self.current_parameters)
            new_weights = [current + update for current, update in zip(current_weights, aggregated_updates)]
            parameters_aggregated = fl.common.ndarrays_to_parameters(new_weights)
            self.current_parameters = parameters_aggregated  # Update stored parameters
            metrics_aggregated = {}
            return parameters_aggregated, metrics_aggregated

        # Standard FedAvg aggregation
        print("Using standard FedAvg aggregation")
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        if parameters_aggregated is not None:
            self.current_parameters = parameters_aggregated  # Update stored parameters
        return parameters_aggregated, metrics_aggregated

    def aggregate_quantized_updates(self, results):
        print("Aggregating quantized updates...")
        total_examples = sum(res.num_examples for _, res in results)
        aggregated_updates = None
        
        for _, fit_res in results:
            update = fl.common.parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples
            metrics = fit_res.metrics
            scale = metrics.get("quant_scale")
            min_val = metrics.get("quant_min")
            
            if scale is not None and min_val is not None:
                dequantized_update = [u * scale + min_val for u in update]
            else:
                dequantized_update = update
                
            weighted_update = [u * (num_examples / total_examples) for u in dequantized_update]
            
            if aggregated_updates is None:
                aggregated_updates = weighted_update
            else:
                aggregated_updates = [agg + w for agg, w in zip(aggregated_updates, weighted_update)]
                
        return aggregated_updates

    def configure_fit(self, server_round, parameters, client_manager):
        print(f"\n=== Server Round {server_round} - Configuring Fit ===")
        
        config = {
            "server_round": server_round, 
            "local_ep": args.local_ep, 
            "batch_size": args.batch_size, 
            "qmode": args.qmode, 
            "update_mode": args.update_mode, 
            "algorithm": args.algorithm
        }
        
        if args.adaptive_bitwidth:
            if self.val_loss is not None and self.f0 is not None:
                bm = b0 + int(np.floor(self.C * np.log2(max(1.0, (self.f0 + 1) / (self.val_loss + 1)))))
                config["Wbitwidth"] = bm
                print(f"Adaptive bitwidth: {bm}")
            else:
                config["Wbitwidth"] = args.Wbitwidth
                print(f"Using default bitwidth: {args.Wbitwidth}")
        else:
            config["Wbitwidth"] = args.Wbitwidth
            
        fit_ins = fl.common.FitIns(parameters, config)
        clients = client_manager.sample(num_clients=self.min_fit_clients, min_num_clients=self.min_fit_clients)
        print(f"Selected {len(clients)} clients for training")
        
        return [(client, fit_ins) for client in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        print(f"\n=== Server Round {server_round} - Aggregating Evaluation Results ===")
        
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
        
        if loss_aggregated is not None:
            self.val_loss = loss_aggregated
            if self.f0 is None:
                self.f0 = loss_aggregated
                self.C = 16 / (np.log2(self.f0 + 1)) if self.f0 > 0 else 16
                print(f"Initialized f0: {self.f0}, C: {self.C}")

            # Logging via MQTT
            # Log server round results
            avg_comm_time = np.mean(self.comm_times) if self.comm_times else 0
            avg_train_time = np.mean(self.train_times) if self.train_times else 0
            accuracy = metrics_aggregated.get("accuracy", 0.0)
            
            # Log using loguru logger
            self.logger.info(
                f"Round {server_round} | "
                f"Algorithm: {args.algorithm} | "
                f"Accuracy: {accuracy:.4f} | "
                f"Avg Comm Time: {avg_comm_time:.8f}s | "
                f"Avg Train Time: {avg_train_time:.4f}s"
            )
            
            # Also print to console for immediate feedback
            print(f"Round {server_round} | Avg Comm Time: {avg_comm_time:.8f}s | Avg Train Time: {avg_train_time:.4f}s")
            # Clear metrics for next round
            self.comm_times, self.train_times = [], []
            
        return loss_aggregated, metrics_aggregated

def get_evaluate_fn(model, test_ds, criterion):
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=args.num_workers)
    
    def evaluate(server_round, parameters, config):
        print(f"=== Server Evaluation Round {server_round} ===")
        device = args.device
        
        # Load parameters into model
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_loss = total_loss / total if total > 0 else 0
        val_prec1 = correct / total if total > 0 else 0
        
        print(f"Server evaluation - Loss: {val_loss:.4f}, Accuracy: {val_prec1:.4f}")
        return val_loss, {"accuracy": val_prec1}
    
    return evaluate

if __name__ == '__main__':
    print(f"=== Starting Server for {args.algorithm} ===")
    
    # Configure algorithm settings
    config = args.algorithm
    if args.algorithm == 'FedAVG':
        args.qmode = 2
        args.update_mode = 0
    elif 'FedPAQ' in args.algorithm or 'FedQT' in args.algorithm or 'Q-FedUpdate' in args.algorithm:
        args.update_mode = 1
        args.quantize_comm = True
        if 'BA' in args.algorithm:
            args.adaptive_bitwidth = True
        if 'FedQNN' in args.algorithm:
            args.qmode = 0
            args.Wbitwidth = 1
        elif 'FedQT' in args.algorithm:
            args.qmode = 1
        elif 'Q-FedUpdate' in args.algorithm:
            args.qmode = 0
        elif 'FedPAQ' in args.algorithm:
            args.qmode = 2
    else:
        raise ValueError(f"Algorithm {args.algorithm} not supported")

    print(f"Algorithm settings - qmode: {args.qmode}, update_mode: {args.update_mode}, quantize_comm: {args.quantize_comm}")

    # Create global model
    global_model = build_fp_model(
        dataset_cfg[args.dataset]['input_channel'], 
        dataset_cfg[args.dataset]['input_size'], 
        dataset_cfg[args.dataset]['output_size'], 
        args.model, 
        args.lr, 
        args.device
    )
    
    if args.init:
        print(f"Loading initial weights from {args.init}")
        global_model.load_state_dict(torch.load(args.init))
    
    print(f"Global model created: {type(global_model)}")
    print(f"Global model parameters: {sum(p.numel() for p in global_model.parameters())}")
    
    # Get initial parameters
    initial_parameters = fl.common.ndarrays_to_parameters([
        val.detach().cpu().numpy() for _, val in global_model.state_dict().items()
    ])
    
    print(f"Initial parameters created: {len(initial_parameters.tensors)} tensors")
    
    # Get test dataset
    _, _, test_ds = get_fl_dataset(args, args.local_data, args.num_clients)
    
    # Create strategy with initial parameters
    strategy = FedQTStrategy(
        initial_parameters=initial_parameters,
        fraction_fit=0.2, 
        fraction_evaluate=0.2, 
        min_fit_clients=max(2, int(args.num_clients * 0.2)), 
        min_evaluate_clients=max(2, int(args.num_clients * 0.2)), 
        min_available_clients=args.num_clients, 
        evaluate_fn=get_evaluate_fn(global_model, test_ds, nn.CrossEntropyLoss())
    )
    
    print(f"Strategy created with {strategy.min_fit_clients} min fit clients")
    print(f"Starting server on {args.server_address}...")
    
    # Start server
    fl.server.start_server(
        server_address=args.server_address, 
        config=fl.server.ServerConfig(num_rounds=args.total_steps), 
        strategy=strategy
    )
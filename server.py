import flwr as fl
import torch
from torch import nn
from data_util import get_fl_dataset, dataset_cfg
from src.model import build_fp_model, build_model
import argparse
import numpy as np
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional, Union
import json
import time
import logging
from datetime import datetime

# Configure logging
log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(filename=f'logs/fedqt_{log_timestamp}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Parsers ---
parser = argparse.ArgumentParser()
parser.add_argument('--num_clients', type=int, default=5)
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

args = parser.parse_args()
b0 = 4

class FedQTStrategy(fl.server.strategy.FedAvg):
    def __init__(self, initial_parameters, global_model, *fl_args, **kwargs):
        super().__init__(*fl_args, **kwargs)
        self.val_loss = None
        self.f0 = None
        self.C = None
        self.comm_times = []
        self.train_times = []
        self.current_parameters = initial_parameters  # Store current global parameters
        self.global_model = global_model  # Store reference to global model
        self.current_bitwidth = args.Wbitwidth  # Track current bitwidth


    def quantize_tensor(self, tensor, bitwidth):
        """Quantize a tensor to specified bitwidth"""
        min_val, max_val = tensor.min(), tensor.max()
        qmin, qmax = 0, 2**bitwidth - 1
        scale = (max_val - min_val) / (qmax - qmin) if max_val != min_val else 1.0
        zero_point = qmin - min_val / scale
        zero_point = torch.clamp(zero_point, qmin, qmax).round()
        
        # Quantize
        quantized = torch.clamp(torch.round(tensor / scale + zero_point), 0, 2**bitwidth - 1)
        return quantized, scale, zero_point

    def dequantize_tensor(self, quantized_tensor, scale, zero_point):
        """Dequantize a tensor"""
        return scale * (quantized_tensor - zero_point)

    def aggregate_fit(self, server_round, results, failures):
        print(f"\n=== Server Round {server_round} - Aggregating Fit Results ===")
        logging.info(f"Server Round {server_round} - Aggregating fit results from {len(results)} clients (with {len(failures)} failures)")
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
            
            # Apply aggregated updates to current parameters
            current_weights = fl.common.parameters_to_ndarrays(self.current_parameters)
            new_weights = [current + update for current, update in zip(current_weights, aggregated_updates)]
            
            # Determine new bitwidth for next round
            new_bitwidth = self.select_bitwidth(server_round)
            
            # Quantize the new global model if needed
            if args.quantize_comm and ('FedQT' in args.algorithm or 'FedPAQ' in args.algorithm or 'Q-FedUpdate' in args.algorithm):
                # For quantized algorithms, quantize the global parameters
                quantized_weights = []
                quant_metadata = {'scales': [], 'zero_points': [], 'bitwidth': new_bitwidth}
                
                for weight in new_weights:
                    weight_tensor = torch.tensor(weight)
                    q_weight, scale, zero_point = self.quantize_tensor(weight_tensor, new_bitwidth)
                    quantized_weights.append(q_weight.numpy())
                    quant_metadata['scales'].append(float(scale))
                    quant_metadata['zero_points'].append(float(zero_point))
                
                parameters_aggregated = fl.common.ndarrays_to_parameters(new_weights)  # Send FP for now
                self.current_parameters = parameters_aggregated
                
                print(f"Server: Quantized global model with {new_bitwidth}-bit precision")
                metrics_aggregated = {"quant_metadata": quant_metadata}
            else:
                parameters_aggregated = fl.common.ndarrays_to_parameters(new_weights)
                self.current_parameters = parameters_aggregated
                metrics_aggregated = {}
            
            return parameters_aggregated, metrics_aggregated

        # Standard FedAvg aggregation
        print("Using standard FedAvg aggregation")
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        if parameters_aggregated is not None:
            self.current_parameters = parameters_aggregated
        return parameters_aggregated, metrics_aggregated

    def aggregate_quantized_updates(self, results):
        """Aggregate quantized updates by dequantizing first"""
        print("Dequantizing and aggregating updates...")
        total_examples = sum(res.num_examples for _, res in results)
        aggregated_updates = None
        
        for _, fit_res in results:
            update_arrays = fl.common.parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples
            metrics = fit_res.metrics
            
            # Check if updates are quantized
            if 'scales' in metrics and 'zero_points' in metrics:
                print(f"Dequantizing updates from client (bitwidth: {metrics.get('bitwidth', 'unknown')})")
                # Dequantize updates
                dequantized_updates = []
                scales = metrics['scales']
                zero_points = metrics['zero_points']
                
                for i, update_array in enumerate(update_arrays):
                    if i < len(scales) and i < len(zero_points):
                        update_tensor = torch.tensor(update_array)
                        dequant_update = self.dequantize_tensor(update_tensor, scales[i], zero_points[i])
                        dequantized_updates.append(dequant_update.numpy())
                    else:
                        # Fallback if metadata is incomplete
                        dequantized_updates.append(update_array)
                
                update_arrays = dequantized_updates
            else:
                print("Updates are already in full precision")
            
            # Weight by number of examples
            weighted_updates = [u * (num_examples / total_examples) for u in update_arrays]
            
            if aggregated_updates is None:
                aggregated_updates = weighted_updates
            else:
                aggregated_updates = [agg + w for agg, w in zip(aggregated_updates, weighted_updates)]
                
        print("Aggregation completed")
        return aggregated_updates

    def select_bitwidth(self, server_round):
        """Select bitwidth for next round based on adaptive strategy"""
        if args.adaptive_bitwidth:
            if self.val_loss is not None and self.f0 is not None:
                bm = b0 + int(np.floor(self.C * np.log2(max(1.0, (self.f0 + 1) / (self.val_loss + 1)))))
                # Clamp bitwidth to reasonable range
                bm = max(1, min(32, bm))
                self.current_bitwidth = bm
                print(f"Adaptive bitwidth selected: {bm} bits")
                return bm
            else:
                print(f"Using default bitwidth: {args.Wbitwidth} bits")
                return args.Wbitwidth
        else:
            return args.Wbitwidth

    def configure_fit(self, server_round, parameters, client_manager):
        print(f"\n=== Server Round {server_round} - Configuring Fit ===")
        
        config = {
            "server_round": server_round, 
            "local_ep": args.local_ep, 
            "batch_size": args.batch_size, 
            "qmode": args.qmode, 
            "update_mode": args.update_mode, 
            "algorithm": args.algorithm,
            "Wbitwidth": self.current_bitwidth
        }
        
        # For quantized update modes, include quantization metadata
        if args.update_mode == 1 and args.quantize_comm and server_round > 1:
            # After first round, send quantized updates
            if hasattr(self, 'last_quant_metadata'):
                config['quant_metadata'] = self.last_quant_metadata
                print(f"Sending quantized updates to clients with {self.current_bitwidth}-bit precision")
        
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

            avg_comm_time = np.mean(self.comm_times) if self.comm_times else 0
            avg_train_time = np.mean(self.train_times) if self.train_times else 0
            accuracy = metrics_aggregated.get("accuracy", 0.0)
            log_data = {
                "global_round": server_round, 
                "algorithm": args.algorithm, 
                "accuracy": accuracy, 
                "average_communication_time": avg_comm_time, 
                "training_overhead": avg_train_time,
                "bitwidth": self.current_bitwidth
            }
            
            # Log to file
            logging.info(f"Round {server_round} results: {json.dumps(log_data)}")
            
            print(f"Round {server_round} | Accuracy: {accuracy:.4f} | Bitwidth: {self.current_bitwidth} | Avg Comm Time: {avg_comm_time:.8f}s | Avg Train Time: {avg_train_time:.4f}s")
            
            # Clear metrics for next round
            self.comm_times, self.train_times = [], []
            
        return loss_aggregated, metrics_aggregated

def get_evaluate_fn(global_model, test_ds, criterion):
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=args.num_workers)
    
    def evaluate(server_round, parameters, config):
        print(f"=== Server Evaluation Round {server_round} ===")
        device = args.device
        
        try:
            # Handle different model types
            if hasattr(global_model, 'dequantize'):
                # Quantized model - dequantize for evaluation
                print("Using quantized model - dequantizing for evaluation")
                eval_model = global_model.dequantize()
                eval_model.to(device)
                eval_model.eval()
                
                # Load parameters into the dequantized model
                try:
                    params_dict = zip(eval_model.state_dict().keys(), parameters)
                    state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict})
                    eval_model.load_state_dict(state_dict, strict=False)
                except Exception as e:
                    print(f"Warning: Could not load all parameters into dequantized model: {e}")
                    # Use the global model's current state
                    eval_model = global_model.dequantize()
                    
            else:
                # Full precision model
                print("Using full precision model")
                eval_model = global_model
                params_dict = zip(eval_model.state_dict().keys(), parameters)
                state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict})
                eval_model.load_state_dict(state_dict, strict=True)
                eval_model.to(device)
                eval_model.eval()
            
            total_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = eval_model(data)
                    loss = criterion(output, target)
                    total_loss += loss.item() * data.size(0)
                    
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            val_loss = total_loss / total if total > 0 else 0
            val_prec1 = correct / total if total > 0 else 0
            
            print(f"Server evaluation - Loss: {val_loss:.4f}, Accuracy: {val_prec1:.4f}")
            return val_loss, {"accuracy": val_prec1}
            
        except Exception as e:
            print(f"Error in server evaluation: {e}")
            # Return dummy values if evaluation fails
            return 0.0, {"accuracy": 0.0}
    
    return evaluate

if __name__ == '__main__':
    print(f"=== Starting Server for {args.algorithm} ===")
    
    # Configure algorithm settings
    config = args.algorithm
    if args.algorithm == 'FedAVG':
        args.qmode = 2
        args.update_mode = 0
        args.quantize_comm = False
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

    # Create global model - use build_model for quantized algorithms, build_fp_model for FedAVG
    if args.algorithm == 'FedAVG':
        print("Creating full precision global model for FedAVG")
        global_model = build_fp_model(
            dataset_cfg[args.dataset]['input_channel'], 
            dataset_cfg[args.dataset]['input_size'], 
            dataset_cfg[args.dataset]['output_size'], 
            args.model, 
            args.lr, 
            args.device
        )
    else:
        print(f"Creating quantized global model for {args.algorithm}")
        global_model = build_model(
            dataset_cfg[args.dataset]['input_channel'], 
            dataset_cfg[args.dataset]['input_size'], 
            dataset_cfg[args.dataset]['output_size'], 
            args
        )
    
    if args.init:
        print(f"Loading initial weights from {args.init}")
        global_model.load_state_dict(torch.load(args.init))
    
    print(f"Global model created: {type(global_model)}")
    
    # Get initial parameters - always send full precision for initial broadcast
    if hasattr(global_model, 'dequantize'):
        # For quantized models, use dequantized parameters for initial communication
        print("Extracting parameters from quantized model")
        fp_model = global_model.dequantize()
        initial_parameters = fl.common.ndarrays_to_parameters([
            val.detach().cpu().numpy() for _, val in fp_model.state_dict().items()
        ])
    else:
        initial_parameters = fl.common.ndarrays_to_parameters([
            val.detach().cpu().numpy() for _, val in global_model.state_dict().items()
        ])
    
    print(f"Initial parameters created: {len(initial_parameters.tensors)} tensors")
    
    # Get test dataset
    _, _, test_ds = get_fl_dataset(args, args.local_data, args.num_clients)
    
    # Create strategy with initial parameters
    strategy = FedQTStrategy(
        initial_parameters=initial_parameters,
        global_model=global_model,
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
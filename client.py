import flwr as fl
import torch
from torch import nn
from data_util import get_fl_dataset, dataset_cfg
from src.model import build_fp_model, build_model
import argparse
import numpy as np
from collections import OrderedDict
import time

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
parser.add_argument('--total_steps', type=int, default=20, help="number of rounds of training")
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
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--server_address', type=str, default="127.0.0.1:8080")
parser.add_argument('--cid', type=int, required=True)

args = parser.parse_args()

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, train_ds, test_ds, criterion):
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.criterion = criterion
        self.model_BW_hist = []
        
        # Create fresh model each time to avoid state issues
        self._create_fresh_model()

    def _create_fresh_model(self):
        """Create a fresh model based on algorithm"""
        if args.algorithm == 'FedAVG':
            # Use full precision model for FedAVG
            self.model = build_fp_model(
                dataset_cfg[args.dataset]['input_channel'],
                dataset_cfg[args.dataset]['input_size'],
                dataset_cfg[args.dataset]['output_size'],
                args.model,
                args.lr,
                args.device,
            )
        else:
            # Use quantized model for quantized algorithms
            self.model = build_model(
                dataset_cfg[args.dataset]['input_channel'], 
                dataset_cfg[args.dataset]['input_size'], 
                dataset_cfg[args.dataset]['output_size'], 
                args
            )
        
        self.model.to(args.device)
        
        # Ensure all parameters require gradients
        for name, param in self.model.named_parameters():
            param.requires_grad_(True)
            
        print(f"Client {args.cid}: Created fresh model: {type(self.model)}")
        print(f"Client {args.cid}: Total parameters: {sum(p.numel() for p in self.model.parameters())}")
        print(f"Client {args.cid}: Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

    def quantize_tensor(self, tensor, bitwidth, scale=None, zero_point=None):
        """Quantize a tensor to specified bitwidth"""
        if scale is None or zero_point is None:
            # Calculate scale and zero point
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

    def get_parameters(self, config):
        """Return parameters for communication"""
        if args.update_mode == 1 and args.quantize_comm:
            # For quantized communication, we need to handle this in fit()
            # This method is called for initial parameter broadcast
            print(f"Client {args.cid}: get_parameters called for initial broadcast")
            
        # Always return full precision parameters for initial setup
        if hasattr(self.model, 'dequantize'):
            fp_model = self.model.dequantize()
            return [val.detach().cpu().numpy() for _, val in fp_model.state_dict().items()]
        else:
            return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Set parameters - handle both quantized updates and full parameters"""
        try:
            if hasattr(self.model, 'state_dict'):
                current_state = self.model.state_dict()
                
                # Handle quantized model case
                if hasattr(self.model, 'dequantize'):
                    # For quantized models, load into dequantized version then re-quantize
                    fp_model = self.model.dequantize()
                    params_dict = zip(fp_model.state_dict().keys(), parameters)
                    state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict})
                    fp_model.load_state_dict(state_dict)
                    
                    # Re-quantize the model (this depends on your quantization implementation)
                    # You may need to implement a method to quantize from fp_model back to self.model
                    print(f"Client {args.cid}: Loaded parameters into quantized model")
                else:
                    # Full precision model
                    params_dict = zip(current_state.keys(), parameters)
                    state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict})
                    self.model.load_state_dict(state_dict)
                    
                    # Ensure all parameters require gradients
                    for param in self.model.parameters():
                        param.requires_grad_(True)
                    
                    print(f"Client {args.cid}: Loaded parameters into full precision model")
                    
        except Exception as e:
            print(f"Client {args.cid}: Error in set_parameters: {e}")
            print(f"Client {args.cid}: Recreating model...")
            self._create_fresh_model()

    def apply_quantized_update(self, quantized_updates, quant_metadata):
        """Apply quantized updates directly to model parameters"""
        try:
            # Dequantize updates first
            dequantized_updates = []
            for i, q_update in enumerate(quantized_updates):
                scale = quant_metadata['scales'][i]
                zero_point = quant_metadata['zero_points'][i]
                deq_update = self.dequantize_tensor(
                    torch.tensor(q_update), scale, zero_point
                )
                dequantized_updates.append(deq_update)
            
            # Apply updates to model parameters
            current_state = self.model.state_dict()
            new_state = OrderedDict()
            
            for (name, param), update in zip(current_state.items(), dequantized_updates):
                new_state[name] = param + update.to(param.device).type(param.dtype)
            
            self.model.load_state_dict(new_state)
            
            # Ensure gradients are enabled
            for param in self.model.parameters():
                param.requires_grad_(True)
                
            print(f"Client {args.cid}: Applied quantized updates successfully")
            
        except Exception as e:
            print(f"Client {args.cid}: Error applying quantized updates: {e}")
            # Fallback to treating as full parameters
            self.set_parameters(quantized_updates)

    def fit(self, parameters, config):
        print(f"\n=== Starting fit for client {args.cid} ===")
        comm_start_time = time.time()
        
        # Handle parameter setting based on whether we're receiving updates or full parameters
        update_mode = config.get("update_mode", 0)
        algorithm = config.get("algorithm", args.algorithm)
        
        if update_mode == 1 and 'quant_metadata' in config:
            # We're receiving quantized updates
            print(f"Client {args.cid}: Receiving quantized updates")
            self.apply_quantized_update(parameters, config['quant_metadata'])
        else:
            # We're receiving full parameters (first round or FedAVG)
            print(f"Client {args.cid}: Receiving full parameters")
            self.set_parameters(parameters)
        
        comm_end_time = time.time()
        comm_time = comm_end_time - comm_start_time
        
        self.model.to(args.device)

        local_ep = config["local_ep"]
        batch_size = config["batch_size"]
        qmode = config["qmode"]
        Wbitwidth = config.get("Wbitwidth", args.Wbitwidth)

        if qmode != 2:
            self.model_BW_hist.append(Wbitwidth if Wbitwidth != 1 else 1)
        else:
            self.model_BW_hist.append(32)

        # Store initial model state for computing updates
        init_model_state = OrderedDict({k: v.clone().detach() for k, v in self.model.state_dict().items()})

        train_start_time = time.time()

        # ==================== TRAINING LOGIC ====================
        train_loader = torch.utils.data.DataLoader(
            self.train_ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers
        )
        
        # Create optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise RuntimeError("No parameters require gradients! Cannot train.")
        
        optimizer = torch.optim.SGD(trainable_params, lr=args.lr, momentum=args.momentum)
        self.model.train()

        epoch_losses = []
        epoch_accs = []
        
        for epoch in range(local_ep):
            batch_losses = []
            correct, total = 0, 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(args.device), target.to(args.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                
                batch_losses.append(loss.item())
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            epoch_losses.append(np.mean(batch_losses))
            epoch_accs.append(correct / total if total > 0 else 0)
            print(f"Client {args.cid}: Epoch {epoch+1}/{local_ep} - Loss: {epoch_losses[-1]:.4f}, Acc: {epoch_accs[-1]:.4f}")

        train_time = time.time() - train_start_time
        
        # Compute metrics
        li = np.mean(epoch_losses) if epoch_losses else 0
        ai = np.mean(epoch_accs) if epoch_accs else 0
        metrics = {"loss": li, "accuracy": ai, "train_time": train_time, "comm_time": comm_time}

        # ==================== COMPUTE AND SEND UPDATES ====================
        if update_mode == 1:
            # Compute parameter updates
            print(f"Client {args.cid}: Computing parameter updates")
            new_state_dict = self.model.state_dict()
            updates = []
            quant_metadata = {'scales': [], 'zero_points': [], 'bitwidth': Wbitwidth}
            
            for name, param in new_state_dict.items():
                if name in init_model_state:
                    update = param - init_model_state[name]
                    
                    if args.quantize_comm and ('FedQT' in algorithm or 'FedPAQ' in algorithm or 'Q-FedUpdate' in algorithm):
                        # Quantize the update
                        quantized_update, scale, zero_point = self.quantize_tensor(update, Wbitwidth)
                        updates.append(quantized_update.cpu().numpy())
                        quant_metadata['scales'].append(float(scale))
                        quant_metadata['zero_points'].append(float(zero_point))
                    else:
                        # Send unquantized update
                        updates.append(update.cpu().numpy())
                        quant_metadata['scales'].append(1.0)
                        quant_metadata['zero_points'].append(0.0)
            
            if args.quantize_comm and ('FedQT' in algorithm or 'FedPAQ' in algorithm or 'Q-FedUpdate' in algorithm):
                metrics.update(quant_metadata)
                print(f"Client {args.cid}: Sending quantized updates with {Wbitwidth}-bit precision")
            else:
                print(f"Client {args.cid}: Sending unquantized updates")
                
            return updates, len(self.train_ds), metrics
        else:
            # Send full parameters (FedAVG mode)
            print(f"Client {args.cid}: Sending full parameters")
            if hasattr(self.model, 'dequantize'):
                fp_model = self.model.dequantize()
                updates = [val.detach().cpu().numpy() for _, val in fp_model.state_dict().items()]
            else:
                updates = [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]
            
            return updates, len(self.train_ds), metrics

    def evaluate(self, parameters, config):
        print(f"=== Starting evaluation for client {args.cid} ===")
        self.set_parameters(parameters)
        self.model.to(args.device)
        self.model.eval()
        
        test_loader = torch.utils.data.DataLoader(
            self.test_ds, batch_size=128, shuffle=False, num_workers=args.num_workers
        )
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(args.device), target.to(args.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item() * data.size(0)
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / total if total > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        print(f"=== Client {args.cid} evaluation complete - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f} ===")
        return float(avg_loss), len(self.test_ds), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
        print("CUDA not available, switching to CPU.")

    train_ds_clients, test_ds_clients, _ = get_fl_dataset(args, args.local_data, args.num_clients)
    client_train_ds, client_test_ds = train_ds_clients[args.cid], test_ds_clients[args.cid]

    client = FlowerClient(client_train_ds, client_test_ds, nn.CrossEntropyLoss())
    
    print(f"Starting client {args.cid} for {args.algorithm}...")
    fl.client.start_numpy_client(server_address=args.server_address, client=client)
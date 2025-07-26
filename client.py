import flwr as fl
import torch
from torch import nn
from data_util import get_fl_dataset, dataset_cfg
from src.model import build_fp_model
import argparse
import numpy as np
from collections import OrderedDict
import time

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
parser.add_argument('--total_steps', type=int, default=200, help="number of rounds of training")
parser.add_argument('--local_ep', type=int, default=1)
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
        """Create a fresh full precision model"""
        self.model = build_fp_model(
            dataset_cfg[args.dataset]['input_channel'],
            dataset_cfg[args.dataset]['input_size'],
            dataset_cfg[args.dataset]['output_size'],
            args.model,
            args.lr,
            args.device,
            momentum=args.momentum
        )
        
        # Ensure all parameters require gradients
        for name, param in self.model.named_parameters():
            param.requires_grad_(True)
            
        print(f"Created fresh model: {type(self.model)}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters())}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Safely set parameters while preserving gradient requirements"""
        try:
            # Get current state dict structure
            current_state = self.model.state_dict()
            
            # Create new state dict with proper tensor conversion
            new_state = OrderedDict()
            for (name, _), param_array in zip(current_state.items(), parameters):
                # Convert numpy to tensor with proper dtype
                tensor = torch.tensor(param_array, dtype=torch.float32)
                new_state[name] = tensor
            
            # Load the state dict
            self.model.load_state_dict(new_state, strict=True)
            
            # CRITICAL: Re-enable gradients after loading
            for param in self.model.parameters():
                param.requires_grad_(True)
                
            print(f"Successfully loaded parameters. Trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
            
        except Exception as e:
            print(f"Error in set_parameters: {e}")
            print("Creating fresh model...")
            self._create_fresh_model()

    def fit(self, parameters, config):
        print(f"\n=== Starting fit for client {args.cid} ===")
        comm_start_time = time.time()
        
        # Set parameters from server
        self.set_parameters(parameters)
        self.model.to(args.device)

        local_ep = config["local_ep"]
        batch_size = config["batch_size"]
        qmode = config["qmode"]
        Wbitwidth = config.get("Wbitwidth", args.Wbitwidth)
        update_mode = config["update_mode"]
        algorithm = config["algorithm"]

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
        
        # Double-check parameters before training
        trainable_params = list(self.model.parameters())
        grad_params = [p for p in trainable_params if p.requires_grad]
        
        print(f"Total parameters: {len(trainable_params)}")
        print(f"Parameters requiring gradients: {len(grad_params)}")
        
        if len(grad_params) == 0:
            raise RuntimeError("No parameters require gradients! Cannot train.")
        
        # Create optimizer
        optimizer = torch.optim.SGD(grad_params, lr=args.lr, momentum=args.momentum)
        self.model.train()

        epoch_losses = []
        epoch_accs = []
        
        for epoch in range(local_ep):
            batch_losses = []
            correct, total = 0, 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(args.device), target.to(args.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                batch_losses.append(loss.item())
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # Debug first batch
                if epoch == 0 and batch_idx == 0:
                    print(f"First batch - Loss: {loss.item():.4f}, Accuracy: {100.*correct/total:.2f}%")

            epoch_losses.append(np.mean(batch_losses))
            epoch_accs.append(correct / total if total > 0 else 0)
            print(f"Epoch {epoch+1}/{local_ep} - Loss: {epoch_losses[-1]:.4f}, Acc: {epoch_accs[-1]:.4f}")

        # Average loss and accuracy over epochs
        li = np.mean(epoch_losses) if epoch_losses else 0
        ai = np.mean(epoch_accs) if epoch_accs else 0

        train_time = time.time() - train_start_time

        metrics = {"loss": li, "accuracy": ai, "train_time": train_time}
        
        if update_mode == 0:
            # Return full model parameters
            updates = self.get_parameters({})
        else:
            # Compute parameter updates
            new_state_dict = self.model.state_dict()
            diff = OrderedDict()
            for name, param in new_state_dict.items():
                diff[name] = param - init_model_state[name]

            if 'FedQT' in algorithm or 'FedPAQ' in algorithm:
                quantized_diff = []
                all_diff_tensors = torch.cat([d.view(-1) for d in diff.values()])
                min_val, max_val = all_diff_tensors.min(), all_diff_tensors.max()
                scale = (max_val - min_val) / (2**Wbitwidth - 1) if (max_val - min_val) > 0 else 1.0
                for name, d_tensor in diff.items():
                    if scale > 0:
                        Q = (d_tensor - min_val) / scale
                        floor, ceil = torch.floor(Q), torch.ceil(Q)
                        Q = torch.where(torch.rand_like(Q) > (Q - floor), floor, ceil)
                        quantized_diff.append(Q.cpu().numpy())
                    else:
                        quantized_diff.append(torch.zeros_like(d_tensor).cpu().numpy())
                updates = quantized_diff
                metrics["quant_scale"] = float(scale)
                metrics["quant_min"] = float(min_val)
            else:
                updates = [val.detach().cpu().numpy() for _, val in diff.items()]
        
        comm_time = time.time() - comm_start_time
        metrics["comm_time"] = comm_time
        
        print(f"=== Completed fit - Loss: {li:.4f}, Acc: {ai:.4f} ===\n")
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
        
        print(f"=== Evaluation complete - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f} ===")
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
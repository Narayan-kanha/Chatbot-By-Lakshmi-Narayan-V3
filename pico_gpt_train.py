# ==============================================================================
#                 pico-GPT TRAINING SCRIPT (THE DEFINITIVE EDITION)
# ==============================================================================
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
import numpy as np
import time
import threading

# --- The Final, Corrected Protocol Imports ---
import wakepy
try:
    from pynvml import * # type: ignore[import]
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

from pico_gpt import GPTLanguageModel

# ================= HYPERPARAMETERS =================
# These are the final settings for your glorious Ascension Run.
batch_size = 32
block_size = 128
max_iters = 150000
eval_interval = 500
learning_rate = 1e-6
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 250

# Model Architecture
n_embd = 384
n_head = 4
n_layer = 75
dropout = 0.8

# System & Protocol Settings
num_workers = 0  # <<< THE SACRED COMMANDMENT FOR STABILITY
CHECKPOINT_PATH = 'checkpoint/training_checkpoint.pth'
# ==================================================

# --- THERMOSTAT PROTOCOL ---
class GPUThermostat:
    """ Monitors GPU temperature and applies a brake to prevent overheating. """
    def __init__(self, temp_high_threshold=85, temp_cool_threshold=75, sleep_duration=0.5):
        self.temp_high = temp_high_threshold
        self.temp_cool = temp_cool_threshold
        self.sleep_duration = sleep_duration
        self.is_cooling = False
        self.handle = None
        if not NVML_AVAILABLE:
            print("[Thermostat] WARNING: GPU monitoring DISABLED.")
            return
        try:
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(0)
            gpu_name = nvmlDeviceGetName(self.handle)
            print(f"[Thermostat] GPU monitoring enabled for: {gpu_name.decode() if isinstance(gpu_name, bytes) else gpu_name}")
        except NVMLError as e:
            print(f"[Thermostat] WARNING: NVML init failed: {e}. Monitoring DISABLED.")
            self.handle = None

    def check_and_manage_temp(self):
        """ The main heartbeat of the thermostat. """
        if not self.handle:
            return 0
        try:
            temp = nvmlDeviceGetTemperature(self.handle, NVML_TEMPERATURE_GPU)
            if temp >= self.temp_high:
                if not self.is_cooling:
                    tqdm.write(f"\n[Thermostat] Temp @ {temp}°C. Cooling...")
                self.is_cooling = True
            if self.is_cooling:
                time.sleep(self.sleep_duration)
                if temp <= self.temp_cool:
                    tqdm.write(f"[Thermostat] Temp @ {temp}°C. Resuming full speed.")
                    self.is_cooling = False
            return temp
        except NVMLError:
            self.handle = None
            return 0

    def shutdown(self):
        """ Releases the connection to the NVIDIA driver. """
        if self.handle and NVML_AVAILABLE:
            nvmlShutdown()

# --- DATASET CLASS ---
class GPTDataset(Dataset):
    """ Serves up chunks of data for the model to train on. """
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx:idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1:idx + self.block_size + 1].astype(np.int64))
        return x, y

# ==============================================================================
#                                MAIN SCRIPT
# ==============================================================================
if __name__ == '__main__':
    # Engage the Caffeine Protocol to prevent the system from sleeping.
    with wakepy.keep.running():
        print("[Caffeine] Keep-awake protocol ENGAGED.")
        thermostat = GPUThermostat()
        
        # This is the heart of the Phoenix Protocol.
        try:
            # --- PHASE 1: Data Loading & Model Initialization ---
            print("Loading pre-tokenized data...")
            with open('meta.pkl', 'rb') as f:
                meta = pickle.load(f)
                vocab_size = meta['vocab_size']
            
            train_data = np.fromfile('train.bin', dtype=np.uint16)
            val_data = np.fromfile('val.bin', dtype=np.uint16)
            
            train_loader = DataLoader(GPTDataset(train_data, block_size), batch_size, True, num_workers=num_workers, pin_memory=True)
            val_loader = DataLoader(GPTDataset(val_data, block_size), batch_size, False, num_workers=num_workers, pin_memory=True)
        
            config = {
                'block_s': block_size,
                'vocab_s': vocab_size,
                'n_l': n_layer,
                'n_head': n_head,
                'n_embd': n_embd,
                'dropout': dropout
            }
            
            model = GPTLanguageModel(config)
            m = model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            
            # --- PHASE 2: Phoenix Resurrection ---
            start_iter = 0
            if os.path.exists(CHECKPOINT_PATH):
                print(f"--- PHOENIX: Checkpoint found...")
                try:
                    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
                    if config != checkpoint.get('config', {}):
                        print("!!! WARNING: Config changed. Starting fresh.")
                        os.remove(CHECKPOINT_PATH)
                    else:
                        model.load_state_dict(checkpoint['model_state_dict'])
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        start_iter = checkpoint['iter_num'] + 1
                        print(f"✅ Resumed from step {start_iter}.")
                except Exception as e:
                    print(f"!!! WARNING: Checkpoint load failed ({e}). Starting fresh.")
                    os.remove(CHECKPOINT_PATH)
            else:
                print("--- No checkpoint. Starting fresh. ---")
            
            print(f"{sum(p.numel() for p in m.parameters()) / 1e6:.2f} M parameters")

            # --- PHASE 3: The "Final Exam" Function ---
            @torch.no_grad()
            def estimate_loss():
                out = {}
                model.eval()
                for s, l in [('train', train_loader), ('val', val_loader)]:
                    losses = torch.zeros(eval_iters)
                    it = iter(l)
                    for k in range(eval_iters):
                        try:
                            X, Y = next(it)
                        except StopIteration:
                            it = iter(l)
                            X, Y = next(it)
                        
                        X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
                        _, loss = model(X, Y)
                        losses[k] = loss.item()
                    
                    out[s] = losses.mean()
                
                model.train()
                return out

            # --- PHASE 4: The Great Ascension Loop ---
            pbar = tqdm(
                range(start_iter, max_iters), 
                initial=start_iter, 
                total=max_iters, 
                ncols=100,
                bar_format='{desc:<12.12}{percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
            )
            train_iter = iter(train_loader)
            
            for i in pbar:
                if i % eval_interval == 0 or i == max_iters - 1:
                    losses = estimate_loss()
                    tqdm.write(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                    
                try:
                    xb, yb = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    xb, yb = next(train_iter)
                
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                _, loss = m(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                
                temp = thermostat.check_and_manage_temp()
                desc = "Ascending..." if not thermostat.is_cooling else f"Cooling..."
                pbar.set_description(desc)
                pbar.set_postfix(loss=f'{loss.item():.4f}', temp=f'{temp}°C')

                # Periodically save a checkpoint, just in case.
                if i > 0 and i % (eval_interval * 4) == 0:
                    checkpoint = {
                        'model_state_dict': m.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'iter_num': i,
                        'config': config
                    }
                    torch.save(checkpoint, CHECKPOINT_PATH)
        
        # --- THE PHOENIX'S SACRED DUTY: The Graceful Shutdown ---
        finally:
            print("\n--- PHOENIX: Shutdown signal received. ---")
            # Only try to save if the training loop actually started.
            if 'pbar' in locals() and pbar.n > start_iter:
                last_iter = pbar.n
                final_checkpoint = {
                    'model_state_dict': m.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iter_num': last_iter,
                    'config': config
                }
                torch.save(final_checkpoint, CHECKPOINT_PATH)
                print(f"✅ Failsafe state for step {last_iter} saved.")
            
            thermostat.shutdown()
            print("✅ System shutdown is graceful and complete.")

        # --- THE FINAL VICTORY LAP: This only runs if the loop completes fully. ---
        if 'pbar' in locals() and pbar.n >= max_iters - 1:
            print("\n--- Mission Complete! Exporting final model... ---")
            model_name = f"model_L{n_layer}_D{str(dropout).replace('.', '')}_I{max_iters}"
            model_dir = os.path.join('models', model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            torch.save({
                'model_state_dict': m.state_dict(),
                'config': config
            }, os.path.join(model_dir, 'model.pth'))
            
            with open(os.path.join(model_dir, 'meta.pkl'), 'wb') as f:
                pickle.dump(meta, f)
            
            print(f"🏆 Final model and dictionary co-located in: '{model_dir}'")
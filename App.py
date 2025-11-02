# ==============================================================================
#                 FLASK AI HUB - FLEET ADMIRAL EDITION (app.py)
# ==============================================================================
# VERSION: 2.2 - The Beautiful, Readable, and Final Edition
# ==============================================================================

# --- Core Python & Flask Imports ---
import os
import signal
import threading
import subprocess
import pickle
import torch
from flask import Flask, render_template, request, jsonify, abort
from flask_socketio import SocketIO, emit

# --- The Brain Import ---
from pico_gpt import GPTLanguageModel

# --- Configuration ---
app = Flask(__name__)
app.secret_key = "the_secret_key_of_the_fleet_admiral"
socketio = SocketIO(app, async_mode='eventlet')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Global State Management ---
training_process = None
training_lock = threading.Lock()
MODEL_CACHE = {'path': None, 'model': None, 'meta': None}

# ==============================================================================
#                            MODEL MANAGEMENT
# ==============================================================================

def find_models():
    """ Scans the './models' directory for all trained .pth files. """
    model_paths = []
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    for root, _, files in os.walk(models_dir):
        for file in files:
            if file.endswith('.pth'):
                full_path = os.path.join(root, file)
                model_paths.append(os.path.relpath(full_path, os.path.dirname(__file__)))
                
    return sorted(model_paths)

def load_model(path):
    """ Loads a model and its matching meta.pkl into the global cache. """
    global MODEL_CACHE
    if MODEL_CACHE['path'] == path:
        return True
        
    try:
        print(f"Loading model from: {path}")
        model_dir = os.path.dirname(path)
        meta_path = os.path.join(model_dir, 'meta.pkl')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"CRITICAL: Could not find 'meta.pkl' in '{model_dir}'")

        checkpoint = torch.load(path, map_location=DEVICE)
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        config = checkpoint.get('config', {})
        if 'vocab_size' in config: config['vocab_s'] = meta['vocab_size']
        if 'n_layer' in config: config['n_l'] = config.pop('n_layer')
        if 'block_size' in config: config['block_s'] = config.pop('block_size')

        model = GPTLanguageModel(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()
        MODEL_CACHE = {'path': path, 'model': model, 'meta': meta}
        print(f"✅ Successfully loaded '{os.path.basename(path)}'")
        return True
    except Exception as e:
        print(f"!!! ERROR loading model '{path}': {e}")
        MODEL_CACHE = {'path': None, 'model': None, 'meta': None}
        return False

# ==============================================================================
#                            FLASK WEB ROUTES
# ==============================================================================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat")
def chat_page():
    models = find_models()
    return render_template("chat.html", models=models)

@app.route("/train")
def train_page():
    is_training = training_process is not None and training_process.poll() is None
    return render_template("train.html", is_training=is_training)

# ==============================================================================
#                        REAL-TIME SOCKETIO EVENTS
# ==============================================================================

def stream_training_output(process):
    """ The heart of the live telemetry feed. """
    global training_process
    for line in process.stdout:
        socketio.emit('training_update', {'data': line})
        socketio.sleep(0)
    process.wait()
    print("Training process finished.")
    socketio.emit('training_finished', {'data': '✅ Mission Complete!'})
    with training_lock:
        training_process = None

@socketio.on('start_training')
def handle_start_training(json_data):
    """ Called when the user clicks the "Launch" button. """
    global training_process
    with training_lock:
        if training_process and training_process.poll() is None:
            emit('training_error', {'error': 'A training session is already in progress.'})
            return
            
        print("Received request to start training expedition.")
        command = ['python', '-u', 'pico_gpt_train.py']
        training_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace'
        )
        
    socketio.start_background_task(target=stream_training_output, process=training_process)
    print(f"Launched training process with PID: {training_process.pid}")
    emit('training_started', {'data': '🚀 Launch sequence initiated...'})

@socketio.on('stop_training')
def handle_stop_training(json_data):
    """ The merciful kill switch. """
    global training_process
    with training_lock:
        if training_process and training_process.poll() is None:
            print(f"Received request to stop training process {training_process.pid}")
            os.kill(training_process.pid, signal.SIGINT)
            emit('training_update', {'data': "\n\n🛑 Shutdown signal sent. Awaiting graceful exit...\n\n"})
        else:
            emit('training_error', {'error': 'No active training session to stop.'})

# --- API Endpoint for Chatting ---
@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json
    model_path = data.get('model_path')
    message = data.get('message')
    temperature = float(data.get('temperature', 0.8))
    max_tokens = int(data.get('max_tokens', 512))

    if not model_path or not message:
        abort(400, "Model/message required.")
        
    if not load_model(model_path):
        abort(500, f"Failed to load model: {model_path}")
        
    model = MODEL_CACHE['model']
    meta = MODEL_CACHE['meta']
    encode = lambda s: [meta['stoi'].get(c, 0) for c in s]
    decode = lambda l: ''.join([meta['itos'].get(i, '') for i in l])
    
    x = (torch.tensor(encode(message), dtype=torch.long, device=DEVICE)[None, ...])
    
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=max_tokens, temperature=temperature, top_k=None)
        response_text = decode(y[0].tolist())
        bot_response = response_text[len(message):] if response_text.startswith(message) else response_text
        
    return jsonify({"response": bot_response})

# --- Application Launcher ---
if __name__ == '__main__':
    print("Starting Flask AI Hub v2.2 (Fleet Admiral Edition)...")
    print("Local access: http://127.0.0.1:5000")
    
    socketio.run(app, host='0.0.0.0', port=5000)
import subprocess
import mlflow
import os
import re
from mlflow.tracking import MlflowClient

# --- 0. Configuration and Credentials (NO CHANGE) ---
# NOTE: Credentials must be set via OS environment variables (export commands)
os.environ["MLFLOW_TRACKING_USERNAME"] = "yashwatwani28"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "d82940c2138e8370272889954f54b5647c92b9bf"

MLFLOW_URI = "https://dagshub.com/yashwatwani28/mlop2.mlflow"
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("politeness-bot-experiment")

# --- 1. Define Paths and Parameters (MODIFIED FOR CPU/GGUF) ---
# Since we can't train with MLX, we will use a hypothetical command for CPU training.
# In a real environment, you would use a tool like Llama Factory or Llama.cpp for this.
MODEL_PATH = "/Users/yash.watwani/Documents/mlops2/gguf-models/tinyllama-cpu.gguf"
ADAPTERS_PATH = "/Users/yash.watwani/Documents/mlops2/adapters"
DATA_PATH = "/Users/yash.watwani/Documents/mlops2/data/mlx_format"
ITERATIONS = "10" # Keep low for fast testing

# The simulated command using a generic 'train' script with CPU flags
# NOTE: This is a placeholder command structure, as direct training of Llama.cpp 
# models often requires specialized tools like Llama Factory.
MLX_COMMAND = [
    "python", 
    "mlx-examples/lora/lora.py", # Using the MLX script structure as a wrapper/placeholder
    "--model", MODEL_PATH,
    "--train",
    "--data", DATA_PATH,
    "--adapter-file", f"{ADAPTERS_PATH}/politeness_adapters.npz",
    "--iters", ITERATIONS,
    "--batch-size", "1",
    #"--device", "cpu", # Explicitly request CPU processing
]

# --- 2. Subprocess Streaming Function (NO CHANGE - This works as the bridge) ---
def parse_and_log_stream(process):
    """
    Captures the output stream from the MLX training script, 
    parses for the Loss metric, and logs it to MLflow.
    """
    # NEW REGEX PATTERNS: Look for 'Train loss' or 'Val loss' followed by a number
    TRAIN_LOSS_PATTERN = re.compile(r"Train loss\s*([\d\.]+)") 
    VAL_LOSS_PATTERN = re.compile(r"Val loss\s*([\d\.]+)") 
    ITER_PATTERN = re.compile(r"Iter\s*(\d+):") # To capture the correct step number
    
    current_iter = 0

    print("--- Starting LLM CPU Stream Parser & Metric Logger ---")
    
    while True:
        output = process.stdout.readline()
        if not output and process.poll() is not None:
            break
        
        if output:
            output = output.strip()
            print(output)

            # 1. Capture the iteration number to use as the step
            iter_match = ITER_PATTERN.search(output)
            if iter_match:
                try:
                    # MLX outputs 'Iter 10:' but sometimes skips numbers, so we rely on the printed number
                    current_iter = int(iter_match.group(1))
                except:
                    pass

            # 2. Search for Validation Loss (Occurs first, usually right after Iter)
            val_loss_match = VAL_LOSS_PATTERN.search(output)
            if val_loss_match:
                try:
                    loss_value = float(val_loss_match.group(1))
                    # Log the metric with the specific name and captured step number
                    mlflow.log_metric("val_loss", loss_value, step=current_iter)
                except:
                    pass

            # 3. Search for Training Loss
            train_loss_match = TRAIN_LOSS_PATTERN.search(output)
            if train_loss_match:
                try:
                    loss_value = float(train_loss_match.group(1))
                    # Log the metric with the specific name and captured step number
                    mlflow.log_metric("train_loss", loss_value, step=current_iter)
                except:
                    pass

# --- 3. Execution Block (NO CHANGE) ---
with mlflow.start_run() as run:
    print(f"Starting MLflow Run ID: {run.info.run_id}")
    
    # Log fixed parameters
    mlflow.log_param("model_type", "CPU_GGUF_TinyLlama")
    mlflow.log_param("iterations", ITERATIONS)
    mlflow.log_param("data_source", "DVC_Drive")
    mlflow.log_param("device", "CPU")

    # 1. Start the MLX script using Popen (for streaming and metric capture)
    try:
        process = subprocess.Popen(
            MLX_COMMAND, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            # We must run this from the project root where 'mlx-examples' lives
            cwd=os.getcwd() 
        )

        # 2. Run the parser to capture and log metrics until the process finishes
        parse_and_log_stream(process)
        
        # 3. Wait for the subprocess to finish and check status
        process.wait()

        if process.returncode != 0:
            # If the MLX script exited with an error code (1), fail the MLflow run
            raise subprocess.CalledProcessError(process.returncode, MLX_COMMAND)
            
        # 4. Log the final trained artifact
        mlflow.log_artifact(f"{ADAPTERS_PATH}/politeness_adapters.npz", artifact_path="model_adapters")
        print(f"✅ Run successful. Metrics and Artifacts logged to Dagshub.")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error code {e.returncode}. Check local console output.")
        # End the MLflow run as 'Failed'
        mlflow.set_tag("mlflow.runStatus", "Failed")
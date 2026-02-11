import os
import subprocess
import yaml
import glob

# Models to evaluate
models = [
    "crop_ablation_anyres_e2p",
    "crop_ablation_cubemap",
    "crop_ablation_e2p",
    "crop_ablation_resize"
]

base_path = "CORA/outputs"
csv_input = "CORA/data/quic360/test_fixed.csv"
python_exec = "/data/3_lib/miniconda3/envs/pano/bin/python"

# Find CUDA library path for Mamba
cuda_libs = glob.glob("/data/3_lib/miniconda3/envs/pano/lib/python*/site-packages/nvidia/cuda_runtime/lib")
cuda_lib_path = cuda_libs[0] if cuda_libs else None
if cuda_lib_path:
    print(f"Found CUDA lib: {cuda_lib_path}")
else:
    print("Warning: CUDA lib not found")

# Prepare configs and launch
procs = []
for i, model in enumerate(models):
    config_path = f"{base_path}/{model}/finetune/config.yaml"
    
    # Read config
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config not found for {model}, skipping")
        continue
    
    # Update batch size
    if 'training' not in config: config['training'] = {}
    config['training']['eval_batch_size'] = 64  # Increased batch size
    
    temp_config = f"temp_config_{model}.yaml"
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    # Determine GPU (distribute 4 models across 2 GPUs)
    gpu_id = i % 2
    
    # Command
    cmd = [
        python_exec, "scripts/eval.py",
        "--checkpoint-dir", f"{base_path}/{model}/finetune/",
        "--csv-input", csv_input,
        "--config", temp_config
    ]
    
    # Environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":" + os.path.abspath("CORA")
    
    if cuda_lib_path:
        current_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{cuda_lib_path}:{current_ld}"
    
    # Launch
    log_file_name = f"eval_{model}.log"
    log_file = open(log_file_name, "w")
    print(f"Launching {model} on GPU {gpu_id} > {log_file_name}")
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    procs.append(proc)

print("All evaluations launched in background.")

from huggingface_hub import file_exists

models_to_check = [
    "OpenGVLab/InternVL2-1B",
    "OpenGVLab/InternVL2_5-1B",
    "OpenGVLab/InternVL-Chat-V1-5",
    "OpenGVLab/InternVL2-2B",
    "OpenGVLab/InternVL2-4B",
]

print("Checking models...")
for model in models_to_check:
    try:
        exists = file_exists(repo_id=model, filename="config.json")
        print(f"{model}: {exists}")
    except Exception as e:
        print(f"{model}: Error {e}")

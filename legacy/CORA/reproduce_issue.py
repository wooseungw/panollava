
import torch
import os

# Define a global class ensuring it's in the module scope
class GlobalClass:
    def __init__(self, x):
        self.x = x

def reproduction():
    obj = GlobalClass(10)
    path = "test_ckpt.pt"
    torch.save(obj, path)
    
    print("Saved checkpoint.")
    
    # Test 1: weights_only=True (default in PT 2.6) - SHOULD FAIL
    print("Test 1: Loading with weights_only=True...")
    try:
        torch.load(path, weights_only=True)
        print("Test 1: SUCCESS (Unexpected if strict)")
    except Exception as e:
        print(f"Test 1: FAILED as expected with error: {e}")

    # Test 2: weights_only=False - SHOULD SUCCEED
    print("Test 2: Loading with weights_only=False...")
    try:
        torch.load(path, weights_only=False)
        print("Test 2: SUCCESS")
    except Exception as e:
        print(f"Test 2: FAILED with error: {e}")
        
    # Clean up
    if os.path.exists(path):
        os.remove(path)

if __name__ == "__main__":
    reproduction()

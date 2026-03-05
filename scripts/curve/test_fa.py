import torch
from flash_attn import flash_attn_func

# Set up the environment for Blackwell
device = "cuda"
dtype = torch.bfloat16 # Blackwell's preferred data type

print(f"--- System Check ---")
print(f"PyTorch Version: {torch.__version__}")
print(f"GPU Detected: {torch.cuda.get_device_name(0)}")

try:
    print("\nInitializing Tensors (Batch: 2, SeqLen: 1024, Heads: 16, Dim: 128)...")
    # Flash Attention expects shape: (batch_size, seq_len, num_heads, head_dim)
    q = torch.randn(2, 1024, 16, 128, device=device, dtype=dtype)
    k = torch.randn(2, 1024, 16, 128, device=device, dtype=dtype)
    v = torch.randn(2, 1024, 16, 128, device=device, dtype=dtype)

    print("Executing FlashAttention Forward Pass...")
    # Run the custom CUDA kernel
    output = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
    
    print("\n[SUCCESS] FlashAttention is fully operational!")
    print(f"Output Tensor Shape: {output.shape}")

except Exception as e:
    print("\n[FAILED] FlashAttention threw an error:")
    print(e)

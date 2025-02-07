def test_cupy():

    import cupy as cp

    # Check if CuPy can detect CUDA
    try:
        gpu_id = cp.cuda.Device(0).id
        gpu_name = cp.cuda.Device(0)
        print(f"CuPy is using GPU: {gpu_name} (ID: {gpu_id})")
    except cp.cuda.runtime.CUDARuntimeError as e:
        print("CuPy cannot access the GPU. Ensure CUDA is installed correctly.")
        print(e)
        exit()

    # Perform a simple GPU computation
    try:
        a = cp.random.rand(1000, 1000)  # Create a random matrix on the GPU
        b = cp.random.rand(1000, 1000)
        
        c = cp.dot(a, b)  # Matrix multiplication using the GPU
        
        print("GPU computation successful! Matrix multiplication result shape:", c.shape)
    except Exception as e:
        print("CuPy encountered an issue during computation.")
        print(e)

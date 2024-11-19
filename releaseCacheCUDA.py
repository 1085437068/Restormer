import os
import sys
import torch
import gc
import time

def release_and_restart():
    """
    释放 CUDA 显存并重启 Python 进程
    """
    print("Releasing all CUDA resources...")
    
    # 强制进行垃圾回收
    gc.collect()
    
    # 清除 PyTorch 的 CUDA 缓存
    torch.cuda.empty_cache()
    
    # 确保所有 CUDA 操作完成
    torch.cuda.synchronize()

    print("CUDA resources released. Restarting the process...")
    
    # 稍微等待一会，确保显存释放完成
    time.sleep(2)
    
    # 重启 Python 进程
    os.execv(sys.executable, ['python'] + sys.argv)

if __name__ == "__main__":
    # 确保在有 GPU 的情况下执行
    if torch.cuda.is_available():
        release_and_restart()
    else:
        print("CUDA is not available. No need to restart the process.")

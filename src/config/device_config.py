import os
import tensorflow as tf

def configure_device(cpu_only=True):
    if cpu_only:
        
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        tf.config.set_visible_devices([], "GPU")
        print("Running in CPU-only mode")
    else:
        print("ℹGPU mode requested (not used in this setup)")

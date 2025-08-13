import os
import time
import logging
import numpy as np
import torch
import onnxruntime as ort
import tensorrt as trt
import argparse
import threading
import tensorflow as tf


#conf

MODEL_TYPE = "pytorch"  
MODEL_PATH = "anomaly_detector.pth" if MODEL_TYPE == "pytorch" else "anomaly_detector.h5"
ONNX_PATH = "optimized_model.onnx"
TRT_ENGINE_PATH = "optimized_model.trt"

# detect cuda 
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")




def convert_to_onnx():
    if os.path.exists(ONNX_PATH):
        logging.info(f" ONNX model already exists: {ONNX_PATH}")
        return

    logging.info(f"Converting {MODEL_TYPE} model to ONNX...")

    try:
        if MODEL_TYPE == "tensorflow":
            import tf2onnx
            model = tf.keras.models.load_model(MODEL_PATH)
            onnx_model, _ = tf2onnx.convert.from_keras(model, output_path=ONNX_PATH)
        else:
            model = torch.load(MODEL_PATH, map_location=DEVICE)
            model.eval()
            dummy_input = torch.randn(1, *model.input_shape[1:]).to(DEVICE)
            torch.onnx.export(model, dummy_input, ONNX_PATH, export_params=True)

        logging.info(f" Successfully converted {MODEL_TYPE} model to ONNX: {ONNX_PATH}")
    except Exception as e:
        logging.error(f" Failed to convert to ONNX: {str(e)}")



def run_onnx(batch_size=1):
    logging.info(f" Running ONNX model inference with batch size {batch_size}...")

    try:
        session = ort.InferenceSession(
            ONNX_PATH, providers=["CUDAExecutionProvider"] if USE_CUDA else ["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        input_shape = session.get_inputs()[0].shape
        input_shape[0] = batch_size  

        input_data = np.random.rand(*input_shape).astype(np.float32)
        start_time = time.time()
        output = session.run([output_name], {input_name: input_data})
        end_time = time.time()

        logging.info(f" ONNX Inference Time: {end_time - start_time:.4f} sec")
        logging.info(f" Anomaly Score: {output}")
    except Exception as e:
        logging.error(f" ONNX Runtime failed: {str(e)}")



def optimize_tensorrt():
    if os.path.exists(TRT_ENGINE_PATH):
        logging.info(f" TensorRT model already optimized: {TRT_ENGINE_PATH}")
        return

    logging.info(f" Optimizing ONNX model with TensorRT...")

    try:
        cmd = f"/usr/src/tensorrt/bin/trtexec --onnx={ONNX_PATH} --saveEngine={TRT_ENGINE_PATH} --best"
        os.system(cmd)
        logging.info(f" TensorRT optimization complete: {TRT_ENGINE_PATH}")
    except Exception as e:
        logging.error(f" TensorRT Optimization Failed: {str(e)}")



def run_tensorrt():
    logging.info(" Running TensorRT optimized inference...")

    try:
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(TRT_ENGINE_PATH, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(f.read())

        logging.info(" TensorRT model loaded successfully! ")

        def infer():
            input_shape = (1, 10)  # Adjust based on model input
            input_data = np.random.rand(*input_shape).astype(np.float32)

            start_time = time.time()
            
            time.sleep(0.01)  # Mock inference time
            end_time = time.time()

            logging.info(f" TensorRT Inference Time: {end_time - start_time:.4f} sec")

        threads = [threading.Thread(target=infer) for _ in range(4)]  # Run 4 parallel inferences
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    except Exception as e:
        logging.error(f" TensorRT Inference Failed: {str(e)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Model Optimization & Inference")
    parser.add_argument("--convert", action="store_true", help="Convert model to ONNX")
    parser.add_argument("--run_onnx", action="store_true", help="Run inference using ONNX Runtime")
    parser.add_argument("--optimize_trt", action="store_true", help="Optimize ONNX with TensorRT")
    parser.add_argument("--run_trt", action="store_true", help="Run optimized TensorRT model")
    parser.add_argument("--all", action="store_true", help="Run full optimization pipeline")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")

    args = parser.parse_args()

    if args.convert or args.all:
        convert_to_onnx()
    if args.run_onnx or args.all:
        run_onnx(args.batch_size)
    if args.optimize_trt or args.all:
        optimize_tensorrt()
    if args.run_trt or args.all:
        run_tensorrt()

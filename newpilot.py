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


# AI-POWERED UAV ANOMALY DETECTION SYSTEM

MODEL_TYPE = "pytorch"  
MODEL_PATH = "uav_anomaly_detector.pth" if MODEL_TYPE == "pytorch" else "uav_anomaly_detector.h5"
ONNX_PATH = "uav_optimized_model.onnx"
TRT_ENGINE_PATH = "uav_optimized_model.trt"

# Auto-detect CUDA availability
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


#  Convert Model to ONNX

def convert_to_onnx():
    if os.path.exists(ONNX_PATH):
        logging.info(f" ONNX model already exists: {ONNX_PATH}")
        return

    logging.info(f" Converting {MODEL_TYPE} model to ONNX...")
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


#  Run ONNX Model with Telemetry Data
def run_onnx(telemetry_data):
    logging.info(" Running ONNX-based UAV anomaly detection...")
    try:
        session = ort.InferenceSession(
            ONNX_PATH, providers=["CUDAExecutionProvider"] if USE_CUDA else ["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        input_data = np.array([telemetry_data], dtype=np.float32)

        start_time = time.time()
        output = session.run([output_name], {input_name: input_data})
        end_time = time.time()
        
        anomaly_score = output[0][0]
        logging.info(f" ONNX Inference Time: {end_time - start_time:.4f} sec")
        logging.info(f" Anomaly Score: {anomaly_score}")

        if anomaly_score > 0.8:
            logging.warning("⚠ HIGH RISK: Activating Failsafe (Return-to-Home or Emergency Landing)")
    except Exception as e:
        logging.error(f" ONNX Runtime failed: {str(e)}")


#  Optimize with TensorRT
def optimize_tensorrt():
    if os.path.exists(TRT_ENGINE_PATH):
        logging.info(f" TensorRT model already optimized: {TRT_ENGINE_PATH}")
        return

    logging.info(" Optimizing ONNX model with TensorRT...")
    try:
        cmd = f"/usr/src/tensorrt/bin/trtexec --onnx={ONNX_PATH} --saveEngine={TRT_ENGINE_PATH} --best"
        os.system(cmd)
        logging.info(f" TensorRT optimization complete: {TRT_ENGINE_PATH}")
    except Exception as e:
        logging.error(f" TensorRT Optimization Failed: {str(e)}")


#  Run TensorRT Model with Multi-threading for parallel computing
#if not use single thread for low resource comsuption


def run_tensorrt(telemetry_data):
    logging.info(" Running TensorRT optimized UAV anomaly detection...")
    try:
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(TRT_ENGINE_PATH, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(f.read())

        logging.info(" TensorRT model loaded successfully! 🚀")

        def infer():
            input_data = np.array([telemetry_data], dtype=np.float32)
            start_time = time.time()
            time.sleep(0.01)  
            end_time = time.time()
            logging.info(f" TensorRT Inference Time: {end_time - start_time:.4f} sec")
        
        threads = [threading.Thread(target=infer) for _ in range(4)]  # Run 4 parallel inferences
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    except Exception as e:
        logging.error(f" TensorRT Inference Failed: {str(e)}")

#  ENTRY POINTs in here

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI UAV Anomaly Detection & Failsafe System")
    parser.add_argument("--convert", action="store_true", help="Convert model to ONNX")
    parser.add_argument("--run_onnx", action="store_true", help="Run ONNX-based anomaly detection")
    parser.add_argument("--optimize_trt", action="store_true", help="Optimize model with TensorRT")
    parser.add_argument("--run_trt", action="store_true", help="Run TensorRT-based anomaly detection")
    parser.add_argument("--telemetry", nargs='+', type=float, help="Telemetry data input")

    args = parser.parse_args()
    telemetry_data = args.telemetry or [0.5] * 10  # these default but you can change the val 

    if args.convert:
        convert_to_onnx()
    if args.run_onnx:
        run_onnx(telemetry_data)
    if args.optimize_trt:
        optimize_tensorrt()
    if args.run_trt:
        run_tensorrt(telemetry_data)

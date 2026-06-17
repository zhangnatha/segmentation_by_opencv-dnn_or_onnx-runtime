import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

DEFAULT_CUDA_BENCHMARK_ITERATIONS = 20
DEFAULT_CUDA_WARMUP_ITERATIONS = 5
DEFAULT_TIMING_WARMUP_ITERATIONS = 1


def result_path(file_name):
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, file_name)


def load_class_names(model_path):
    names_file_path = os.path.join(os.path.dirname(os.path.abspath(model_path)), "classes.names")
    class_names = {}
    if not os.path.exists(names_file_path):
        print(f"WARN: '{names_file_path}' not found. Using default numeric IDs.")
        return class_names

    with open(names_file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            name = line.strip()
            if name:
                class_names[idx] = name
    return class_names


def hsv_to_rgb(h, s, v):
    rgb_max = v * 2.55
    rgb_min = rgb_max * (100 - s) / 100.0
    i = h // 60
    difs = h % 60
    rgb_adj = (rgb_max - rgb_min) * difs / 60.0

    if i == 0:
        r, g, b = rgb_max, rgb_min + rgb_adj, rgb_min
    elif i == 1:
        r, g, b = rgb_max - rgb_adj, rgb_max, rgb_min
    elif i == 2:
        r, g, b = rgb_min, rgb_max, rgb_min + rgb_adj
    elif i == 3:
        r, g, b = rgb_min, rgb_max - rgb_adj, rgb_max
    elif i == 4:
        r, g, b = rgb_min + rgb_adj, rgb_min, rgb_max
    else:
        r, g, b = rgb_max, rgb_min, rgb_max - rgb_adj

    return int(r), int(g), int(b)


def generate_palette_bgr(size_of_result):
    size = max(1, int(size_of_result))
    palette = []
    for i in range(size):
        h = int(360.0 / size * i)
        r, g, b = hsv_to_rgb(h, 100, 100)
        palette.append((b, g, r))
    return palette


def create_session(model_path, provider):
    available = ort.get_available_providers()
    if provider in ("cuda", "gpu"):
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is not available. "
                f"onnxruntime={ort.__version__}, available providers={available}"
            )
        providers = ["CUDAExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(model_path, sess_options=options, providers=providers)
    if provider in ("cuda", "gpu") and "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError(
            "CUDAExecutionProvider could not be activated. "
            f"onnxruntime={ort.__version__}, active providers={session.get_providers()}"
        )
    return session


def run_yolo11_seg_ort_inference(
    model_path,
    image_path,
    show_boxes,
    conf_threshold,
    iou_threshold,
    provider,
    benchmark_iterations=1,
    warmup_iterations=0,
):
    class_names = load_class_names(model_path)
    num_classes = len(class_names) if class_names else 4
    palette_bgr = generate_palette_bgr(num_classes)

    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not read image '{image_path}'.")
        return
    original_image = image.copy()
    original_h, original_w = original_image.shape[:2]

    requested_cuda = provider in ("cuda", "gpu")
    benchmark_iterations = max(1, int(benchmark_iterations))
    warmup_iterations = max(0, int(warmup_iterations))

    start_total_time = time.perf_counter()
    session = create_session(model_path, provider)
    print(f" |- Active Providers: {session.get_providers()}")
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    if requested_cuda:
        warmup_blob = cv2.dnn.blobFromImage(original_image, 1 / 255.0, (640, 640), swapRB=True, crop=False)
        try:
            for _ in range(warmup_iterations):
                session.run(output_names, {input_name: warmup_blob})
        except Exception as exc:
            raise RuntimeError(f"CUDA warmup failed: {exc}") from exc

    timing_warmup_blob = cv2.dnn.blobFromImage(original_image, 1 / 255.0, (640, 640), swapRB=True, crop=False)
    try:
        for _ in range(DEFAULT_TIMING_WARMUP_ITERATIONS):
            session.run(output_names, {input_name: timing_warmup_blob})
    except Exception as exc:
        raise RuntimeError(f"Timing warmup failed: {exc}") from exc

    forward_total_ms = 0.0
    pipeline_total_ms = 0.0
    detected_count = 0
    image = original_image.copy()

    for _ in range(benchmark_iterations):
        iter_start = time.perf_counter()
        image = original_image.copy()
        mask_overlay = np.zeros_like(image, dtype=np.uint8)
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (640, 640), swapRB=True, crop=False)

        start_forward = time.perf_counter()
        try:
            outputs = session.run(output_names, {input_name: blob})
        except Exception as exc:
            if requested_cuda:
                raise RuntimeError(f"CUDA inference failed: {exc}") from exc
            raise
        end_forward = time.perf_counter()
        forward_total_ms += (end_forward - start_forward) * 1000.0

        if outputs[0].shape[1] == 32:
            proto = outputs[0]
            preds = outputs[1]
        else:
            preds = outputs[0]
            proto = outputs[1]

        preds = np.squeeze(preds)
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T
        proto = np.squeeze(proto)

        boxes = []
        confidences = []
        class_ids = []
        mask_coefficients = []
        x_factor = original_w / 640.0
        y_factor = original_h / 640.0

        for pred in preds:
            scores = pred[4 : 4 + num_classes]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence <= conf_threshold:
                continue

            cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
            boxes.append([int((cx - w / 2) * x_factor), int((cy - h / 2) * y_factor), int(w * x_factor), int(h * y_factor)])
            confidences.append(confidence)
            class_ids.append(class_id)
            mask_coefficients.append(pred[4 + num_classes :])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)
        text_draw_queue = []

        if len(indices) > 0:
            indices = indices.flatten()
            c, proto_h, proto_w = proto.shape
            proto_flat = proto.reshape(c, -1)

            for i in indices:
                left, top, width, height = boxes[i]
                class_id = class_ids[i]
                color = palette_bgr[class_id % len(palette_bgr)]

                coeffs = np.array(mask_coefficients[i])
                single_mask = np.dot(coeffs, proto_flat)
                single_mask = 1 / (1 + np.exp(-single_mask))
                single_mask = single_mask.reshape(proto_h, proto_w)

                scale_mask = cv2.resize(single_mask, (640, 640), interpolation=cv2.INTER_LINEAR)
                mx1, my1 = max(0, int(left / x_factor)), max(0, int(top / y_factor))
                mx2, my2 = min(640, int((left + width) / x_factor)), min(640, int((top + height) / y_factor))

                crop_mask = np.zeros_like(scale_mask)
                crop_mask[my1:my2, mx1:mx2] = scale_mask[my1:my2, mx1:mx2]
                actual_mask = cv2.resize(crop_mask, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
                mask_overlay[actual_mask > 0.5] = color

                if show_boxes:
                    cv2.rectangle(mask_overlay, (left, top), (left + width, top + height), color, 2)

                label = f"{class_names.get(class_id, str(class_id))}: {confidences[i]:.2f}"
                font_scale = 0.55
                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                if show_boxes:
                    text_x = left
                    text_y = top - 8
                    if text_y < text_h + 5:
                        text_y = top + height - 8
                        if text_y > original_h - 5:
                            text_y = top + text_h + 5
                else:
                    text_x = left + (width // 2) - (text_w // 2)
                    text_y = top + (height // 2) + (text_h // 2)

                text_x = max(5, min(text_x, original_w - text_w - 5))
                text_y = max(text_h + 5, min(text_y, original_h - baseline - 5))
                text_draw_queue.append((label, (text_x, text_y), font_scale))

            cv2.addWeighted(mask_overlay, 0.4, image, 1.0, 0, image)
            for label, (x, y), scale in text_draw_queue:
                cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            print("WARN: No targets detected.")

        detected_count = len(indices) if len(indices) > 0 else 0
        pipeline_total_ms += (time.perf_counter() - iter_start) * 1000.0

    forward_ms = forward_total_ms / benchmark_iterations
    total_ms = pipeline_total_ms / benchmark_iterations
    print("-------------------------------------------")
    print(f"PERF: Pure inference (forward) cost: {forward_ms:.2f} ms")
    print(f"PERF: Total algorithm pipeline cost: {total_ms:.2f} ms (including pre/post-processing)")
    print(f"INFO: Inference finished. Successfully detected {detected_count} targets.")
    print("-------------------------------------------")

    active_provider = "cuda" if "CUDAExecutionProvider" in session.get_providers() else "cpu"
    output_name = result_path(f"result_onnxruntime_python_{active_provider}.png")
    cv2.imwrite(output_name, image)
    print(f"INFO: Result image saved to: {output_name}")
    if os.environ.get("OPENCV_ONNX_SHOW") == "1":
        try:
            cv2.imshow("ONNX Runtime Python YOLO11-Seg", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as exc:
            print(f"WARN: Could not open display window: {exc}")


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("ERROR: Insufficient arguments!")
        print(
            "USAGE: python main_ort.py <model_path.onnx> <image_path.png/.jpg> "
            "<True/False> <conf_threshold> <iou_threshold> [cpu|cuda]"
        )
        sys.exit(-1)

    model_arg = sys.argv[1]
    image_arg = sys.argv[2]
    show_boxes_arg = sys.argv[3].lower() in ["true", "1", "t", "y", "yes"]
    conf_threshold_arg = float(sys.argv[4])
    iou_threshold_arg = float(sys.argv[5])
    provider_arg = sys.argv[6].lower() if len(sys.argv) >= 7 else "cpu"
    benchmark_iterations_arg = DEFAULT_CUDA_BENCHMARK_ITERATIONS if provider_arg in ("cuda", "gpu") else 1
    warmup_iterations_arg = DEFAULT_CUDA_WARMUP_ITERATIONS if provider_arg in ("cuda", "gpu") else 0

    print("INFO: Starting ONNX Runtime Python inference...")
    print(f" |- Model: {model_arg}")
    print(f" |- Image: {image_arg}")
    print(f" |- Provider: {provider_arg}")
    print(f" |- ONNX Runtime Version: {ort.__version__}")
    print(f" |- Available Providers: {ort.get_available_providers()}")
    print(f" |- Show Bounding Boxes: {show_boxes_arg}")
    print(f" |- Confidence Threshold: {conf_threshold_arg}")
    print(f" |- IoU NMS Threshold: {iou_threshold_arg}")
    print(f" |- Benchmark Iterations: {benchmark_iterations_arg}")
    print(f" |- Warmup Iterations: {warmup_iterations_arg}")

    run_yolo11_seg_ort_inference(
        model_arg,
        image_arg,
        show_boxes_arg,
        conf_threshold_arg,
        iou_threshold_arg,
        provider_arg,
        benchmark_iterations_arg,
        warmup_iterations_arg,
    )

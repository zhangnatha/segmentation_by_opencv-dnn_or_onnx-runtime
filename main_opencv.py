import sys
import os
import time
import cv2
import numpy as np


def has_display():
    return os.environ.get("OPENCV_ONNX_SHOW") == "1"


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


def run_yolo11_seg_full_inference(model_path, image_path, show_boxes, conf_threshold, iou_threshold):
    class_names = load_class_names(model_path)
    num_classes = len(class_names) if class_names else 4
    palette_bgr = generate_palette_bgr(num_classes)

    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not read image '{image_path}'.")
        return
    original_image = image.copy()
    original_h, original_w = original_image.shape[:2]

    start_total_time = time.perf_counter()

    input_size = (640, 640)
    blob = cv2.dnn.blobFromImage(image, 1/255.0, input_size, swapRB=True, crop=False)

    net = cv2.dnn.readNetFromONNX(model_path)
    net.setInput(blob)
    out_names = net.getUnconnectedOutLayersNames()

    start_forward = time.perf_counter()
    outputs = net.forward(out_names)
    end_forward = time.perf_counter()
    forward_ms = (end_forward - start_forward) * 1000.0

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

        if confidence > conf_threshold:
            cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]

            left = int((cx - w / 2) * x_factor)
            top = int((cy - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            boxes.append([left, top, width, height])
            confidences.append(confidence)
            class_ids.append(class_id)
            mask_coefficients.append(pred[4 + num_classes :])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)
    text_draw_queue = []

    if len(indices) > 0:
        indices = indices.flatten()
        c, proto_h, proto_w = proto.shape
        proto_flat = proto.reshape(c, -1)

        mask_overlay = np.zeros_like(original_image, dtype=np.uint8)

        for i in indices:
            left, top, width, height = boxes[i]
            class_id = class_ids[i]
            color = palette_bgr[class_id % len(palette_bgr)]

            coeffs = np.array(mask_coefficients[i])
            single_mask = np.dot(coeffs, proto_flat)
            single_mask = single_mask.reshape(proto_h, proto_w)

            # Sigmoid
            single_mask = 1 / (1 + np.exp(-single_mask))

            scale_mask = cv2.resize(single_mask, (640, 640), interpolation=cv2.INTER_LINEAR)

            # box 区域（无 padding，避免小目标异常）
            mx1 = max(0, int(left / x_factor))
            my1 = max(0, int(top / y_factor))
            mx2 = min(640, int((left + width) / x_factor))
            my2 = min(640, int((top + height) / y_factor))

            crop_mask = np.zeros_like(scale_mask)
            if mx2 > mx1 and my2 > my1:
                crop_mask[my1:my2, mx1:mx2] = scale_mask[my1:my2, mx1:mx2]

            actual_mask = cv2.resize(crop_mask, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

            binary_mask = actual_mask > 0.5
            mask_overlay[binary_mask] = color

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

        image = original_image.copy()
        cv2.addWeighted(mask_overlay, 0.4, image, 1.0, 0, image)

        for label, (x, y), scale in text_draw_queue:
            cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)

        end_total_time = time.perf_counter()
        total_ms = (end_total_time - start_total_time) * 1000.0

        print("-------------------------------------------")
        print(f"PERF: Pure inference (forward) cost: {forward_ms:.2f} ms")
        print(f"PERF: Total algorithm pipeline cost: {total_ms:.2f} ms (including pre/post-processing)")
        print(f"INFO: Inference finished. Successfully detected {len(indices)} targets.")
        print("-------------------------------------------")
    else:
        print("WARN: No targets detected.")
        image = original_image.copy()

    output_name = result_path("result_opencv_dnn_python_cpu.png")
    cv2.imwrite(output_name, image)
    print(f"INFO: Result image saved to: {output_name}")
    if has_display():
        try:
            cv2.imshow("OpenCV 5 Python YOLO11-Seg", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as exc:
            print(f"WARN: Could not open display window: {exc}")


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("ERROR: Insufficient arguments!")
        print("USAGE: python main_opencv.py <model_path.onnx> <image_path.png/.jpg> <True/False> <conf_threshold> <iou_threshold>")
        sys.exit(-1)

    model_arg = sys.argv[1]
    image_arg = sys.argv[2]
    show_boxes_arg = sys.argv[3].lower() in ['true', '1', 't', 'y', 'yes']
    conf_threshold_arg = float(sys.argv[4])
    iou_threshold_arg = float(sys.argv[5])

    print("INFO: Starting OpenCV 5 Python inference...")
    print(f" └─ Model: {model_arg}")
    print(f" └─ Image: {image_arg}")
    print(f" └─ Show Bounding Boxes: {show_boxes_arg}")
    print(f" └─ Confidence Threshold: {conf_threshold_arg}")
    print(f" └─ IoU NMS Threshold: {iou_threshold_arg}")

    run_yolo11_seg_full_inference(model_arg, image_arg, show_boxes_arg, conf_threshold_arg, iou_threshold_arg)

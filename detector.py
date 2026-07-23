import cv2
import numpy as np
import config
import os
import time

class PersonDetector:
    def __init__(self):
        self.use_openvino = False
        self.use_ultralytics = False
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_abs_path = os.path.join(base_dir, config.MODEL_PATH)
        
        try:
            import openvino as ov
            core = ov.Core()
            if 'GPU' in core.available_devices:
                gpu_name = core.get_property('GPU', 'FULL_DEVICE_NAME')
                print(f"[INFO] Ditemukan GPU: {gpu_name}")
                if "Intel" in gpu_name or "UHD Graphics" in gpu_name:
                    self.use_openvino = True
                    self.ov_core = core
                    print("[INFO] Menggunakan library: openvino (Intel iGPU)")
        except ImportError:
            pass

        if self.use_openvino:
            import openvino as ov
            # OpenVINO 2023.0+ can read .tflite files directly
            ov_model = self.ov_core.read_model(model_abs_path)
            self.compiled_model = self.ov_core.compile_model(ov_model, "GPU")
            self.infer_request = self.compiled_model.create_infer_request()

            input_node = self.compiled_model.inputs[0]
            input_shape = input_node.shape
            
            self.is_nchw = (input_shape[1] == 3)
            self.input_height = input_shape[2] if self.is_nchw else input_shape[1]
            self.input_width = input_shape[3] if self.is_nchw else input_shape[2]
            
            if input_node.element_type == ov.Type.f32:
                self.input_dtype = np.float32
            else:
                self.input_dtype = np.uint8

            output_node = self.compiled_model.outputs[0]
            output_shape = output_node.shape
        else:
            tflite = None
            try:
                import ai_edge_litert.interpreter as tflite
                print("[INFO] Library TFLite: ai_edge_litert")
            except ImportError:
                try:
                    import tflite_runtime.interpreter as tflite
                    print("[INFO] Library TFLite: tflite_runtime")
                except ImportError:
                    try:
                        import tensorflow.lite as tflite
                        print("[INFO] Library TFLite: tensorflow.lite")
                    except ImportError:
                        try:
                            from ultralytics import YOLO
                            print("[INFO] Menggunakan library: ultralytics (Fallback)")
                            self.use_ultralytics = True
                            self.yolo_model = YOLO(model_abs_path, task='detect')
                            print(f"[INFO] Model loaded successfully via Ultralytics: {model_abs_path}")
                            return
                        except ImportError:
                            raise ImportError("Tidak dapat menemukan library TFLite atau Ultralytics! Pastikan 'ai-edge-litert', 'tflite-runtime', 'tensorflow', atau 'ultralytics' terinstall.")

            self.interpreter = tflite.Interpreter(model_path=model_abs_path, num_threads=config.NUM_THREADS)
            self.interpreter.allocate_tensors()

            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            self.input_shape = self.input_details[0]['shape']
            self.is_nchw = (self.input_shape[1] == 3)
            self.input_height = self.input_shape[2] if self.is_nchw else self.input_shape[1]
            self.input_width = self.input_shape[3] if self.is_nchw else self.input_shape[2]
            self.input_dtype = self.input_details[0]['dtype']

            output_shape = self.output_details[0]['shape']
        # YOLOv8/v11 output tensor biasanya berdimensi [1, channels, boxes]
        # channels mewakili 4 koordinat box + jumlah class
        num_channels = output_shape[1] if output_shape[1] < output_shape[2] else output_shape[2]
        
        if num_channels == 6:
            print(f"[INFO] Loaded model output channels: {num_channels} (Detected End-to-End NMS-Free model)")
        else:
            num_classes = num_channels - 4
            if num_classes == 1:
                print(f"[INFO] Loaded model output channels: {num_channels} (Detected Person-Only model)")
            else:
                print(f"[INFO] Loaded model output channels: {num_channels} (Detected {num_classes}-class COCO model)")

        # State tracking per camera/device to handle temporal confirmation and cooldown
        self.tracks = {}
        self.last_send_time = {}

    def run_inference(self, img_bgr, device_id="default", bypass_temporal=False):
        if self.use_ultralytics:
            results = self.yolo_model(img_bgr, conf=config.CONFIDENCE_THRESHOLD, verbose=False)[0]
            final_boxes = []
            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if cls_id == 0:
                    xyxy = box.xyxyn[0].tolist()
                    final_boxes.append({
                        "confidence": round(conf, 2),
                        "posisi": [
                            round(float(max(0.0, min(xyxy[0], 1.0))), 4),
                            round(float(max(0.0, min(xyxy[1], 1.0))), 4),
                            round(float(max(0.0, min(xyxy[2], 1.0))), 4),
                            round(float(max(0.0, min(xyxy[3], 1.0))), 4),
                        ]
                    })
            return self._apply_pipeline_filters(final_boxes, img_bgr, device_id, bypass_temporal=bypass_temporal)

        # Preprocessing
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        if self.is_nchw:
            img_resized = np.transpose(img_resized, (2, 0, 1))
        img_input = np.expand_dims(img_resized, axis=0)

        if self.input_dtype == np.float32:
            img_input = img_input.astype(np.float32) / 255.0
        else:
            img_input = img_input.astype(self.input_dtype)

        # Inferensi
        if self.use_openvino:
            import openvino as ov
            # Convert contiguous numpy array to OpenVINO tensor
            ov_tensor = ov.Tensor(np.ascontiguousarray(img_input))
            self.infer_request.set_input_tensor(ov_tensor)
            self.infer_request.infer()
            
            output_tensor = self.infer_request.get_output_tensor(0)
            output_data = output_tensor.data
            output_data = np.squeeze(output_data)
        else:
            self.interpreter.set_tensor(self.input_details[0]['index'], img_input)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            output_data = np.squeeze(output_data)

        # Transpose: (84, 8400) → (8400, 84) atau (6, N) -> (N, 6)
        if output_data.shape[0] < output_data.shape[1]:
            output_data = output_data.T

        orig_h, orig_w = img_bgr.shape[:2]
        num_channels = output_data.shape[1]

        boxes_for_nms = []
        confidences = []
        box_coords_norm = []

        if num_channels == 6:
            # YOLO End-to-End Format: [x1, y1, x2, y2, confidence, class_id]
            class_confs = output_data[:, 4]
            class_ids = output_data[:, 5].astype(np.int32)

            mask = (class_ids == 0) & (class_confs > config.CONFIDENCE_THRESHOLD)

            if np.any(mask):
                matching_rows = output_data[mask]
                matching_confs = class_confs[mask]

                max_coord = np.max(matching_rows[:, :4])
                if max_coord <= 1.0:
                    x1_n = matching_rows[:, 0]
                    y1_n = matching_rows[:, 1]
                    x2_n = matching_rows[:, 2]
                    y2_n = matching_rows[:, 3]
                else:
                    x1_n = matching_rows[:, 0] / self.input_width
                    y1_n = matching_rows[:, 1] / self.input_height
                    x2_n = matching_rows[:, 2] / self.input_width
                    y2_n = matching_rows[:, 3] / self.input_height

                x1_abs = x1_n * orig_w
                y1_abs = y1_n * orig_h
                w_abs  = (x2_n - x1_n) * orig_w
                h_abs  = (y2_n - y1_n) * orig_h

                boxes_for_nms = np.column_stack((x1_abs, y1_abs, w_abs, h_abs)).tolist()
                confidences = matching_confs.tolist()
                box_coords_norm = np.column_stack((x1_n, y1_n, x2_n, y2_n))
        else:
            # Standard YOLOv8/v11 Format: [xc, yc, w, h, class_0, class_1, ...]
            class_scores = output_data[:, 4:]
            
            class_ids = np.argmax(class_scores, axis=1)
            class_confs = class_scores[np.arange(len(class_scores)), class_ids]

            # 1. Filter by confidence threshold & class 0 (person)
            mask = (class_ids == 0) & (class_confs > config.CONFIDENCE_THRESHOLD)

            if np.any(mask):
                matching_rows = output_data[mask]
                matching_confs = class_confs[mask]

                max_coord = np.max(matching_rows[:, :4])
                if max_coord <= 1.0:
                    xc_n = matching_rows[:, 0]
                    yc_n = matching_rows[:, 1]
                    w_n  = matching_rows[:, 2]
                    h_n  = matching_rows[:, 3]
                else:
                    xc_n = matching_rows[:, 0] / self.input_width
                    yc_n = matching_rows[:, 1] / self.input_height
                    w_n  = matching_rows[:, 2] / self.input_width
                    h_n  = matching_rows[:, 3] / self.input_height

                x1_n = xc_n - w_n / 2.0
                y1_n = yc_n - h_n / 2.0
                x2_n = xc_n + w_n / 2.0
                y2_n = yc_n + h_n / 2.0

                x1_abs = x1_n * orig_w
                y1_abs = y1_n * orig_h
                w_abs  = w_n * orig_w
                h_abs  = h_n * orig_h

                boxes_for_nms = np.column_stack((x1_abs, y1_abs, w_abs, h_abs)).tolist()
                confidences = matching_confs.tolist()
                box_coords_norm = np.column_stack((x1_n, y1_n, x2_n, y2_n))

        # 2. Non-Maximum Suppression (NMS) Filtering
        final_boxes = []
        if len(boxes_for_nms) > 0:
            indices = cv2.dnn.NMSBoxes(
                boxes_for_nms, confidences,
                score_threshold=float(config.CONFIDENCE_THRESHOLD),
                nms_threshold=float(config.NMS_IOU_THRESHOLD)
            )
            if len(indices) > 0:
                for i in indices.flatten():
                    x1_n, y1_n, x2_n, y2_n = box_coords_norm[i]
                    final_boxes.append({
                        "confidence": round(float(confidences[i]), 2),
                        "posisi": [
                            round(float(max(0.0, min(x1_n, 1.0))), 4),
                            round(float(max(0.0, min(y1_n, 1.0))), 4),
                            round(float(max(0.0, min(x2_n, 1.0))), 4),
                            round(float(max(0.0, min(y2_n, 1.0))), 4)
                        ]
                    })

        # Bersihkan memori tensor dan input
        del img_rgb, img_resized, img_input, output_data
        return self._apply_pipeline_filters(final_boxes, img_bgr, device_id, bypass_temporal=bypass_temporal)

    def _apply_pipeline_filters(self, final_boxes, img_bgr, device_id, bypass_temporal=False):
        """
        Applies geometric filtering (box area and aspect ratio), temporal tracking
        confirmation, and per-camera cooldown logic to reduce false positives.
        """
        orig_h, orig_w = img_bgr.shape[:2]
        
        # 1. Geometric Filters (Area and Aspect Ratio)
        filtered_boxes = []
        for box in final_boxes:
            x1_n, y1_n, x2_n, y2_n = box["posisi"]
            
            # Convert normalized coords to absolute pixels
            w_abs = (x2_n - x1_n) * orig_w
            h_abs = (y2_n - y1_n) * orig_h
            
            # Filter by area: reject if box is too small (e.g. background noise, birds, chickens)
            area = w_abs * h_abs
            aspect_ratio = h_abs / (w_abs + 1e-6)
            
            if area < config.MIN_BOX_AREA:
                continue
                
            # Filter by aspect ratio: humans are generally taller than wide
            # ratio = height / width
            if aspect_ratio < config.MIN_ASPECT_RATIO or aspect_ratio > config.MAX_ASPECT_RATIO:
                continue
                
            filtered_boxes.append(box)
            
        # 2. Temporal Confirmation (Multi-Object Tracking using IoU)
        confirmed_boxes = []
        if device_id not in self.tracks:
            self.tracks[device_id] = []
            
        active_tracks = self.tracks[device_id]
        new_tracks = []
        
        # Helper to compute IoU between normalized box coordinates
        def compute_iou(b1, b2):
            xi1 = max(b1[0], b2[0])
            yi1 = max(b1[1], b2[1])
            xi2 = min(b1[2], b2[2])
            yi2 = min(b1[3], b2[3])
            
            inter_area = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
            b1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
            b2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])
            union_area = b1_area + b2_area - inter_area
            
            if union_area <= 0:
                return 0.0
            return inter_area / union_area

        # Greedy match current detections to active tracks
        pairs = []
        for t_idx, track in enumerate(active_tracks):
            for c_idx, box in enumerate(filtered_boxes):
                iou = compute_iou(track["box"], box["posisi"])
                if iou >= 0.3: # IoU threshold to associate detection with existing track
                    pairs.append((iou, t_idx, c_idx))
                    
        pairs.sort(key=lambda x: x[0], reverse=True)
        
        matched_tracks = set()
        matched_boxes = set()
        
        for iou, t_idx, c_idx in pairs:
            if t_idx not in matched_tracks and c_idx not in matched_boxes:
                matched_tracks.add(t_idx)
                matched_boxes.add(c_idx)
                
                track = active_tracks[t_idx]
                track["box"] = filtered_boxes[c_idx]["posisi"]
                track["frames"] += 1
                new_tracks.append(track)
                
                # Check if object appeared consistently for TEMPORAL_CONFIRMATION_FRAMES
                if bypass_temporal or track["frames"] >= config.TEMPORAL_CONFIRMATION_FRAMES:
                    confirmed_boxes.append(filtered_boxes[c_idx])
                    
        # Start new tracks for unmatched detections (not confirmed yet)
        for c_idx, box in enumerate(filtered_boxes):
            if c_idx not in matched_boxes:
                new_tracks.append({
                    "box": box["posisi"],
                    "frames": 1
                })
                if bypass_temporal:
                    confirmed_boxes.append(box)
                
        # Update device-specific tracking state
        self.tracks[device_id] = new_tracks
        
        # 3. Detection Cooldown (Disabled to allow continuous real-time tracking and bounding box burning)
        # now = time.time()
        # last_send = self.last_send_time.get(device_id, 0)
        # cooldown_active = (now - last_send) < config.DETECTION_COOLDOWN_SECONDS
        # 
        # if len(confirmed_boxes) > 0:
        #     if cooldown_active:
        #         # Suppress reporting during cooldown
        #         return []
        #     else:
        #         # Accept detection, update last send time, and trigger cooldown
        #         self.last_send_time[device_id] = now
        #         return confirmed_boxes
        # else:
        #     # Reset cooldown because no person is detected in this frame
        #     self.last_send_time[device_id] = 0
        #     return []
        
        return confirmed_boxes

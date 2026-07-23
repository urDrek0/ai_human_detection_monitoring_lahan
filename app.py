print("yolo started")

import asyncio
import websockets
import json
import cv2
import numpy as np
import gc
import time

import config
from detector import PersonDetector
from annotator import annotate_image


# Inisialisasi detector secara global (dimuat sekali saat server start)
detector = PersonDetector()

# State store dictionary mapping device_id -> state dict
# e.g., states[device_id] = { 'static_back': float_array, 'prev_frame': gray_blurred }
states = {}

# Helper to group bounding boxes close to each other
def cluster_boxes(boxes, max_dist=200):
    if not boxes:
        return []

    n = len(boxes)
    parent = list(range(n))

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    def are_close(b1, b2):
        x1, y1, w1, h1 = b1[:4]
        x2, y2, w2, h2 = b2[:4]
        
        left1, right1 = x1, x1 + w1
        top1, bottom1 = y1, y1 + h1
        left2, right2 = x2, x2 + w2
        top2, bottom2 = y2, y2 + h2

        h_dist = 0
        if right1 < left2:
            h_dist = left2 - right1
        elif right2 < left1:
            h_dist = left1 - right2
            
        v_dist = 0
        if bottom1 < top2:
            v_dist = top2 - bottom1
        elif bottom2 < top1:
            v_dist = top1 - bottom2

        return h_dist <= max_dist and v_dist <= max_dist

    # Compare all pairs
    for i in range(n):
        for j in range(i + 1, n):
            if are_close(boxes[i], boxes[j]):
                union(i, j)

    # Group boxes by parent
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(boxes[i])

    merged_boxes = []
    for group in groups.values():
        min_x = min(b[0] for b in group)
        min_y = min(b[1] for b in group)
        max_x = max(b[0] + b[2] for b in group)
        max_y = max(b[1] + b[3] for b in group)
        merged_boxes.append((min_x, min_y, max_x - min_x, max_y - min_y))

    return merged_boxes

async def handle_client(websocket, path=None):
    print(f"[INFO] Terhubung ke server, mulai mendengarkan stream...")
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # Binary request (high-performance stream)
                try:
                    # Bytes 0-3: request_id (UInt32BE)
                    # Byte 4: annotate (UInt8)
                    # Byte 5: deviceId length (UInt8)
                    # Bytes 6 to 6 + deviceId length: deviceId string
                    # Bytes 6 + deviceId length+: raw binary JPEG image
                    req_id = int.from_bytes(message[0:4], byteorder='big')
                    annotate_flag = message[4]
                    annotate = annotate_flag in (1, 3)
                    force_yolo = annotate_flag in (2, 3)
                    dev_id_len = message[5]
                    device_id = message[6:6+dev_id_len].decode('utf-8')
                    img_bytes = message[6+dev_id_len:]

                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if img is None:
                        await websocket.send(json.dumps({
                            "requestId": req_id,
                            "status": "error",
                            "message": "File gambar korup"
                        }))
                        continue

                    # Cek mode deteksi aktif dari config
                    mode = getattr(config, "CAMERA_DETECTION_MODE", "AI")
                    
                    if force_yolo:
                        mode = "AI"

                    if mode == "Pixel" or mode == "Hybrid":
                        # Pixel Comparison Motion Detection logic
                        orig_h, orig_w = img.shape[:2]

                        # 1. Grayscale and Gaussian Blur
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)

                        if device_id not in states:
                            states[device_id] = {
                                'static_back': None,
                                'prev_frame': None,
                                'last_reset_time': time.time(),
                                'last_boxes': []
                            }
                        device_state = states[device_id]

                        if device_state['static_back'] is not None and device_state['static_back'].shape != gray_blurred.shape:
                            print(f"[INFO] Static background shape mismatch for {device_id}. Resetting...")
                            device_state['static_back'] = None

                        if device_state['prev_frame'] is not None and device_state['prev_frame'].shape != gray_blurred.shape:
                            print(f"[INFO] Previous frame shape mismatch for {device_id}. Resetting...")
                            device_state['prev_frame'] = None

                        if annotate and 'last_boxes' in device_state and device_state['last_boxes']:
                            # Reuse bounding boxes from the stream frame trigger
                            koordinat_kotak = device_state['last_boxes']
                        else:
                            pixel_mode = getattr(config, "PIXEL_MOTION_MODE", 0)
                            sensitivity = getattr(config, "PIXEL_MOTION_SENSITIVITY", 25)
                            min_area_pixels = getattr(config, "PIXEL_MOTION_MIN_AREA", 10.0)
                            merge = getattr(config, "PIXEL_MOTION_MERGE", False)
                            cluster_dist = getattr(config, "PIXEL_MOTION_CLUSTER_DIST", 50)
                            min_size = getattr(config, "PIXEL_MOTION_MIN_SIZE", 10)

                            # 2. Compute absolute difference berdasarkan mode
                            if pixel_mode == 0:
                                # Periodic background reference reset
                                reset_interval_sec = getattr(config, "PIXEL_MOTION_RESET_INTERVAL", 1)
                                
                                now_time = time.time()
                                if 'last_reset_time' not in device_state:
                                    device_state['last_reset_time'] = now_time
                                    
                                if now_time - device_state['last_reset_time'] >= reset_interval_sec:
                                    print(f"[INFO] Mereset referensi background statis setelah {reset_interval_sec} detik untuk {device_id}")
                                    device_state['static_back'] = None
                                    device_state['last_reset_time'] = now_time

                                # Static Reference mode (dengan running weighted background update lambat)
                                if device_state['static_back'] is None:
                                    device_state['static_back'] = gray_blurred.copy().astype(np.float32)
                                else:
                                    # Update background dengan bobot 0.02 (learning rate 2%) untuk adaptasi cahaya lambat
                                    cv2.accumulateWeighted(gray_blurred, device_state['static_back'], 0.02)
                                
                                static_back_display = cv2.convertScaleAbs(device_state['static_back'])
                                diff = cv2.absdiff(static_back_display, gray_blurred)
                            else:
                                # Frame-to-Frame difference
                                if device_state['prev_frame'] is None:
                                    device_state['prev_frame'] = gray_blurred.copy()
                                
                                diff = cv2.absdiff(device_state['prev_frame'], gray_blurred)
                                if not annotate:
                                    device_state['prev_frame'] = gray_blurred.copy()

                            # 3. Thresholding dan Dilation
                            _, thresh = cv2.threshold(diff, sensitivity, 255, cv2.THRESH_BINARY)
                            thresh = cv2.dilate(thresh, None, iterations=2)

                            # 4. Cari Contours
                            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                            # Filter contours berdasarkan min area dan min size (jika merge aktif)
                            qualifying_boxes = []
                            for contour in contours:
                                area = cv2.contourArea(contour)
                                if area >= min_area_pixels:
                                    (x, y, w, h) = cv2.boundingRect(contour)
                                    if merge:
                                        # Only include if either width or height meets the min size requirement
                                        if w >= min_size or h >= min_size:
                                            qualifying_boxes.append((x, y, w, h, area))
                                    else:
                                        qualifying_boxes.append((x, y, w, h, area))

                            koordinat_kotak = []
                            if len(qualifying_boxes) > 0:
                                if merge:
                                    # Menggabungkan semua bounding box gerakan menjadi 1 box besar
                                    min_x = min(box[0] for box in qualifying_boxes)
                                    min_y = min(box[1] for box in qualifying_boxes)
                                    max_x = max(box[0] + box[2] for box in qualifying_boxes)
                                    max_y = max(box[1] + box[3] for box in qualifying_boxes)
                                    
                                    koordinat_kotak.append({
                                        "confidence": 1.0,
                                        "posisi": [
                                            round(float(min_x / orig_w), 4),
                                            round(float(min_y / orig_h), 4),
                                            round(float(max_x / orig_w), 4),
                                            round(float(max_y / orig_h), 4)
                                        ]
                                    })
                                else:
                                    # Bounding box dikelompokkan (clustering) untuk gerakan yang berdekatan
                                    print(f"[DEBUG] Menjalankan cluster_boxes dengan max_dist={cluster_dist}")
                                    clustered_boxes = cluster_boxes(qualifying_boxes, max_dist=cluster_dist)
                                    for (x, y, w, h) in clustered_boxes:
                                        koordinat_kotak.append({
                                            "confidence": 1.0,
                                            "posisi": [
                                                round(float(x / orig_w), 4),
                                                round(float(y / orig_h), 4),
                                                round(float((x + w) / orig_w), 4),
                                                round(float((y + h) / orig_h), 4)
                                            ]
                                        })
                            
                            # Cache the coordinates
                            device_state['last_boxes'] = koordinat_kotak

                        ada_orang = len(koordinat_kotak) > 0
                        jumlah_orang = len(koordinat_kotak)
                        label_prefix = "Gerakan"
                        pesan_success = "Gerakan terdeteksi!" if ada_orang else "Aman, tidak ada gerakan."

                    else:
                        # Deteksi orang menggunakan subsystem detector (YOLO)
                        koordinat_kotak = detector.run_inference(img, device_id=device_id, bypass_temporal=annotate)
                        jumlah_orang = len(koordinat_kotak)
                        ada_orang = jumlah_orang > 0
                        label_prefix = "Orang"
                        pesan_success = "AWAS: Orang terdeteksi!" if ada_orang else "Aman, tidak ada orang."

                    if annotate:
                        # Anotasi bounding box menggunakan subsystem annotator
                        img_hasil = annotate_image(img, koordinat_kotak, label_prefix=label_prefix)

                        # Encode gambar hasil outlining ke JPEG
                        _, buffer = cv2.imencode('.jpg', img_hasil)
                        img_bytes_out = buffer.tobytes()

                        # Konstruksi binary response
                        metadata = {
                            "status": "success",
                            "pesan": pesan_success,
                            "ada_orang": ada_orang,
                            "jumlah_orang": jumlah_orang,
                            "koordinat_kotak": koordinat_kotak
                        }
                        metadata_bytes = json.dumps(metadata).encode('utf-8')
                        json_len = len(metadata_bytes)

                        # Header: req_id (4 bytes) + json_len (4 bytes)
                        header = req_id.to_bytes(4, byteorder='big') + json_len.to_bytes(4, byteorder='big')
                        response_bytes = header + metadata_bytes + img_bytes_out
                        await websocket.send(response_bytes)

                        del img_hasil, buffer
                    else:
                        # Non-annotated: kirim standard JSON response string
                        response = {
                            "requestId": req_id,
                            "status": "success",
                            "pesan": pesan_success,
                            "ada_orang": ada_orang,
                            "jumlah_orang": jumlah_orang,
                            "koordinat_kotak": koordinat_kotak
                        }
                        await websocket.send(json.dumps(response))

                    # Bersihkan sisa memori RAM untuk performa optimal di Raspberry Pi
                    del img_bytes, nparr, img
                    if mode == "Pixel" or mode == "Hybrid":
                        if 'gray' in locals():
                            del gray
                        if 'gray_blurred' in locals():
                            del gray_blurred
                        if 'diff' in locals():
                            del diff
                        if 'thresh' in locals():
                            del thresh
                        if 'contours' in locals():
                            del contours
                    gc.collect()

                except Exception as e:
                    print(f"[ERROR] Gagal memproses binary request: {e}")
                    try:
                        await websocket.send(json.dumps({
                            "requestId": req_id if 'req_id' in locals() else None,
                            "status": "error",
                            "message": f"Server error: {str(e)}"
                        }))
                    except:
                        pass
            else:
                # Text requests represent JSON config updates
                try:
                    data = json.loads(message)
                    if data.get("type") == "config_update":
                        new_config = data.get("config", {})
                        print(f"[INFO] Menerima pembaruan konfigurasi: {new_config}")
                        if "cameraDetectionMode" in new_config:
                            config.CAMERA_DETECTION_MODE = new_config["cameraDetectionMode"]
                        if "pixelMotionSensitivity" in new_config:
                            config.PIXEL_MOTION_SENSITIVITY = int(new_config["pixelMotionSensitivity"])

                        if "pixelMotionMode" in new_config:
                            config.PIXEL_MOTION_MODE = int(new_config["pixelMotionMode"])
                        if "pixelMotionMerge" in new_config:
                            config.PIXEL_MOTION_MERGE = bool(new_config["pixelMotionMerge"])
                        if "pixelMotionResetInterval" in new_config:
                            config.PIXEL_MOTION_RESET_INTERVAL = int(new_config["pixelMotionResetInterval"])
                        if "pixelMotionClusterDist" in new_config:
                            config.PIXEL_MOTION_CLUSTER_DIST = int(new_config["pixelMotionClusterDist"])
                            # print(f"[DEBUG] config.PIXEL_MOTION_CLUSTER_DIST diperbarui menjadi: {config.PIXEL_MOTION_CLUSTER_DIST}")
                        if "pixelMotionMinSize" in new_config:
                            config.PIXEL_MOTION_MIN_SIZE = int(new_config["pixelMotionMinSize"])
                            # print(f"[DEBUG] config.PIXEL_MOTION_MIN_SIZE diperbarui menjadi: {config.PIXEL_MOTION_MIN_SIZE}")
                except Exception as e:
                    print(f"[ERROR] Gagal memproses pesan konfigurasi teks: {e}")

    except websockets.exceptions.ConnectionClosed:
        print(f"[INFO] Koneksi terputus dari server.")

async def main():
    uri = f"ws://{config.BACKEND_HOST}:{config.BACKEND_PORT}"
    print(f"\n[INFO] Menghubungkan ke Node.js backend di {uri}...")
    while True:
        try:
            async with websockets.connect(uri, max_size=config.MAX_WS_SIZE) as websocket:
                print("[INFO] Terhubung ke Node.js backend successfully.")
                await handle_client(websocket)
        except Exception as e:
            print(f"[WARNING] Koneksi terputus/gagal: {e}. Menghubungkan kembali dalam 3 detik...")
            await asyncio.sleep(3)

if __name__ == '__main__':
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[INFO] Server dihentikan.")
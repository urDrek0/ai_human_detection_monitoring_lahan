import cv2

MIN_CONFIDENCE = 0.5
MIN_AREA = 1500   # pixel²

def annotate_image(img, koordinat_kotak, label_prefix="Person"):

    img_hasil = img.copy()
    h, w = img.shape[:2]

    for box in koordinat_kotak:

        conf = box["confidence"]

        if conf < MIN_CONFIDENCE:
            continue

        x1, y1, x2, y2 = box["posisi"]

        x1 = int(x1 * w)
        y1 = int(y1 * h)
        x2 = int(x2 * w)
        y2 = int(y2 * h)

        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))

        area = (x2-x1)*(y2-y1)

        if area < MIN_AREA:
            continue

        cv2.rectangle(img_hasil,(x1,y1),(x2,y2),(0,0,255),2)

        label=f"{label_prefix} {conf:.0%}"

        (tw,th),base=cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1
        )

        cv2.rectangle(
            img_hasil,
            (x1,y1-th-6),
            (x1+tw+6,y1),
            (0,0,255),
            -1
        )

        cv2.putText(
            img_hasil,
            label,
            (x1+3,y1-4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,255,255),
            1
        )

    return img_hasil
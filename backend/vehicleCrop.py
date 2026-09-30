import cv2

def crop_vehicle(frame,bbox,padding=10):
    if frame is None:
        return None

    if bbox is None:
        return None

    if len(bbox) != 4:
        return None

    frame_height, frame_width = frame.shape[:2]

    x1, y1, x2, y2 = [
        int(value)
        for value in bbox
    ]
    x1 -= padding
    y1 -= padding
    x2 += padding
    y2 += padding

    x1 = max(
        0,
        min(
            x1,
            frame_width - 1,
        ),
    )

    y1 = max(0,min(y1,frame_height - 1))

    x2 = max(0,min(x2,frame_width))

    y2 = max(0,min(y2,frame_height))
    if x2 <= x1:
        return None
    if y2 <= y1:
        return None
    crop = frame[y1:y2,x1:x2]
    if crop.size == 0:
        return None
    return crop
import math
import cv2
from config import MIN_POSE_VISIBILITY, MIN_VISIBLE_KEYPOINTS

NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

SKELETON = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
]


def kp(kpts, idx, conf=0.35):
    if kpts is None or idx >= len(kpts):
        return None

    x, y, c = kpts[idx]

    if float(c) < conf:
        return None

    return float(x), float(y)


def avg(points):
    valid = [p for p in points if p is not None]

    if not valid:
        return None

    return (
        sum(p[0] for p in valid) / len(valid),
        sum(p[1] for p in valid) / len(valid),
    )


def visibility_score(kpts):
    if kpts is None:
        return 0.0

    important = [
        NOSE,
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_HIP,
        RIGHT_HIP,
        LEFT_KNEE,
        RIGHT_KNEE,
        LEFT_ANKLE,
        RIGHT_ANKLE,
    ]

    confs = [float(kpts[i][2]) for i in important if i < len(kpts)]

    return sum(confs) / len(confs) if confs else 0.0


def visible_keypoint_count(kpts, conf=0.35):
    if kpts is None:
        return 0

    return sum(1 for p in kpts if len(p) >= 3 and float(p[2]) >= conf)


def human_pose_valid(kpts):
    if kpts is None:
        return False

    if visibility_score(kpts) < MIN_POSE_VISIBILITY:
        return False

    if visible_keypoint_count(kpts) < MIN_VISIBLE_KEYPOINTS:
        return False

    has_shoulder = kp(kpts, LEFT_SHOULDER) or kp(kpts, RIGHT_SHOULDER)
    has_hip = kp(kpts, LEFT_HIP) or kp(kpts, RIGHT_HIP)

    return bool(has_shoulder and has_hip)


def running_pose_valid(kpts):
    if not human_pose_valid(kpts):
        return False

    legs = [
        kp(kpts, LEFT_KNEE),
        kp(kpts, RIGHT_KNEE),
        kp(kpts, LEFT_ANKLE),
        kp(kpts, RIGHT_ANKLE),
    ]

    arms = [
        kp(kpts, LEFT_ELBOW),
        kp(kpts, RIGHT_ELBOW),
        kp(kpts, LEFT_WRIST),
        kp(kpts, RIGHT_WRIST),
    ]

    return sum(1 for p in legs if p) >= 3 and sum(1 for p in arms if p) >= 2


def aggressive_pose_score(kpts):
    if not human_pose_valid(kpts):
        return 0.0

    shoulders = avg([kp(kpts, LEFT_SHOULDER), kp(kpts, RIGHT_SHOULDER)])
    wrists = [kp(kpts, LEFT_WRIST), kp(kpts, RIGHT_WRIST)]
    elbows = [kp(kpts, LEFT_ELBOW), kp(kpts, RIGHT_ELBOW)]

    score = 0.0

    if shoulders:
        raised_wrists = sum(1 for w in wrists if w and w[1] < shoulders[1])
        score += raised_wrists * 0.25

    valid_arm_points = sum(1 for p in wrists + elbows if p)

    if valid_arm_points >= 3:
        score += 0.30

    return min(score, 1.0)


def body_horizontal_score(kpts):
    if not human_pose_valid(kpts):
        return 0.0

    shoulders = avg([kp(kpts, LEFT_SHOULDER), kp(kpts, RIGHT_SHOULDER)])
    hips = avg([kp(kpts, LEFT_HIP), kp(kpts, RIGHT_HIP)])

    if not shoulders or not hips:
        return 0.0

    dx = abs(shoulders[0] - hips[0])
    dy = abs(shoulders[1] - hips[1])

    ratio = dx / max(1.0, dy)

    return min(1.0, ratio / 2.5)


def hand_regions(kpts, box):
    regions = []
    x1, y1, x2, y2 = box

    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad = int(max(28, min(width, height) * 0.14))

    for idx in [LEFT_WRIST, RIGHT_WRIST]:
        p = kp(kpts, idx)

        if p:
            regions.append(
                (
                    int(max(x1, p[0] - pad)),
                    int(max(y1, p[1] - pad)),
                    int(min(x2, p[0] + pad)),
                    int(min(y2, p[1] + pad)),
                )
            )

    return regions


def norm_dist(p1, p2, w, h):
    return math.dist(p1, p2) / max(1.0, math.sqrt(w * w + h * h))


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    bb = max(0, bx2 - bx1) * max(0, by2 - by1)

    return inter / max(1, aa + bb - inter)


def overlap_ratio(inner_box, outer_box):
    ix1 = max(inner_box[0], outer_box[0])
    iy1 = max(inner_box[1], outer_box[1])
    ix2 = min(inner_box[2], outer_box[2])
    iy2 = min(inner_box[3], outer_box[3])

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    inner_area = max(1, inner_box[2] - inner_box[0]) * max(
        1, inner_box[3] - inner_box[1]
    )

    return inter / inner_area


def boxes_overlap(a, b, min_ratio=0.15):
    return overlap_ratio(a, b) >= min_ratio


def draw_pose(frame, kpts):
    if kpts is None:
        return frame

    for a, b in SKELETON:
        pa = kp(kpts, a)
        pb = kp(kpts, b)

        if pa and pb:
            cv2.line(
                frame,
                (int(pa[0]), int(pa[1])),
                (int(pb[0]), int(pb[1])),
                (0, 255, 255),
                2,
            )

    for i in range(len(kpts)):
        p = kp(kpts, i)

        if p:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)

    return frame
from __future__ import annotations
import cv2, numpy as np, os

def analyze(image:np.ndarray)->dict:
    if image is None or image.size==0: return {"usable":False,"brightness":0,"blur":0,"contrast":0,"width":0,"height":0,"enhancement":"reject"}
    g=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    h,w=g.shape[:2]; brightness=float(g.mean()); blur=float(cv2.Laplacian(g,cv2.CV_64F).var()); contrast=float(g.std())
    dark=brightness<float(os.getenv('DARKIR_BRIGHTNESS_THRESHOLD','55'))
    small=w<int(os.getenv('EDSR_MIN_ROI_WIDTH','96')) or h<int(os.getenv('EDSR_MIN_ROI_HEIGHT','48'))
    enhancement='darkir+edsr' if dark and small else 'darkir' if dark else 'edsr' if small else 'original'
    return {"usable":w>4 and h>4,"brightness":brightness,"blur":blur,"contrast":contrast,"width":w,"height":h,"enhancement":enhancement}

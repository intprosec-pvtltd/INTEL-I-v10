from __future__ import annotations
from typing import Any, Dict, List
import cv2
import numpy as np
from services.modelEnhancement import (
    load_superres_model,
    upscale_crop,
    enhance_plate_crop,
)

def _u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x,0,255).astype(np.uint8)

def gamma_correct(frame: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(0.35, min(2.8, float(gamma)))
    inv = 1.0/gamma
    table = np.array([((i/255.0)**inv)*255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(frame, table)

def apply_clahe(frame: np.ndarray, clip_limit: float = 2.2, tile_grid=(8,8)) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=max(0.5,float(clip_limit)), tileGridSize=tile_grid).apply(l)
    return cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)

def mild_denoise(frame: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(frame,5,30,30)

def unsharp_mask(frame: np.ndarray, amount: float=0.45, sigma: float=1.0) -> np.ndarray:
    amount=max(0,min(1.2,float(amount)))
    blur=cv2.GaussianBlur(frame,(0,0),sigmaX=sigma)
    return _u8(cv2.addWeighted(frame,1+amount,blur,-amount,0))

def contrast_stretch(frame: np.ndarray) -> np.ndarray:
    lab=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(lab)
    lo=float(np.percentile(l,2)); hi=float(np.percentile(l,98))
    if hi-lo < 8: return frame
    l=_u8((l.astype(np.float32)-lo)*255/(hi-lo))
    return cv2.cvtColor(cv2.merge((l,a,b)),cv2.COLOR_LAB2BGR)

def fast_dehaze(frame: np.ndarray, strength: float=0.4) -> np.ndarray:
    strength=max(0,min(0.8,float(strength)))
    h,w=frame.shape[:2]; src=frame
    if w>720:
        s=720/w; src=cv2.resize(frame,(720,max(1,int(h*s))),interpolation=cv2.INTER_AREA)
    rgb=src.astype(np.float32)/255.0
    dark=cv2.erode(np.min(rgb,axis=2),cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)))
    A=max(0.55,min(0.98,float(np.percentile(dark,98))))
    t=np.clip(1-strength*(1-dark/A),0.45,1.0)
    rec=np.clip((rgb-A)/t[...,None]+A,0,1)
    out=(rec*255).astype(np.uint8)
    if out.shape[:2]!=frame.shape[:2]:
        out=cv2.resize(out,(w,h),interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(frame,1-strength*0.55,out,strength*0.55,0)

def reduce_glare(frame: np.ndarray) -> np.ndarray:
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,np.array([0,0,235],np.uint8),np.array([180,80,255],np.uint8))
    if np.mean(mask>0)<0.01: return frame
    lab=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab)
    lf=l.astype(np.float32); m=mask>0; lf[m]=225+(lf[m]-225)*0.35
    return cv2.cvtColor(cv2.merge((_u8(lf),a,b)),cv2.COLOR_LAB2BGR)

def adaptive_enhance(frame: np.ndarray, quality: Dict[str,Any]) -> Dict[str,Any]:
    if not isinstance(frame,np.ndarray) or frame.size==0:
        return {"frame":frame,"enhanced":False,"operations":[]}
    flags=set(quality.get("flags",[])); mode=str(quality.get("mode") or "").upper()
    cur=frame; ops=[]
    if {"very_dark","low_light"} & flags or mode=="LOW_LIGHT":
        b=float(quality.get("metrics",{}).get("mean_brightness",80))
        cur=gamma_correct(cur,0.52 if b<35 else 0.62 if b<55 else 0.72)
        cur=apply_clahe(cur,2.5); cur=mild_denoise(cur); ops += ["gamma","clahe","bilateral_denoise"]
    if "possible_fog_haze" in flags or mode=="FOG_HAZE":
        cur=fast_dehaze(cur,0.42); cur=contrast_stretch(cur); ops += ["fast_dehaze","contrast_stretch"]
    if "low_contrast" in flags and "possible_fog_haze" not in flags:
        cur=apply_clahe(cur,2.1); ops.append("clahe")
    if "overexposed_or_glare" in flags or mode=="GLARE":
        cur=reduce_glare(cur); ops.append("glare_compression")
    if "high_noise" in flags and "very_dark" not in flags:
        cur=mild_denoise(cur); ops.append("bilateral_denoise")
    if "blur" in flags and "severe_blur" not in flags:
        cur=unsharp_mask(cur); ops.append("unsharp")
    return {"frame":np.ascontiguousarray(_u8(cur)),"enhanced":bool(ops),"operations":ops}

def enhance_vehicle_crop(crop: np.ndarray) -> np.ndarray:
    if not isinstance(crop,np.ndarray) or crop.size==0: return crop
    return np.ascontiguousarray(apply_clahe(crop,1.8))

def build_plate_variants(crop: np.ndarray) -> List[np.ndarray]:
    if not isinstance(crop,np.ndarray) or crop.size==0: return []
    bgr=cv2.cvtColor(crop,cv2.COLOR_GRAY2BGR) if crop.ndim==2 else crop
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    clahe=cv2.createCLAHE(2.0,(8,8)).apply(gray)
    den=cv2.bilateralFilter(clahe,5,30,30)
    sharp=unsharp_mask(cv2.cvtColor(clahe,cv2.COLOR_GRAY2BGR))
    return [np.ascontiguousarray(x) for x in [
        bgr, cv2.cvtColor(clahe,cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(den,cv2.COLOR_GRAY2BGR), sharp
    ]]


def adaptive_model_enhance(
    frame,
    quality,
):
    mode = str(
        quality.get("mode", "")
    ).upper()

    flags = set(
        quality.get("flags", [])
    )

    # Only invoke neural SR for genuinely difficult frames.
    should_use_sr = (
        "severe_blur" in flags
        or "blur" in flags
        or "very_dark" in flags
        or mode in {
            "LOW_LIGHT",
            "BLUR",
            "SEVERE_BLUR",
        }
    )

    if not should_use_sr:
        return {
            "frame": frame,
            "model_used": False,
        }

    if load_superres_model():
        enhanced = upscale_crop(
            frame,
            max_input_width=960,
            max_input_height=540,
        )

        return {
            "frame": enhanced,
            "model_used": True,
        }

    return {
        "frame": frame,
        "model_used": False,
    }
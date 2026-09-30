import os
import time
from unittest.mock import patch

import cv2
import numpy as np

from services.faceEnhancement import FaceDarkIRGate


def candidate(brightness=18, width=80):
    rng=np.random.default_rng(42)
    return np.clip(rng.normal(brightness,12,(width,width,3)),0,255).astype(np.uint8)


def gate(restorer, now=None):
    with patch.dict(os.environ,{"FRS_DARKIR_BRIGHTNESS_THRESHOLD":"65","FRS_DARKIR_MIN_BLUR":"10"}):
        return FaceDarkIRGate(restorer,clock=now or time.monotonic)


def test_good_face_does_not_invoke_darkir():
    calls=[]
    face=np.tile(np.array([95,220],dtype=np.uint8),(80,40))
    face=np.repeat(face[:,:,None],3,axis=2)
    service=gate(lambda image:calls.append(1) or {"image":image,"model_used":True})
    assert service.maybe_enhance('a','1',face,detected=True,detection_score=.9) is None
    assert calls==[]


def test_unusable_face_never_becomes_accepted():
    calls=[]
    service=gate(lambda image:calls.append(1))
    assert service.maybe_enhance('a','1',candidate(18,16),detected=True,detection_score=.95) is None
    assert service.maybe_enhance('a','2',candidate(18),detected=True,detection_score=.3) is None
    assert calls==[]


def test_dark_face_is_cropped_cooldown_limited_and_falls_back_on_invalid_output():
    clock=[100.0]
    calls=[]
    original=candidate(17)
    def restore(image):
        calls.append(image.copy())
        return {"image":cv2.convertScaleAbs(image,alpha=1,beta=26),"model_used":True}
    service=gate(restore,now=lambda:clock[0])
    selected=service.maybe_enhance('a','1',original,detected=True,detection_score=.95)
    assert selected is not None and selected.shape==original.shape
    assert selected.mean()>original.mean()
    assert np.array_equal(calls[0],original)
    assert service.maybe_enhance('a','1',original,detected=True,detection_score=.95) is None
    assert len(calls)==1
    clock[0]+=1
    assert service.maybe_enhance('a','1',original,detected=True,detection_score=.95) is not None
    bad=gate(lambda image:{"image":np.zeros((1,1),np.uint8),"model_used":True})
    assert bad.maybe_enhance('b','1',original,detected=True,detection_score=.95) is None


def test_enhanced_only_alert_requires_independent_frames_and_matching_identity():
    service=gate(lambda image:None)
    assert not service.confirmed_enhanced_only('cam',1,'a',100)
    assert not service.confirmed_enhanced_only('cam',1,'a',100)  # same frame
    assert not service.confirmed_enhanced_only('cam',1,'b',100.5)
    assert not service.confirmed_enhanced_only('cam',1,'a',101)
    assert service.confirmed_enhanced_only('cam',1,'a',102)
    assert not service.confirmed_enhanced_only('cam',2,'a',102)

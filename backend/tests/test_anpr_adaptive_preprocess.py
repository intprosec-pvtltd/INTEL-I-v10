import numpy as np
from services.anprAdaptivePreprocess import analyze_conditions, build_adaptive_variants

def test_dark_blurred_plate_generates_bounded_variants():
    image=np.full((48,160,3),25,dtype=np.uint8)
    info=analyze_conditions(image)
    variants,meta=build_adaptive_variants(image,max_variants=8)
    assert info['low_light'] is True
    assert info['blurred'] is True
    assert 1 <= len(variants) <= 8
    assert all(v.shape==image.shape for v in variants)

def test_variant_builder_rejects_empty():
    variants,meta=build_adaptive_variants(np.empty((0,0,3),dtype=np.uint8))
    assert variants==[] and meta['invalid'] is True

import psutil


def _gpu_capacity():
    """Read GPU capacity when PyTorch is installed in an analytics image."""
    try:
        import torch
        available = bool(torch.cuda.is_available())
        if not available:
            return False, None, 0
        props = torch.cuda.get_device_properties(0)
        return True, torch.cuda.get_device_name(0), round(props.total_memory / (1024 ** 3), 2)
    except (ImportError, RuntimeError):
        return False, None, 0


def get_system_capacity():
    cpu_cores = psutil.cpu_count(logical=True)
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)

    gpu_available, gpu_name, gpu_vram_gb = _gpu_capacity()

    if gpu_available:
        if gpu_vram_gb >= 8 and ram_gb >= 16:
            max_live_cameras = 8
        elif gpu_vram_gb >= 4 and ram_gb >= 8:
            max_live_cameras = 4
        else:
            max_live_cameras = 2
    else:
        if ram_gb >= 16 and cpu_cores >= 8:
            max_live_cameras = 3
        elif ram_gb >= 8 and cpu_cores >= 4:
            max_live_cameras = 2
        else:
            max_live_cameras = 1

    return {
        "cpu_cores": cpu_cores,
        "ram_gb": ram_gb,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "gpu_vram_gb": gpu_vram_gb,
        "max_live_cameras": max_live_cameras,
    }


def can_start_camera(active_camera_count: int):
    capacity = get_system_capacity()
    allowed = active_camera_count < capacity["max_live_cameras"]

    return {
        "allowed": allowed,
        "active_camera_count": active_camera_count,
        "max_live_cameras": capacity["max_live_cameras"],
        "system": capacity,
        "message": (
            "Camera can be started"
            if allowed
            else f"System limit reached. Only {capacity['max_live_cameras']} live cameras can run smoothly."
        )
    }

from .dtf import DTF
ARCHITECTURES = {"DTF": DTF}

def build_model(cfg):
    return ARCHITECTURES[cfg.MODEL.ARCHITECTURE](cfg)

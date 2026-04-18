from src.models.B1.B1_model import ResNetFineTune



MODELS = {
    "B1": ResNetFineTune
}



def build_model(cfg: dict):
    """
    Instantiates the correct model from cfg["model"]["name"].

    Args:
        cfg: full config dict loaded from YAML

    Returns:
        nn.Module instance

    Example YAML:
        model:
          name: B3
          num_classes: 8
    """

    name = cfg['model']['name']
    if name not in MODELS:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {list(MODELS.keys())}\n"
            f"Register it in src/models/model_registry.py"
        )

    model_cls = MODELS[name]
    return model_cls(num_classes=cfg["model"]["num_classes"])


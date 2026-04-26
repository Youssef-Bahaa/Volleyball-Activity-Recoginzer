import argparse
import importlib
import torch
import torch.nn as nn

from src.engine.trainer import train
from src.utils.paths import Paths
from src.engine.utils import get_config, set_seed, build_optimizer, build_scheduler

LOADER_REGISTRY = {
    "B1": "src.dataset.DataLoader.B1_loader",
    "B2": "src.dataset.DataLoader.B2_loader",
}

MODEL_REGISTRY = {
    "B1": ("src.models.B1.B1_model", "ResNetFineTune"),
    "B2": ("src.models.B2.B2_model", "."),

}

def load_model(name, nclasses, pretrained= True):
    module_path, class_name = MODEL_REGISTRY[name]
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(num_classes=nclasses, pretrained=pretrained)


def load_loaders(model_name, cfg):
    module = importlib.import_module(LOADER_REGISTRY[model_name])
    return module.build_loaders(cfg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_REGISTRY.keys())
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = get_config(args.config)
    p = Paths('.', model_name=args.model)

    num_classes = cfg['model']['num_classes']
    model = load_model(args.model, nclasses =num_classes,pretrained= cfg['model']['pretrained']).to(device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    train_loader, val_loader, _ = load_loaders(model_name=args.model, cfg=cfg)

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=device,
        num_classes=num_classes,
        num_epochs=cfg['training']['epochs'],
        model_name=args.model,
        path=p,
        scheduler=scheduler,
        seed=cfg['experiment']['seed'],
    )


if __name__ == "__main__":
    main()









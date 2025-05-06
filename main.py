"""
Main entry point for training, validating, and testing Mask R-CNN with Swin + FPN + PAN.
"""

import os
import argparse
import multiprocessing

import torch
from torch.backends import cudnn
from torch.optim import AdamW, lr_scheduler

from model import build_swin_maskrcnn
from train import train_model, load_data, validate_model
from test import test_model


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """Main function for handling train, validate, and test modes."""
    cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Mask R-CNN Swin + PAN Training")

    parser.add_argument("data_path", type=str, help="Root path to dataset.")
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=2,
        help="Batch size.")
    parser.add_argument("-e", "--epochs", type=int, default=50, help="Epochs.")
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate.")
    parser.add_argument(
        "-em",
        "--eta_min",
        type=float,
        default=1e-6,
        help="Min learning rate (for cosine annealing).")
    parser.add_argument(
        "-d",
        "--decay",
        type=float,
        default=5e-3,
        help="Weight decay.")
    parser.add_argument(
        "-s",
        "--saved_path",
        type=str,
        default="saved_models",
        help="Save directory.")
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=[
            "train",
            "validate",
            "test"],
        default="train",
        help="Execution mode.")
    parser.add_argument(
        "-v",
        "--is_valid",
        type=bool,
        default=True,
        help="Enable validation split.")

    args = parser.parse_args()

    os.makedirs(args.saved_path, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)

    train_loader, valid_loader, num_classes = load_data(
        os.path.join(args.data_path, "train"), args=args)
    model = build_swin_maskrcnn(
        num_classes=num_classes,
        device=device)

    print(
        f"Model has {count_parameters(model) / 1e6:.2f}M trainable parameters")

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.decay)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.eta_min)

    if args.mode == 'train':
        train_model(device=device, model=model, optimizer=optimizer,
                    train_loader=train_loader, valid_loader=valid_loader,
                    scheduler=scheduler, args=args)
    elif args.mode == 'validate':
        validate_model(
            device=device,
            model=model,
            valid_loader=valid_loader,
            args=args)
    elif args.mode == 'test':
        test_model(device=device, model=model, args=args)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    multiprocessing.freeze_support()
    main()

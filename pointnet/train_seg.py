import torch
import torch.nn.functional as F
import argparse
from datetime import datetime
from tqdm import tqdm
from model import PointNetPartSeg, get_orthogonal_loss
from dataloaders.shapenet_partseg import get_data_loaders
from utils.metrics import Accuracy, mIoU
from utils.model_checkpoint import CheckpointManager
from torch.autograd import Variable
from utils.misc import save_samples
import os.path as osp


def step(points, pc_labels, class_labels, model):
    """
    Input : 
        - points [B, N, 3]
        - pc_labels [B, N]
        - class_labels [B]
    Output : loss
        - loss []
        - logits [B, C, N] (C: num_class)
        - preds [B, N]
    """
    # TODO : Implement step function for segmentation.
    logits, feature_transform_matrix = model(points.cuda())
    preds = torch.argmax(logits, dim=1)
    loss = F.cross_entropy(logits.cuda(), pc_labels.cuda().to(torch.int64)) \
        + get_orthogonal_loss(feature_transform_matrix)

    return loss, logits, preds


def train_step(points, pc_labels, class_labels, model, optimizer, train_acc_metric):
    loss, logits, preds = step(
        points, pc_labels, class_labels, model
    )
    logits, preds, pc_labels = logits.cuda(), preds.cuda(), pc_labels.cuda()
    train_batch_acc = train_acc_metric(preds, pc_labels)

    # TODO : Implement backpropagation using optimizer and loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, train_batch_acc


def validation_step(
        points, pc_labels, class_labels, model, val_acc_metric, val_iou_metric
):
    loss, logits, preds = step(
        points, pc_labels, class_labels, model
    )
    logits, preds, pc_labels = logits.cuda(), preds.cuda(), pc_labels.cuda()
    val_batch_acc = val_acc_metric(preds, pc_labels)
    val_batch_iou, masked_preds = val_iou_metric(logits, pc_labels, class_labels)

    return loss, masked_preds, val_batch_acc, val_batch_iou


def main(args):
    global device
    device = "cpu" if args.gpu == -1 else f"cuda:{args.gpu}"

    # TODO: Implement the model call
    model = PointNetPartSeg(num_classes=50, input_transform=True, feature_transform=True)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30, 80], gamma=0.5
    )
    if args.save:
        checkpoint_manager = CheckpointManager(
            dirpath=datetime.now().strftime("checkpoints/segmentation/%m-%d_%H-%M-%S"),
            metric_name="val_iou",
            mode="max",
            topk=2,
            verbose=True,
        )

    # It will download Shapenet Dataset at the first time.
    (train_ds, val_ds, test_ds), (train_dl, val_dl, test_dl) = get_data_loaders(
        data_dir="./data", batch_size=args.batch_size, phases=["train", "val", "test"]
    )

    train_acc_metric = Accuracy()
    val_acc_metric = Accuracy()
    val_iou_metric = mIoU()

    for epoch in range(args.epochs):
        # training step
        model.train()
        pbar = tqdm(train_dl)
        train_epoch_loss = []
        for points, pc_labels, class_labels in pbar:
            train_batch_loss, train_batch_acc = train_step(
                points, pc_labels, class_labels, model, optimizer, train_acc_metric
            )
            train_epoch_loss.append(train_batch_loss)
            pbar.set_description(
                f"{epoch + 1}/{args.epochs} epoch | loss: {train_batch_loss:.4f} | accuracy: {train_batch_acc * 100:.1f}%"
            )

        train_epoch_loss = sum(train_epoch_loss) / len(train_epoch_loss)
        train_epoch_acc = train_acc_metric.compute_epoch()

        # validataion step
        model.eval()
        with torch.no_grad():
            val_epoch_loss = []
            for points, pc_labels, class_labels in val_dl:
                val_batch_loss, val_batch_masked_preds, val_batch_acc, val_batch_iou = validation_step(
                    points,
                    pc_labels,
                    class_labels,
                    model,
                    val_acc_metric,
                    val_iou_metric,
                )
                val_epoch_loss.append(val_batch_loss)

            val_epoch_loss = sum(val_epoch_loss) / len(val_epoch_loss)
            val_epoch_acc = val_acc_metric.compute_epoch()
            val_epoch_iou = val_iou_metric.compute_epoch()
            print(
                f"train loss: {train_epoch_loss:.4f} | train acc: {train_epoch_acc * 100:.1f}% | val loss: {val_epoch_loss:.4f} | val acc: {val_epoch_acc * 100:.1f}% | val mIoU: {val_epoch_iou * 100:.1f}%"
            )

            if args.save:
                checkpoint_manager.update(
                    model, epoch, round(val_epoch_iou.item() * 100, 2)
                )
        scheduler.step()

    # After training, test on testset
    if args.save:
        checkpoint_manager.load_best_ckpt(model, device)
    model.eval()
    with torch.no_grad():
        test_acc_metric = Accuracy()
        test_iou_metric = mIoU()
        for points, pc_labels, class_labels in test_dl:
            test_batch_loss, test_batch_masked_preds, test_batch_acc, test_batch_iou = validation_step(
                points,
                pc_labels,
                class_labels,
                model,
                test_acc_metric,
                test_iou_metric,
            )
        test_acc = test_acc_metric.compute_epoch()
        test_iou = test_iou_metric.compute_epoch()

        print(f"test acc: {test_acc * 100:.1f}% | test mIoU: {test_iou * 100:.1f}%")
        save_samples(points[4:8], pc_labels[4:8], test_batch_masked_preds[4:8], "segmentation_samples.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PointNet ShapeNet Part Segmentation")
    parser.add_argument(
        "--gpu", type=int, default=0, help="gpu device num. -1 is for cpu"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--save", action="store_true", help="Whether to save topk checkpoints or not"
    )

    args = parser.parse_args()

    main(args)

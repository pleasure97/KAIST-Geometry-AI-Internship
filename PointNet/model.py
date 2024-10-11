import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class STNKd(nn.Module):
    # T-Net a.k.a. Spatial Transformer Network
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        """
        Input: [B,N,k]
        Output: [B,k,k]
        """
        B = x.shape[0]
        device = x.device
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]

        x = self.fc(x)

        # Followed the original implementation to initialize a matrix as I.
        identity = (
            Variable(torch.eye(self.k, dtype=torch.float))
            .reshape(1, self.k * self.k)
            .expand(B, -1)
            .to(device)
        )
        x = x + identity
        x = x.reshape(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    """
    Corresponds to the part that extracts max-pooled features.
    """

    def __init__(
            self,
            input_transform: bool = False,
            feature_transform: bool = False,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        # point-wise mlp
        # TODO : Implement point-wise mlp model based on PointNet Architecture.

        self.mlp64 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.mlp128 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.mlp1024 = nn.Sequential(
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
        )

        self.feature_transform_matrix = None

        if self.input_transform:
            self.stn3 = STNKd(k=3)
        if self.feature_transform:
            self.stn64 = STNKd(k=64)

    def forward(self, pointcloud):
        """
        Input:
            pointcloud: [B,N,3]
        Output:
            Global feature: [B,1024]
        """

        # TODO : Implement forward function.
        x = pointcloud.transpose(2, 1)
        if self.input_transform:
            input_transform_matrix = self.stn3(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, input_transform_matrix)
            x = x.transpose(2, 1)
        x = self.mlp64(x)
        if self.feature_transform:
            feature_transform_matrix = self.stn64(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, feature_transform_matrix)
            x = x.transpose(2, 1)
            self.feature_transform_matrix = feature_transform_matrix
        else:
            self.feature_transform_matrix = None
        local_feature = x
        x = self.mlp128(x)
        x = self.mlp1024(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.reshape(-1, 1024)
        global_feature = x

        return global_feature, self.feature_transform_matrix, local_feature


class PointNetCls(nn.Module):
    def __init__(self, num_classes, input_transform, feature_transform):
        super().__init__()
        self.num_classes = num_classes

        # extracts max-pooled features
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)

        # returns the final logits from the max-pooled features.
        # TODO : Implement MLP that takes global feature as an input and return logits.
        self.mlp512 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.mlp256 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.3),
            nn.ReLU()
        )

        self.mlp_k = nn.Sequential(
            nn.Linear(256, self.num_classes),
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - ...
        """
        # TODO : Implement forward function.
        x, feature_transform_matrix, _ = self.pointnet_feat(pointcloud)
        x = self.mlp512(x)
        x = self.mlp256(x)
        x = self.mlp_k(x)
        return x, feature_transform_matrix


class PointNetAutoEncoder(nn.Module):
    def __init__(self, num_points, input_transform, feature_transform):
        super().__init__()
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)

        # Decoder is just a simple MLP that outputs N x 3 (x,y,z) coordinates.
        # TODO : Implement decoder.
        self.num_points = num_points

        self.mlp1 = nn.Sequential(
            nn.Linear(1024, self.num_points // 4),
            nn.BatchNorm1d(self.num_points // 4),
            nn.ReLU()
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(self.num_points // 4, self.num_points // 2),
            nn.BatchNorm1d(self.num_points // 2),
            nn.ReLU()
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(self.num_points // 2, self.num_points),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(self.num_points),
            nn.ReLU()
        )
        self.mlp4 = nn.Linear(self.num_points, self.num_points * 3)

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - pointcloud [B,N,3]
            - ...
        """
        # TODO : Implement forward function.
        x, _, _ = self.pointnet_feat(pointcloud)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        x = x.reshape(-1, self.num_points, 3)

        return x


class PointNetPartSeg(nn.Module):
    def __init__(self, num_classes, input_transform, feature_transform):
        # TODO: Implement this
        super().__init__()
        self.num_classes = num_classes
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)

        self.mlp512 = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.mlp256 = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.mlp128 = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.mlp_m = nn.Sequential(
            nn.Conv1d(128, self.num_classes, 1),
            nn.BatchNorm1d(self.num_classes),
            nn.ReLU()
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - logits: [B,50,N] | 50: # of point labels
            - ...
        """
        # TODO: Implement this
        global_feature, feature_transform_matrix, local_feature = self.pointnet_feat(pointcloud)
        global_feature = global_feature.unsqueeze(dim=-1).repeat(1, 1, local_feature.shape[-1])
        concat_feature = torch.cat((local_feature, global_feature), dim=1)
        concat_feature = self.mlp512(concat_feature)
        concat_feature = self.mlp256(concat_feature)
        concat_feature = self.mlp128(concat_feature)
        concat_feature = self.mlp_m(concat_feature)

        logits = concat_feature
        return logits, feature_transform_matrix


def get_orthogonal_loss(feat_trans, reg_weight=1e-3):
    """
    a regularization loss that enforces a transformation matrix to be a rotation matrix.
    Property of rotation matrix A: A*A^T = I
    """
    if feat_trans is None:
        return 0

    B, K = feat_trans.shape[:2]
    device = feat_trans.device

    identity = torch.eye(K).to(device)[None].expand(B, -1, -1)
    mat_square = torch.bmm(feat_trans, feat_trans.transpose(1, 2))

    mat_diff = (identity - mat_square).reshape(B, -1)

    return reg_weight * mat_diff.norm(dim=1).mean()

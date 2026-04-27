# Relational learner in tinygrad + Wasserstein detection head + YOLO loader.

from __future__ import annotations

import os
import math
import argparse
import glob
import pathlib
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np

# tinygrad env
# os.environ.setdefault("CPU", "1")
os.environ.setdefault("JIT", "0")

from tinygrad import Tensor, dtypes
from tinygrad.nn import Linear

try:
    from tinygrad.nn.optim import Adam
except Exception:
    from tinygrad.optim import Adam

from tinygrad.nn.state import get_parameters

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import yaml
except Exception:
    yaml = None


# Seeds
Tensor.manual_seed(42)
np.random.seed(42)


# Utils
def Softmax(x: Tensor, axis: int = -1) -> Tensor:
    m = x.max(axis=axis, keepdim=True)
    e = (x - m).exp()
    return e / (e.sum(axis=axis, keepdim=True) + 1e-8)


def LogSoftmax(x: Tensor, axis: int = -1) -> Tensor:
    m = x.max(axis=axis, keepdim=True)
    z = x - m
    lse = z.exp().sum(axis=axis, keepdim=True).log()
    return z - lse


def PairwiseDistances(x: Tensor, eps: float = 1e-8) -> Tensor:
    xx = (x * x).sum(axis=1, keepdim=True)
    yy = (x * x).sum(axis=1, keepdim=True).transpose()
    d2 = xx + yy - 2 * (x @ x.transpose())
    return (d2 + eps).relu()


def NormalizeRdm(d: Tensor) -> Tensor:
    b = d.shape[0]
    mask = Tensor.ones(b, b) - Tensor.eye(b)
    mean = (d * mask).sum() / (mask.sum() + 1e-8)
    ex2 = ((d * mask) ** 2).sum() / (mask.sum() + 1e-8)
    std = (ex2 - mean * mean).relu().sqrt() + 1e-8
    return (d - mean) / std


def L2Norm(x: Tensor, axis: int, keepdim: bool) -> Tensor:
    return ((x * x).sum(axis=axis, keepdim=keepdim) + 1e-8).sqrt()


def ElemMax(a: Tensor, b: Tensor) -> Tensor:
    return 0.5 * (a + b + (a - b).abs())


def ElemMin(a: Tensor, b: Tensor) -> Tensor:
    return 0.5 * (a + b - (a - b).abs())


# Sinkhorn OT
def Sinkhorn(
    a: Tensor,
    b: Tensor,
    C: Tensor,
    epsilon: float = 0.05,
    iters: int = 60,
) -> Tuple[Tensor, Tensor]:
    K = (-C / epsilon).exp()
    u = Tensor.ones(C.shape[0]) / C.shape[0]
    v = Tensor.ones(C.shape[1]) / C.shape[1]

    for _ in range(iters):
        Kv = K @ v
        u = a / (Kv + 1e-8)
        Ku = K.transpose() @ u
        v = b / (Ku + 1e-8)

    P = u.reshape(-1, 1) * K * v.reshape(1, -1)
    return P, (P * C).sum()


def SinkhornUniform(
    x: Tensor,
    y: Tensor,
    epsilon: float = 0.05,
    iters: int = 60,
) -> Tensor:
    n, m = x.shape[0], y.shape[0]
    a = Tensor.ones(n) / n
    b = Tensor.ones(m) / m

    x2 = (x * x).sum(axis=1, keepdim=True)
    y2 = (y * y).sum(axis=1, keepdim=True)
    C = (x2 + y2.transpose() - 2 * (x @ y.transpose())).relu()

    _, cost = Sinkhorn(a, b, C, epsilon=epsilon, iters=iters)
    return cost


# Layers
class SimpleLayerNorm:
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        self.gamma = Tensor.ones(dim, requires_grad=True)
        self.beta = Tensor.zeros(dim, requires_grad=True)
        self.eps = eps

    def __call__(self, x: Tensor) -> Tensor:
        m = x.mean(axis=-1, keepdim=True)
        v = ((x - m) * (x - m)).mean(axis=-1, keepdim=True)
        xhat = (x - m) / (v + self.eps).sqrt()
        return xhat * self.gamma + self.beta


class MultiHeadAttention:
    def __init__(self, dModel: int, nHead: int) -> None:
        assert dModel % nHead == 0
        self.nh = nHead
        self.dk = dModel // nHead
        self.q = Linear(dModel, dModel, bias=False)
        self.k = Linear(dModel, dModel, bias=False)
        self.v = Linear(dModel, dModel, bias=False)
        self.o = Linear(dModel, dModel, bias=False)

    def __call__(
        self,
        qx: Tensor,
        kx: Tensor,
        vx: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        B, Nq, D = qx.shape
        Nk = kx.shape[1]

        Q = self.q(qx).reshape(B, Nq, self.nh, self.dk).permute(0, 2, 1, 3)
        K = self.k(kx).reshape(B, Nk, self.nh, self.dk).permute(0, 2, 1, 3)
        V = self.v(vx).reshape(B, Nk, self.nh, self.dk).permute(0, 2, 1, 3)

        scores = (Q @ K.transpose(2, 3)) / math.sqrt(self.dk)

        if mask is not None:
            scores = scores + mask

        attn = Softmax(scores, axis=-1)
        head = attn @ V
        out = head.permute(0, 2, 1, 3).reshape(B, Nq, D)

        return self.o(out)


class PatchEmbed:
    def __init__(self, inCh: int = 3, patch: int = 4, dModel: int = 128) -> None:
        self.patch = patch
        self.lin = Linear(inCh * patch * patch, dModel)

    def __call__(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        ps = self.patch

        assert H % ps == 0 and W % ps == 0

        x = x.reshape(B, C, H // ps, ps, W // ps, ps).permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, (H // ps) * (W // ps), ps * ps * C)

        return self.lin(x).relu()


class SlotExtractor:
    def __init__(self, dModel: int = 128, numSlots: int = 6, nHead: int = 4) -> None:
        self.numSlots = numSlots
        self.slots = Tensor.randn(numSlots, dModel, requires_grad=True) * 0.02
        self.mha = MultiHeadAttention(dModel, nHead)
        self.norm = SimpleLayerNorm(dModel)
        self.fc1 = Linear(dModel, 2 * dModel)
        self.fc2 = Linear(2 * dModel, dModel)

    def __call__(self, tokens: Tensor) -> Tensor:
        B, _, D = tokens.shape
        q = self.slots.reshape(1, self.numSlots, D).repeat(B, 1, 1)
        x = self.mha(q, tokens, tokens) + q
        x = self.fc2(self.fc1(self.norm(x)).relu()) + x
        return x


class RoleFillerBinder:
    def __init__(self, dModel: int = 128, numRoles: int = 3) -> None:
        self.numRoles = numRoles
        self.roleParams = Tensor.randn(numRoles, dModel, requires_grad=True) * 0.02
        self.proj = Linear(dModel, dModel, bias=False)

    def __call__(self, slots: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        B, _, _ = slots.shape
        roles = self.roleParams
        fill = self.proj(slots)

        roleDen = L2Norm(roles, axis=1, keepdim=True)
        fillDen = L2Norm(fill, axis=2, keepdim=True)
        roleN = roles / roleDen
        fillN = fill / fillDen

        scores = fillN @ roleN.transpose()
        attn = Softmax(scores, axis=1)
        bound = attn.transpose(1, 2) @ fill
        scene = bound.reshape(B, -1)

        return scene, roles, bound


class RelationalBackbone:
    def __init__(
        self,
        dModel: int = 128,
        patch: int = 4,
        numSlots: int = 6,
        numRoles: int = 3,
        nHead: int = 4,
    ) -> None:
        self.patch = PatchEmbed(inCh=3, patch=patch, dModel=dModel)
        self.slotter = SlotExtractor(dModel=dModel, numSlots=numSlots, nHead=nHead)
        self.binder = RoleFillerBinder(dModel=dModel, numRoles=numRoles)
        self.fc1 = Linear(numRoles * dModel, 2 * dModel)
        self.fc2 = Linear(2 * dModel, dModel)

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        tokens = self.patch(x)
        slots = self.slotter(tokens)
        scene, roles, bound = self.binder(slots)
        z = self.fc2(self.fc1(scene).relu())
        return z, roles, bound

    def Parameters(self) -> List[Tensor]:
        return get_parameters(self)


class TargetProjection:
    def __init__(self, dIn: int, dOut: int) -> None:
        self.lin = Linear(dIn, dOut)

    def __call__(self, y: Tensor) -> Tensor:
        return self.lin(y)

    def Parameters(self) -> List[Tensor]:
        return get_parameters(self)


class DetectionHead:
    def __init__(self, dModel: int, numRoles: int, numClasses: int) -> None:
        self.box = Linear(dModel, 4)
        self.cls = Linear(dModel, numClasses)
        self.obj = Linear(dModel, 1)

    def __call__(self, roleFeats: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        boxes = self.box(roleFeats).sigmoid()
        logits = self.cls(roleFeats)
        objLog = self.obj(roleFeats).reshape(roleFeats.shape[0], roleFeats.shape[1])
        return boxes, logits, objLog

    def Parameters(self) -> List[Tensor]:
        return get_parameters(self)


# Losses
@dataclass
class LossWeights:
    lambdaRdm: float = 1.0
    lambdaWass: float = 0.1
    lambdaRole: float = 0.01
    lambdaL2: float = 1e-4
    lambdaDet: float = 1.0


def RelationalLoss(z: Tensor, yProj: Tensor, roles: Tensor, w: LossWeights) -> Tensor:
    Dz = PairwiseDistances(z).sqrt()
    Dy = PairwiseDistances(yProj).sqrt()

    L_rdm = (NormalizeRdm(Dz) - NormalizeRdm(Dy)).pow(2).mean()
    L_wass = SinkhornUniform(z, yProj, epsilon=0.05, iters=40)

    G = roles @ roles.transpose()
    I = Tensor.eye(G.shape[0])
    L_role = (G - I).pow(2).mean()
    L_l2 = (z * z).mean()

    return (
        w.lambdaRdm * L_rdm
        + w.lambdaWass * L_wass
        + w.lambdaRole * L_role
        + w.lambdaL2 * L_l2
    )


def CxcywhToXyxy(boxes: Tensor) -> Tensor:
    cx = boxes[..., 0]
    cy = boxes[..., 1]
    w = boxes[..., 2]
    h = boxes[..., 3]

    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    return Tensor.stack([x1, y1, x2, y2], dim=-1)


def IoUMatrix(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    b1 = CxcywhToXyxy(boxes1).reshape(boxes1.shape[0], 1, 4)
    b2 = CxcywhToXyxy(boxes2).reshape(1, boxes2.shape[0], 4)

    x1 = ElemMax(b1[..., 0], b2[..., 0])
    y1 = ElemMax(b1[..., 1], b2[..., 1])
    x2 = ElemMin(b1[..., 2], b2[..., 2])
    y2 = ElemMin(b1[..., 3], b2[..., 3])

    interW = (x2 - x1).relu()
    interH = (y2 - y1).relu()
    inter = interW * interH

    area1 = (b1[..., 2] - b1[..., 0]).relu() * (b1[..., 3] - b1[..., 1]).relu()
    area2 = (b2[..., 2] - b2[..., 0]).relu() * (b2[..., 3] - b2[..., 1]).relu()
    union = area1 + area2 - inter + 1e-8

    return inter / union


def PairwiseL1(a: Tensor, b: Tensor) -> Tensor:
    diff = (a.reshape(a.shape[0], 1, 4) - b.reshape(1, b.shape[0], 4)).abs()
    return diff.sum(axis=-1)


def OneHot(labels: Tensor, C: int) -> Tensor:
    counter = Tensor.arange(C, dtype=dtypes.int32).reshape(1, C)
    return (labels.reshape(-1, 1) == counter) * 1.0


def DetectionCostMatrix(
    predBoxes: Tensor,
    predLogits: Tensor,
    gtBoxes: Tensor,
    gtLabels: Tensor,
    lamCls: float = 1.0,
    lamBox: float = 1.0,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Tensor:
    N, C = predLogits.shape[0], predLogits.shape[1]
    M = gtLabels.shape[0]

    logp = LogSoftmax(predLogits, axis=-1)
    yoh = OneHot(gtLabels, C=C)
    CE = -(logp.reshape(N, 1, C) * yoh.reshape(1, M, C)).sum(axis=-1)

    L1 = PairwiseL1(predBoxes, gtBoxes)
    iou = IoUMatrix(predBoxes, gtBoxes)
    reg = alpha * L1 + beta * (1.0 - iou)

    return lamCls * CE + lamBox * reg


def PadWithDustbin(C: Tensor, dustbinCost: float) -> Tensor:
    N, M = C.shape
    lastRow = Tensor.full((1, M), dustbinCost)
    lastCol = Tensor.full((N + 1, 1), dustbinCost)

    Cpad = C.cat(lastRow, dim=0)
    Cpad = Cpad.cat(lastCol, dim=1)

    return Cpad


def ExtendWithDustbin(a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    aReal = a * 0.5
    bReal = b * 0.5

    aDust = (1.0 - aReal.sum()).reshape(1)
    bDust = (1.0 - bReal.sum()).reshape(1)

    aExt = aReal.cat(aDust, dim=0)
    bExt = bReal.cat(bDust, dim=0)

    return aExt, bExt


def OtDetectionLoss(
    predBoxes: Tensor,
    predLogits: Tensor,
    gtBoxes: Tensor,
    gtLabels: Tensor,
    objLogit: Optional[Tensor] = None,
    eps: float = 0.07,
    uot: bool = True,
    dustbinCost: float = 0.5,
) -> Tensor:
    N = predBoxes.shape[0]
    M = gtBoxes.shape[0]

    a = Tensor.ones(N) / N if objLogit is None else Softmax(objLogit)

    if M == 0:
        C = Tensor.full((N, 0), 0.0)
        C = PadWithDustbin(C, dustbinCost)
        a, b = ExtendWithDustbin(a, Tensor.ones(0))
        _, cost = Sinkhorn(a, b, C, epsilon=eps, iters=50)
        return cost

    b = Tensor.ones(M) / M
    C = DetectionCostMatrix(predBoxes, predLogits, gtBoxes, gtLabels)

    if uot:
        C = PadWithDustbin(C, dustbinCost)
        a, b = ExtendWithDustbin(a, b)

    _, cost = Sinkhorn(a, b, C, epsilon=eps, iters=50)
    return cost


# Synthetic Data fallback
def SyntheticBatch(
    B: int,
    H: int = 64,
    W: int = 64,
    numClasses: int = 3,
) -> Tuple[Tensor, Tensor, List[Tensor], List[Tensor], int]:
    imgs = []
    ys = []
    gtBoxes = []
    gtLabels = []

    Dn = 64
    rr = max(1, min(H, W) // 8)

    for i in range(B):
        c = i % numClasses
        arr = np.zeros((3, H, W), dtype=np.float32)

        if c == 0:
            cy, cx = H // 3, W // 3
        elif c == 1:
            cy, cx = H // 2, 2 * W // 3
        else:
            cy, cx = 2 * H // 3, W // 2

        yy, xx = np.ogrid[:H, :W]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= rr * rr

        for ch in range(3):
            val = (0.3 + 0.3 * c) if ch == c % 3 else 0.1
            arr[ch][mask] = val

        img = Tensor(arr) + 0.03 * Tensor.randn(3, H, W)
        img = img.clip(0, 1)
        imgs.append(img)

        base = np.zeros(Dn, dtype=np.float32)
        base[c * Dn // numClasses : (c + 1) * Dn // numClasses] = 1.0
        base += 0.1 * np.random.randn(Dn).astype(np.float32)
        ys.append(Tensor(base))

        cx_n = cx / W
        cy_n = cy / H
        w_n = (2 * rr) / W
        h_n = (2 * rr) / H

        gtBoxes.append(Tensor(np.array([cx_n, cy_n, w_n, h_n], np.float32).reshape(-1, 4)))
        gtLabels.append(Tensor(np.array([c], np.int32)))

    X = Tensor.stack(imgs)
    Y = Tensor.stack(ys)
    Y = (Y - Y.mean(axis=1, keepdim=True)) / (Y.std(axis=1, keepdim=True) + 1e-8)

    return X, Y, gtBoxes, gtLabels, numClasses


# YOLO Loader
def ResolveLabelsDir(imagesDir: str) -> str:
    p = pathlib.Path(imagesDir)
    parts = list(p.parts)

    try:
        idx = parts.index("images")
        parts[idx] = "labels"
        return str(pathlib.Path(*parts))
    except ValueError:
        return str(p.parent.parent / "labels" / p.name)


def ReadYaml(path: str) -> dict:
    if yaml is None:
        raise ImportError("PyYAML not installed. pip install pyyaml")

    with open(path, "r") as f:
        return yaml.safe_load(f)


def GatherImageFiles(imagesDir: str) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
    files = []

    for ext in exts:
        files.extend(glob.glob(str(pathlib.Path(imagesDir) / "**" / ext), recursive=True))

    return sorted(files)


def LabelPathFor(imagePath: str, labelsDir: str) -> str:
    stem = pathlib.Path(imagePath).with_suffix("").name
    return str(pathlib.Path(labelsDir) / f"{stem}.txt")


def ReadYoloTxt(labelPath: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(labelPath):
        return np.zeros((0,), np.int32), np.zeros((0, 4), np.float32)

    clsIds = []
    boxes = []

    with open(labelPath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                continue

            c, x, y, w, h = parts
            clsIds.append(int(float(c)))
            boxes.append([float(x), float(y), float(w), float(h)])

    if len(clsIds) == 0:
        return np.zeros((0,), np.int32), np.zeros((0, 4), np.float32)

    return np.array(clsIds, np.int32), np.array(boxes, np.float32)


def LoadImage(path: str, H: int, W: int) -> Tensor:
    assert Image is not None, "Pillow not installed. pip install pillow"

    img = Image.open(path).convert("RGB").resize((W, H))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))

    return Tensor(arr)


def YoloIterBatches(
    dataYaml: str,
    split: str,
    H: int,
    W: int,
    batch: int,
):
    y = ReadYaml(dataYaml)

    if split not in y:
        raise ValueError(f"split '{split}' not found in {dataYaml}")

    root = pathlib.Path(y.get("path", "."))
    splitPath = pathlib.Path(y[split])
    imagesDir = str(splitPath if splitPath.is_absolute() else root / splitPath)
    labelsDir = ResolveLabelsDir(imagesDir)

    names = y.get("names", [])
    numClasses = len(names) if isinstance(names, (list, tuple)) else int(y.get("nc", 0))

    files = GatherImageFiles(imagesDir)
    if len(files) == 0:
        raise FileNotFoundError(f"No images found under: {imagesDir}")

    i = 0
    while True:
        batchFiles = files[i : i + batch]

        if len(batchFiles) < batch:
            rem = batch - len(batchFiles)
            batchFiles += files[0:rem]
            i = rem
        else:
            i += batch

        Xs = []
        gtBoxesList = []
        gtLabelsList = []

        for fp in batchFiles:
            Xs.append(LoadImage(fp, H, W))

            lp = LabelPathFor(fp, labelsDir)
            clsNp, boxNp = ReadYoloTxt(lp)

            gtBoxesList.append(Tensor(boxNp.reshape(-1, 4)))
            gtLabelsList.append(Tensor(clsNp.reshape(-1).astype(np.int32)))

        Dn = max(64, numClasses * 8)
        Ys = []

        for lbl in gtLabelsList:
            vec = np.zeros(Dn, np.float32)

            if lbl.shape[0] == 0:
                vec[:1] = 1.0
            else:
                bucketCount = max(1, Dn // 2)
                for c in lbl.numpy().astype(np.int32):
                    vec[c % bucketCount] += 1.0

            Ys.append(Tensor((vec - vec.mean()) / (vec.std() + 1e-8)))

        X = Tensor.stack(Xs)
        Y = Tensor.stack(Ys)

        yield X, Y, gtBoxesList, gtLabelsList, numClasses


def EnableGrad(params: List[Tensor]) -> None:
    for p in params:
        p.requires_grad_(True)


def LogTrainingStep(it: int, loss: Tensor, detLoss: Tensor, Z: Tensor, Yp: Tensor) -> None:
    Dz = PairwiseDistances(Z.detach()).sqrt().numpy()
    Dy = PairwiseDistances(Yp.detach()).sqrt().numpy()
    iu = np.triu_indices(Dz.shape[0], k=1)

    if len(iu[0]) == 0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(Dz[iu], Dy[iu])[0, 1])

    print(
        f"[{it:03d}] "
        f"total={float(loss.detach().numpy()):.4f} "
        f"det={float(detLoss.detach().numpy()):.4f} "
        f"RSA≈{corr:.3f}"
    )


# Train
def TrainDemo(
    steps: int = 40,
    batch: int = 8,
    imgHw: Tuple[int, int] = (640, 640),
    dModel: int = 128,
    numRoles: int = 20,
    dataYaml: Optional[str] = None,
    split: str = "train",
) -> None:
    H, W = imgHw

    if H % 8 != 0 or W % 8 != 0:
        raise ValueError("Image height and width must both be divisible by patch size 8.")

    if dataYaml is None:
        X0, Y0, gtBoxes0, gtLabels0, numClasses = SyntheticBatch(batch, H, W)

        proj = TargetProjection(dIn=Y0.shape[1], dOut=dModel)
        model = RelationalBackbone(
            dModel=dModel,
            patch=8,
            numSlots=2 * numRoles,
            numRoles=numRoles,
            nHead=4,
        )
        det = DetectionHead(dModel=dModel, numRoles=numRoles, numClasses=numClasses)

        params = model.Parameters() + proj.Parameters() + det.Parameters()
        EnableGrad(params)

        opt = Adam(params, lr=2e-3)
        wts = LossWeights(lambdaDet=1.0)

        with Tensor.train():
            Z0, roles0, bound0 = model(X0)
            Y0p = proj(Y0)
            lossRep0 = RelationalLoss(Z0, Y0p, roles0, wts)
            boxes0, logits0, obj0 = det(bound0)

            detLoss0 = Tensor(0.0)
            for b in range(batch):
                detLoss0 = detLoss0 + OtDetectionLoss(
                    boxes0[b],
                    logits0[b],
                    gtBoxes0[b],
                    gtLabels0[b],
                    objLogit=obj0[b],
                    eps=0.07,
                    uot=True,
                )

            detLoss0 = detLoss0 / batch
            loss0 = lossRep0 + wts.lambdaDet * detLoss0

            opt.zero_grad()
            loss0.backward()
            opt.step()

            for it in range(1, steps + 1):
                X, Y, gtBoxesList, gtLabelsList, _ = SyntheticBatch(batch, H, W, numClasses)

                Z, roles, bound = model(X)
                Yp = proj(Y)
                lossRep = RelationalLoss(Z, Yp, roles, wts)
                boxes, logits, obj = det(bound)

                detLoss = Tensor(0.0)
                for b in range(batch):
                    detLoss = detLoss + OtDetectionLoss(
                        boxes[b],
                        logits[b],
                        gtBoxesList[b],
                        gtLabelsList[b],
                        objLogit=obj[b],
                        eps=0.07,
                        uot=True,
                    )

                detLoss = detLoss / batch
                loss = lossRep + wts.lambdaDet * detLoss

                opt.zero_grad()
                loss.backward()
                opt.step()

                if it % 10 == 0:
                    LogTrainingStep(it, loss, detLoss, Z, Yp)

        return

    assert yaml is not None, "PyYAML required for --data YAML. pip install pyyaml"
    assert Image is not None, "Pillow required to load images. pip install pillow"

    gen = YoloIterBatches(dataYaml, split, H, W, batch)
    X0, Y0, gtBoxes0, gtLabels0, numClasses = next(gen)

    proj = TargetProjection(dIn=Y0.shape[1], dOut=dModel)
    model = RelationalBackbone(
        dModel=dModel,
        patch=8,
        numSlots=2 * numRoles,
        numRoles=numRoles,
        nHead=4,
    )
    det = DetectionHead(dModel=dModel, numRoles=numRoles, numClasses=numClasses)

    params = model.Parameters() + proj.Parameters() + det.Parameters()
    EnableGrad(params)

    opt = Adam(params, lr=2e-4)
    wts = LossWeights(lambdaDet=1.0)

    with Tensor.train():
        Z0, roles0, bound0 = model(X0)
        Y0p = proj(Y0)
        lossRep0 = RelationalLoss(Z0, Y0p, roles0, wts)
        boxes0, logits0, obj0 = det(bound0)

        detLoss0 = Tensor(0.0)
        for b in range(batch):
            detLoss0 = detLoss0 + OtDetectionLoss(
                boxes0[b],
                logits0[b],
                gtBoxes0[b],
                gtLabels0[b],
                objLogit=obj0[b],
                eps=0.07,
                uot=True,
            )

        detLoss0 = detLoss0 / batch
        loss0 = lossRep0 + wts.lambdaDet * detLoss0

        opt.zero_grad()
        loss0.backward()
        opt.step()

        for it in range(1, steps + 1):
            X, Y, gtBoxesList, gtLabelsList, _ = next(gen)

            Z, roles, bound = model(X)
            Yp = proj(Y)
            lossRep = RelationalLoss(Z, Yp, roles, wts)
            boxes, logits, obj = det(bound)

            detLoss = Tensor(0.0)
            for b in range(batch):
                detLoss = detLoss + OtDetectionLoss(
                    boxes[b],
                    logits[b],
                    gtBoxesList[b],
                    gtLabelsList[b],
                    objLogit=obj[b],
                    eps=0.07,
                    uot=True,
                )

            detLoss = detLoss / batch
            loss = lossRep + wts.lambdaDet * detLoss

            opt.zero_grad()
            loss.backward()
            opt.step()

            if it % 10 == 0:
                LogTrainingStep(it, loss, detLoss, Z, Yp)


# CLI
def ParseArgs() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None, help="path to YOLO data.yaml")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--img", type=int, nargs=2, default=[640, 640], help="H W")
    ap.add_argument("--roles", type=int, default=20, help="num roles / max proposals")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=128)
    return ap.parse_args()

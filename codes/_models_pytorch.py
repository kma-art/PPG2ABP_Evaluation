
# PyTorch 2.7 implementation of the provided Keras/TF1.5 


import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetDS64(nn.Module):
    """
    Эквивалент Keras UNetDS64 (TF1.5): Conv1D(relu)->BN, MaxPool1D(2), UpSampling1D(2)+concat по каналам,
    выходы: [out, level1, level2, level3, level4] (как в Keras).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 bn_eps: float = 1e-3, bn_momentum_keras: float = 0.99):
        super().__init__()

        # Важно: momentum в PyTorch и Keras считаются "в разные стороны".
        # Keras: moving = moving*momentum + batch*(1-momentum)
        # PyTorch: running = running*(1-mom) + batch*mom  (см. доки) -> mom_pt = 1 - mom_keras
        # Для инференса с загруженными running_mean/var это не критично, но для дообучения — да.
        bn_momentum_pt = 1.0 - bn_momentum_keras

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # (idx, in_ch, out_ch) — строго по вашей Keras UNetDS64 (x=64)
        conv_specs = [
            (1,  in_channels, 64),
            (2,  64, 64),

            (3,  64, 128),
            (4,  128, 128),

            (5,  128, 256),
            (6,  256, 256),

            (7,  256, 512),
            (8,  512, 512),

            (9,  512, 1024),
            (10, 1024, 1024),

            (11, 1536, 512),
            (12, 512, 512),

            (13, 768, 256),
            (14, 256, 256),

            (15, 384, 128),
            (16, 128, 128),

            (17, 192, 64),
            (18, 64, 64),
        ]

        for idx, cin, cout in conv_specs:
            setattr(self, f"conv1d_{idx}", nn.Conv1d(cin, cout, kernel_size=3, padding=1, bias=True))
            setattr(self, f"batch_normalization_{idx}",
                    nn.BatchNorm1d(cout, eps=bn_eps, momentum=bn_momentum_pt, affine=True, track_running_stats=True))

        # 1x1 головы как в Keras: out/level1..level4
        self.out    = nn.Conv1d(64,   out_channels, kernel_size=1, bias=True)
        self.level1 = nn.Conv1d(128,  out_channels, kernel_size=1, bias=True)
        self.level2 = nn.Conv1d(256,  out_channels, kernel_size=1, bias=True)
        self.level3 = nn.Conv1d(512,  out_channels, kernel_size=1, bias=True)
        self.level4 = nn.Conv1d(1024, out_channels, kernel_size=1, bias=True)

    @staticmethod
    def _to_ncl(x: torch.Tensor):
        """
        Приводим вход к (N,C,L). Если пришло (N,L,C) и C маленький (обычно 1), транспонируем.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input, got shape={tuple(x.shape)}")
        # Keras обычно (N, L, C) с C=1
        if x.shape[1] != 1 and x.shape[2] == 1:
            return x.transpose(1, 2), True  # (N,C,L), channels_last=True
        return x, False

    @staticmethod
    def _match_len(a: torch.Tensor, b: torch.Tensor):
        """
        Подгоняем длину (L) при concat, если из-за нечётных длин где-то +/-1.
        Обрезаем по min(L).
        """
        la, lb = a.shape[-1], b.shape[-1]
        if la == lb:
            return a, b
        m = min(la, lb)
        return a[..., :m], b[..., :m]

    def _cbr(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        conv = getattr(self, f"conv1d_{idx}")
        bn   = getattr(self, f"batch_normalization_{idx}")
        x = conv(x)
        x = F.relu(x, inplace=True)
        x = bn(x)
        return x

    def forward(self, x: torch.Tensor):
        x, was_channels_last = self._to_ncl(x)

        # Encoder
        c1 = self._cbr(x, 1)
        c1 = self._cbr(c1, 2)
        p1 = self.pool(c1)

        c2 = self._cbr(p1, 3)
        c2 = self._cbr(c2, 4)
        p2 = self.pool(c2)

        c3 = self._cbr(p2, 5)
        c3 = self._cbr(c3, 6)
        p3 = self.pool(c3)

        c4 = self._cbr(p3, 7)
        c4 = self._cbr(c4, 8)
        p4 = self.pool(c4)

        # Bridge (conv5 в Keras)
        c5 = self._cbr(p4, 9)
        c5 = self._cbr(c5, 10)
        level4 = self.level4(c5)

        # Decoder
        u6 = self.up(c5)
        u6, c4m = self._match_len(u6, c4)
        u6 = torch.cat([u6, c4m], dim=1)  # concat по каналам
        c6 = self._cbr(u6, 11)
        c6 = self._cbr(c6, 12)
        level3 = self.level3(c6)

        u7 = self.up(c6)
        u7, c3m = self._match_len(u7, c3)
        u7 = torch.cat([u7, c3m], dim=1)
        c7 = self._cbr(u7, 13)
        c7 = self._cbr(c7, 14)
        level2 = self.level2(c7)

        u8 = self.up(c7)
        u8, c2m = self._match_len(u8, c2)
        u8 = torch.cat([u8, c2m], dim=1)
        c8 = self._cbr(u8, 15)
        c8 = self._cbr(c8, 16)
        level1 = self.level1(c8)

        u9 = self.up(c8)
        u9, c1m = self._match_len(u9, c1)
        u9 = torch.cat([u9, c1m], dim=1)
        c9 = self._cbr(u9, 17)
        c9 = self._cbr(c9, 18)

        out = self.out(c9)

        # Вернём в Keras-формате (N, L, C), если вход был channels_last
        if was_channels_last:
            out    = out.transpose(1, 2)
            level1 = level1.transpose(1, 2)
            level2 = level2.transpose(1, 2)
            level3 = level3.transpose(1, 2)
            level4 = level4.transpose(1, 2)

        return [out, level1, level2, level3, level4]




def _split_filters(U: int, alpha: float = 2.5):
    """
    Keras code:
      W = alpha * U
      f1 = int(W*0.167)
      f2 = int(W*0.333)
      f3 = int(W*0.5)
      out = f1 + f2 + f3
    """
    W = alpha * U
    f1 = int(W * 0.167)
    f2 = int(W * 0.333)
    f3 = int(W * 0.5)
    out = f1 + f2 + f3
    return f1, f2, f3, out


class ConvBNAct1d(nn.Module):
    """
    Mirrors conv2d_bn() from the TF code:
      Conv1D(filters, kernel=3, padding='same')
      BatchNormalization()
      optional Activation('relu')
    Important: In the TF code kernel is ALWAYS 3 (even for "1x1" calls).
    """
    def __init__(self, in_ch: int, out_ch: int, activation: bool, eps: float = 1e-3, bn_momentum: float = 0.01):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn = nn.BatchNorm1d(out_ch, eps=eps, momentum=bn_momentum, affine=True, track_running_stats=True)
        self.act = nn.ReLU(inplace=False) if activation else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class MultiResBlock1D(nn.Module):
    """
    Mirrors MultiResBlock() from TF code:
      shortcut = conv2d_bn(inp, out_ch, activation=None)
      conv3 = conv2d_bn(inp, f1, relu)
      conv5 = conv2d_bn(conv3, f2, relu)
      conv7 = conv2d_bn(conv5, f3, relu)
      out = concat([conv3, conv5, conv7], axis=channels)
      out = BN(out)
      out = add([shortcut, out])
      out = relu(out)
      out = BN(out)
    """
    def __init__(self, U: int, in_ch: int, alpha: float = 2.5, eps: float = 1e-3, bn_momentum: float = 0.01):
        super().__init__()
        f1, f2, f3, out_ch = _split_filters(U, alpha)

        # IMPORTANT: shortcut uses kernel=3 too (matches your h5 shapes)
        self.shortcut = ConvBNAct1d(in_ch, out_ch, activation=False, eps=eps, bn_momentum=bn_momentum)

        self.conv3 = ConvBNAct1d(in_ch, f1, activation=True, eps=eps, bn_momentum=bn_momentum)
        self.conv5 = ConvBNAct1d(f1,  f2, activation=True, eps=eps, bn_momentum=bn_momentum)
        self.conv7 = ConvBNAct1d(f2,  f3, activation=True, eps=eps, bn_momentum=bn_momentum)

        self.bn_after_concat = nn.BatchNorm1d(out_ch, eps=eps, momentum=bn_momentum, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=False)
        self.bn_after_relu = nn.BatchNorm1d(out_ch, eps=eps, momentum=bn_momentum, affine=True, track_running_stats=True)

        self.out_ch = out_ch

    def forward(self, x):
        sc = self.shortcut(x)

        c3 = self.conv3(x)
        c5 = self.conv5(c3)
        c7 = self.conv7(c5)

        out = torch.cat([c3, c5, c7], dim=1)  # channels-first
        out = self.bn_after_concat(out)

        out = out + sc
        out = self.relu(out)
        out = self.bn_after_relu(out)
        return out


class ResPath1D(nn.Module):
    """
    Mirrors ResPath(filters, length, inp) from TF code.
    Each step:
      shortcut = conv2d_bn(out, filters, activation=None)  # kernel=3
      out      = conv2d_bn(out, filters, relu)             # kernel=3
      out = add + relu + BN
    Note: length=1 means just the first step.
    """
    def __init__(self, filters: int, length: int, in_ch: int, eps: float = 1e-3, bn_momentum: float = 0.01):
        super().__init__()
        assert length >= 1
        self.length = length

        blocks = []
        cur_in = in_ch
        for _ in range(length):
            shortcut = ConvBNAct1d(cur_in, filters, activation=False, eps=eps, bn_momentum=bn_momentum)
            conv     = ConvBNAct1d(cur_in, filters, activation=True,  eps=eps, bn_momentum=bn_momentum)
            post_bn  = nn.BatchNorm1d(filters, eps=eps, momentum=bn_momentum, affine=True, track_running_stats=True)
            blocks.append(nn.ModuleDict({
                "shortcut": shortcut,
                "conv": conv,
                "relu": nn.ReLU(inplace=False),
                "post_bn": post_bn,
            }))
            cur_in = filters

        self.blocks = nn.ModuleList(blocks)
        self.out_ch = filters

    def forward(self, x):
        out = x
        for b in self.blocks:
            sc = b["shortcut"](out)
            y  = b["conv"](out)
            out = b["relu"](y + sc)
            out = b["post_bn"](out)
        return out


class MultiResUNet1D(nn.Module):
    """
    PyTorch version of your Keras MultiResUNet1D(length, n_channel=1).
    Input:  (N, n_channel, L)
    Output: (N, 1, L)

    IMPORTANT practical note:
      With 4x MaxPool(stride=2), L should usually be divisible by 16
      so that upsampling and concatenations align perfectly.
    """
    def __init__(self, n_channel: int = 1, alpha: float = 2.5, eps: float = 1e-3, bn_momentum: float = 0.01):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # Encoder
        self.mres1 = MultiResBlock1D(32, in_ch=n_channel, alpha=alpha, eps=eps, bn_momentum=bn_momentum)   # -> 79
        self.rpath1 = ResPath1D(filters=32,  length=4, in_ch=self.mres1.out_ch, eps=eps, bn_momentum=bn_momentum)  # -> 32

        self.mres2 = MultiResBlock1D(64, in_ch=self.mres1.out_ch, alpha=alpha, eps=eps, bn_momentum=bn_momentum)   # -> 159
        self.rpath2 = ResPath1D(filters=64,  length=3, in_ch=self.mres2.out_ch, eps=eps, bn_momentum=bn_momentum)  # -> 64

        self.mres3 = MultiResBlock1D(128, in_ch=self.mres2.out_ch, alpha=alpha, eps=eps, bn_momentum=bn_momentum)  # -> 319
        self.rpath3 = ResPath1D(filters=128, length=2, in_ch=self.mres3.out_ch, eps=eps, bn_momentum=bn_momentum)  # -> 128

        self.mres4 = MultiResBlock1D(256, in_ch=self.mres3.out_ch, alpha=alpha, eps=eps, bn_momentum=bn_momentum)  # -> 639
        self.rpath4 = ResPath1D(filters=256, length=1, in_ch=self.mres4.out_ch, eps=eps, bn_momentum=bn_momentum)  # -> 256

        self.mres5 = MultiResBlock1D(512, in_ch=self.mres4.out_ch, alpha=alpha, eps=eps, bn_momentum=bn_momentum)  # -> 1279

        # Decoder (concats are channel-wise)
        self.mres6 = MultiResBlock1D(256, in_ch=self.mres5.out_ch + self.rpath4.out_ch, alpha=alpha, eps=eps, bn_momentum=bn_momentum)  # 1279+256=1535 -> 639
        self.mres7 = MultiResBlock1D(128, in_ch=self.mres6.out_ch + self.rpath3.out_ch, alpha=alpha, eps=eps, bn_momentum=bn_momentum)  # 639+128=767  -> 319
        self.mres8 = MultiResBlock1D(64,  in_ch=self.mres7.out_ch + self.rpath2.out_ch, alpha=alpha, eps=eps, bn_momentum=bn_momentum)  # 319+64=383   -> 159
        self.mres9 = MultiResBlock1D(32,  in_ch=self.mres8.out_ch + self.rpath1.out_ch, alpha=alpha, eps=eps, bn_momentum=bn_momentum)  # 159+32=191   -> 79

        # Final conv10 = Conv1D(1, 1)
        self.final = nn.Conv1d(self.mres9.out_ch, 1, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        # x: (N, C, L)
        x1 = self.mres1(x)        # 79
        p1 = self.pool(x1)
        s1 = self.rpath1(x1)      # 32 (skip)

        x2 = self.mres2(p1)       # 159
        p2 = self.pool(x2)
        s2 = self.rpath2(x2)      # 64

        x3 = self.mres3(p2)       # 319
        p3 = self.pool(x3)
        s3 = self.rpath3(x3)      # 128

        x4 = self.mres4(p3)       # 639
        p4 = self.pool(x4)
        s4 = self.rpath4(x4)      # 256

        x5 = self.mres5(p4)       # 1279

        u6 = torch.cat([self.up(x5), s4], dim=1)  # 1535
        x6 = self.mres6(u6)                       # 639

        u7 = torch.cat([self.up(x6), s3], dim=1)  # 767
        x7 = self.mres7(u7)                       # 319

        u8 = torch.cat([self.up(x7), s2], dim=1)  # 383
        x8 = self.mres8(u8)                       # 159

        u9 = torch.cat([self.up(x8), s1], dim=1)  # 191
        x9 = self.mres9(u9)                       # 79

        out = self.final(x9)                      # 1
        return out


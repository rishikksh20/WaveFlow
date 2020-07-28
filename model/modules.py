import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, set_grad_enabled, grad, gradcheck
from utils.utils import add_weight_norms
from functools import reduce
from operator import mul


class InvertibleConv1x1(nn.Conv1d):
    def __init__(self, c, memory_efficient=False):
        super().__init__(c, c, 1, bias=False)
        W = torch.randn(c, c).qr()[0]
        self.weight.data = W[..., None]
        if memory_efficient:
            self.efficient_forward = Conv1x1Func.apply
            self.efficient_inverse = InvConv1x1Func.apply

    def forward(self, x):
        if hasattr(self, 'efficient_forward'):
            z, log_det_W = self.efficient_forward(x, self.weight)
            x.storage().resize_(0)
            return z, log_det_W
        else:
            *_, n_of_groups = x.shape
            log_det_W = n_of_groups * self.weight.squeeze().slogdet()[1]  # should fix nan logdet
            z = super().forward(x)
            return z, log_det_W

    def inverse(self, z):
        if hasattr(self, 'efficient_inverse'):
            x, log_det_W = self.efficient_inverse(z, self.weight)
            z.storage().resize_(0)
            return x, log_det_W
        else:
            weight = self.weight.squeeze()
            *_, n_of_groups = z.shape
            log_det_W = -n_of_groups * weight.slogdet()[1]  # should fix nan logdet
            x = F.conv1d(z, weight.inverse().unsqueeze(-1))
            return x, log_det_W


class AffineCouplingBlock(nn.Module):
    def __init__(self,
                 transform_type,
                 memory_efficient=True,
                 **kwargs):
        super().__init__()

        self.F = transform_type(**kwargs)
        if memory_efficient:
            self.efficient_forward = AffineCouplingFunc.apply
            self.efficient_inverse = InvAffineCouplingFunc.apply

    def forward(self, x, y):
        if hasattr(self, 'efficient_forward'):
            z, log_s = self.efficient_forward(x, y, self.F, *self.F.parameters())
            x.storage().resize_(0)
            return z, log_s
        else:
            xa, xb = x.chunk(2, 1)
            za = xa
            log_s, t = self.F(xa, y)
            zb = xb * log_s.exp() + t
            z = torch.cat((za, zb), 1)
            return z, log_s

    def inverse(self, z, y):
        if hasattr(self, 'efficient_inverse'):
            x, log_s = self.efficient_inverse(z, y, self.F, *self.F.parameters())
            z.storage().resize_(0)
            return x, log_s
        else:
            za, zb = z.chunk(2, 1)
            xa = za
            log_s, t = self.F(za, y)
            xb = (zb - t) / log_s.exp()
            x = torch.cat((xa, xb), 1)
            return x, -log_s


class AffineCouplingFunc(Function):
    @staticmethod
    def forward(ctx, x, y, F, *F_weights):
        ctx.F = F
        with torch.no_grad():
            xa, xb = x.chunk(2, 1)
            xa, xb = xa.contiguous(), xb.contiguous()

            log_s, t = F(xa, y)
            zb = xb * log_s.exp() + t
            za = xa
            z = torch.cat((za, zb), 1)

        ctx.save_for_backward(x.data, y, z)
        return z, log_s

    @staticmethod
    def backward(ctx, z_grad, log_s_grad):
        F = ctx.F
        x, y, z = ctx.saved_tensors

        za, zb = z.chunk(2, 1)
        za, zb = za.contiguous(), zb.contiguous()
        dza, dzb = z_grad.chunk(2, 1)
        dza, dzb = dza.contiguous(), dzb.contiguous()

        with set_grad_enabled(True):
            xa = za
            xa.requires_grad = True
            log_s, t = F(xa, y)

        with torch.no_grad():
            s = log_s.exp()
            xb = (zb - t) / s
            x.storage().resize_(reduce(mul, xb.shape) * 2)
            torch.cat((xa, xb), 1, out=x)  # .contiguous()
            #x.copy_(xout)  # .detach()

        with set_grad_enabled(True):
            param_list = [xa] + list(F.parameters())
            if ctx.needs_input_grad[1]:
                param_list += [y]
            dtsdxa, *dw = grad(torch.cat((log_s, t), 1), param_list,
                               grad_outputs=torch.cat((dzb * xb * s + log_s_grad, dzb), 1))

            dxa = dza + dtsdxa
            dxb = dzb * s
            dx = torch.cat((dxa, dxb), 1)
            if ctx.needs_input_grad[1]:
                *dw, dy = dw
            else:
                dy = None

        return (dx, dy, None) + tuple(dw)


class InvAffineCouplingFunc(Function):
    @staticmethod
    def forward(ctx, z, y, F, *F_weights):
        ctx.F = F
        with torch.no_grad():
            za, zb = z.chunk(2, 1)
            za, zb = za.contiguous(), zb.contiguous()

            log_s, t = F(za, y)
            xb = (zb - t) / log_s.exp()
            xa = za
            x = torch.cat((xa, xb), 1)

        ctx.save_for_backward(z.data, y, x)
        return x, -log_s

    @staticmethod
    def backward(ctx, x_grad, log_s_grad):
        F = ctx.F
        z, y, x = ctx.saved_tensors

        xa, xb = x.chunk(2, 1)
        xa, xb = xa.contiguous(), xb.contiguous()
        dxa, dxb = x_grad.chunk(2, 1)
        dxa, dxb = dxa.contiguous(), dxb.contiguous()

        with set_grad_enabled(True):
            za = xa
            za.requires_grad = True
            log_s, t = F(za, y)
            s = log_s.exp()

        with torch.no_grad():
            zb = xb * s + t

            z.storage().resize_(reduce(mul, zb.shape) * 2)
            torch.cat((za, zb), 1, out=z)
            #z.copy_(zout)

        with set_grad_enabled(True):
            param_list = [za] + list(F.parameters())
            if ctx.needs_input_grad[1]:
                param_list += [y]
            dtsdza, *dw = grad(torch.cat((-log_s, -t / s), 1), param_list,
                               grad_outputs=torch.cat((dxb * zb / s.detach() + log_s_grad, dxb), 1))

            dza = dxa + dtsdza
            dzb = dxb / s.detach()
            dz = torch.cat((dza, dzb), 1)
            if ctx.needs_input_grad[1]:
                *dw, dy = dw
            else:
                dy = None
        return (dz, dy, None) + tuple(dw)


class Conv1x1Func(Function):
    @staticmethod
    def forward(ctx, x, weight):
        with torch.no_grad():
            *_, n_of_groups = x.shape
            log_det_W = weight.squeeze().slogdet()[1]
            log_det_W *= n_of_groups
            z = F.conv1d(x, weight)

        ctx.save_for_backward(x.data, weight, z)
        return z, log_det_W

    @staticmethod
    def backward(ctx, z_grad, log_det_W_grad):
        x, weight, z = ctx.saved_tensors
        *_, n_of_groups = z.shape

        with torch.no_grad():
            inv_weight = weight.squeeze().inverse()
            x.storage().resize_(reduce(mul, z.shape))
            x[:] = F.conv1d(z, inv_weight.unsqueeze(-1))

            dx = F.conv1d(z_grad, weight[..., 0].t().unsqueeze(-1))
            dw = z_grad.transpose(0, 1).contiguous().view(weight.shape[0], -1) @ x.transpose(1, 2).contiguous().view(
                -1, weight.shape[1])
            dw += inv_weight.t() * log_det_W_grad * n_of_groups

        return dx, dw.unsqueeze(-1)


class InvConv1x1Func(Function):
    @staticmethod
    def forward(ctx, x, inv_weight):
        with torch.no_grad():
            sqr_inv_weight = inv_weight.squeeze()
            *_, n_of_groups = x.shape
            log_det_W = -sqr_inv_weight.slogdet()[1]
            log_det_W *= n_of_groups
            z = F.conv1d(x, sqr_inv_weight.inverse().unsqueeze(-1))

        ctx.save_for_backward(x.data, inv_weight, z)
        return z, log_det_W

    @staticmethod
    def backward(ctx, z_grad, log_det_W_grad):
        x, inv_weight, z = ctx.saved_tensors
        *_, n_of_groups = z.shape

        with torch.no_grad():
            x.storage().resize_(reduce(mul, z.shape))
            x[:] = F.conv1d(z, inv_weight)

            inv_weight = inv_weight.squeeze()
            weight_T = inv_weight.inverse().t()
            dx = F.conv1d(z_grad, weight_T.unsqueeze(-1))
            dw = z_grad.transpose(0, 1).contiguous().view(weight_T.shape[0], -1) @ \
                 x.transpose(1, 2).contiguous().view(-1, weight_T.shape[1])
            dinvw = - weight_T @ dw @ weight_T
            dinvw -= weight_T * log_det_W_grad * n_of_groups

        return dx, dinvw.unsqueeze(-1)


class _NonCausalLayer(nn.Module):
    def __init__(self,
                 dilation,
                 dilation_channels,
                 residual_channels,
                 skip_channels,
                 aux_channels,
                 radix,
                 bias,
                 last_layer=False):
        super().__init__()
        pad_size = dilation * (radix - 1) // 2
        self.WV = nn.Conv1d(residual_channels + aux_channels, dilation_channels * 2, kernel_size=radix,
                            padding=pad_size, dilation=dilation, bias=bias)

        self.chs_split = [skip_channels]
        if last_layer:
            self.W_o = nn.Conv1d(dilation_channels, skip_channels, 1, bias=bias)
        else:
            self.W_o = nn.Conv1d(dilation_channels, residual_channels + skip_channels, 1, bias=bias)
            self.chs_split.insert(0, residual_channels)

    def forward(self, x, y):
        xy = torch.cat((x, y), 1)
        zw, zf = self.WV(xy).chunk(2, 1)
        z = zw.tanh_().mul(zf.sigmoid_())
        *z, skip = self.W_o(z).split(self.chs_split, 1)
        return z[0] + x if len(z) else None, skip


class WN(nn.Module):
    def __init__(self,
                 in_channels,
                 aux_channels,
                 dilation_channels=256,
                 residual_channels=256,
                 skip_channels=256,
                 depth=8,
                 radix=3,
                 bias=False,
                 zero_init=True):
        super().__init__()
        dilations = 2 ** torch.arange(depth)
        self.dilations = dilations.tolist()
        self.in_chs = in_channels
        self.res_chs = residual_channels
        self.dil_chs = dilation_channels
        self.skp_chs = skip_channels
        self.aux_chs = aux_channels
        self.rdx = radix
        self.r_field = sum(self.dilations) + 1

        self.start = nn.Conv1d(in_channels, residual_channels, 1, bias=bias)
        self.start.apply(add_weight_norms)

        self.layers = nn.ModuleList(_NonCausalLayer(d,
                                                    dilation_channels,
                                                    residual_channels,
                                                    skip_channels,
                                                    aux_channels,
                                                    radix,
                                                    bias) for d in self.dilations[:-1])
        self.layers.append(_NonCausalLayer(self.dilations[-1],
                                           dilation_channels,
                                           residual_channels,
                                           skip_channels,
                                           aux_channels,
                                           radix,
                                           bias,
                                           last_layer=True))
        self.layers.apply(add_weight_norms)

        self.end = nn.Conv1d(skip_channels, in_channels * 2, 1, bias=bias)
        if zero_init:
            self.end.weight.data.zero_()
            if bias:
                self.end.bias.data.zero_()

    def forward(self, x, y):
        x = self.start(x)
        cum_skip = None
        for layer in self.layers:
            x, skip = layer(x, y)
            if cum_skip is None:
                cum_skip = skip
            else:
                cum_skip += skip
        return self.end(cum_skip).chunk(2, 1)


class _NonCausalLayer2D(nn.Module):
    def __init__(self,
                 h_dilation,
                 dilation,
                 aux_channels,
                 dilation_channels,
                 residual_channels,
                 skip_channels,
                 radix,
                 bias,
                 last_layer=False):
        super().__init__()
        self.h_pad_size = h_dilation * (radix - 1)
        self.pad_size = dilation * (radix - 1) // 2

        self.V = nn.Conv1d(aux_channels, dilation_channels * 2, radix, dilation=dilation, padding=self.pad_size,
                           bias=bias)

        self.W = nn.Conv2d(residual_channels, dilation_channels * 2,
                           kernel_size=radix,
                           dilation=(h_dilation, dilation), bias=bias)

        self.chs_split = [skip_channels]
        if last_layer:
            self.W_o = nn.Conv2d(dilation_channels, skip_channels, 1, bias=bias)
        else:
            self.W_o = nn.Conv2d(dilation_channels, residual_channels + skip_channels, 1, bias=bias)
            self.chs_split.insert(0, residual_channels)

    def forward(self, x, y):
        tmp = F.pad(x, [self.pad_size] * 2 + [self.h_pad_size, 0])
        xy = self.W(tmp) + self.V(y).unsqueeze(2)
        zw, zf = xy.chunk(2, 1)
        z = zw.tanh_().mul(zf.sigmoid_())
        *z, skip = self.W_o(z).split(self.chs_split, 1)
        if len(z):
            output = z[0]
            return output + x[:, :, -output.size(2):], skip
        else:
            return None, skip

    def inverse_forward(self, x, y, buffer=None):
        if buffer is None:
            buffer = F.pad(x, [0, 0, self.h_pad_size, 0])
        else:
            buffer = torch.cat((buffer[:, :, 1:], x), 2)
        tmp = F.pad(buffer, [self.pad_size] * 2)
        xy = self.W(tmp) + self.V(y).unsqueeze(2)
        zw, zf = xy.chunk(2, 1)
        z = zw.tanh_().mul(zf.sigmoid_())
        *z, skip = self.W_o(z).split(self.chs_split, 1)
        if len(z):
            output = z[0]
            return output + x, skip, buffer
        else:
            return None, skip, buffer


class WN2D(nn.Module):
    def __init__(self,
                 n_group,
                 aux_channels,
                 dilation_channels=256,
                 residual_channels=256,
                 skip_channels=256,
                 bias=False,
                 zero_init=True):
        super().__init__()

        dilation_dict = {
            8: [1] * 8,
            16: [1] * 8,
            32: [1, 2, 4] * 2 + [1, 2],
            64: [1, 2, 4, 8, 16, 1, 2, 4],
            128: [1, 2, 4, 8, 16, 32, 64, 1],
        }

        self.h_dilations = dilation_dict[n_group]
        dilations = 2 ** torch.arange(8)
        self.dilations = dilations.tolist()
        self.n_group = n_group
        self.res_chs = residual_channels
        self.dil_chs = dilation_channels
        self.skp_chs = skip_channels
        self.aux_chs = aux_channels
        self.r_field = sum(self.dilations) * 2 + 1
        self.h_r_field = sum(self.h_dilations) * 2 + 1

        self.start = nn.Conv2d(1, residual_channels, 1, bias=bias)
        self.start.apply(add_weight_norms)

        self.layers = nn.ModuleList(_NonCausalLayer2D(hd, d,
                                                      aux_channels,
                                                      dilation_channels,
                                                      residual_channels,
                                                      skip_channels,
                                                      3,
                                                      bias) for hd, d in
                                    zip(self.h_dilations[:-1], self.dilations[:-1]))
        self.layers.append(_NonCausalLayer2D(self.h_dilations[-1], self.dilations[-1],
                                             aux_channels,
                                             dilation_channels,
                                             residual_channels,
                                             skip_channels,
                                             3,
                                             bias,
                                             last_layer=True))
        self.layers.apply(add_weight_norms)

        self.end = nn.Conv2d(skip_channels, 2, 1, bias=bias)
        if zero_init:
            self.end.weight.data.zero_()
            if bias:
                self.end.bias.data.zero_()

    def forward(self, x, y):
        x = self.start(x)
        cum_skip = None
        for layer in self.layers:
            x, skip = layer(x, y)
            if cum_skip is None:
                cum_skip = skip
            else:
                cum_skip += skip
        return self.end(cum_skip).chunk(2, 1)

    def inverse_forward(self, x, y, buffer_list=None):
        x = self.start(x)
        new_buffer_list = []
        if buffer_list is None:
            buffer_list = [None] * len(self.layers)

        cum_skip = None
        for layer, buf in zip(self.layers, buffer_list):
            x, skip, buf = layer.inverse_forward(x, y, buf)
            new_buffer_list.append(buf)
            if cum_skip is None:
                cum_skip = skip
            else:
                cum_skip += skip

        return self.end(cum_skip).chunk(2, 1) + (new_buffer_list,)
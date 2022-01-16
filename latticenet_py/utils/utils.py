import torch
import torch.nn as nn
import torch.nn.functional as F
from easypbr  import *
import sys
import math
import functools

import copy
import inspect
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch as th
import torch.nn.functional as thf
from torch import Tensor
from torch.nn.utils.weight_norm import WeightNorm, remove_weight_norm
from torch.nn.modules.utils import _pair
# from utils.learn import gaussian_kernel

# from latticenet_py.lattice.lattice_funcs import *
# from latticenet_py.lattice.lattice_modules import ConvLatticeIm2RowModule
# import  latticenet_py.lattice as ln
# import  latticenet_py.lattice.lattice_modules as ln
# from latticenet_py.lattice.lattice_funcs import *
# from latticenet_py.lattice.lattice_modules import *
# from latticenet_py.lattice.lattice_modules import ConvLatticeIm2RowModule

# help( ln )

# module=ConvLatticeIm2RowModule(nr_filters=20, neighbourhood_size=1, dilation=1, bias=True)



def check_args_shadowing(name, method, arg_names):
    spec = inspect.getfullargspec(method)
    init_args = {*spec.args, *spec.kwonlyargs}
    for arg_name in arg_names:
        if arg_name in init_args:
            raise TypeError(f"{name} attempted to shadow a wrapped argument: {arg_name}")


# For backward compatibility.
class TensorMappingHook(object):
    def __init__(
        self,
        name_mapping: List[Tuple[str, str]],
        expected_shape: Optional[Dict[str, List[int]]] = None,
    ):
        """This hook is expected to be used with "_register_load_state_dict_pre_hook" to
        modify names and tensor shapes in the loaded state dictionary.
        Args:
            name_mapping: list of string tuples
            A list of tuples containing expected names from the state dict and names expected
            by the module.
            expected_shape: dict
            A mapping from parameter names to expected tensor shapes.
        """
        self.name_mapping = name_mapping
        self.expected_shape = expected_shape if expected_shape is not None else {}

    def __call__(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        for old_name, new_name in self.name_mapping:
            if prefix + old_name in state_dict:
                tensor = state_dict.pop(prefix + old_name)
                if new_name in self.expected_shape:
                    tensor = tensor.view(*self.expected_shape[new_name])
                state_dict[prefix + new_name] = tensor


def weight_norm_wrapper(cls, name="weight", g_dim=0, v_dim=0):
    """Wraps a torch.nn.Module class to support weight normalization. The wrapped class
    is compatible with the fuse/unfuse syntax and is able to load state dict from previous
    implementations.
    Args:
        name: str
        Name of the parameter to apply weight normalization.
        g_dim: int
        Learnable dimension of the magnitude tensor. Set to None or -1 for single scalar magnitude.
        Default values for Linear and Conv2d layers are 0s and for ConvTranspose2d layers are 1s.
        v_dim: int
        Of which dimension of the direction tensor is calutated independently for the norm. Set to
        None or -1 for calculating norm over the entire direction tensor (weight tensor). Default
        values for most of the WN layers are None to preserve the existing behavior.
    """

    class Wrap(cls):
        def __init__(self, *args, name=name, g_dim=g_dim, v_dim=v_dim, **kwargs):
            # Check if the extra arguments are overwriting arguments for the wrapped class
            check_args_shadowing(
                "weight_norm_wrapper", super().__init__, ["name", "g_dim", "v_dim"]
            )
            super().__init__(*args, **kwargs)

            # Sanitize v_dim since we are hacking the built-in utility to support
            # a non-standard WeightNorm implementation.
            if v_dim is None:
                v_dim = -1
            self.weight_norm_args = {"name": name, "g_dim": g_dim, "v_dim": v_dim}
            self.is_fused = True
            self.unfuse()

            # For backward compatibility.
            self._register_load_state_dict_pre_hook(
                TensorMappingHook(
                    [(name, name + "_v"), ("g", name + "_g")],
                    {name + "_g": getattr(self, name + "_g").shape},
                )
            )

        def fuse(self):
            if self.is_fused:
                return
            # Check if the module is frozen.
            param_name = self.weight_norm_args["name"] + "_g"
            if hasattr(self, param_name) and param_name not in self._parameters:
                raise ValueError("Trying to fuse frozen module.")
            remove_weight_norm(self, self.weight_norm_args["name"])
            self.is_fused = True

        def unfuse(self):
            if not self.is_fused:
                return
            # Check if the module is frozen.
            param_name = self.weight_norm_args["name"]
            if hasattr(self, param_name) and param_name not in self._parameters:
                raise ValueError("Trying to unfuse frozen module.")
            wn = WeightNorm.apply(
                self, self.weight_norm_args["name"], self.weight_norm_args["g_dim"]
            )
            # Overwrite the dim property to support mismatched norm calculate for v and g tensor.
            if wn.dim != self.weight_norm_args["v_dim"]:
                wn.dim = self.weight_norm_args["v_dim"]
                # Adjust the norm values.
                weight = getattr(self, self.weight_norm_args["name"] + "_v")
                norm = getattr(self, self.weight_norm_args["name"] + "_g")
                norm.data[:] = th.norm_except_dim(weight, 2, wn.dim)
            self.is_fused = False

        def __deepcopy__(self, memo):
            # Delete derived tensor to avoid deepcopy error.
            if not self.is_fused:
                delattr(self, self.weight_norm_args["name"])

            # Deepcopy.
            cls = self.__class__
            result = cls.__new__(cls)
            memo[id(self)] = result
            for k, v in self.__dict__.items():
                setattr(result, k, copy.deepcopy(v, memo))

            if not self.is_fused:
                setattr(result, self.weight_norm_args["name"], None)
                setattr(self, self.weight_norm_args["name"], None)
            return result

    return Wrap

def is_weight_norm_wrapped(module):
    for hook in module._forward_pre_hooks.values():
        if isinstance(hook, WeightNorm):
            return True
    return False

class Conv1dUB(th.nn.Conv1d):
    def __init__(self, in_channels, out_channels, width, *args, bias=True, **kwargs):
        """ Conv2d with untied bias. """
        super().__init__(in_channels, out_channels, *args, bias=False, **kwargs)
        self.bias = th.nn.Parameter(th.zeros(out_channels, width)) if bias else None

    # TODO: remove this method once upgraded to pytorch 1.8
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        # Copied from pt1.8 source code.
        if self.padding_mode != 'zeros':
            input = thf.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            return thf.conv1d(
                input, weight, bias, self.stride, _pair(0), self.dilation, self.groups
            )
        return thf.conv1d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input: Tensor) -> Tensor:
        output = self._conv_forward(input, self.weight, None)
        bias = self.bias
        if bias is not None:
            # Assertion for jit script.
            assert bias is not None
            output = output + bias[None]
        return output


class Conv2dUB(th.nn.Conv2d):
    def __init__(self, in_channels, out_channels, height, width, *args, bias=True, **kwargs):
        """ Conv2d with untied bias. """
        super().__init__(in_channels, out_channels, *args, bias=False, **kwargs)
        self.bias = th.nn.Parameter(th.zeros(out_channels, height, width)) if bias else None

    # TODO: remove this method once upgraded to pytorch 1.8
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        # Copied from pt1.8 source code.
        if self.padding_mode != 'zeros':
            input = thf.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            return thf.conv2d(
                input, weight, bias, self.stride, _pair(0), self.dilation, self.groups
            )
        return thf.conv2d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input: Tensor) -> Tensor:
        output = self._conv_forward(input, self.weight, None)
        bias = self.bias
        if bias is not None:
            # Assertion for jit script.
            assert bias is not None
            output = output + bias[None]
        return output


class ConvTranspose1dUB(th.nn.ConvTranspose1d):
    def __init__(self, in_channels, out_channels, width, *args, bias=True, **kwargs):
        """ ConvTranspose1d with untied bias. """
        super().__init__(in_channels, out_channels, *args, bias=False, **kwargs)
        self.bias = th.nn.Parameter(th.zeros(out_channels, width)) if bias else None

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        # Copied from pt1.8 source code.
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')

        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation
        )

        output = thf.conv_transpose1d(
            input,
            self.weight,
            None,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
        bias = self.bias
        if bias is not None:
            # Assertion for jit script.
            assert bias is not None
            output = output + bias[None]
        return output


class ConvTranspose2dUB(th.nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, height, width, *args, bias=True, **kwargs):
        """ ConvTranspose2d with untied bias. """
        super().__init__(in_channels, out_channels, *args, bias=False, **kwargs)
        self.bias = th.nn.Parameter(th.zeros(out_channels, height, width)) if bias else None

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        # Copied from pt1.8 source code.
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation
        )

        output = thf.conv_transpose2d(
            input,
            self.weight,
            None,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
        bias = self.bias
        if bias is not None:
            # Assertion for jit script.
            assert bias is not None
            output = output + bias[None]
        return output



# Set default g_dim=0 (Conv2d) or 1 (ConvTranspose2d) and v_dim=None to preserve
# the current weight norm behavior.
LinearWN = weight_norm_wrapper(th.nn.Linear, g_dim=0, v_dim=None)
Conv1dWN = weight_norm_wrapper(th.nn.Conv1d, g_dim=0, v_dim=None)
Conv1dWNUB = weight_norm_wrapper(Conv1dUB, g_dim=0, v_dim=None)
Conv2dWN = weight_norm_wrapper(th.nn.Conv2d, g_dim=0, v_dim=None)
Conv2dWNUB = weight_norm_wrapper(Conv2dUB, g_dim=0, v_dim=None)
ConvTranspose1dWN = weight_norm_wrapper(th.nn.ConvTranspose1d, g_dim=1, v_dim=None)
ConvTranspose1dWNUB = weight_norm_wrapper(ConvTranspose1dUB, g_dim=1, v_dim=None)
ConvTranspose2dWN = weight_norm_wrapper(th.nn.ConvTranspose2d, g_dim=1, v_dim=None)
ConvTranspose2dWNUB = weight_norm_wrapper(ConvTranspose2dUB, g_dim=1, v_dim=None)



class GatedConv2dWNSwish(torch.nn.Module):
    def __init__(self, in_channels, out_channels, *args, bias=True, **kwargs):
        super(GatedConv2dWNSwish, self).__init__()

        self.conv=Conv2dWN(in_channels, out_channels, *args, bias=bias, **kwargs )
        self.gated_conv=Conv2dWN(in_channels, out_channels, *args, bias=bias, **kwargs )

        self.swish=th.nn.SiLU()
        self.sigmoid=torch.nn.Sigmoid()

    def forward(self, input: Tensor) -> Tensor:
        conv_out=self.swish(self.conv(input))
        gated_out=self.sigmoid(self.gated_conv(input))

        output=conv_out*gated_out

        return output



class InterpolateHook(object):
    def __init__(self, size=None, scale_factor=None, mode="bilinear"):
        """An object storing options for interpolate function"""
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def __call__(self, module, x):
        assert len(x) == 1, "Module should take only one input for the forward method."
        return thf.interpolate(
            x[0],
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=False,
        )


def interpolate_wrapper(cls):
    """Wraps a torch.nn.Module class and perform additional interpolation on the
    first and only positional input of the forward method.
    """

    class Wrap(cls):
        def __init__(self, *args, size=None, scale_factor=None, mode="bilinear", **kwargs):
            check_args_shadowing(
                "interpolate_wrapper", super().__init__, ["size", "scale_factor", "mode"]
            )
            super().__init__(*args, **kwargs)
            self.register_forward_pre_hook(
                InterpolateHook(size=size, scale_factor=scale_factor, mode=mode)
            )

    return Wrap


UpConv2d = interpolate_wrapper(th.nn.Conv2d)
UpConv2dWN = interpolate_wrapper(Conv2dWN)
UpConv2dWNUB = interpolate_wrapper(Conv2dWNUB)


class GlobalAvgPool(th.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], x.shape[1], -1).mean(dim=2)

class Upsample(th.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return thf.interpolate(x, *self.args, **self.kwargs)


def glorot(m, alpha=0.2):
    gain = np.sqrt(2.0 / (1.0 + alpha ** 2))

    if isinstance(m, th.nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, th.nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, th.nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // 2
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, th.nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // 4
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, th.nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] // 8
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, th.nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * np.sqrt(2.0 / (n1 + n2))
    else:
        return

    is_wnw = is_weight_norm_wrapped(m)
    if is_wnw:
        m.fuse()

    m.weight.data.uniform_(-std * np.sqrt(3.0), std * np.sqrt(3.0))
    if m.bias is not None:
        m.bias.data.zero_()

    if isinstance(m, th.nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    if is_wnw:
        m.unfuse()



def swish_init(m, is_linear, scale=1):

    #mport here in rder to avoid circular dependency
    from latticenet_py.lattice.lattice_modules import ConvLatticeIm2RowModule


    # is_wnw = is_weight_norm_wrapped(m)
    # if is_wnw:
    #     m.fuse()
    # if hasattr(m, 'weight'):
    #     torch.nn.init.kaiming_normal_(m.weight)
    # if hasattr(m, 'bias'):
    #     if m.bias is not None:
    #         m.bias.data.zero_()
    # if is_wnw:
    #     m.unfuse()
    # return

    #nromally relu has a gain of sqrt(2)
    #however swish has a gain of sqrt(2.952) as per the paper https://arxiv.org/pdf/1805.08266.pdf
    gain=np.sqrt(2.952)
    # gain=np.sqrt(3.2)
    # gain=np.sqrt(3)
    # gain=np.sqrt(2)
    if is_linear:
        gain=1
        # gain = np.sqrt(2.0 / (1.0 + 1 ** 2))
        # print("is lienar")



    if isinstance(m, th.nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        # std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
        std = gain / np.sqrt( ((n1 ) * ksize))
        # std = gain / np.sqrt( ((n2 ) * ksize))
    elif isinstance(m, th.nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        # std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
        std = gain / np.sqrt( ((n1 ) * ksize))
        # std = gain / np.sqrt( ((n2 ) * ksize))
    # elif isinstance(m, PacConv2d):
    #     print("pac init")
    #     ksize = m.kernel_size[0] * m.kernel_size[1]
    #     n1 = m.in_channels
    #     n2 = m.out_channels

    #     # std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    #     std = gain / np.sqrt( ((n1 ) * ksize))
    #     # std = gain / np.sqrt( ((n2 ) * ksize))
    elif isinstance(m, th.nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // 2
        n1 = m.in_channels
        n2 = m.out_channels

        # std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
        std = gain / np.sqrt( ((n1 ) * ksize))
        # std = gain / np.sqrt( ((n2 ) * ksize))
    elif isinstance(m, th.nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // 4
        n1 = m.in_channels
        n2 = m.out_channels

        # std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
        std = gain / np.sqrt( ((n1) * ksize))
        # std = gain / np.sqrt( ((n2) * ksize))
    elif isinstance(m, th.nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] // 8
        n1 = m.in_channels
        n2 = m.out_channels

        # std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
        std = gain / np.sqrt( ((n1 ) * ksize))
        # std = gain / np.sqrt( ((n2 ) * ksize))
    elif isinstance(m, th.nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        # std = gain * np.sqrt(2.0 / (n1 + n2))
        std = gain / np.sqrt( (n1 ))
        # std = gain / np.sqrt( (n2 ))
    #LATTICE THINGS
    elif isinstance(m, ConvLatticeIm2RowModule):
        # n1 = m.in_features
        # n2 = m.out_features
        # std = gain / np.sqrt( (n1 ))
        # print("init convlattice")

        # print("conv lattice weight is ", m.weight.shape)
        n1 = m.in_channels
        n2 = m.nr_filters
        filter_extent=m.filter_extent
        # print("filter_extent", filter_extent)
        # print("n1", n1)
        std = gain / np.sqrt( ((n1 ) * filter_extent))
        # return
    else:
        return


    # print("applying init to a ",m)

    is_wnw = is_weight_norm_wrapped(m)
    if is_wnw:
        m.fuse()

    # m.weight.data.uniform_(-std * np.sqrt(3.0), std * np.sqrt(3.0))
    # print("scale is ", scale)
    # print("normal is ", std*scale)
    m.weight.data.normal_(0, std*scale)
    if m.bias is not None:
        m.bias.data.zero_()
        # m.bias.data.normal_(0, np.sqrt(0.04))

    if isinstance(m, th.nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    if is_wnw:
        m.unfuse()

    m.weights_initialized=True

#init the positional encoding layers, should be done after the swish_init
def pe_layers_init(m):





    if isinstance(m, LearnedPE):
        m.init_weights()
        print("init LearnedPE")
    else:
        return



def apply_weight_init_fn(m, fn, is_linear=False, scale=1):

    should_initialize_weight=True
    if not hasattr(m, "weights_initialized"): #if we don't have this then we need to intiialzie
        # fn(m, is_linear, scale)
        should_initialize_weight=True
    elif m.weights_initialized==False: #if we have it but it's set to false
        # fn(m, is_linear, scale)
        should_initialize_weight=True
    else:
        # print("skipping weight init on ", m)
        should_initialize_weight=False

    if should_initialize_weight:
        fn(m, is_linear, scale)
        m.weights_initialized=True
        for module in m.children():
            apply_weight_init_fn(module, fn, is_linear, scale)


def apply_weight_init_fn_glorot(m, fn):

    should_initialize_weight=True
    if not hasattr(m, "weights_initialized"): #if we don't have this then we need to intiialzie
        # fn(m, is_linear, scale)
        should_initialize_weight=True
    elif m.weights_initialized==False: #if we have it but it's set to false
        # fn(m, is_linear, scale)
        should_initialize_weight=True
    else:
        # print("skipping weight init on ", m)
        should_initialize_weight=False

    if should_initialize_weight:
        fn(m)
        for module in m.children():
            apply_weight_init_fn_glorot(module, fn)
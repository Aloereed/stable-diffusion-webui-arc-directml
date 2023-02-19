import sys
import contextlib
import torch
import torch_directml
from modules import errors
from modules.sd_hijack_utils import CondFunc
from packaging import version
from functools import reduce
import operator
from modules import errors

if sys.platform == "darwin":
    from modules import mac_specific


def has_mps() -> bool:
    if sys.platform != "darwin":
        return False
    else:
        return mac_specific.has_mps

def extract_device_id(args, name):
    for x in range(len(args)):
        if name in args[x]:
            return args[x + 1]

    return None


def get_cuda_device_string():
    from modules import shared

    if shared.cmd_opts.device_id is not None:
        return f"cuda:{shared.cmd_opts.device_id}"

    return "cuda"


def get_dml_device_string():
    from modules import shared

    if shared.cmd_opts.device_id is not None:
        return f"privateuseone:{shared.cmd_opts.device_id}"

    return "privateuseone:0"


def get_optimal_device_name():
    if torch.cuda.is_available():
        return get_cuda_device_string()

    if has_mps():
        return "mps"

    if torch_directml.is_available():
        return get_dml_device_string()

    return "cpu"


def get_optimal_device():
    return torch.device(get_optimal_device_name())


def get_device_for(task):
    from modules import shared

    if task in shared.cmd_opts.use_cpu:
        return cpu

    return get_optimal_device()


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(get_cuda_device_string()):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def enable_tf32():
    if torch.cuda.is_available():

        # enabling benchmark option seems to enable a range of cards to do fp16 when they otherwise can't
        # see https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/4407
        if any([torch.cuda.get_device_capability(devid) == (7, 5) for devid in range(0, torch.cuda.device_count())]):
            torch.backends.cudnn.benchmark = True

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


errors.run(enable_tf32, "Enabling TF32")

cpu = torch.device("cpu")
adl = None
hMEM = None
try:
    dml = torch_directml.device()
    # if dml.type == "privateuseone" and "AMD" in torch_directml.device_name(dml.index):
    #     from modules import atiadlxx
    #     adl = atiadlxx.ATIADLxx()
    #     hMEM = adl.getMemoryInfo2(0).iHyperMemorySize
    # else:
    #     print("Intel! Warning: experimental graphic memory optimizations are disabled due to gpu vendor.")
except RuntimeError as e:
    if str(e) == 'NOT_WINDOWS':
        print("Memory optimization for DirectML is disabled. Because this is not Windows platform.")
    else:
        print("Memory optimization for DirectML is disabled. Because there is an unknown error.")
device = device_interrogate = device_gfpgan = device_esrgan = device_codeformer = None
dtype = torch.float16
dtype_vae = torch.float16
dtype_unet = torch.float16
unet_needs_upcast = False


def cond_cast_unet(input):
    return input.to(dtype_unet) if unet_needs_upcast else input


def cond_cast_float(input):
    return input.float() if unet_needs_upcast else input


def randn(seed, shape):
    torch.manual_seed(seed)
    if device.type == 'mps':
        return torch.randn(shape, device=cpu).to(device)
    return torch.randn(shape, device=device)


def randn_without_seed(shape):
    if device.type == 'mps':
        return torch.randn(shape, device=cpu).to(device)
    return torch.randn(shape, device=device)


def autocast(disable=False):
    from modules import shared

    if disable:
        return contextlib.nullcontext()

    if dtype == torch.float32 or shared.cmd_opts.precision == "full":
        return contextlib.nullcontext()

    return torch.autocast("cuda")


def without_autocast(disable=False):
    return torch.autocast("cuda", enabled=False) if torch.is_autocast_enabled() and not disable else contextlib.nullcontext()


class NansException(Exception):
    pass


def test_for_nans(x, where):
    from modules import shared

    if shared.cmd_opts.disable_nan_check:
        return

    if not torch.all(torch.isnan(x)).item():
        return

    if where == "unet":
        message = "A tensor with all NaNs was produced in Unet."

        if not shared.cmd_opts.no_half:
            message += " This could be either because there's not enough precision to represent the picture, or because your video card does not support half type. Try setting the \"Upcast cross attention layer to float32\" option in Settings > Stable Diffusion or using the --no-half commandline argument to fix this."

    elif where == "vae":
        message = "A tensor with all NaNs was produced in VAE."

        if not shared.cmd_opts.no_half and not shared.cmd_opts.no_half_vae:
            message += " This could be because there's not enough precision to represent the picture. Try adding --no-half-vae commandline argument to fix this."
    else:
        message = "A tensor with all NaNs was produced."

    message += " Use --disable-nan-check commandline argument to disable this check."

    raise NansException(message)


# MPS workaround for https://github.com/pytorch/pytorch/issues/89784
def cumsum_fix(input, cumsum_func, *args, **kwargs):
    if input.device.type == 'mps':
        output_dtype = kwargs.get('dtype', input.dtype)
        if output_dtype == torch.int64:
            return cumsum_func(input.cpu(), *args, **kwargs).to(input.device)
        elif cumsum_needs_bool_fix and output_dtype == torch.bool or cumsum_needs_int_fix and (output_dtype == torch.int8 or output_dtype == torch.int16):
            return cumsum_func(input.to(torch.int32), *args, **kwargs).to(torch.int64)
    return cumsum_func(input, *args, **kwargs)


class GroupNorm(torch.nn.GroupNorm):
    def forward(self, x):
        if (x.dtype == torch.float16 or self.weight.dtype == torch.float16) and x.device.type == 'privateuseone':
            self.weight = torch.nn.Parameter(self.weight.float())
            self.bias = torch.nn.Parameter(self.bias.float())
            return super().forward(x.float()).type(x.dtype)
        else:
            return super().forward(x)


class LayerNorm(torch.nn.LayerNorm):
    def forward(self, x):
        if (x.dtype == torch.float16 or self.weight.dtype == torch.float16) and x.device.type == 'privateuseone':
            self.weight = torch.nn.Parameter(self.weight.float())
            if self.bias is not None and self.bias.dtype == torch.float16:
                self.bias = torch.nn.Parameter(self.bias.float())
            return super().forward(x.float()).type(x.dtype)
        else:
            return super().forward(x)


class Linear(torch.nn.Linear):
    def forward(self, x):
        if (x.dtype == torch.float16 or self.weight.dtype == torch.float16) and x.device.type == 'privateuseone':
            self.weight = torch.nn.Parameter(self.weight.float())
            if self.bias is not None and self.bias.dtype == torch.float16:
                self.bias = torch.nn.Parameter(self.bias.float())
            return super().forward(x.float()).type(x.dtype)
        else:
            return super().forward(x)


class Conv2d(torch.nn.Conv2d):
    def forward(self, x):
        if (x.dtype == torch.float16 or self.weight.dtype == torch.float16) and x.device.type == 'privateuseone':
            self.weight = torch.nn.Parameter(self.weight.float())
            if self.bias is not None and self.bias.dtype == torch.float16:
                self.bias = torch.nn.Parameter(self.bias.float())
            return super().forward(x.float()).type(x.dtype)
        else:
            return super().forward(x)


_pad = torch.nn.functional.pad
def pad(input, pad, mode='constant', value=None):
    if input.dtype == torch.float16 and input.device.type == 'privateuseone':
        return _pad(input.float(), pad, mode, value).type(input.dtype)
    else:
        return _pad(input, pad, mode, value)


_cumsum = torch.Tensor.cumsum
def cumsum(self, *args, **kwargs):
    if self.dtype == torch.bool:
        return _cumsum(self.int(), *args, **kwargs)
    else:
        return _cumsum(self, *args, **kwargs)


if torch_directml.is_available():
    torch.nn.GroupNorm = GroupNorm
    torch.nn.LayerNorm = LayerNorm
    torch.nn.Linear = Linear
    torch.nn.Conv2d = Conv2d
    torch.nn.functional.pad = pad

    torch.Tensor.cumsum = cumsum

    CondFunc('torchsde._brownian.brownian_interval._randn', lambda _, size, dtype, device, seed: torch.randn(size, dtype=dtype, device=torch.device("cpu"), generator=torch.Generator(torch.device("cpu")).manual_seed(int(seed))).to(device), lambda _, size, dtype, device, seed: device.type == 'privateuseone')

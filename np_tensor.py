from typing import Dict, List, Tuple
from collections.abc import Iterable
import numpy as np

import torch
from torch.utils._pytree import tree_map
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.logging_tensor import no_dispatch

from collections import namedtuple


# What are all the layers to this?
# --------------------------------
# NPTensorWrapper: Interface to the rest of PyTorch (e.g. Python frontend, transforms like Autograd)
#                  Given `func` calls into the corresponding function implemented by the backend
#                  which is expected to have a torch-like interface (i.e. ops match the semantics of the
#                  corresponding PyTorch op).
# Decomposition  : Lowers higher-level PyTorch ops to a smaller set of more primitive PyTorch ops.
# NPTensor       : Wraps around np.ndarray to provide a torch-like interface
# np.ndarray     : Backend implementation

# Table of contents
# -----------------
# 0) Helper functions
# 1) Operators
#   1.1) Register operators that call straight into primitive backend operations (prims)
#   1.2) Register decompositions that call into other "composite" operators as well as prims
#   1.3) Small testing helper function for collecting example runs
# 2) Mock backend (backend by numpy array but implements torch semantics)
# 3) Subclass wrapping backend
# 4) XOR Example
# 5) Testing our implemented operators/decomopsition using collected examples runs from (1.3)

# To Do
# -----
# - dtypes and casting
# - interaction with parameter
# - inplace (used by optimizers)
#   - inplace a subclass into a non-subclass (what does the backend do here?)
# - view tests


# 0) Helpers

def make_tuple(outs):
    return (outs,) if not isinstance(outs, Iterable) or isinstance(outs, torch.Tensor) else tuple(outs)

def contig_strides_from_shape(shape):
    return torch.zeros(shape).stride()

# From torch/testing/_internal/common_utils.py
numpy_to_torch_dtype_dict = {
    np.bool       : torch.bool,
    np.uint8      : torch.uint8,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128
}
torch_to_numpy_dtype_dict = {v: k for k, v in numpy_to_torch_dtype_dict.items()}

# Uh oh, why are we getting ints?
def torch_to_numpy_dtype_dict_int(dtype_int):
    if dtype_int == 6:
        return np.float32  # Not sure if this is even the right one
    else:
        assert False, f"got: {dtype_int=}"

# 1) Operators

# Maps torch op names to operators that our backend has implemented
all_funcs = dict()

def register_func(name):
    # Helpful decorator to colocate registration and definition
    def register_func_inner(f):
        assert name not in all_funcs, f"{name} has already been registered"
        all_funcs[name] = f
        return f
    return register_func_inner

# 1.1) Register adapters for numpy backend

# These simply just redispatch
@register_func("add.Tensor")
def add(x, y):
    return x + y

@register_func("mul.Tensor")
def mul(x, y):
    # NB: Explicitly call __mul__ to make sure NPTensor (and not numpy) handles the call
    if isinstance(x, np.ndarray):
        return y.__mul__(x)
    else:
        out = x.__mul__(y)
        return out

@register_func("view.default")
def view(x, shape):
    return x.view(shape)

@register_func("exp.default")
def exp(x):
    return x.exp()

@register_func("sum.dim_IntList")
def sum(x, dims, keepdims=False):
    return x.sum(dims, keepdims)

@register_func("mm.default")
def mm(x, y):
    return x @ y

@register_func("squeeze.dim")
def squeeze(x, dim):
    return x.squeeze(dim)

@register_func("unsqueeze.default")
def unsqueeze(x, dim):
    return x.unsqueeze(dim)

@register_func("ones_like.default")  # Maybe this shouldn't be a method
def ones_like(x, dtype, layout, device, pin_memory, memory_format):
    # TODO: what do we do with all these arguments?
    return x.ones_like(dtype=dtype)

# @register_func("_to_copy.default")
# def _to_copy(x, dtype):
#     pass

# 1.2) Register decompositions

# These redispatch into more primitive ops, e.g. sigmoid -> exp
# This means that the backend does not have to directly support all of these

@register_func("relu.default")
def relu(x):
    return x * (x > 0)

@register_func("addmm.default")
def addmm(bias, a, b):
    return a @ b + bias

@register_func("sigmoid.default")
def sigmoid(x):
    return 1 / (1 + (x * -1).exp())

@register_func("sigmoid_backward.default")
def sigmoid_backward(grad_out, x):
    return grad_out * sigmoid(x) * (1 - sigmoid(x))

@register_func("detach.default")
def detach(x):
    # To the backend, detach is simply a view
    return x.alias()

@register_func("mse_loss.default")
def mse_loss(x, y, reduction="mean"):
    if reduction == "none":
        return (x - y)**2
    elif reduction == "sum":
        return ((x - y)**2).sum()
    elif reduction == "mean":
        return ((x - y)**2).sum() / x.numel()
    else:
        allowed_reductions = ("none", "mean", "sum")
        assert False, (
            f"Expected 'reduction' to be one of {allowed_reductions}, but got: {reduction}")

@register_func("mse_loss_backward.default")
def mse_loss_backward(grad_out, self, target, reduction):
    def unsqueeze_multiple(x, nr_times):
        # NB: Make sure broadcasting works -
        #     forward always reduces out to a scalar, but in NPTensor, .arr should have at least dim of 1
        for _ in range(nr_times):
            x = x.unsqueeze(-1)
        return x
    if reduction == 0:    # Reduction::None
        ret = grad_out
    elif reduction == 1:  # Reduction::Mean
        ret = unsqueeze_multiple(grad_out, self.ndim() - grad_out.ndim()) / self.numel()
    elif reduction == 2:  # Reduction::Sum
        ret = unsqueeze_multiple(grad_out, self.ndim() - grad_out.ndim())
    else:
        assert False, f"This is a bug. Got reduction: {reduction}"

    return  2 * (self - target) * ret

@register_func("threshold_backward.default")
def threshold_backward(grad_out, self, threshold):
    return grad_out * (self > threshold)

@register_func("t.default")
def t(x):
    if x.ndim() == 0:
        assert False, "Huh? Scalars .arr should not exist"
    elif x.ndim() == 1:
        return x
    elif x.ndim() == 2:
        return x.permute((1, 0))
    else:
        assert False


# 1.3) Testing for Operators:

# Helper for collecting example runs into a dict

OpInputOutput = namedtuple("OpInputOutput", "op_name input_args input_kwargs outputs")
op_in_out_examples: Dict[str, OpInputOutput] = dict()

def collect_example_run(op_name, args, kwargs, outs) -> None:
    outs = (outs,) if not isinstance(outs, Iterable) or isinstance(outs, torch.Tensor) else tuple(outs)
    new_example = OpInputOutput(op_name, args, kwargs, outs)
    op_in_out_examples[op_name] = op_in_out_examples.get(op_name, []) + [new_example]


# 2) Mock backend

class NPTensor():
    # This is a "backend" wraps a numpy array and implements a tensor-like interface
    #
    # Notes:
    # - The semantics of this object will try to follow that of torch. The idea is that
    #   decompositions (mapping from torch ops to more primtive torch ops) will eventually
    #   be something that PyTorch should eventually provide as well so that shouldn't be the
    #   place where the numpy-torch translation happens.
    # - Autograd-aware ops like detach should not exist here
    # - This "Tensor" supports operations between NPTensor and unwrapped numpy array
    #
    # TODO:
    # - shape/view operators (NPTenosrWrapper may need metadata sync)
    # - casting operators
    # - we only have methods, what about functions?

    def __init__(self, arr):
        self.arr = np.atleast_1d(arr)  # TODO: maybe not do this?
        self.public_shape = lambda: arr.shape
        self.public_dtype = lambda: str(arr.dtype)

    def maybe_unwrap(self, arg):
        assert isinstance(arg, np.ndarray) or isinstance(arg, NPTensor) or \
            isinstance(arg, float) or isinstance(arg, int)
        return arg.arr if isinstance(arg, NPTensor) else arg

    def __add__(self, other):
        return NPTensor(self.arr + self.maybe_unwrap(other))

    def __radd__(self, other):
        return NPTensor(self.arr + self.maybe_unwrap(other))

    # def __iadd__(self, other):
    #     self.arr += self.maybe_unwrap(other)
    #     return  NPTensor(self.arr)

    def __sub__(self, other):
        return NPTensor(self.arr - self.maybe_unwrap(other))

    def __rsub__(self, other):
        return NPTensor(self.maybe_unwrap(other) - self.arr)

    def __mul__(self, other):
        return NPTensor(self.arr * self.maybe_unwrap(other))

    def __rmul__(self, other):
        return NPTensor(self.arr * self.maybe_unwrap(other))

    def __pow__(self, other):
        return NPTensor(self.arr ** self.maybe_unwrap(other))

    def __rtruediv__(self, other):
        return NPTensor(self.maybe_unwrap(other) / self.arr)

    def __truediv__(self, other):
        return NPTensor(self.arr / self.maybe_unwrap(other))

    def __matmul__(self, other):
        return NPTensor(self.arr @ self.maybe_unwrap(other))

    def __gt__(self, other):
        return NPTensor(self.arr > self.maybe_unwrap(other))

    def view(self, shape):
        # TODO: Add a test for the error case (it would be a numpy error, /should/ it be?)
        # NB: np.view is the same as torch.view
        # https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
        ret = self.arr.view()
        ret.shape = shape
        return NPTensor(ret)

    def alias(self):
        return NPTensor(self.arr.view())

    def exp(self):
        return NPTensor(np.exp(self.arr))

    def sum(self, dims=None, keepdims=False):
        # TODO: This not only handles sum_dim_IntList, but also dims=None
        # Maybe there is a better way to deal with overloads?
        assert isinstance(dims, List) or dims is None
        if dims is None:
            return NPTensor(np.sum(self.arr, keepdims=keepdims))
        return NPTensor(np.sum(self.arr, axis=tuple(dims), keepdims=keepdims))

    def squeeze(self, dim):
        return NPTensor(self.arr.squeeze(dim))

    def unsqueeze(self, dim):
        return NPTensor(np.expand_dims(self.arr, dim))

    def permute(self, perm: Tuple[int]):
        # np.transpose is torch.permute when axes is specified
        return NPTensor(np.transpose(self.arr, axes=perm))

    def ones_like(self, dtype):
        return NPTensor(np.ones_like(self.arr, dtype=torch_to_numpy_dtype_dict_int(dtype)))

    def __repr__(self):
        return repr(self.arr)

    def publish(self, sigma=None):
        # For PySyft
        return self.arr

    # Methods that are supposed to be on Tensor, but don't redispatch
    # Our decompositions shouldn't really rely on these?

    def numel(self):
        return self.arr.size

    def ndim(self):
        return len(self.arr.shape)

# conversion functions with dtype sserts
def torch_to_numpy(t):
    assert isinstance(t, torch.Tensor), f"Expected torch.Tensor, got: {type(t)}"
    assert not isinstance(t, NPTensorWrapper)
    ret = t.numpy()
    assert ret.dtype.type == torch_to_numpy_dtype_dict[t.dtype]
    return ret

def numpy_to_torch(arr):
    assert isinstance(arr, np.ndarray), f"Expected nd.array, got: {type(arr)}"  # What about scalars?
    ret = torch.from_numpy(arr)
    # dtype object actually encapsulates more information such as byte order, etc.
    assert ret.dtype == numpy_to_torch_dtype_dict[arr.dtype.type]
    return ret

# 3) The subclass wrapping our mock backend

class NPTensorWrapper(torch.Tensor):
    # Wrap the tensor-like interface in an actual tensor so that we can use autograd

    @staticmethod
    def __new__(cls, pointer, requires_grad=False):
        assert isinstance(pointer, NPTensor)
        # For logging tensor: "it should advertise the same device as before", but for
        # this tensor was no previous device, so maybe just default to CPU?

        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            pointer.public_shape(),
            strides=contig_strides_from_shape(pointer.public_shape()),
            storage_offset=0,  # elem.storage_offset(),
            # TODO: clone storage aliasing
            dtype=numpy_to_torch_dtype_dict[getattr(np, pointer.public_dtype())],
            layout=torch.strided,
            device="cpu",  # elem.device
            requires_grad=requires_grad
        )
        r.pointer = pointer

        return r

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def wrap(t):  # NPTensor -> NPTensorWrapper
            if isinstance(t, NPTensor):
                return NPTensorWrapper(t)
            elif isinstance(t, np.ndarray):
                # assert func.__name__ in ("add_.Tensor", "addcmul_.default")
                # return NPTensorWrapper(NPTensor(t))
                assert False
            else:
                return t

        def unwrap(t):  # NPTensorWrapper -> NPTensor, torch.Tensor -> np.ndarray
            # NB: Adding a numpy array to NPTensor returns a NPTensor, the analog for this wrapper
            #     is that users can add torch.Tensor to NPTensorWrapper and obtain NPTensorWrapper
            if isinstance(t, NPTensorWrapper):
                return t.pointer
            elif isinstance(t, torch.Tensor):
                return torch_to_numpy(t.detach())
            else:
                # assert False, f"We shouldn't reach here right? Got: type {type(t)}"
                return t

        # NPTensorWrapper -> torch.Tensor
        def unwrap2(t):
            return numpy_to_torch(t.pointer.arr) if isinstance(t, NPTensorWrapper) else t  # .to(dtype=torch.float32)

        # torch.Tensor -> NPTensorWrapper
        def wrap2(t):
            return NPTensorWrapper(NPTensor(torch_to_numpy(t.detach()))) if isinstance(t, torch.Tensor) else t

        # print([type(t) for t in args])
        # print([t.dtype for t in args])

        # Always run so that we can record for the test!
        tensor_args = tree_map(unwrap2, args)
        tensor_kwargs = tree_map(unwrap2, kwargs)

        # print([t.dtype for t in tensor_args if isinstance(t, torch.Tensor)])

        def detach_tensors(x):  # Avoid keeping the graph alive. But it also fails for other reasons (why?)
            # assert isinstance(x, tuple) or isinstance(x, dict)
            with no_dispatch():  # Cannot use this because we may have other subclasses like parameter!
                if isinstance(x, tuple):  # Clone or else it breaks (does it still happen? and why?)
                    return [t.clone().detach() if isinstance(t, torch.Tensor) else t for t in x]
                elif isinstance(x, dict):
                    return {k: (v.clone().detach() if isinstance(v, torch.Tensor) else v) for k, v in x.items()}
                elif isinstance(x, torch.Tensor):
                    return x.clone().detach()
            return x

        # Clone inputs before running func in case the operation is in-place
        detached_args = detach_tensors(tensor_args)
        detached_kwargs = detach_tensors(tensor_kwargs)

        raw_out = func(*tensor_args, **tensor_kwargs)

        detached_outs = detach_tensors(raw_out)
        collect_example_run(func.__name__, detached_args, detached_kwargs, detached_outs)

        try:
            out = tree_map(wrap, all_funcs[func.__name__](*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))
        except KeyError as e:
            out = None

        # Fallback if necessary
        return tree_map(wrap2, raw_out) if out is None else out

    def __repr__(self):
        # TODO: also print autograd information?
        return repr(self.pointer)

    # Non-tensor APIs from the backend to just forward
    def synthetic(self):
        return self.pointer.synthetic

    def publish(self, sigma=1e5):
        # We lose the graph though. TODO: find some way to reattach the graph?
        return numpy_to_torch(self.pointer.publish(sigma=1e5))


# 4) Run through the XOR training example

import torch.nn as nn
import torch.nn.functional as F

SIGMA = 0.4
N_PER_QUAD = 100
N_ITER = 100

train_X_centers = [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
train_y = [0, 1, 1, 0]

shuffle_idx = np.arange(N_PER_QUAD * 4)
np.random.shuffle(shuffle_idx)
train_X_list = []
train_y_list = []

for center, label in zip(train_X_centers, train_y):
    train_X_list.append(np.random.randn(N_PER_QUAD, 2) * SIGMA + np.array(center))
    train_y_list.append(np.zeros(N_PER_QUAD) if label == 0 else np.ones(N_PER_QUAD))

# Linear layer does not support double
train_X_np = np.concatenate(train_X_list)[shuffle_idx].astype(np.float32)
train_y_np = np.concatenate(train_y_list)[shuffle_idx].astype(np.float32)

# Convert input data to torch.Tensor

train_X = NPTensorWrapper(NPTensor(train_X_np))
train_y = NPTensorWrapper(NPTensor(train_y_np))

# 4.1) Simple MLP

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(2, 10)
        self.linear2 = nn.Linear(10, 10)
        self.linear3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x.squeeze(1).sigmoid()

# 4.2) Train

model =  Net()
sgd = torch.optim.Adam(model.parameters(), lr=1e-2)

# Register hook that publishes gradients (unwraps)
for p in model.parameters():
    p.register_hook(lambda x: x.publish())

losses = []
for _ in range(N_ITER):
    out = model(train_X)
    loss = F.mse_loss(out, train_y)
    loss.backward()
    sgd.step()
    sgd.zero_grad()
    losses.append(loss.detach())

# 4.3) Test

test_X = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
test_y = torch.tensor([[0., 1., 1., 0.]])
predicted = model(test_X)

assert torch.allclose(predicted, test_y, atol=1e-2)
print("\nTraining works! (Successfully predicted test set)\n")


# 5) Test the functions in our backend using collected examples runs

def nptensor_to_torch(t):
    return numpy_to_torch(t.arr) if isinstance(t, NPTensor) else t  # .to(dtype=torch.float32)

def torch_to_nptensor(t):
    return NPTensor(torch_to_numpy(t.detach())) if isinstance(t, torch.Tensor) else t

for name, func in all_funcs.items():
    print(name)
    if name in ("add.Tensor", "mul.Tensor", "exp.default",):
        # TODO: xfail this
        # These may be called indirectly
        print("SKIP")
        continue
    example_runs = op_in_out_examples[name]
    for name, input_args, input_kwargs, expected_outputs in example_runs:
        raw_actual = func(*tree_map(torch_to_nptensor, input_args), **tree_map(torch_to_nptensor, input_kwargs))
        torch_actual = tree_map(nptensor_to_torch, raw_actual)

        expected_outputs = make_tuple(expected_outputs)
        torch_actual = make_tuple(torch_actual)

        assert len(expected_outputs) == len(torch_actual)

        for expected, actual in zip(expected_outputs, torch_actual):
            if isinstance(expected, torch.Tensor):
                assert isinstance(actual, torch.Tensor)

                if name in ("ones_like.default"):
                    actual = actual.to(dtype=expected.dtype)

                # Why such a high atol! TODO: op-specific atol? (e.g. for sigmoid_backward...)
                assert torch.allclose(expected, actual, atol=3e-3), (
                    f"Test failed for {name}:\n{expected = }, {actual =} "
                    f"\n{input_args = }, {input_kwargs = }"
                    f"\nMax difference: {(expected-actual).abs().max()}")
            else:
                assert expected == actual, f"Expected {name} to return {expected} outputs, but got: {actual}"
    print("OK!")
print("\nTests Pass!")

# List ops still using the fallback (e.g. that aren't supported by the backend)

print("\nList of ops that are using the fallback")
for name in op_in_out_examples.keys():
    if name in all_funcs:
        continue
    print(name)

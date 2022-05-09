from typing import List, Tuple
import unittest

import numpy as np
import torch
from torch.utils._pytree import tree_map
from torch.testing._internal.common_utils import TestCase, run_tests, numpy_to_torch_dtype_dict
from torch.testing._internal.common_device_type import ops, instantiate_device_type_tests
from torch.testing._internal.common_methods_invocations import op_db, DecorateInfo
from torch._decomp import decomposition_table

aten = torch.ops.aten

# 1) Mock backend

# NPTensor implements an tensor-like interface backed by NumPy arrays. We try to follow
# torch semantics as closely as possible because the layers above rely on that behavior.

class NPTensor():
    def __init__(self, arr):
        self.arr = arr  # TODO: maybe not do this?

    def __repr__(self):
        return f"NPTensor(arr={repr(self.arr)})"

    def publish(self):
        return self.arr

np_tensor_backend_impl = dict()

def register_func(name, ns):
    # Helpful decorator to colocate registration and definition
    def register_func_inner(f):
        assert name not in ns, f"{name} has already been registered"
        ns[name] = f
        return f
    return register_func_inner

def maybe_unwrap(arg):
    assert isinstance(arg, np.ndarray) or isinstance(arg, NPTensor) or \
        isinstance(arg, float) or isinstance(arg, int)
    return arg.arr if isinstance(arg, NPTensor) else arg

def fix_binary_type_promotion(f):
    # numpy and torch do binary type promotion differently
    expected_dtypes = (
        (np.int64, np.float32, np.float32),  # numpy promotes to np.float64
    )

    def check_dtype(x, y, lhs, rhs):
        # HACK: if lhs/rhs are int, make it a numpy array, so that we can check dtype
        # this might not be the right way to do it
        x = np.array(x) if isinstance(x, int) or isinstance(x, float) else maybe_unwrap(x)
        y = np.array(y) if isinstance(y, int) or isinstance(y, float) else maybe_unwrap(y)
        return x.dtype.type == lhs and y.dtype.type == rhs

    def wrapper(x, y):
        ret = f(x, y)
        for lhs, rhs, expected in expected_dtypes:
            if check_dtype(x, y, lhs, rhs) or check_dtype(y, x, lhs, rhs):
                ret = NPTensor(ret.arr.astype(dtype=expected))
        return ret

    return wrapper

torch_to_numpy_dtype_dict = {v: k for k, v in numpy_to_torch_dtype_dict.items()}

def torch_to_numpy_dtype_dict_int(dtype_int):
    if dtype_int == 6:
        return np.float32
    else:
        assert False, f"got: {dtype_int=}"

def numpy_to_torch(t):
    # Replacement for torch.from_numpy that actually handles scalars
    if np.isscalar(t):
        return torch.tensor(t)
    else:
        return torch.from_numpy(t)

@register_func(aten.add.Tensor, np_tensor_backend_impl)
@fix_binary_type_promotion
def np_tensor_add_impl(x, y):
    return NPTensor(maybe_unwrap(x) + maybe_unwrap(y))

@register_func(aten.sub.Tensor, np_tensor_backend_impl)
@fix_binary_type_promotion
def np_tensor_sub_impl(x, y):
    return NPTensor(maybe_unwrap(x) - maybe_unwrap(y))

@register_func(aten.rsub.Scalar, np_tensor_backend_impl)
@fix_binary_type_promotion
def np_tensor_rsub_impl(x, y):
    return NPTensor(maybe_unwrap(y) - maybe_unwrap(x))

@register_func(aten.mul.Tensor, np_tensor_backend_impl)
@fix_binary_type_promotion
def np_tensor_mul_impl(x, y):
    return NPTensor(maybe_unwrap(x) * maybe_unwrap(y))

@register_func(aten.div.Tensor, np_tensor_backend_impl)
@fix_binary_type_promotion
def np_tensor_div_impl(x, y):
    return NPTensor(maybe_unwrap(x) / maybe_unwrap(y))

@register_func(aten.view.default, np_tensor_backend_impl)
def np_tensor_view_impl(x, shape):
    ret = x.arr.view()
    ret.shape = shape
    return NPTensor(ret)

@register_func(aten.permute.default, np_tensor_backend_impl)
def np_tensor_permute_impl(x, perm):
    return NPTensor(np.transpose(x.arr, axes=perm))

@register_func(aten.pow.Tensor_Scalar, np_tensor_backend_impl)
def np_tensor_pow_impl(x, a):
    ret = maybe_unwrap(x) ** maybe_unwrap(a)
    if x.arr.dtype.type == np.float32:
        # Correct type promotion to match torch
        ret = ret.astype(dtype=np.float32)
    return NPTensor(ret)

@register_func(aten.sum.default, np_tensor_backend_impl)
def np_tensor_sum_default_impl(x, keepdims=False):
    return NPTensor(np.sum(x.arr, keepdims=keepdims))

@register_func(aten.sum.dim_IntList, np_tensor_backend_impl)
def np_tensor_sum_dim_IntList_impl(x, dims, keepdims=False):
    if x.arr.ndim == 0:
        ret = x.arr
    else:
        ret = np.sum(x.arr, axis=tuple(dims), keepdims=keepdims)
    return NPTensor(ret)

@register_func(aten.gt.Scalar, np_tensor_backend_impl)
def np_tensor_gt_impl(x, y):
    return NPTensor(maybe_unwrap(x) > maybe_unwrap(y))

@register_func(aten.reciprocal.default, np_tensor_backend_impl)
def np_tensor_reciprocal_impl(x):
    ret = 1 / x.arr
    if x.arr.dtype.type == np.float32:
        # Correct type promotion to match torch
        ret = ret.astype(dtype=np.float32)
    return NPTensor(ret)

@register_func(aten.exp.default, np_tensor_backend_impl)
def np_tensor_exp_impl(x):
    return NPTensor(np.exp(x.arr))

@register_func(aten.mm.default, np_tensor_backend_impl)
def np_tensor_mm_impl(x, y):
    return NPTensor(maybe_unwrap(x) @ maybe_unwrap(y))

@register_func(aten.squeeze.dim, np_tensor_backend_impl)
def np_tensor_squeeze_impl(x, dim):
    return NPTensor(x.arr.squeeze(dim))

@register_func(aten.unsqueeze.default, np_tensor_backend_impl)
def np_tensor_unsqueeze_impl(x, dim):
    return NPTensor(np.expand_dims(x.arr, dim))

@register_func(aten.ones_like.default, np_tensor_backend_impl)
def np_tensor_ones_like_impl(x, dtype, layout, device, pin_memory, memory_format):
    # TODO: do we care about the other arguments
    return NPTensor(np.ones_like(x.arr, dtype=torch_to_numpy_dtype_dict_int(dtype)))

@register_func(aten.detach.default, np_tensor_backend_impl)
def np_tensor_detach_impl(x):
    # To the backend, detach is simply a view
    return NPTensor(x.arr.view())

@register_func(aten.to.dtype, np_tensor_backend_impl)
def np_tensor_to_dtype_impl(x, dtype):
    # To the backend, detach is simply a view
    ret = x.arr.astype(dtype=torch_to_numpy_dtype_dict_int(dtype))
    return NPTensor(ret)

# For l1 loss (instead of mse loss)

@register_func(aten.abs.default, np_tensor_backend_impl)
def np_tensor_abs_default_impl(x):
    return NPTensor(np.abs(x.arr))

@register_func(aten.mean.default, np_tensor_backend_impl)
def np_tensor_mean_default_impl(x):
    return NPTensor(np.mean(x.arr))

@register_func(aten.sign.default, np_tensor_backend_impl)
def np_tensor_mean_default_impl(x):
    return NPTensor(np.sign(x.arr))

# 2) Decompositions

# Decomposing ops into more primitive ones reduces the number a backend will
# need to implement. Eventually torch should provide a set of "official"
# decompositions so we wouldn't need to write them ourselves here.

decompositions = dict()

@register_func(aten.relu.default, decompositions)
def relu(x):
    return x * (x > 0)

@register_func(aten.addmm.default, decompositions)
def addmm(bias, a, b):
    return torch.mm(a, b) + bias

@register_func(aten.sigmoid.default, decompositions)
def sigmoid(x):
    return 1 / (1 + torch.exp(x * -1))

@register_func(aten.sigmoid_backward.default, decompositions)
def sigmoid_backward(grad_out, result):
    return grad_out * result * (1 - result)

@register_func(aten.mse_loss.default, decompositions)
def mse_loss(x, y, reduction="mean"):
    if reduction == "none" or reduction == 0:
        return (x - y)**2
    elif reduction == "mean" or reduction == 1:
        return ((x - y)**2).sum() / x.numel()
    elif reduction == "sum" or reduction == 2:
        return ((x - y)**2).sum()
    else:
        allowed_reductions = ("none", "mean", "sum")
        assert False, (
            f"Expected 'reduction' to be one of {allowed_reductions}, but got: {reduction}")

@register_func(aten.mse_loss_backward.default, decompositions)
def mse_loss_backward(grad_out, self, target, reduction):
    def unsqueeze_multiple(x, nr_times):
        for _ in range(nr_times):
            x = x.unsqueeze(-1)
        return x
    if reduction == 0:    # Reduction::None
        ret = grad_out
    elif reduction == 1:  # Reduction::Mean
        ret = unsqueeze_multiple(grad_out, self.ndim - grad_out.ndim) / self.numel()
    elif reduction == 2:  # Reduction::Sum
        ret = unsqueeze_multiple(grad_out, self.ndim - grad_out.ndim)
    else:
        assert False

    return  2 * (self - target) * ret

@register_func(aten.threshold_backward.default, decompositions)
def threshold_backward(grad_out, self, threshold):
    return grad_out * (self > threshold)

@register_func(aten.t.default, decompositions)
def t(x):
    if x.ndim <= 1:
        return x
    elif x.ndim == 2:
        return x.permute((1, 0))
    else:
        assert False

# 3) Subclass Wrapper for Mock Backend

class NPTensorWrapper(torch.Tensor):
    # Wrap the tensor-like interface in an actual tensor so that we can use autograd

    @staticmethod
    def __new__(cls, pointer, requires_grad=False):
        # Pretend to be a contiguous tensor on cpu
        assert isinstance(pointer, NPTensor)
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            pointer.arr.shape,
            strides=torch.zeros(pointer.arr.shape).stride(),
            storage_offset=0,
            # TODO: clone storage aliasing
            dtype=numpy_to_torch_dtype_dict[pointer.arr.dtype.type],
            layout=torch.strided,
            device="cpu",
            requires_grad=requires_grad
        )
        r.pointer = pointer

        return r

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def wrap(t):
            if isinstance(t, NPTensor):
                return NPTensorWrapper(t)
            else:
                return t

        def unwrap(t):
            if isinstance(t, NPTensorWrapper):
                return t.pointer
            elif isinstance(t, torch.Tensor):
                if t.requires_grad:
                    t = t.detach()
                ret = t.numpy()
                return ret
            else:
                return t

        if func in np_tensor_backend_impl:
            return tree_map(wrap, np_tensor_backend_impl[func](*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))
        elif func in decompositions:
            return decompositions[func](*args, **kwargs)
        elif func in decomposition_table:
            return decomposition_table[func](*args, **kwargs)
        else:
            raise NotImplementedError(f"Backend has not implemented {func.__name__}")

    def __repr__(self):
        return f"NPTensorWrapper(pointer={repr(self.pointer)}, requires_grad={self.requires_grad})"

    def publish(self):
        return numpy_to_torch(self.pointer.publish())


# 4) Testing

def get_op_list(tested_op_list: List[Tuple]):
    return [opinfo for opinfo in op_db if (opinfo.name, opinfo.variant_test_name) in tested_op_list]

# Some testing utils from funtorch repo (test/common_utils.py)

def skipOps(test_case_name, base_test_name, to_skip):
    all_opinfos = op_db
    for xfail in to_skip:
        op_name, variant_name, device_type, dtypes, expected_failure = xfail
        matching_opinfos = [o for o in all_opinfos
                            if o.name == op_name and o.variant_test_name == variant_name]
        assert len(matching_opinfos) >= 1, f"Couldn't find OpInfo for {xfail}"
        for opinfo in matching_opinfos:
            decorators = list(opinfo.decorators)
            if expected_failure:
                decorator = DecorateInfo(unittest.expectedFailure,
                                         test_case_name, base_test_name,
                                         device_type=device_type, dtypes=dtypes)
                decorators.append(decorator)
            else:
                decorator = DecorateInfo(unittest.skip("Skipped!"),
                                         test_case_name, base_test_name,
                                         device_type=device_type, dtypes=dtypes)
                decorators.append(decorator)
            opinfo.decorators = tuple(decorators)

    # This decorator doesn't modify fn in any way
    def wrapped(fn):
        return fn
    return wrapped

def xfail(op_name, variant_name='', *, device_type=None, dtypes=None):
    return (op_name, variant_name, device_type, dtypes, True)

class NPTensorTest(TestCase):

    def nptensorwrapper_to_torch(self, t):
        return numpy_to_torch(t.pointer.arr) if isinstance(t, NPTensorWrapper) else t

    def torch_to_nptensorwrapper(self, t):
        if isinstance(t, torch.Tensor):
            return NPTensorWrapper(NPTensor(t.detach().numpy())).requires_grad_(t.requires_grad)
        else:
            return t

   # 4.1) XOR Training Example

    def test_train_XOR(self):
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

        # Simple MLP

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.linear1 = nn.Linear(2, 10)
                self.linear2 = nn.Linear(10, 10)
                self.linear3 = nn.Linear(10, 1)
                self.relu = nn.ReLU()
                self.gelu = nn.GELU()

            def forward(self, x):
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                x = self.relu(x)
                x = self.linear3(x)
                return x.squeeze(1).sigmoid()

        # Train

        model =  Net()
        sgd = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Register hook that publishes gradients (unwraps)
        for p in model.parameters():
            p.register_hook(lambda x: x.publish())

        losses = []
        for _ in range(N_ITER):
            out = model(train_X)
            loss = F.l1_loss(out, train_y)
            loss.backward()
            sgd.step()
            sgd.zero_grad()
            losses.append(loss.detach())

        # Test

        test_X = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        test_y = torch.tensor([[0., 1., 1., 0.]])
        predicted = model(test_X)

        # TODO: Check if parameters have been updated
        assert torch.allclose(predicted, test_y, atol=1e-2)


    # 4.2) More Testing

    @ops(op_list=get_op_list([
        # Backend ops
        ('sum',''),
        ('add',''),
        ('sub',''),
        ('rsub',''),
        ('div',''),
        ('mul',''),

        # Decompositions
        ('sigmoid',''),
        ('nn.functional.mse_loss',''),
        ('nn.functional.l1_loss',''),
    ]), allowed_dtypes=(torch.float,))
    @skipOps('NPTensorTest', 'test_np_tensor_parity', {
        # Need to accept `alpha` param
        xfail('add'),
        xfail('sub'),

        # We didn't implement a specific overload
        xfail('rsub',''),  # need: rsub.Tensor
    })
    def test_np_tensor_parity(self, device, dtype, op):
        # Tests if tensor and subclass tensor compute the same values
        assert device == 'cpu' and  dtype == torch.float

        samples = op.sample_inputs(device, dtype, requires_grad=False)

        for sample in samples:
            arg_values = [sample.input] + list(sample.args)
            kwarg_values = sample.kwargs

            subclass_arg_values = tree_map(self.torch_to_nptensorwrapper, arg_values)
            subclass_kwarg_values = tree_map(self.torch_to_nptensorwrapper, kwarg_values)

            fn = op.get_op()
            result = self.nptensorwrapper_to_torch(fn(*subclass_arg_values, **subclass_kwarg_values))
            expected = fn(*arg_values, **kwarg_values)
            # print(result, expected, (result - expected).abs().max())
            # print(arg_values)
            self.assertTrue(torch.allclose(result, expected))

    @ops(op_list=get_op_list([
        # Backend ops
        ('sum',''),
        ('add',''),
        ('sub',''),
        ('div',''),
        ('mul',''),

        # Decompositions
        ('sigmoid',''),
        ('nn.functional.mse_loss',''),
    ]), allowed_dtypes=(torch.float,))
    @skipOps('NPTensorTest', 'test_np_tensor_gradients_parity', {
        # Forward is already failing for these
        xfail('add'),
        xfail('sub'),
        xfail('rsub',''),
    })
    def test_np_tensor_gradients_parity(self, device, dtype, op):
        # Tests if tensor and subclass tensor compute the same gradient
        assert device == 'cpu' and dtype == torch.float

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        for sample in samples:
            arg_values = [sample.input] + list(sample.args)
            kwarg_values = sample.kwargs

            subclass_arg_values = tree_map(self.torch_to_nptensorwrapper, arg_values)
            subclass_kwarg_values = tree_map(self.torch_to_nptensorwrapper, kwarg_values)

            fn = op.get_op()

            subclass_out = fn(*subclass_arg_values, **subclass_kwarg_values)
            out = fn(*arg_values, **kwarg_values)

            assert isinstance(subclass_out, torch.Tensor), f"Expected op to return a single tensor, but got: {type(subclass_out)}"

            grad_outputs = torch.rand_like(out)
            subclass_grad_inputs = torch.autograd.grad(subclass_out, subclass_arg_values, grad_outputs=grad_outputs)
            subclass_grad_inputs = tree_map(self.nptensorwrapper_to_torch, subclass_grad_inputs)

            grad_inputs = torch.autograd.grad(out, arg_values, grad_outputs=grad_outputs)

            self.assertEqual(len(subclass_grad_inputs), len(grad_inputs))

            for subclass_grad_input, grad_input in zip(subclass_grad_inputs, grad_inputs):
                # print(subclass_grad_input, grad_input, (subclass_grad_input - grad_input).abs().max())
                # print(arg_values)
                self.assertTrue(torch.allclose(grad_input, subclass_grad_input))

instantiate_device_type_tests(NPTensorTest, globals(), only_for=("cpu,"))

if __name__ == "__main__":
    run_tests()
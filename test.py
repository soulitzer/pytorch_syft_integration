from typing import List, Tuple, Any
import unittest
from uuid import uuid1
import time

import numpy as np

import torch 
from torch.testing._internal.common_utils import TestCase, run_tests, numpy_to_torch_dtype_dict
from torch.testing._internal.common_device_type import ops, instantiate_device_type_tests
from torch.testing._internal.common_methods_invocations import op_db, DecorateInfo
from torch.utils._pytree import tree_map

import syft as sy
from syft.core.tensor.autodp.phi_tensor import TensorWrappedPhiTensorPointer, TensorWrappedGammaTensorPointer
sy.logger.remove()

aten = torch.ops.aten

# Mapping between aten ops and backend ops
# This is where all the code would go to adapt the semantics of the 

np_tensor_backend_impl = dict()

def register_func(name, ns):
    # Helpful decorator to colocate registration and definition
    def register_func_inner(f):
        assert name not in ns, f"{name} has already been registered"
        ns[name] = f
        return f
    return register_func_inner

# def fix_binary_type_promotion(f):
#     # numpy and torch do binary type promotion differently
#     expected_dtypes = (
#         (np.int64, np.float32, np.float32),  # numpy promotes to np.float64
#     )

#     def check_dtype(x, y, lhs, rhs):
#         # HACK: if lhs/rhs are int, make it a numpy array, so that we can check dtype
#         # this might not be the right way to do it
#         x = np.array(x) if isinstance(x, int) or isinstance(x, float) else maybe_unwrap(x)
#         y = np.array(y) if isinstance(y, int) or isinstance(y, float) else maybe_unwrap(y)
#         return x.dtype.type == lhs and y.dtype.type == rhs

#     def wrapper(x, y):
#         ret = f(x, y)
#         for lhs, rhs, expected in expected_dtypes:
#             if check_dtype(x, y, lhs, rhs) or check_dtype(y, x, lhs, rhs):
#                 ret = NPTensor(ret.arr.astype(dtype=expected))
#         return ret

#     return wrapper

# torch_to_numpy_dtype_dict = {v: k for k, v in numpy_to_torch_dtype_dict.items()}

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
def np_tensor_add_impl(x, y):  # @fix_binary_type_promotion
    return x + y

@register_func(aten.sub.Tensor, np_tensor_backend_impl)
def np_tensor_sub_impl(x, y):  # @fix_binary_type_promotion
    return x - y

@register_func(aten.rsub.Scalar, np_tensor_backend_impl)
def np_tensor_rsub_impl(x, y):  # @fix_binary_type_promotion
    return y - x

@register_func(aten.mul.Tensor, np_tensor_backend_impl)
def np_tensor_mul_impl(x, y):  # @fix_binary_type_promotion
    return x * y

@register_func(aten.div.Tensor, np_tensor_backend_impl)
def np_tensor_div_impl(x, y):  # @fix_binary_type_promotion
    return x / y

# def np_tensor_pow_impl(x, a):
#     ret = maybe_unwrap(x) ** maybe_unwrap(a)
#     if x.arr.dtype.type == np.float32:
#         # Correct type promotion to match torch
#         ret = ret.astype(dtype=np.float32)
#     return NPTensor(ret)

@register_func(aten.sum.default, np_tensor_backend_impl)
def np_tensor_sum_default_impl(x, keepdims=False):
    print('np_tensor_sum_default_impl')
    assert not keepdims
    return x.sum()

# @register_func(aten.sum.dim_IntList, np_tensor_backend_impl)
# def np_tensor_sum_dim_IntList_impl(x, dims, keepdims=False):
#     if x.arr.ndim == 0:
#         ret = x.arr
#     else:
#         ret = np.sum(x.arr, axis=tuple(dims), keepdims=keepdims)
#     return NPTensor(ret)

@register_func(aten.gt.Scalar, np_tensor_backend_impl)
def np_tensor_gt_impl(x, y):
    return x > y

@register_func(aten.reciprocal.default, np_tensor_backend_impl)
def np_tensor_reciprocal_impl(x):
    return x.reciprocal()

@register_func(aten.exp.default, np_tensor_backend_impl)
def np_tensor_exp_impl(x):
    return x.exp()

@register_func(aten.mm.default, np_tensor_backend_impl)
def np_tensor_mm_impl(x, y):
    return x @ y

# @register_func(aten.softmax.Tenso, np_tensor_backend_impl)
# def np_tensor_softmax_impl(x):
#     return x.softmax()

print(np_tensor_backend_impl[aten.add.Tensor])
# View Ops

# @register_func(aten.view.default, np_tensor_backend_impl)
# def np_tensor_view_impl(x, shape):
#     ret = x.arr.view()
#     ret.shape = shape
#     return NPTensor(ret)

# @register_func(aten.permute.default, np_tensor_backend_impl)
# def np_tensor_permute_impl(x, perm):
#     return NPTensor(np.transpose(x.arr, axes=perm))

# @register_func(aten.pow.Tensor_Scalar, np_tensor_backend_impl)

# @register_func(aten.squeeze.dim, np_tensor_backend_impl)
# def np_tensor_squeeze_impl(x, dim):
#     return NPTensor(x.arr.squeeze(dim))

# @register_func(aten.unsqueeze.default, np_tensor_backend_impl)
# def np_tensor_unsqueeze_impl(x, dim):
#     return NPTensor(np.expand_dims(x.arr, dim))

# @register_func(aten.detach.default, np_tensor_backend_impl)
# def np_tensor_detach_impl(x):
#     # To the backend, detach is simply a view
#     return NPTensor(x.arr.view())

# Factory functions

# @register_func(aten.ones_like.default, np_tensor_backend_impl)
# def np_tensor_ones_like_impl(x, dtype, layout, device, pin_memory, memory_format):
#     # TODO: do we care about the other arguments
#     return NPTensor(np.ones_like(x.arr, dtype=torch_to_numpy_dtype_dict_int(dtype)))

# Casting

# @register_func(aten.to.dtype, np_tensor_backend_impl)
# def np_tensor_to_dtype_impl(x, dtype):
#     # To the backend, detach is simply a view
#     ret = x.arr.astype(dtype=torch_to_numpy_dtype_dict_int(dtype))
#     return NPTensor(ret)

# (Optional) For l1 loss (instead of mse loss)

# @register_func(aten.abs.default, np_tensor_backend_impl)
# def np_tensor_abs_default_impl(x):
#     return NPTensor(np.abs(x.arr))

# @register_func(aten.mean.default, np_tensor_backend_impl)
# def np_tensor_mean_default_impl(x):
#     return NPTensor(np.mean(x.arr))

# @register_func(aten.sign.default, np_tensor_backend_impl)
# def np_tensor_mean_default_impl(x):
#     return NPTensor(np.sign(x.arr))

def contig_strides_from_shape(shape):
    return torch.zeros(shape).stride()

class DPTensor(torch.Tensor):
    # This class wraps a TensorPointer so autograd can be used
    #
    # TODO:
    # - Think about the gamma case
    # - nit: maybe use slots here to be more efficient
    @staticmethod
    def __new__(cls, pointer, requires_grad=False):
        assert isinstance(pointer, TensorWrappedPhiTensorPointer) or \
            isinstance(pointer, TensorWrappedGammaTensorPointer)
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            pointer.public_shape,
            strides=contig_strides_from_shape(pointer.public_shape),
            storage_offset=0,  # elem.storage_offset(),
            # TODO: clone storage aliasing
            # FIXME: why don't we just return a np type instead of a string?
            dtype=numpy_to_torch_dtype_dict[getattr(np, pointer.public_dtype)],
            layout=torch.strided,
            device="cpu",  # elem.device
            requires_grad=requires_grad
        )
        r.pointer = pointer

        return r
        
    __torch_function__ = torch._C._disabled_torch_function_impl
    
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def wrap(t):
            if isinstance(t, TensorWrappedPhiTensorPointer):
                return DPTensor(t)
            elif isinstance(t, TensorWrappedGammaTensorPointer):
                return DPTensor(t)
            else:
                assert False, f"__torch_dispatch__ wrap got unexpected type: {type(t)}" 
                # return t
        def unwrap(t):
            if isinstance(t, DPTensor):
                return t.pointer
            elif isinstance(t, torch.Tensor):
                # Sometimes we get tensors (either because autograd creates tensors internally)
                #.or scalars get wrapped into tensors - wrapped number
                return t.numpy()
            else:
                assert False

        return tree_map(wrap, np_tensor_backend_impl[func](*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))

    def __repr__(self):
        # TODO: maybe we should include autograd information?
        return repr(self.pointer)

    def synthetic(self):
        return self.pointer.synthetic
    
    def publish(self, sigma=None):
        return self.pointer.publish(sigma=sigma)
    
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

_syf_owner = None
_syf_user = None

def syf_login():
    global _syf_owner
    global _syf_user

    if _syf_owner is not None:
        return _syf_owner, _syf_user

    _syf_owner = sy.login(email="info@openmined.org", password="changethis", port=8081)
    email = str(uuid1())
    password = "pw"
    if email not in [entry['email'] for entry in _syf_owner.users.all()]:
        _syf_owner.users.create(
            **{
                "name": "Sheldon2 Cooper",
                "email": email,
                "password": password,
                "budget": 1e9
            }
        )
    _syf_user = sy.login(email=email, password=password, port=8081)
    return _syf_owner, _syf_user

def _allclose_with_type_promotion(a, b, rtol, atol):
    # FIXME: float32 is not supported by Syft
    promoted_type = torch.promote_types(a.dtype, b.dtype)
    a = a.to(dtype=promoted_type)
    b = b.to(dtype=promoted_type)
    return torch.allclose(a, b, rtol, atol)

class NPTensorTest(TestCase):
    def setUp(self):
        owner, user = syf_login()
        assert user.privacy_budget > 1e7

    def nptensorwrapper_to_torch(self, t, sigma=1e-4):
        # There will be noise!
        owner, user = syf_login()
        before = user.privacy_budget
        np_arr = t.publish(sigma=sigma).block_with_timeout(10).get()
        after = user.privacy_budget
        assert before - after > 1
        assert isinstance(np_arr, np.ndarray)
        return numpy_to_torch(np_arr)

    def numpy_to_tensor_pointer(self, np_arr):
        owner, user = syf_login()
        backend_phi_tensor = sy.Tensor(np_arr)
        assert np_arr.ndim != 0
        data_subjects = ["abc"] * np_arr.shape[0] if np_arr.ndim != 0 else ["abc"]
        single_data_subject = backend_phi_tensor.private(
            min_val=10,
            max_val=90,
            data_subjects=data_subjects)
        data_asset_name = str(uuid1())
        owner.load_dataset(
            assets={
                data_asset_name: single_data_subject,
            },
            name="my_data",
            description="description"
        )
        tensor_pointer = user.datasets[-1][data_asset_name]
        assert isinstance(tensor_pointer, TensorWrappedPhiTensorPointer)
        return tensor_pointer

    def wrap_samples_with_subclass(self, samples):
        # Given OpInfo sample, reproduce sample, but with arguments wrapped
        # Have all the sample inputs in a single data set, where each
        # asset corresponds to a single sample
        # Returns tuple of just 
        def handle_arg(t):
            if isinstance(t, torch.Tensor):
                if t.dtype == torch.float32:
                    # FIXME: float32 is not supported by Syft
                    t = t.to(torch.float64)
                np_arr = t.detach().numpy()
                assert np_arr.ndim != 0

                backend_phi_tensor = sy.Tensor(np_arr)
                data_subjects = ["abc"] * np_arr.shape[0] if np_arr.ndim != 0 else ["abc"]
                single_data_subject = backend_phi_tensor.private(
                    min_val=10,
                    max_val=90,
                    data_subjects=data_subjects)
                return (str(uuid1()), single_data_subject)
            else:
                return (None, t)

        owner, user = syf_login()
        assets = {}
        all_wrapped_args = []
        all_wrapped_kwargs = []

        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs= sample.kwargs
            wrapped_args = [handle_arg(arg) for arg in args]
            wrapped_kwargs = {k: handle_arg(v) for k,v in kwargs.items()}
            for k, v in wrapped_args + list(wrapped_kwargs.values()):
                if k is not None:
                    assets[k] = v
            all_wrapped_args.append(wrapped_args)
            all_wrapped_kwargs.append(wrapped_kwargs)

        owner.load_dataset(
            assets=assets,
            name="my_data",
            description="description"
        )
        # Shouldn't need a sleep here
        user_assets = user.datasets[-1]

        out_sample_arg_kwargs = []
        for wrapped_args, wrapped_kwargs in zip(all_wrapped_args, all_wrapped_kwargs):
            single_out_args = [v if k is None else DPTensor(user_assets[k]) for k, v in wrapped_args]
            single_out_kwargs = {k: (v_v if v_k is None else DPTensor(user_assets[v_k])) for k, (v_v, v_k) in wrapped_kwargs.items()}
            out_sample_arg_kwargs.append((single_out_args, single_out_kwargs))
    
        return out_sample_arg_kwargs

    def torch_to_nptensorwrapper(self, t, single_data_subject=True):
        if not single_data_subject:
            raise NotImplementedError

        if isinstance(t, torch.Tensor):
            if t.dtype == torch.float32:
                # FIXME: float32 is not supported by Syft
                t = t.to(torch.float64)
            np_arr = t.detach().numpy()
            tensor_pointer = self.numpy_to_tensor_pointer(np_arr)
            return DPTensor(tensor_pointer).requires_grad_(t.requires_grad)
        else:
            return t

    # def test_conversion(self):
    #     # NB: cannot be scalar, dtype must be float64, int64, or bool
    #     t = torch.tensor([100.], dtype=torch.float64)
    #     wrapped = self.torch_to_nptensorwrapper(t)
    #     t_back = self.nptensorwrapper_to_torch(wrapped)
    #     self.assertTrue(torch.allclose(t, t_back, atol=10))

    # def test_dtypes(self):
    #     # Only float64 (and int64 and bool) are allowed
    #     pass

    # def test_broadcasting(self):
    #     pass

   # 4.1) XOR Training Example

    # def test_train_XOR(self):
    #     import torch.nn as nn
    #     import torch.nn.functional as F

    #     SIGMA = 0.4
    #     N_PER_QUAD = 100
    #     N_ITER = 100

    #     train_X_centers = [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
    #     train_y = [0, 1, 1, 0]

    #     shuffle_idx = np.arange(N_PER_QUAD * 4)
    #     np.random.shuffle(shuffle_idx)
    #     train_X_list = []
    #     train_y_list = []

    #     for center, label in zip(train_X_centers, train_y):
    #         train_X_list.append(np.random.randn(N_PER_QUAD, 2) * SIGMA + np.array(center))
    #         train_y_list.append(np.zeros(N_PER_QUAD) if label == 0 else np.ones(N_PER_QUAD))

    #     # Linear layer does not support double
    #     train_X_np = np.concatenate(train_X_list)[shuffle_idx].astype(np.float32)
    #     train_y_np = np.concatenate(train_y_list)[shuffle_idx].astype(np.float32)

    #     # Convert input data to torch.Tensor

    #     train_X = NPTensorWrapper(NPTensor(train_X_np))
    #     train_y = NPTensorWrapper(NPTensor(train_y_np))

    #     # Simple MLP

    #     class Net(nn.Module):
    #         def __init__(self):
    #             super(Net, self).__init__()
    #             self.linear1 = nn.Linear(2, 10)
    #             self.linear2 = nn.Linear(10, 10)
    #             self.linear3 = nn.Linear(10, 1)
    #             self.relu = nn.ReLU()
    #             self.gelu = nn.GELU()

    #         def forward(self, x):
    #             x = self.linear1(x)
    #             x = self.relu(x)
    #             x = self.linear2(x)
    #             x = self.relu(x)
    #             x = self.linear3(x)
    #             return x.squeeze(1).sigmoid()

    #     # Train

    #     model =  Net()
    #     sgd = torch.optim.Adam(model.parameters(), lr=1e-2)

    #     # Register hook that publishes gradients (unwraps)
    #     for p in model.parameters():
    #         p.register_hook(lambda x: x.publish())

    #     losses = []
    #     for _ in range(N_ITER):
    #         out = model(train_X)
    #         loss = F.l1_loss(out, train_y)
    #         loss.backward()
    #         sgd.step()
    #         sgd.zero_grad()
    #         losses.append(loss.detach())

    #     # Test

    #     test_X = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    #     test_y = torch.tensor([[0., 1., 1., 0.]])
    #     predicted = model(test_X)

    #     # TODO: Check if parameters have been updated
    #     assert torch.allclose(predicted, test_y, atol=1e-2)


    # 4.2) Op Testing

    @ops(op_list=get_op_list([
        # Backend ops
        # ('sum',''),  # int dim list?
        # ('add',''),
        # ('sub',''),
        # ('rsub',''),
        # ('div',''),
        ('mul',''),
        # Decompositions
        # ('softmax',''),
        # ('nn.functional.mse_loss',''),
        # ('nn.functional.l1_loss',''),
    ]), allowed_dtypes=(torch.float,))
    @skipOps('NPTensorTest', 'test_np_tensor_parity', {
        # Need to accept `alpha` param
        xfail('add'),
        xfail('sub'),
        # We didn't implement a specific overload
        xfail('rsub',''),  # need: rsub.Tensor
    })
    def test_np_tensor_parity(self, device, dtype, op):
        # TODO: faster way to run tests, load all datasets at the same time?
        # Tests if tensor and subclass tensor compute the same values
        assert device == 'cpu' and  dtype == torch.float

        samples = list(op.sample_inputs(device, dtype, requires_grad=False))

        def supported_by_syft(sample):
            element_wise_ops = "add sub rsub div mul".split(" ")
            arg_values = [sample.input] + list(sample.args)
            kwarg_values = sample.kwargs
            if any(t.numel() == 0 for t in arg_values if isinstance(t, torch.Tensor)) or \
                any(t.numel() == 0 for t in kwarg_values.values() if isinstance(t, torch.Tensor)):
                # FIXME: zero-numel tensor errors (data subject issue?)
                return False
            if any(t.ndim == 0 for t in arg_values if isinstance(t, torch.Tensor)) or \
                any(t.ndim == 0 for t in kwarg_values.values() if isinstance(t, torch.Tensor)):
                # FIXME: Scalar errors (data subject issue?)
                return False
            if op.name in element_wise_ops and arg_values[0].shape[0] != arg_values[1].shape[0]:
                # FIXME: element-wise op when data_subjects_indexed arrays are differently shaped
                return False
            return True

        samples = list(filter(supported_by_syft, samples))
        wrapped_samples = self.wrap_samples_with_subclass(samples)
        for wrapped_sample, sample in zip(wrapped_samples, samples):
            print("Sample: ", sample)
            print("Wrapped sample: ", wrapped_sample)
            # Wrapped
            wrapped_args, wrapped_kwargs = wrapped_sample
            fn = op.get_op()
            raw_output = fn(*wrapped_args, **wrapped_kwargs)
            result = self.nptensorwrapper_to_torch(raw_output)

            # Non-Wrapped
            arg_values = [sample.input] + list(sample.args)
            kwarg_values = sample.kwargs
            expected = fn(*arg_values, **kwarg_values)
            # print(result, expected, (result - expected).abs().max())
            # print(arg_values)

            self.assertTrue(_allclose_with_type_promotion(result, expected, rtol=1e-3, atol=1e-1))

    # @ops(op_list=get_op_list([
    #     # Backend ops
    #     ('sum',''),
    #     ('add',''),
    #     ('sub',''),
    #     ('div',''),
    #     ('mul',''),
    #     # Decompositions
    #     ('sigmoid',''),
    #     ('nn.functional.mse_loss',''),
    # ]), allowed_dtypes=(torch.float,))
    # @skipOps('NPTensorTest', 'test_np_tensor_gradients_parity', {
    #     # Forward is already failing for these
    #     xfail('add'),
    #     xfail('sub'),
    #     xfail('rsub',''),
    # })
    # def test_np_tensor_gradients_parity(self, device, dtype, op):
    #     # Tests if tensor and subclass tensor compute the same gradient
    #     assert device == 'cpu' and dtype == torch.float

    #     samples = op.sample_inputs(device, dtype, requires_grad=True)

    #     for sample in samples:
    #         arg_values = [sample.input] + list(sample.args)
    #         kwarg_values = sample.kwargs

    #         subclass_arg_values = tree_map(self.torch_to_nptensorwrapper, arg_values)
    #         subclass_kwarg_values = tree_map(self.torch_to_nptensorwrapper, kwarg_values)

    #         fn = op.get_op()

    #         subclass_out = fn(*subclass_arg_values, **subclass_kwarg_values)
    #         out = fn(*arg_values, **kwarg_values)

    #         assert isinstance(subclass_out, torch.Tensor), f"Expected op to return a single tensor, but got: {type(subclass_out)}"

    #         grad_outputs = torch.rand_like(out)
    #         subclass_grad_inputs = torch.autograd.grad(subclass_out, subclass_arg_values, grad_outputs=grad_outputs)
    #         subclass_grad_inputs = tree_map(self.nptensorwrapper_to_torch, subclass_grad_inputs)

    #         grad_inputs = torch.autograd.grad(out, arg_values, grad_outputs=grad_outputs)

    #         self.assertEqual(len(subclass_grad_inputs), len(grad_inputs))

    #         for subclass_grad_input, grad_input in zip(subclass_grad_inputs, grad_inputs):
    #             # print(subclass_grad_input, grad_input, (subclass_grad_input - grad_input).abs().max())
    #             # print(arg_values)
    #             self.assertTrue(torch.allclose(grad_input, subclass_grad_input))

instantiate_device_type_tests(NPTensorTest, globals(), only_for=("cpu,"))

if __name__ == "__main__":
    run_tests()

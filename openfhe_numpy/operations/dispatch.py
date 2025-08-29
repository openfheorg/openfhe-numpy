# ==================================================================================
#  BSD 2-Clause License
#
#  Copyright (c) 2014-2025, NJIT, Duality Technologies Inc. and other contributors
#
#  All rights reserved.
#
#  Author TPOC: contact@openfhe.org
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this
#     list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==================================================================================

from typing import Tuple, Callable, Any
import functools
import os


# Operation registry - stores implementations keyed by (operation_name, signature)
TENSOR_FUNCTIONS = {}

# Registry for explicitly commutative operations
COMMUTATIVE_OPS = set()
DEBUG = False


def mark_commutative(*names):
    """Mark one or more operations as commutative."""
    COMMUTATIVE_OPS.update(names)


def register_tensor_function(name, type_signatures):
    """Register operation implementation for specific types.

    Supports both single and multiple type signatures.
    Also attaches function metadata like docstring and return type.
    """
    import inspect
    from typing import get_type_hints

    def decorator(func):
        func._tensor_function_name = name
        func._tensor_function_docstring = inspect.getdoc(func)
        func._tensor_function_type_hints = get_type_hints(func)
        if isinstance(type_signatures, tuple):
            signatures = [type_signatures]
        else:
            signatures = type_signatures

        for signature in signatures:
            TENSOR_FUNCTIONS[(name, signature)] = func
        return func

    return decorator


def dispatch_tensor_function(
    func_name: str, args: Tuple[Any, ...], kwargs=None, return_hint=False, verbose=DEBUG
) -> Any:
    """Dispatch implementation for the given function name and argument types.

    If return_hint is True, return a string describing what failed instead of raising.
    """
    kwargs = kwargs or {}

    # Step 1: Extract type names
    sig = tuple(getattr(arg, "dtype", type(arg).__name__) for arg in args)

    if verbose:
        print(f"DEBUG: dispatch_tensor_function called with func_name='{func_name}', sig={sig}")
        print(
            f"DEBUG: Available registrations for '{func_name}': {[k[1] for k in TENSOR_FUNCTIONS if k[0] == func_name]}"
        )
    if verbose:
        if os.getenv("DISPATCH_DEBUG", "0") == "1":
            print(f"[DISPATCH] Trying {func_name}{sig}")

    # Step 2: Try exact match
    key = (func_name, sig)
    if verbose:
        print(f"DEBUG: Trying exact match with key: {key}")
    if key in TENSOR_FUNCTIONS:
        if verbose:
            print(f"DEBUG: Found exact match! Calling {TENSOR_FUNCTIONS[key]}")
        return TENSOR_FUNCTIONS[key](*args, **kwargs)

    # Step 3: Normalize scalars to 'scalar'
    normalized_sig = tuple("scalar" if t in {"int", "float", "complex", "bool"} else t for t in sig)
    key = (func_name, normalized_sig)
    if key in TENSOR_FUNCTIONS:
        return TENSOR_FUNCTIONS[key](*args, **kwargs)

    # Step 4: Try commutative fallback
    if func_name in COMMUTATIVE_OPS and len(args) == 2:
        a_pri = getattr(args[0], "tensor_priority", 0)
        b_pri = getattr(args[1], "tensor_priority", 0)
        if b_pri > a_pri:
            swapped_sig = (normalized_sig[1], normalized_sig[0])
            key = (func_name, swapped_sig)
            if key in TENSOR_FUNCTIONS:
                return TENSOR_FUNCTIONS[key](args[1], args[0], **kwargs)

    # Step 5: Try MRO fallback on the first argument
    if hasattr(args[0], "__class__"):
        for parent in args[0].__class__.__mro__[1:]:
            if parent.__name__ == "object":
                continue
            parent_sig = (parent.__name__,) + normalized_sig[1:]
            key = (func_name, parent_sig)
            if key in TENSOR_FUNCTIONS:
                return TENSOR_FUNCTIONS[key](*args, **kwargs)

    if return_hint:
        return (
            f"No implementation found for {func_name} with signature {sig}.\n"
            f"Registered for {func_name}: {[k[1] for k in TENSOR_FUNCTIONS if k[0] == func_name]}"
        )

    # import pdb;pdb.set_trace()
    raise NotImplementedError(
        f"{str(func_name)} not implemented for types {sig}.\n"
        f"Registered signatures for '{func_name}':\n  "
        + "\n  ".join(str(k[1]) for k in TENSOR_FUNCTIONS if k[0] == func_name)
    )


def tensor_function_api(op_name: str, binary: bool = True, verbose: bool = DEBUG) -> Callable:
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if verbose:
                print(
                    f"DEBUG: tensor_function_api called for operation '{op_name}' with args: {[type(arg).__name__ for arg in args]}"
                )
            if binary:
                if len(args) < 2:
                    raise TypeError(f"{op_name} requires 2 arguments")
                a, b = args[0], args[1]

                # Check if both have __tensor_function__
                a_tf = hasattr(a, "__tensor_function__")
                b_tf = hasattr(b, "__tensor_function__")
                if verbose:
                    print(
                        f"DEBUG: a has __tensor_function__: {a_tf}, b has __tensor_function__: {b_tf}"
                    )

                if a_tf and b_tf:
                    # Use tensor_priority to decide which implementation to use
                    a_pri = getattr(a, "tensor_priority", 0)
                    b_pri = getattr(b, "tensor_priority", 0)

                    if b_pri > a_pri:
                        return b.__tensor_function__(op_name, args, kwargs)
                    else:
                        return a.__tensor_function__(op_name, args, kwargs)

                # Original code continues...
                if a_tf:
                    return a.__tensor_function__(op_name, args, kwargs)
                if b_tf:
                    return b.__tensor_function__(op_name, args, kwargs)

                return dispatch_tensor_function(op_name, args, kwargs)

            else:
                if not args:
                    raise TypeError(f"{op_name} requires at least 1 argument")
                a = args[0]

                if hasattr(a, "__tensor_function__"):
                    return a.__tensor_function__(op_name, args, kwargs)

                return dispatch_tensor_function(op_name, args, kwargs)

        return wrapper

    return decorator


# === Mark commutative operations ===
mark_commutative("add", "multiply")

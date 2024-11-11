# Copyright 2024 Rebellions Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Portions of this software are licensed under the Apache License,
# Version 2.0. See the NOTICE file distributed with this work for
# additional information regarding copyright ownership.

# All other portions of this software, including proprietary code,
# are the intellectual property of Rebellions Inc. and may not be
# copied, modified, or distributed without prior written permission
# from Rebellions Inc.


from tvm import get_global_func


def npu_is_available(device: int = 0) -> bool:
    """
    Return a bool indicating whether the RBLN device is currently available.

    Args:
        device (int, optional): Index of the npu. Defaults to 0.

    Returns:
        A bool indicating whether the RBLN device is currently available
    """
    return bool(get_global_func("rebel.is_available")(device))


def get_npu_name(device: int = 0) -> str:
    """Return the name of RBLN npu.

    Args:
        device (int, optional): Index of the npu. Defaults to 0.

    Returns:
        Corresponding name of the npu. e.g. "RBLN-CA12"
    """
    return get_global_func("rebel.get_npu")(device)

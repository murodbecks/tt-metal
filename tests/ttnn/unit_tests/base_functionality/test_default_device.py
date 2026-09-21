# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The default must not retain a device or return a closed device hierarchy."""

import gc
import os

import pytest

import ttnn


@pytest.fixture(autouse=True)
def clear_default_device():
    ttnn.SetDefaultDevice(None)
    yield
    ttnn.SetDefaultDevice(None)


@pytest.mark.parametrize(
    "close",
    [ttnn.close_device, ttnn.close_mesh_device, ttnn.CloseDevice, lambda device: ttnn.CloseDevices({0: device})],
    ids=["close_device", "close_mesh_device", "CloseDevice", "CloseDevices"],
)
def test_close_device_forgets_matching_default_device(close):
    device = ttnn.open_device(device_id=0)
    try:
        ttnn.SetDefaultDevice(device)
        assert ttnn.GetDefaultDevice() is not None
    finally:
        close(device)
    assert ttnn.GetDefaultDevice() is None
    ttnn.SetDefaultDevice(device)
    assert ttnn.GetDefaultDevice() is None


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_close_mesh_device_forgets_default_set_to_submesh(depth):
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        submesh = mesh
        for _ in range(depth):
            submesh = submesh.create_submesh(ttnn.MeshShape(1, 1))
        ttnn.SetDefaultDevice(submesh)
        assert ttnn.GetDefaultDevice() is not None
    finally:
        ttnn.close_mesh_device(mesh)
    assert ttnn.GetDefaultDevice() is None
    ttnn.SetDefaultDevice(submesh)
    assert ttnn.GetDefaultDevice() is None


def test_closing_a_submesh_keeps_a_default_set_to_its_parent():
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        submesh = mesh.create_submesh(ttnn.MeshShape(1, 1))
        ttnn.SetDefaultDevice(mesh)
        ttnn.close_mesh_device(submesh)
        assert ttnn.GetDefaultDevice() is not None
    finally:
        ttnn.close_mesh_device(mesh)
    assert ttnn.GetDefaultDevice() is None


def test_destroying_device_forgets_default():
    device = ttnn.open_device(device_id=0)
    ttnn.SetDefaultDevice(device)
    del device
    gc.collect()
    assert ttnn.GetDefaultDevice() is None


def test_rejected_close_preserves_default(expect_error):
    if "TT_METAL_SLOW_DISPATCH_MODE" in os.environ:
        pytest.skip("Command-queue conflict validation requires fast dispatch")
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    submesh = mesh.create_submesh(ttnn.MeshShape(1, 1))
    try:
        # Drain each host completion before switching meshes: the completion queue
        # is shared, but both meshes retain their in-use flags for close validation.
        ttnn.event_synchronize(ttnn.record_event(mesh, 0))
        ttnn.event_synchronize(ttnn.record_event(submesh, 0))
        ttnn.SetDefaultDevice(mesh)
        with expect_error(RuntimeError, "in use by child submesh"):
            ttnn.close_mesh_device(mesh)
        assert ttnn.GetDefaultDevice() is mesh
    finally:
        mesh.quiesce_devices()
        ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh)
    assert ttnn.GetDefaultDevice() is None

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <thread>

#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>

#include "ttnn/device.hpp"

namespace {

using tt::tt_metal::MetalEnv;
using tt::tt_metal::MetalEnvDescriptor;
using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::distributed::MeshDeviceConfig;

class DefaultDeviceTest : public ::testing::Test {
protected:
    std::unique_ptr<MetalEnv> env_;
    std::shared_ptr<MeshDevice> mesh_;

    void SetUp() override {
        ttnn::SetDefaultDevice(nullptr);
        env_ = std::make_unique<MetalEnv>(
            MetalEnvDescriptor(tt::tt_metal::experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1)));
        mesh_ = env_->create_mesh_device(MeshDeviceConfig(env_->get_system_mesh().shape()));
    }

    void TearDown() override {
        ttnn::SetDefaultDevice(nullptr);
        mesh_.reset();
        env_.reset();
    }
};

TEST_F(DefaultDeviceTest, DirectCloseInvalidatesDefault) {
    ttnn::SetDefaultDevice(mesh_.get());
    mesh_->close();
    EXPECT_EQ(ttnn::GetDefaultDevice(), nullptr);
}

TEST_F(DefaultDeviceTest, DestructionInvalidatesDefaultWithoutKeepingDeviceAlive) {
    std::weak_ptr<MeshDevice> weak = mesh_;
    ttnn::SetDefaultDevice(mesh_.get());
    mesh_.reset();
    EXPECT_TRUE(weak.expired());
    EXPECT_EQ(ttnn::GetDefaultDevice(), nullptr);
}

TEST_F(DefaultDeviceTest, AcquiredDefaultKeepsDeviceAlive) {
    std::weak_ptr<MeshDevice> weak = mesh_;
    ttnn::SetDefaultDevice(mesh_.get());
    {
        auto acquired = ttnn::GetDefaultDevice();
        ASSERT_NE(acquired, nullptr);
        mesh_.reset();
        EXPECT_FALSE(weak.expired());
    }
    EXPECT_TRUE(weak.expired());
    EXPECT_EQ(ttnn::GetDefaultDevice(), nullptr);
}

TEST_F(DefaultDeviceTest, AncestorCloseInvalidatesGrandchildDefault) {
    auto child = mesh_->create_submesh(mesh_->shape());
    auto grandchild = child->create_submesh(child->shape());
    ttnn::SetDefaultDevice(grandchild.get());
    mesh_->close();
    EXPECT_EQ(ttnn::GetDefaultDevice(), nullptr);
    ttnn::SetDefaultDevice(grandchild.get());
    EXPECT_EQ(ttnn::GetDefaultDevice(), nullptr);
}

TEST_F(DefaultDeviceTest, ClosingChildPreservesParentDefault) {
    auto child = mesh_->create_submesh(mesh_->shape());
    ttnn::SetDefaultDevice(mesh_.get());
    child->close();
    EXPECT_NE(ttnn::GetDefaultDevice(), nullptr);
}

TEST_F(DefaultDeviceTest, ClosingChildPreservesSiblingDefault) {
    auto child = mesh_->create_submesh(mesh_->shape());
    auto sibling = mesh_->create_submesh(mesh_->shape());
    ttnn::SetDefaultDevice(sibling.get());
    child->close();
    EXPECT_NE(ttnn::GetDefaultDevice(), nullptr);
}

TEST_F(DefaultDeviceTest, ClosingIntermediateAncestorInvalidatesGrandchildDefault) {
    auto child = mesh_->create_submesh(mesh_->shape());
    auto grandchild = child->create_submesh(child->shape());
    ttnn::SetDefaultDevice(grandchild.get());
    child->close();
    EXPECT_EQ(ttnn::GetDefaultDevice(), nullptr);
}

TEST_F(DefaultDeviceTest, ClosedDeviceCannotBeReinstalled) {
    mesh_->close();
    ttnn::SetDefaultDevice(mesh_.get());
    EXPECT_EQ(ttnn::GetDefaultDevice(), nullptr);
}

TEST_F(DefaultDeviceTest, ConcurrentSetGetAndCloseCannotReinstallDefault) {
    std::atomic<bool> start{false};
    std::atomic<int> ready{0};
    std::atomic<bool> closed{false};
    std::atomic<bool> observed_closed_device{false};
    auto* mesh = mesh_.get();
    std::thread setter([&] {
        while (!start.load()) {
            std::this_thread::yield();
        }
        ttnn::SetDefaultDevice(mesh);
        ready.fetch_add(1);
        while (!closed.load()) {
            ttnn::SetDefaultDevice(mesh);
        }
        for (int i = 0; i < 1000; ++i) {
            ttnn::SetDefaultDevice(mesh);
        }
    });
    std::thread getter([&] {
        while (!start.load()) {
            std::this_thread::yield();
        }
        (void)ttnn::GetDefaultDevice();
        ready.fetch_add(1);
        while (!closed.load()) {
            (void)ttnn::GetDefaultDevice();
        }
        for (int i = 0; i < 1000; ++i) {
            if (ttnn::GetDefaultDevice() != nullptr) {
                observed_closed_device.store(true);
            }
        }
    });
    start.store(true);
    while (ready.load() != 2) {
        std::this_thread::yield();
    }
    EXPECT_NO_THROW(mesh_->close());
    closed.store(true);
    setter.join();
    getter.join();
    EXPECT_FALSE(observed_closed_device.load());
    EXPECT_EQ(ttnn::GetDefaultDevice(), nullptr);
}

}  // namespace

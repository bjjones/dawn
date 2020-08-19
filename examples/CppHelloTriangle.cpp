// Copyright 2020 The Dawn Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "GLFW/glfw3.h"
#include "SampleUtils.h"
#include "Windows.h"
#include "utils/ComboRenderPipelineDescriptor.h"
#include "utils/SystemUtils.h"
#include "utils/Timer.h"
#include "utils/WGPUHelpers.h"

uint32_t frameNumber = 0;

wgpu::Device device;
int drawStart = 0;
wgpu::Buffer indexBuffer;
wgpu::Buffer indexBuffer2;
wgpu::Buffer vertexBuffer;
wgpu::Buffer vertexOffsetBuffer;
wgpu::Buffer offsetBuffer;

std::vector<double> frameTimeArray;
std::vector<wgpu::Texture> textures;
std::vector<wgpu::Sampler> samplers;
std::vector<wgpu::BindGroup> bindGroups;

wgpu::Queue queue;
wgpu::SwapChain swapchain;
wgpu::TextureView depthStencilView;
wgpu::RenderPipeline pipeline;

utils::Timer* mTimer;

wgpu::Texture mStagingTexture;
wgpu::TextureCopyView mStagingTextureCopyView;

// This is a simple workload that is meant to stress Dawn's D3D12 residency management system. The
// workload first creates a pool of textures that saturates physical device memory, then renders
// from a portion of them. In the next frame, the same number of textures are used in rendering,
// however the portion of textures is shifted such that some of the textures used in the beginning
// of the last frame are no longer used, and some new textures will be used at the end of the next
// frame. Because Dawn's residency manager is implemented as an LRU, this forces paging because the
// textures in the next frame should not be in device memory. What this workload tests, is that with
// a changing memory landscape - the most recently used resources remain in local memory.

/*
Scaled down example:

Below # represents the total pool of textures:
##########

The $ represents the total textures that can fit in local memory:
###$$$####

The ! represent the textures rendered from in frame N:
###$!!####

The ! represents the textures rendered from, and @ represents textures paged out in frame N+1:
###@$!!###

The ! represents the textures rendered from, and @ represents textures paged out in frame N+2:
####@$!!##
*/

// TEXTURES_IN_RESOURCE_POOL defines the number of textures to be created. This number should be
// enough to overcommit physical device memory.
#define TEXTURES_IN_RESOURCE_POOL 9000

// TEXTURES_PER_FRAME defines how many resources must be in local memory each frame. Kind of like a
// sliding window over the texture pool.
#define TEXTURES_PER_FRAME 1000

// TEXTURE_CHURN_PER_FRAME defines the number of textures the change between frames. A value of 20
// means that 20 textures currently in local memory will no longer be used and 20 new textures (that
// probably exist in system memory) will be needed for the next frame.
#define TEXTURE_CHURN_PER_FRAME 20

// Initializes buffers for draw indices and vertices.
void initBuffers() {
    static const uint32_t indexData[6] = {0, 1, 2, 3, 4, 5};
    indexBuffer =
        utils::CreateBufferFromData(device, indexData, sizeof(indexData), wgpu::BufferUsage::Index);

    static const float vertexData[18] = {
        -1.0f, 1.0f, 0.0f, 1.0f,  1.0f,  0.0f, 1.0f, -1.0f, 0.0f,
        -1.0f, 1.0f, 0.0f, -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f,
    };

    vertexBuffer = utils::CreateBufferFromData(device, vertexData, sizeof(vertexData),
                                               wgpu::BufferUsage::Vertex);
}

// Creates a simple staging texture with some data that I can initialize the main texture pool with.
void CreateStagingTexture() {
    wgpu::TextureDescriptor descriptor;
    descriptor.dimension = wgpu::TextureDimension::e2D;
    descriptor.size.width = 512;
    descriptor.size.height = 512;
    descriptor.size.depth = 1;
    descriptor.sampleCount = 1;
    descriptor.format = wgpu::TextureFormat::RGBA8Unorm;
    descriptor.mipLevelCount = 1;
    descriptor.usage = wgpu::TextureUsage::CopySrc | wgpu::TextureUsage::CopyDst;
    mStagingTexture = device.CreateTexture(&descriptor);

    std::vector<uint8_t> data(4 * 512 * 512, 0);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(i % 253);
    }

    wgpu::Buffer stagingBuffer = utils::CreateBufferFromData(
        device, data.data(), static_cast<uint32_t>(data.size()), wgpu::BufferUsage::CopySrc);
    wgpu::BufferCopyView bufferCopyView = utils::CreateBufferCopyView(stagingBuffer, 0, 4 * 512, 0);
    mStagingTextureCopyView = utils::CreateTextureCopyView(mStagingTexture, 0, {0, 0, 0});
    wgpu::Extent3D copySize = {512, 512, 1};

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.CopyBufferToTexture(&bufferCopyView, &mStagingTextureCopyView, &copySize);

    wgpu::CommandBuffer copy = encoder.Finish();
    queue.Submit(1, &copy);
}

// Creates a large pool of textures. A portion of these will be used for rendering. The portion used
// for render will change each frame, which will force paging between local and system memory.
void initTextures() {
    for (int i = 0; i < TEXTURES_IN_RESOURCE_POOL; i++) {
        // Create textures 1MB in size each
        wgpu::TextureDescriptor descriptor;
        descriptor.dimension = wgpu::TextureDimension::e2D;
        descriptor.size.width = 512;
        descriptor.size.height = 512;
        descriptor.size.depth = 1;
        descriptor.sampleCount = 1;
        descriptor.format = wgpu::TextureFormat::RGBA8Unorm;
        descriptor.mipLevelCount = 1;
        descriptor.usage = wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::Sampled;
        textures.push_back(device.CreateTexture(&descriptor));

        // Create a sampler for each texture
        wgpu::SamplerDescriptor samplerDesc = utils::GetDefaultSamplerDescriptor();
        samplers.push_back(device.CreateSampler(&samplerDesc));

        wgpu::TextureCopyView textureCopyView =
            utils::CreateTextureCopyView(textures[i], 0, {0, 0, 0});
        wgpu::Extent3D copySize = {512, 512, 1};

        wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
        encoder.CopyTextureToTexture(&mStagingTextureCopyView, &textureCopyView, &copySize);
        wgpu::CommandBuffer copy = encoder.Finish();
        queue.Submit(1, &copy);
    }
}

void init() {
    device = CreateCppDawnDevice();
    mTimer = utils::CreateTimer();
    queue = device.GetDefaultQueue();
    swapchain = GetSwapChain(device);
    swapchain.Configure(GetPreferredSwapChainTextureFormat(), wgpu::TextureUsage::OutputAttachment,
                        640, 480);

    initBuffers();
    CreateStagingTexture();
    initTextures();
    wgpu::ShaderModule vsModule =
        utils::CreateShaderModule(device, utils::SingleShaderStage::Vertex, R"(
        #version 450
        layout(location = 0) in vec3 pos;
        void main() {
            gl_Position = vec4(pos.xy, 0.0f, 1.0f);
        })");

    wgpu::ShaderModule fsModule =
        utils::CreateShaderModule(device, utils::SingleShaderStage::Fragment, R"(
        #version 450
        layout(set = 0, binding = 0) uniform sampler mySampler;
        layout(set = 0, binding = 1) uniform texture2D myTexture;

        layout(location = 0) out vec4 fragColor;
        void main() {
            fragColor = texture(sampler2D(myTexture, mySampler), gl_FragCoord.xy / vec2(640.0, 480.0));
        })");

    auto bgl = utils::MakeBindGroupLayout(
        device, {{0, wgpu::ShaderStage::Fragment, wgpu::BindingType::Sampler},
                 {1, wgpu::ShaderStage::Fragment, wgpu::BindingType::SampledTexture}});

    for (int i = 0; i < TEXTURES_IN_RESOURCE_POOL; i++) {
        wgpu::TextureView view = textures[i].CreateView();
        bindGroups.push_back(utils::MakeBindGroup(device, bgl, {{0, samplers[i]}, {1, view}}));
    }
    wgpu::PipelineLayout pl = utils::MakeBasicPipelineLayout(device, &bgl);

    depthStencilView = CreateDefaultDepthStencilView(device);

    utils::ComboRenderPipelineDescriptor descriptor(device);
    descriptor.layout = utils::MakeBasicPipelineLayout(device, &bgl);
    descriptor.vertexStage.module = vsModule;
    descriptor.cFragmentStage.module = fsModule;
    descriptor.cVertexState.vertexBufferCount = 1;
    descriptor.cVertexState.cVertexBuffers[0].arrayStride = 3 * sizeof(float);
    descriptor.cVertexState.cVertexBuffers[0].attributeCount = 1;
    descriptor.cVertexState.cAttributes[0].format = wgpu::VertexFormat::Float3;
    descriptor.depthStencilState = &descriptor.cDepthStencilState;
    descriptor.cDepthStencilState.format = wgpu::TextureFormat::Depth24PlusStencil8;
    descriptor.cColorStates[0].format = GetPreferredSwapChainTextureFormat();

    pipeline = device.CreateRenderPipeline(&descriptor);
}

void frame() {
    // Render directly to the backbuffer.
    wgpu::TextureView backbufferView = swapchain.GetCurrentTextureView();
    utils::ComboRenderPassDescriptor renderPass({backbufferView}, depthStencilView);

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    {
        wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&renderPass);
        pass.SetPipeline(pipeline);
        int index = drawStart;
        pass.SetIndexBuffer(indexBuffer, 0);
        pass.SetVertexBuffer(0, vertexBuffer);
        // We draw TEXTURES_PER_FRAME times. Each draw samples from a texture in the texture pool.
        // This is meant to be a small amount of work that just ensures the texture does exist in
        // GPU memory.
        for (int i = 0; i < TEXTURES_PER_FRAME; i++) {
            pass.SetBindGroup(0, bindGroups[index], 0, nullptr);
            pass.DrawIndexed(6, 1, 0, 0, 0);
            index++;
            // Go back to the beginning of the texture pool if we reach the end index.
            if (index >= TEXTURES_IN_RESOURCE_POOL) {
                index = 0;
            }
        }
        pass.EndPass();
    }

    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);
    swapchain.Present();
    DoFlush();

    // Increment the drawStart by TEXTURE_CHURN_PER_FRAME. This changes which textures from the pool
    // are being sampled from (i.e. changes what is GPU memory).
    drawStart += TEXTURE_CHURN_PER_FRAME;
    if (drawStart >= TEXTURES_IN_RESOURCE_POOL) {
        drawStart = 0;
    }
}

int main(int argc, const char* argv[]) {
    if (!InitSample(argc, argv)) {
        return 1;
    }
    init();

    while (!ShouldQuit()) {
        // All render work is done in frame()
        frame();

        // Poorly written but working FPS counter for window title bar.
        mTimer->Stop();
        float frameTime = mTimer->GetElapsedTime();
        frameTimeArray.push_back(frameTime);
        if (frameTimeArray.size() >= 20) {
            float movingAverageFrameTime = 0;
            for (int i = 1; i < 20; i++) {
                movingAverageFrameTime += frameTimeArray[i] / 19.0f;
            }
            float a = 1 / movingAverageFrameTime;
            frameTimeArray.clear();
            std::string s = std::to_string(a);
            glfwSetWindowTitle(GetGLFWWindow(), s.c_str());
        }
        mTimer->Start();
    }
}
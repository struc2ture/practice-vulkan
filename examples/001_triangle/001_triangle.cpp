#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <volk/volk.h>
#include <vulkan/vulkan.h>
#include <vma/vk_mem_alloc.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <assert.h>
#include <chrono>
#include <iostream>
#include <format>
#include <fstream>
#include <vector>

#include "base_test.h"
#include "example_base.h"
#include "camera.h"
#include "VulkanInitializers.h"
#include "VulkanTools.h"

class VulkanExample : public VulkanExampleBase
{
public:
    struct Vertex
    {
        float position[3];
        float color[3];
    };

    struct VulkanBuffer
    {
        VkDeviceMemory memory{ VK_NULL_HANDLE };
        VkBuffer handle{ VK_NULL_HANDLE };
    };

    VulkanBuffer vertexBuffer;
    VulkanBuffer indexBuffer;

    uint32_t indexCount{ 0 };

    struct UniformBuffer : VulkanBuffer
    {
        VkDescriptorSet descriptorSet{ VK_NULL_HANDLE };
        uint8_t* mapped{ nullptr };
    };

    std::array<UniformBuffer, maxConcurrentFrames> uniformBuffers;

    struct ShaderData
    {
        glm::mat4 projectionMatrix;
        glm::mat4 modelMatrix;
        glm::mat4 viewMatrix;
    };

    VulkanExample() : VulkanExampleBase()
    {
        title = "Basic indexed triangle using Vulkan 1.3";
        camera.type = Camera::CameraType::lookat;
        camera.setPosition(glm::vec3(0.0f, 0.0f, -2.5f));
        camera.setRotation(glm::vec3(0.0f));
        camera.setPerspective(60.0f, (float)width / (float)height, 1.0f, 256.0f);
    }

    ~VulkanExample() override
    {
        if (device)
        {
            vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
            vkDestroyBuffer(device, vertexBuffer.handle, nullptr);
            vkFreeMemory(device, vertexBuffer.memory, nullptr);
            vkDestroyBuffer(device, indexBuffer.handle, nullptr);
            vkFreeMemory(device, indexBuffer.memory, nullptr);
            vkDestroyCommandPool(device, exampleCommandPool, nullptr);
            for (auto i = 0; i < examplePresentCompleteSemaphores.size(); i++)
            {
                vkDestroySemaphore(device, examplePresentCompleteSemaphores[i], nullptr);
            }
            for (auto i = 0; i < exampleRenderCompleteSemaphores.size(); i++)
            {
                vkDestroySemaphore(device, exampleRenderCompleteSemaphores[i], nullptr);
            }
            for (auto i = 0; i < maxConcurrentFrames; i++)
            {
                vkDestroyFence(device, exampleWaitFences[i], nullptr);
                vkDestroyBuffer(device, uniformBuffers[i].handle, nullptr);
                vkFreeMemory(device, uniformBuffers[i].memory, nullptr);
            }
        }
    }

private:
    std::array<VkFence, maxConcurrentFrames> exampleWaitFences{};
    std::array<VkSemaphore, maxConcurrentFrames> examplePresentCompleteSemaphores{};
    std::vector<VkSemaphore> exampleRenderCompleteSemaphores{};
    VkCommandPool exampleCommandPool{ VK_NULL_HANDLE };
    std::array<VkCommandBuffer, maxConcurrentFrames> exampleCommandBuffers{};
    VkDescriptorPool descriptorPool{ VK_NULL_HANDLE };
    VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };
    VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };
    VkPipeline pipeline{ VK_NULL_HANDLE };

    uint32_t currentFrame{ 0 };

public:
    void prepare() override
    {
        VulkanExampleBase::prepare();
        createExampleSynchronizationPrimitives();
        createExampleCommandBuffers();
        createVertexBuffer();
        createUniformBuffers();
        createDescriptors();
        createPipeline();
        prepared = true;
    }

    void render() override
    {
        vkWaitForFences(device, 1, &exampleWaitFences[currentFrame], VK_TRUE, UINT64_MAX);
        VK_CHECK_RESULT(vkResetFences(device, 1, &exampleWaitFences[currentFrame]));

        uint32_t imageIndex{ 0 };
        VkResult result = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, examplePresentCompleteSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
        if (result == VK_ERROR_OUT_OF_DATE_KHR)
        {
            windowResize();
            return;
        }
        else if ((result != VK_SUCCESS) && (result != VK_SUBOPTIMAL_KHR))
        {
            throw "Could not acquire the next swapchain image";
        }

        ShaderData shaderData{};
        shaderData.projectionMatrix = camera.matrices.perspective;
        shaderData.viewMatrix = camera.matrices.view;
        shaderData.modelMatrix = glm::mat4(1.0f);
        memcpy(uniformBuffers[currentFrame].mapped, &shaderData, sizeof(ShaderData));

        vkResetCommandBuffer(exampleCommandBuffers[currentFrame], 0);
        VkCommandBufferBeginInfo cmdBufInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        const VkCommandBuffer commandBuffer = exampleCommandBuffers[currentFrame];
        VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &cmdBufInfo));

        vks::tools::insertImageMemoryBarrier(commandBuffer, swapchainImages[imageIndex], 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });
        vks::tools::insertImageMemoryBarrier(commandBuffer, depthStencil.image, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VkImageSubresourceRange{ VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1 });

        VkRenderingAttachmentInfo colorAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        colorAttachment.imageView = swapchainImageViews[imageIndex];
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.clearValue.color = { 0.0f, 0.0f, 0.2f, 0.0f };

        VkRenderingAttachmentInfo depthStencilAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        depthStencilAttachment.imageView = depthStencil.view;
        depthStencilAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthStencilAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthStencilAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthStencilAttachment.clearValue.depthStencil = { 1.0f, 0 };

        VkRenderingInfo renderingInfo{ VK_STRUCTURE_TYPE_RENDERING_INFO_KHR };
        renderingInfo.renderArea = { 0, 0, width, height };
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;
        renderingInfo.pDepthAttachment = &depthStencilAttachment;
        renderingInfo.pStencilAttachment = &depthStencilAttachment;

        vkCmdBeginRendering(commandBuffer, &renderingInfo);
        VkViewport viewport{ 0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f };
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
        VkRect2D scissor{ 0, 0, width, height };
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &uniformBuffers[currentFrame].descriptorSet, 0, nullptr);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        VkDeviceSize offsets[1]{ 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer.handle, offsets);
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer.handle, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(commandBuffer, indexCount, 1, 0, 0, 0);
        vkCmdEndRendering(commandBuffer);

        vks::tools::insertImageMemoryBarrier(commandBuffer, swapchainImages[imageIndex], VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_NONE, VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });
        VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));

        VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submitInfo.pWaitDstStageMask = &waitStageMask;
        submitInfo.pCommandBuffers = &commandBuffer;
        submitInfo.commandBufferCount = 1;
        submitInfo.pWaitSemaphores = &examplePresentCompleteSemaphores[currentFrame];
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &exampleRenderCompleteSemaphores[imageIndex];
        submitInfo.signalSemaphoreCount = 1;
        VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, exampleWaitFences[currentFrame]));

        VkPresentInfoKHR presentInfo{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &exampleRenderCompleteSemaphores[imageIndex];
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain;
        presentInfo.pImageIndices = &imageIndex;
        result = vkQueuePresentKHR(queue, &presentInfo);
        if ((result == VK_ERROR_OUT_OF_DATE_KHR) || (result == VK_SUBOPTIMAL_KHR))
        {
            windowResize();
        }
        else if (result != VK_SUCCESS)
        {
            throw "Could not present the image to the swapchain";
        }

        currentFrame = (currentFrame + 1) % maxConcurrentFrames;
    }

private:
    void createExampleSynchronizationPrimitives()
    {
        for (uint32_t i = 0; i < maxConcurrentFrames; i++)
        {
            VkFenceCreateInfo fenceCI{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
            fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            VK_CHECK_RESULT(vkCreateFence(device, &fenceCI, nullptr, &exampleWaitFences[i]));
        }
        for (auto& semaphore : examplePresentCompleteSemaphores)
        {
            VkSemaphoreCreateInfo semaphoreCI{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
            VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCI, nullptr, &semaphore));
        }
        exampleRenderCompleteSemaphores.resize(swapchainImages.size());
        for (auto& semaphore : exampleRenderCompleteSemaphores)
        {
            VkSemaphoreCreateInfo semaphoreCI{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
            VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCI, nullptr, &semaphore));
        }
    }

    void createExampleCommandBuffers()
    {
        VkCommandPoolCreateInfo commandPoolCI{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        commandPoolCI.queueFamilyIndex = queueFamilyIndex;
        commandPoolCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VK_CHECK_RESULT(vkCreateCommandPool(device, &commandPoolCI, nullptr, &exampleCommandPool));
        VkCommandBufferAllocateInfo cmdBufAI = vks::initializers::commandBufferAllocateInfo(exampleCommandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, maxConcurrentFrames);
        VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAI, exampleCommandBuffers.data()));
    }

    void createVertexBuffer()
    {
        const std::vector<Vertex> vertices{
            { {  1.0f,  1.0f, 0.0f}, { 1.0f, 0.0f, 0.0f } },
            { { -1.0f,  1.0f, 0.0f}, { 0.0f, 1.0f, 0.0f } },
            { {  0.0f, -1.0f, 0.0f}, { 0.0f, 0.0f, 1.0f } }
        };
        uint32_t vertexBufferSize = static_cast<uint32_t>(vertices.size()) * sizeof(Vertex);

        std::vector<uint32_t> indices{ 0, 1, 2 };
        indexCount = static_cast<uint32_t>(indices.size());
        uint32_t indexBufferSize = indexCount * sizeof(uint32_t);

        VkMemoryAllocateInfo memAlloc{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        VkMemoryRequirements memReqs;

        VulkanBuffer stagingBuffer;
        VkBufferCreateInfo stagingBufferCI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        stagingBufferCI.size = vertexBufferSize + indexBufferSize;
        stagingBufferCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VK_CHECK_RESULT(vkCreateBuffer(device, &stagingBufferCI, nullptr, &stagingBuffer.handle));
        vkGetBufferMemoryRequirements(device, stagingBuffer.handle, &memReqs);
        memAlloc.allocationSize = memReqs.size;
        memAlloc.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &stagingBuffer.memory));
        VK_CHECK_RESULT(vkBindBufferMemory(device, stagingBuffer.handle, stagingBuffer.memory, 0));

        uint8_t* data{ nullptr };
        VK_CHECK_RESULT(vkMapMemory(device, stagingBuffer.memory, 0, memAlloc.allocationSize, 0, (void**)&data));
        memcpy(data, vertices.data(), vertexBufferSize);
        memcpy(((char*)data) + vertexBufferSize, indices.data(), indexBufferSize);

        VkBufferCreateInfo vertexBufferCI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        vertexBufferCI.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        vertexBufferCI.size = vertexBufferSize;
        VK_CHECK_RESULT(vkCreateBuffer(device, &vertexBufferCI, nullptr, &vertexBuffer.handle));
        vkGetBufferMemoryRequirements(device, vertexBuffer.handle, &memReqs);
        memAlloc.allocationSize = memReqs.size;
        memAlloc.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &vertexBuffer.memory));
        VK_CHECK_RESULT(vkBindBufferMemory(device, vertexBuffer.handle, vertexBuffer.memory, 0));

        VkBufferCreateInfo indexBufferCI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        indexBufferCI.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        indexBufferCI.size = indexBufferSize;
        VK_CHECK_RESULT(vkCreateBuffer(device, &indexBufferCI, nullptr, &indexBuffer.handle));
        vkGetBufferMemoryRequirements(device, indexBuffer.handle, &memReqs);
        memAlloc.allocationSize = memReqs.size;
        memAlloc.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &indexBuffer.memory));
        VK_CHECK_RESULT(vkBindBufferMemory(device, indexBuffer.handle, indexBuffer.memory, 0));

        VkCommandBuffer copyCmd;

        VkCommandBufferAllocateInfo cmdBufAI{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        cmdBufAI.commandPool = exampleCommandPool;
        cmdBufAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdBufAI.commandBufferCount = 1;
        VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAI, &copyCmd));

        auto cbBI = vks::initializers::commandBufferBeginInfo();
        VK_CHECK_RESULT(vkBeginCommandBuffer(copyCmd, &cbBI));
        VkBufferCopy copyRegion{};
        copyRegion.size = vertexBufferSize;
        vkCmdCopyBuffer(copyCmd, stagingBuffer.handle, vertexBuffer.handle, 1, &copyRegion);
        copyRegion.size = indexBufferSize;
        copyRegion.srcOffset = vertexBufferSize;
        vkCmdCopyBuffer(copyCmd, stagingBuffer.handle, indexBuffer.handle, 1, &copyRegion);
        VK_CHECK_RESULT(vkEndCommandBuffer(copyCmd));

        VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &copyCmd;

        VkFenceCreateInfo fenceCI{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        VkFence fence;
        VK_CHECK_RESULT(vkCreateFence(device, &fenceCI, nullptr, &fence));
        VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));
        VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT));
        vkDestroyFence(device, fence, nullptr);
        vkFreeCommandBuffers(device, exampleCommandPool, 1, &copyCmd);

        vkDestroyBuffer(device, stagingBuffer.handle, nullptr);
        vkFreeMemory(device, stagingBuffer.memory, nullptr);
    }

    void createUniformBuffers()
    {
        VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufferInfo.size = sizeof(ShaderData);
        bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

        for (uint32_t i = 0; i < maxConcurrentFrames; i++)
        {
            VK_CHECK_RESULT(vkCreateBuffer(device, &bufferInfo, nullptr, &uniformBuffers[i].handle));
            VkMemoryRequirements memReqs;
            vkGetBufferMemoryRequirements(device, uniformBuffers[i].handle, &memReqs);
            VkMemoryAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
            allocInfo.allocationSize = memReqs.size;
            allocInfo.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            VK_CHECK_RESULT(vkAllocateMemory(device, &allocInfo, nullptr, &(uniformBuffers[i].memory)));
            VK_CHECK_RESULT(vkBindBufferMemory(device, uniformBuffers[i].handle, uniformBuffers[i].memory, 0));
            VK_CHECK_RESULT(vkMapMemory(device, uniformBuffers[i].memory, 0, sizeof(ShaderData), 0, (void**)&uniformBuffers[i].mapped));
        }
    }

    void createDescriptors()
    {
        VkDescriptorPoolSize descriptorTypeCounts[1]{};
        descriptorTypeCounts[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorTypeCounts[0].descriptorCount = maxConcurrentFrames;

        VkDescriptorPoolCreateInfo descriptorPoolCI{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        descriptorPoolCI.poolSizeCount = 1;
        descriptorPoolCI.pPoolSizes = descriptorTypeCounts;
        descriptorPoolCI.maxSets = maxConcurrentFrames;
        VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCI, nullptr, &descriptorPool));

        VkDescriptorSetLayoutBinding layoutBinding{};
        layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        layoutBinding.descriptorCount = 1;
        layoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutCreateInfo descriptorLayoutCI{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        descriptorLayoutCI.bindingCount = 1;
        descriptorLayoutCI.pBindings = &layoutBinding;
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayoutCI, nullptr, &descriptorSetLayout));
        for (uint32_t i = 0; i < maxConcurrentFrames; i++)
        {
            VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            allocInfo.descriptorPool = descriptorPool;
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts = &descriptorSetLayout;
            VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &uniformBuffers[i].descriptorSet));

            VkWriteDescriptorSet writeDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };

            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i].handle;
            bufferInfo.range = sizeof(ShaderData);

            // Binding 0: Uniform buffer
            writeDescriptorSet.dstSet = uniformBuffers[i].descriptorSet;
            writeDescriptorSet.descriptorCount = 1;
            writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writeDescriptorSet.pBufferInfo = &bufferInfo;
            writeDescriptorSet.dstBinding = 0;
            vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
        }
    }

    void createPipeline()
    {
        VkPipelineLayoutCreateInfo pipelineLayoutCI{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipelineLayoutCI.setLayoutCount = 1;
        pipelineLayoutCI.pSetLayouts = &descriptorSetLayout;
        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelineLayout));
        
        VkGraphicsPipelineCreateInfo pipelineCI{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
        pipelineCI.layout = pipelineLayout;

        VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
        inputAssemblyStateCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineRasterizationStateCreateInfo rasterizationStateCI{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
        rasterizationStateCI.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizationStateCI.cullMode = VK_CULL_MODE_NONE;
        rasterizationStateCI.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizationStateCI.depthClampEnable = VK_FALSE;
        rasterizationStateCI.rasterizerDiscardEnable = VK_FALSE;
        rasterizationStateCI.depthBiasEnable = VK_FALSE;
        rasterizationStateCI.lineWidth = 1.0f;

        VkPipelineColorBlendAttachmentState blendAttachmentState{};
        blendAttachmentState.colorWriteMask = 0xf;
        blendAttachmentState.blendEnable = VK_FALSE;
        VkPipelineColorBlendStateCreateInfo colorBlendStateCI{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
        colorBlendStateCI.attachmentCount = 1;
        colorBlendStateCI.pAttachments = &blendAttachmentState;

        VkPipelineViewportStateCreateInfo viewportStateCI{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
        viewportStateCI.viewportCount = 1;
        viewportStateCI.scissorCount = 1;

        std::vector<VkDynamicState> dynamicStateEnables{ VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicStateCI{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
        dynamicStateCI.pDynamicStates = dynamicStateEnables.data();
        dynamicStateCI.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size());

        VkPipelineDepthStencilStateCreateInfo depthStencilStateCI{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
        depthStencilStateCI.depthTestEnable = VK_TRUE;
        depthStencilStateCI.depthWriteEnable = VK_TRUE;
        depthStencilStateCI.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        depthStencilStateCI.depthBoundsTestEnable = VK_FALSE;
        depthStencilStateCI.back.failOp = VK_STENCIL_OP_KEEP;
        depthStencilStateCI.back.passOp = VK_STENCIL_OP_KEEP;
        depthStencilStateCI.back.compareOp = VK_COMPARE_OP_ALWAYS;
        depthStencilStateCI.stencilTestEnable = VK_FALSE;
        depthStencilStateCI.front = depthStencilStateCI.back;

        VkPipelineMultisampleStateCreateInfo multisampleStateCI{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
        multisampleStateCI.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkVertexInputBindingDescription vertexInputBinding{};
        vertexInputBinding.binding = 0;
        vertexInputBinding.stride = sizeof(Vertex);
        vertexInputBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        std::array<VkVertexInputAttributeDescription, 2> vertexInputAttributes{};
        vertexInputAttributes[0].binding = 0;
        vertexInputAttributes[0].location = 0;
        vertexInputAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        vertexInputAttributes[0].offset = offsetof(Vertex, position);
        vertexInputAttributes[1].binding = 0;
        vertexInputAttributes[1].location = 1;
        vertexInputAttributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        vertexInputAttributes[1].offset = offsetof(Vertex, color);

        VkPipelineVertexInputStateCreateInfo vertexInputStateCI{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
        vertexInputStateCI.vertexBindingDescriptionCount = 1;
        vertexInputStateCI.pVertexBindingDescriptions = &vertexInputBinding;
        vertexInputStateCI.vertexAttributeDescriptionCount = 2;
        vertexInputStateCI.pVertexAttributeDescriptions = vertexInputAttributes.data();

        std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};

        shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        shaderStages[0].module = loadSPIRVShader("examples/001_triangle/triangle.vert.spv");
        shaderStages[0].pName = "main";
        assert(shaderStages[0].module != VK_NULL_HANDLE);

        shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        shaderStages[1].module = loadSPIRVShader("examples/001_triangle/triangle.frag.spv");
        shaderStages[1].pName = "main";
        assert(shaderStages[0].module != VK_NULL_HANDLE);

        pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineCI.pStages = shaderStages.data();

        VkPipelineRenderingCreateInfoKHR pipelineRenderingCI{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR };
        pipelineRenderingCI.colorAttachmentCount = 1;
        pipelineRenderingCI.pColorAttachmentFormats = &swapchainImageFormat;
        pipelineRenderingCI.depthAttachmentFormat = depthFormat;
        pipelineRenderingCI.stencilAttachmentFormat = depthFormat;

        pipelineCI.pVertexInputState = &vertexInputStateCI;
        pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
        pipelineCI.pRasterizationState = &rasterizationStateCI;
        pipelineCI.pColorBlendState = &colorBlendStateCI;
        pipelineCI.pMultisampleState = &multisampleStateCI;
        pipelineCI.pViewportState = &viewportStateCI;
        pipelineCI.pDepthStencilState = &depthStencilStateCI;
        pipelineCI.pDynamicState = &dynamicStateCI;
        pipelineCI.pNext = &pipelineRenderingCI;

        VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipeline));

        vkDestroyShaderModule(device, shaderStages[0].module, nullptr);
        vkDestroyShaderModule(device, shaderStages[1].module, nullptr);
    }

    VkShaderModule loadSPIRVShader(const std::string& filename)
    {
        size_t shaderSize{};
        char *shaderCode{ nullptr };

        std::ifstream is(filename, std::ios::binary | std::ios::in | std::ios::ate);

        if (is.is_open())
        {
            shaderSize = is.tellg();
            is.seekg(0, std::ios::beg);
            shaderCode = new char[shaderSize];
            is.read(shaderCode, shaderSize);
            is.close();
            assert(shaderSize > 0);
        }

        if (shaderCode)
        {
            VkShaderModuleCreateInfo shaderModuleCI{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
            shaderModuleCI.codeSize = shaderSize;
            shaderModuleCI.pCode = (uint32_t*)shaderCode;

            VkShaderModule shaderModule;
            VK_CHECK_RESULT(vkCreateShaderModule(device, &shaderModuleCI, nullptr, &shaderModule));

            delete[] shaderCode;

            return shaderModule;
        }
        else
        {
            std::cerr << "Error: Could not open shader file \"" << filename << "\"" << std::endl;
            return VK_NULL_HANDLE;
        }
    }

    // Base class shouldn't generate framebuffers and render passes
    void setupFrameBuffer() override {}
    void setupRenderPass() override {}
};

VulkanExample *vulkanExample;

int main(int argc, char **argv)
{
    std::cout << std::format("Hello world {}!!!\n", 123);
    testMe();

    vulkanExample = new VulkanExample();
    vulkanExample->init();
    vulkanExample->prepare();
    vulkanExample->renderLoop();
    delete(vulkanExample);
    return 0;
}

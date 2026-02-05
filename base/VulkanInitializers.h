#pragma once

#include <vulkan/vulkan.h>

namespace vks
{
    namespace initializers
    {
        inline VkCommandBufferAllocateInfo commandBufferAllocateInfo(
            VkCommandPool commandPool,
            VkCommandBufferLevel level,
            uint32_t bufferCount)
        {
            VkCommandBufferAllocateInfo commandBufferAI{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .commandPool = commandPool,
                .level = level,
                .commandBufferCount = bufferCount   
            };
            return commandBufferAI;
        }

        inline VkCommandBufferBeginInfo commandBufferBeginInfo()
        {
            VkCommandBufferBeginInfo cmdBufferBeginInfo {};
            cmdBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            return cmdBufferBeginInfo;
        }
    }
}
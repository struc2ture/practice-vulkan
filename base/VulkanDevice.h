#pragma once

#include <vulkan/vulkan.h>

namespace vks
{
    struct VulkanDevice
    {
        VkPhysicalDevice physicalDevice{ VK_NULL_HANDLE };
        VkDevice logicalDevice{ VK_NULL_HANDLE };
    };
}
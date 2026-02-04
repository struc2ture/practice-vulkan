#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <volk/volk.h>
#include <vulkan/vulkan.h>
#include <vma/vk_mem_alloc.h>

#include <assert.h>
#include <iostream>
#include <format>
#include <vector>

#include "base_test.h"
#include "VulkanTools.h"

class VulkanExample
{
public:
    std::string title = "Vulkan Example";
    std::string name = "vulkanExample";
	uint32_t apiVersion = VK_API_VERSION_1_3;

private:
    SDL_Window *window;

    VkInstance instance{ VK_NULL_HANDLE };
    VkPhysicalDevice physicalDevice{ VK_NULL_HANDLE };
    VkDevice device{ VK_NULL_HANDLE };
    uint32_t queueFamilyIndex{ 0 };
    VkQueue queue{ VK_NULL_HANDLE };
    VmaAllocator allocator{ VK_NULL_HANDLE };
    VkSurfaceKHR surface{ VK_NULL_HANDLE };
    VkSurfaceCapabilitiesKHR surfaceCaps{};
    VkCommandPool cmdPool{ VK_NULL_HANDLE };

public:
    bool init()
    {
        assert(SDL_Init(SDL_INIT_VIDEO));
        assert(SDL_Vulkan_LoadLibrary(NULL));
        volkInitialize();

        // Create instance
        VK_CHECK_RESULT(createInstance());

        // Physical device
        uint32_t deviceCount { 0 };
        VK_CHECK_RESULT(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr));
        std::vector<VkPhysicalDevice> devices(deviceCount);
        VK_CHECK_RESULT(vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data()));
        uint32_t deviceIndex{ 0 };
        VkPhysicalDeviceProperties2 deviceProperties{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        vkGetPhysicalDeviceProperties2(devices[deviceIndex], &deviceProperties);
        std::cout << "Selected device: " << deviceProperties.properties.deviceName << "\n";
        physicalDevice = devices[deviceIndex];

        // Find a queue family for graphics
        uint32_t queueFamilyCount{ 0 };
        vkGetPhysicalDeviceQueueFamilyProperties(devices[deviceIndex], &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(devices[deviceIndex], &queueFamilyCount, queueFamilies.data());
        for (size_t i = 0; i < queueFamilies.size(); i++)
        {
            if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            {
                queueFamilyIndex = i;
                break;
            }
        }

        // Logical device
        const float qfpriorities{ 1.0f };
        VkDeviceQueueCreateInfo queueCI{
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = queueFamilyIndex,
            .queueCount = 1,
            .pQueuePriorities = &qfpriorities
        };
        const std::vector<const char*> deviceExtensions{ VK_KHR_SWAPCHAIN_EXTENSION_NAME };
        VkPhysicalDeviceVulkan12Features enabledVk12Features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            .descriptorIndexing = true,
            .descriptorBindingVariableDescriptorCount = true,
            .runtimeDescriptorArray = true,
            .bufferDeviceAddress = true
        };
        const VkPhysicalDeviceVulkan13Features enabledVk13Features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            .pNext = &enabledVk12Features,
            .synchronization2 = true,
            .dynamicRendering = true
        };
        const VkPhysicalDeviceFeatures enabledVk10Features{
            .samplerAnisotropy = VK_TRUE
        };
        VkDeviceCreateInfo deviceCI{
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = &enabledVk13Features,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queueCI,
            .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
            .ppEnabledExtensionNames = deviceExtensions.data(),
            .pEnabledFeatures = &enabledVk10Features
        };
        VK_CHECK_RESULT(vkCreateDevice(devices[deviceIndex], &deviceCI, nullptr, &device));
        vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

        // VMA
        VmaVulkanFunctions vkFunctions{
            .vkGetInstanceProcAddr = vkGetInstanceProcAddr,
            .vkGetDeviceProcAddr = vkGetDeviceProcAddr,
            .vkCreateImage = vkCreateImage
        };
        VmaAllocatorCreateInfo allocatorCI{
            .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
            .physicalDevice = devices[deviceIndex],
            .device = device,
            .pVulkanFunctions = &vkFunctions,
            .instance = instance
        };
        VK_CHECK_RESULT(vmaCreateAllocator(&allocatorCI, &allocator));

        window = SDL_CreateWindow(name.c_str(), 1280u, 720u, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
        assert(window);

        return true;
    }

    void prepare()
    {
        createSurface();
        createCommandPool();
        //createSwapChain();
        //createCommandBuffers();
        //createSynchronizationPrimitives();
        //setupDepthStencil();
        //setupRenderPass();
        //createPipelineCache();
        //setupFrameBuffer();
    }

    void renderLoop()
    {

    }

private:
    VkResult createInstance()
    {
        VkApplicationInfo appInfo{
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = name.c_str(),
            .apiVersion = apiVersion
        };

        uint32_t instanceExtensionsCount{ 0 };
        char const* const* instanceExtensions{ SDL_Vulkan_GetInstanceExtensions(&instanceExtensionsCount) };
        VkInstanceCreateInfo instanceCI{
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &appInfo,
            .enabledExtensionCount = instanceExtensionsCount,
            .ppEnabledExtensionNames = instanceExtensions
        };

        VkResult result = vkCreateInstance(&instanceCI, nullptr, &instance);
        volkLoadInstance(instance);
        return result;
    }

    void createSurface()
    {
        assert(SDL_Vulkan_CreateSurface(window, instance, nullptr, &surface));
        VkSurfaceCapabilitiesKHR surfaceCaps{};
        VK_CHECK_RESULT(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &surfaceCaps));
    }

    void createCommandPool()
    {
        VkCommandPoolCreateInfo cmdPoolInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = queueFamilyIndex
        };
        VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &cmdPool));
    }

    void createSwapChain();
    void createCommandBuffers();
    void createSynchronizationPrimitives();
    void setupDepthStencil();
    void setupRenderPass();
    void createPipelineCache();
    void setupFrameBuffer();

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

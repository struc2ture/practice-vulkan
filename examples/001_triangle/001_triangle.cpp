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
#include "VulkanInitializers.h"
#include "VulkanTools.h"

constexpr uint32_t maxConcurrentFrames{ 2 };

class VulkanExampleBase
{
public:
    std::string title = "Vulkan Example";
    std::string name = "vulkanExample";
	uint32_t apiVersion = VK_API_VERSION_1_3;
    uint32_t width{ 1280 };
    uint32_t height{ 720 };

    float frameTimer{ 1.0f };

    // Defines a frame rate independent timer value clamped from -1.0...1.0
    // For use in animations, rotations, etc.
    float timer{ 0.0f };
    // Multiplier for speeding up (or slowing down) the global timer
    float timerSpeed{ 0.25f };
    bool paused{ false };

protected:
    VkInstance instance{ VK_NULL_HANDLE };
    VkDevice device{ VK_NULL_HANDLE };
    uint32_t queueFamilyIndex{ 0 };
    VkQueue queue{ VK_NULL_HANDLE };

    VkSwapchainKHR swapchain{ VK_NULL_HANDLE };
    VkFormat swapchainImageFormat{ VK_FORMAT_UNDEFINED };
    std::vector<VkImage> swapchainImages{};
    std::vector<VkImageView> swapchainImageViews{};
    struct {
        VkImage image;
        VkDeviceMemory memory;
        VkImageView view;
    } depthStencil{};
    VkFormat depthFormat{ VK_FORMAT_UNDEFINED };
    VkPipelineCache pipelineCache{ VK_NULL_HANDLE };
    bool prepared{ false };

private:
    uint32_t frameCounter = 0;
    uint32_t lastFPS = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> lastTimestamp;

    SDL_Window *window = nullptr;

    VkPhysicalDevice physicalDevice{ VK_NULL_HANDLE };
    VkPhysicalDeviceProperties deviceProperties{};
    VkPhysicalDeviceMemoryProperties deviceMemoryProperties{};
    VmaAllocator allocator{ VK_NULL_HANDLE };
    VkSurfaceKHR surface{ VK_NULL_HANDLE };
    VkSurfaceCapabilitiesKHR surfaceCaps{};
    VkCommandPool cmdPool{ VK_NULL_HANDLE };
    std::array<VkCommandBuffer, maxConcurrentFrames> drawCmdBuffers{};
    std::array<VkSemaphore, maxConcurrentFrames> presentCompleteSemaphores{};
    std::vector<VkSemaphore> renderCompleteSemaphores{};
    std::array<VkFence, maxConcurrentFrames> waitFences{};
    VkRenderPass renderPass{ VK_NULL_HANDLE };
    std::vector<VkFramebuffer> frameBuffers{};

    bool resized{ false };

public:
    VulkanExampleBase()
    {

    }

    ~VulkanExampleBase()
    {
        VK_CHECK_RESULT(vkDeviceWaitIdle(device));
        if (swapchain != VK_NULL_HANDLE)
        {
            for (auto i = 0; i < swapchainImages.size(); i++)
            {
                vkDestroyImageView(device, swapchainImageViews[i], nullptr);
            }
            vkDestroySwapchainKHR(device, swapchain, nullptr);
        }

        if (surface != VK_NULL_HANDLE)
        {
            vkDestroySurfaceKHR(instance, surface, nullptr);
        }

        //if (descriptorPool != VK_NULL_HANDLE)
        //{
        //    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        //}

        vkFreeCommandBuffers(device, cmdPool, static_cast<uint32_t>(drawCmdBuffers.size()), drawCmdBuffers.data());

        if (renderPass != VK_NULL_HANDLE)
        {
            vkDestroyRenderPass(device, renderPass, nullptr);
        }
        for (auto& frameBuffer : frameBuffers)
        {
            vkDestroyFramebuffer(device, frameBuffer, nullptr);
        }
        //for (auto& shaderModule : shaderModules)
        //{
        //    vkDestroyShaderModule(device, shaderModule, nullptr);
        //}
        vkDestroyImageView(device, depthStencil.view, nullptr);
        vkDestroyImage(device, depthStencil.image, nullptr);
        vkFreeMemory(device, depthStencil.memory, nullptr);
        vkDestroyPipelineCache(device, pipelineCache, nullptr);
        vkDestroyCommandPool(device, cmdPool, nullptr);
        for (auto& fence : waitFences)
        {
            vkDestroyFence(device, fence, nullptr);
        }
        for (auto& semaphore : presentCompleteSemaphores)
        {
            vkDestroySemaphore(device, semaphore, nullptr);
        }
        for (auto& semaphore : renderCompleteSemaphores)
        {
            vkDestroySemaphore(device, semaphore, nullptr);
        }
        vmaDestroyAllocator(allocator);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);

        SDL_DestroyWindow(window);
        SDL_QuitSubSystem(SDL_INIT_VIDEO);
        SDL_Quit();
    }

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
        vkGetPhysicalDeviceProperties(devices[deviceIndex], &deviceProperties);
        std::cout << "Selected device: " << deviceProperties.deviceName << "\n";
        physicalDevice = devices[deviceIndex];
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &deviceMemoryProperties);

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

        window = SDL_CreateWindow(name.c_str(), width, height, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
        assert(window);

        int w, h;
        assert(SDL_GetWindowSize(window, &w, &h));
        width = static_cast<uint32_t>(w);
        height = static_cast<uint32_t>(h);

        assert(vks::tools::getSupportedDepthFormat(physicalDevice, &depthFormat));
        
        return true;
    }

    virtual void prepare()
    {
        createSurface();
        createCommandPool();
        createSwapChain();
        createCommandBuffers();
        createSynchronizationPrimitives();
        setupDepthStencil();
        setupRenderPass();
        createPipelineCache();
        setupFrameBuffer();
    }

    void renderLoop()
    {
        lastTimestamp = std::chrono::high_resolution_clock::now();
        bool quit{ false };
        while (!quit)
        {
            for (SDL_Event event; SDL_PollEvent(&event);)
            {
                if (event.type == SDL_EVENT_QUIT)
                {
                    quit = true;
                    break;
                }
            }

            bool isMinimized = SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED;
            if (prepared && !isMinimized)
            {
                nextFrame();
            }
        }

        // Flush device to make sure all resources can be freed
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
        }
    }

    virtual void render() = 0;

protected:
    /**
    * Get the index of a memory type that has all the requested property bits set
    * 
    * @param typeBits Bit mask with bits set for each memory type supported by the resource to request for (from VkMemoryRequirements)
    * @param properties Bit mask of properties for the memory type to request
    * @param (Optional) memTypeFound Pointer to a bool that is set to true if a matching memory type has been found
    * 
    * @return Index of the requested memory type
    * 
    * @throw Throws an exception if memTypeFound is null and no memory type could be found that supports the requested properties
    */
    uint32_t getMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties, VkBool32* memTypeFound = nullptr) const
    {
        for (uint32_t i = 0; i < deviceMemoryProperties.memoryTypeCount; i++)
        {
            if ((typeBits & 1) == 1)
            {
                if ((deviceMemoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
                {
                    if (memTypeFound)
                    {
                        *memTypeFound = true;
                    }
                    return i;
                }
            }
            typeBits >>= 1;
        }

        if (memTypeFound)
        {
            *memTypeFound = false;
            return 0;
        }
        else
        {
            throw std::runtime_error("Could not find a matching memory type");
        }
    }

    virtual void setupRenderPass()
    {
        std::array<VkAttachmentDescription, 2> attachments{
            // Color Attachment
            VkAttachmentDescription{
                .format = swapchainImageFormat,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
        },
            // Depth attachment
            VkAttachmentDescription{
                .format = depthFormat,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        }
        };

        VkAttachmentReference colorReference{ .attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
        VkAttachmentReference depthReference{ .attachment = 1, .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

        VkSubpassDescription subpassDescription{
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorReference,
            .pDepthStencilAttachment = &depthReference
        };

        // Subpass dependencies for layout transitions
        std::array<VkSubpassDependency, 2> dependencies{
            VkSubpassDependency{
                .srcSubpass = VK_SUBPASS_EXTERNAL,
                .dstSubpass = 0,
                .srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
        },
        VkSubpassDependency{
                .srcSubpass = VK_SUBPASS_EXTERNAL,
                .dstSubpass = 0,
                .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                .srcAccessMask = 0,
                .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
        }
        };

        VkRenderPassCreateInfo renderPassInfo{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = static_cast<uint32_t>(attachments.size()),
            .pAttachments = attachments.data(),
            .subpassCount = 1,
            .pSubpasses = &subpassDescription,
            .dependencyCount = static_cast<uint32_t>(dependencies.size()),
            .pDependencies = dependencies.data()
        };
        VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass));
    }

    virtual void setupFrameBuffer()
    {
        // Create frame buffers for every swap chain image, only one depth/stencil attachment is required, as this is owned by the application
        frameBuffers.resize(swapchainImages.size());
        for (uint32_t i = 0; i < frameBuffers.size(); i++)
        {
            const VkImageView attachments[2] = { swapchainImageViews[i], depthStencil.view };
            VkFramebufferCreateInfo framebufferCI{
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .renderPass = renderPass,
                .attachmentCount = 2,
                .pAttachments = attachments,
                .width = width,
                .height = height,
                .layers = 1
            };
            VK_CHECK_RESULT(vkCreateFramebuffer(device, &framebufferCI, nullptr, &frameBuffers[i]));
        }
    }

    void windowResize()
    {
        if (!prepared)
        {
            return;
        }

        prepared = false;
        resized = true;

        vkDeviceWaitIdle(device);

        createSwapChain();

        vkDestroyImageView(device, depthStencil.view, nullptr);
        vkDestroyImage(device, depthStencil.image, nullptr);
        vkFreeMemory(device, depthStencil.memory, nullptr);
        setupDepthStencil();
        for (auto& frameBuffer : frameBuffers)
        {
            vkDestroyFramebuffer(device, frameBuffer, nullptr);k
        }
        setupFrameBuffer();

        for (auto& semaphore : presentCompleteSemaphores)
        {
            vkDestroySemaphore(device, semaphore, nullptr);
        }
        for (auto& semaphore : renderCompleteSemaphores)
        {
            vkDestroySemaphore(device, semaphore, nullptr);
        }
        for (auto& fence : waitFences)
        {
            vkDestroyFence(device, fence, nullptr);
        }
        createSynchronizationPrimitives();

        vkDeviceWaitIdle(device);

        if ((width > 0) && (height > 0))
        {
            camera.updateAspectRatio((float)width / (float)height);
        }

        // Notify derived class
         windowResized();

         prepared = true;
    }

    virtual void windowResized() {};

private:
    void nextFrame()
    {
        auto tStart = std::chrono::high_resolution_clock::now();
        render();
        frameCounter++;
        auto tEnd = std::chrono::high_resolution_clock::now();

        auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

        frameTimer = (float)tDiff / 1000.0f;

        if (!paused)
        {
            timer += timerSpeed * frameTimer;
            if (timer > 1.0)
            {
                timer -= 1.0f;
            }
        }
        float fpsTimer = (float)(std::chrono::duration<double, std::milli>(tEnd - lastTimestamp).count());
        if (fpsTimer > 1000.0f)
        {
            lastFPS = static_cast<uint32_t>((float)frameCounter * (1000.0f / fpsTimer));
            frameCounter = 0;
            lastTimestamp = tEnd;
        }
    }

    std::string getWindowTitle() const
    {
        std::string windowTitle{ title + " - " + deviceProperties.deviceName + " - " + std::to_string(frameCounter) + " fps" };
    }

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

    void createSwapChain()
    {
        swapchainImageFormat = VK_FORMAT_B8G8R8A8_SRGB;
        VkSwapchainCreateInfoKHR swapchainCI{
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface = surface,
            .minImageCount = surfaceCaps.minImageCount,
            .imageFormat = swapchainImageFormat,
            .imageColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR,
            .imageExtent{ .width = surfaceCaps.currentExtent.width, .height = surfaceCaps.currentExtent.height },
            .imageArrayLayers = 1,
            .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
            .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = VK_PRESENT_MODE_FIFO_KHR
        };
        VK_CHECK_RESULT(vkCreateSwapchainKHR(device, &swapchainCI, nullptr, &swapchain));

        uint32_t imageCount{ 0 };
        VK_CHECK_RESULT(vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr));
        swapchainImages.resize(imageCount);
        VK_CHECK_RESULT(vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages.data()));
        swapchainImageViews.resize(imageCount);
        for (auto i = 0; i < imageCount; i++)
        {
            VkImageViewCreateInfo viewCI{
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = swapchainImages[i],
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = swapchainImageFormat,
                .subresourceRange{
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .levelCount = 1,
                    .layerCount = 1
                }
            };
            VK_CHECK_RESULT(vkCreateImageView(device, &viewCI, nullptr, &swapchainImageViews[i]));
        }
    }

    void createCommandBuffers()
    {
        VkCommandBufferAllocateInfo cmdBufferAI{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = cmdPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = static_cast<uint32_t>(drawCmdBuffers.size())
        };
        VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufferAI, drawCmdBuffers.data()));
    }

    void createSynchronizationPrimitives()
    {
        // Wait fences to sync command buffer access
        VkFenceCreateInfo fenceCI{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT
        };
        for (auto& fence : waitFences)
        {
            VK_CHECK_RESULT(vkCreateFence(device, &fenceCI, nullptr, &fence));
        }

        // Used to ensure that image presentation is complete before starting to submit again
        VkSemaphoreCreateInfo semaphoreCI{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        for (auto& semaphore : presentCompleteSemaphores)
        {
            VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCI, nullptr, &semaphore));
        }

        // Used to ensure that all commands submitted have been finished before submitting the image to the queue
        renderCompleteSemaphores.resize(swapchainImages.size());
        for (auto& semaphore : renderCompleteSemaphores)
        {
            VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCI, nullptr, &semaphore));
        }
    }

    void setupDepthStencil()
    {
        VkImageCreateInfo imageCI{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = depthFormat,
            .extent = { width, height, 1 },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
        };
        VK_CHECK_RESULT(vkCreateImage(device, &imageCI, nullptr, &depthStencil.image));
        VkMemoryRequirements memReqs{};
        vkGetImageMemoryRequirements(device, depthStencil.image, &memReqs);

        VkMemoryAllocateInfo memAlloc{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memReqs.size,
            .memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        };
        VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &depthStencil.memory));
        VK_CHECK_RESULT(vkBindImageMemory(device, depthStencil.image, depthStencil.memory, 0));

        VkImageViewCreateInfo imageViewCI{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = depthStencil.image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = depthFormat,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };
        VK_CHECK_RESULT(vkCreateImageView(device, &imageViewCI, nullptr, &depthStencil.view));
    }

    void createPipelineCache()
    {
        VkPipelineCacheCreateInfo pipelineCacheCreateInfo{ .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
        VK_CHECK_RESULT(vkCreatePipelineCache(device, &pipelineCacheCreateInfo, nullptr, &pipelineCache));
    }
};

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
        vks::tools::insertImageMemoryBarrier(commandBuffer, depthStencil.image, 0, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VkImageSubresourceRange{ VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1 });

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
        VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, waitFences[currentFrame]));

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

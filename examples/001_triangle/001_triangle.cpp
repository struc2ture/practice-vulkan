#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <volk/volk.h>
#include <vulkan/vulkan.h>
#include <vma/vk_mem_alloc.h>

#include <array>
#include <assert.h>
#include <chrono>
#include <iostream>
#include <format>
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

    VkSwapchainKHR swapchain{ VK_NULL_HANDLE };
    std::vector<VkImage> swapchainImages{};
    std::vector<VkImageView> swapchainImageViews{};
    uint32_t queueFamilyIndex{ 0 };
    VkQueue queue{ VK_NULL_HANDLE };

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
    VkFormat swapchainImageFormat{ VK_FORMAT_UNDEFINED };
    VkFormat depthFormat{ VK_FORMAT_UNDEFINED };
    struct {
        VkImage image;
        VkDeviceMemory memory;
        VkImageView view;
    } depthStencil{};
    VkRenderPass renderPass{ VK_NULL_HANDLE };
    VkPipelineCache pipelineCache{ VK_NULL_HANDLE };
    std::vector<VkFramebuffer> frameBuffers{};
    bool prepared{ false };

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

    void render()
    {

    }

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

    void setupRenderPass()
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
    void createPipelineCache()
    {
        VkPipelineCacheCreateInfo pipelineCacheCreateInfo{ .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
        VK_CHECK_RESULT(vkCreatePipelineCache(device, &pipelineCacheCreateInfo, nullptr, &pipelineCache));
    }

    void setupFrameBuffer()
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

private:
    std::array<VkFence, maxConcurrentFrames> exampleWaitFences{};
    std::array<VkSemaphore, maxConcurrentFrames> examplePresentCompleteSemaphores{};
    std::vector<VkSemaphore> exampleRenderCompleteSemaphores{};
    VkCommandPool exampleCommandPool{ VK_NULL_HANDLE };
    std::array<VkCommandBuffer, maxConcurrentFrames> exampleCommandBuffers{};

public:
    void prepare() override
    {
        VulkanExampleBase::prepare();
        createExampleSynchronizationPrimitives();
        createExampleCommandBuffers();
        createVertexBuffer();
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

#include "example_base.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>
#include <volk/volk.h>
#include <vma/vk_mem_alloc.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <chrono>
#include <string>
#include <vector>

#include "camera.h"
#include "VulkanTools.h"

VulkanExampleBase::VulkanExampleBase()
{
}

VulkanExampleBase::~VulkanExampleBase()
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

bool VulkanExampleBase::init()
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

    std::string windowTitle = getWindowTitle();
    window = SDL_CreateWindow(windowTitle.c_str(), width, height, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    assert(window);

    int w, h;
    assert(SDL_GetWindowSize(window, &w, &h));
    width = static_cast<uint32_t>(w);
    height = static_cast<uint32_t>(h);

    assert(vks::tools::getSupportedDepthFormat(physicalDevice, &depthFormat));

    return true;
}

void VulkanExampleBase::prepare()
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

void VulkanExampleBase::renderLoop()
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
            if (event.type == SDL_EVENT_KEY_DOWN)
            {
                switch (event.key.key)
                {
                case SDLK_P:
                    paused = !paused;
                    break;
                case SDLK_F1:
                    // UI visible
                    break;
                case SDLK_F2:
                    if (camera.type == Camera::CameraType::lookat)
                    {
                        camera.type = Camera::CameraType::firstperson;
                    }
                    else
                    {
                        camera.type = Camera::CameraType::lookat;
                    }
                    break;
                case SDLK_ESCAPE:
                    quit = true;
                    break;
                }

                if (camera.type == Camera::firstperson)
                {
                    switch (event.key.key)
                    {
                    case SDLK_W:
                        camera.keys.up = true;
                        break;
                    case SDLK_S:
                        camera.keys.down = true;
                        break;
                    case SDLK_A:
                        camera.keys.left = true;
                        break;
                    case SDLK_D:
                        camera.keys.right = true;
                        break;
                    }
                }
            }
            if (event.type == SDL_EVENT_KEY_UP)
            {
                if (camera.type == Camera::firstperson)
                {
                    switch (event.key.key)
                    {
                    case SDLK_W:
                        camera.keys.up = false;
                        break;
                    case SDLK_S:
                        camera.keys.down = false;
                        break;
                    case SDLK_A:
                        camera.keys.left = false;
                        break;
                    case SDLK_D:
                        camera.keys.right = false;
                        break;
                    }
                }
            }

            if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN)
            {
                switch (event.button.button)
                {
                case SDL_BUTTON_LEFT:
                    mouseState.buttons.left = true;
                    break;
                case SDL_BUTTON_MIDDLE:
                    mouseState.buttons.middle = true;
                    break;
                case SDL_BUTTON_RIGHT:
                    mouseState.buttons.right = true;
                    break;
                }
            }
            if (event.type == SDL_EVENT_MOUSE_BUTTON_UP)
            {
                switch (event.button.button)
                {
                case SDL_BUTTON_LEFT:
                    mouseState.buttons.left = false;
                    break;
                case SDL_BUTTON_MIDDLE:
                    mouseState.buttons.middle = false;
                    break;
                case SDL_BUTTON_RIGHT:
                    mouseState.buttons.right = false;
                    break;
                }
            }

            if (event.type == SDL_EVENT_MOUSE_MOTION)
            {
                handleMouseMove(event.motion.x, event.motion.y);
            }

            if (event.type == SDL_EVENT_MOUSE_WHEEL)
            {
                camera.translate(glm::vec3(0.0f, 0.0f, (float)event.wheel.y * 0.005f));
            }

            if (event.type == SDL_EVENT_WINDOW_RESIZED)
            {
                int w, h;
                SDL_GetWindowSize(window, &w, &h);
                width = static_cast<uint32_t>(w);
                height = static_cast<uint32_t>(h);
                windowResize();
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

uint32_t VulkanExampleBase::getMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties, VkBool32* memTypeFound) const
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

void VulkanExampleBase::setupRenderPass()
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

void VulkanExampleBase::setupFrameBuffer()
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

void VulkanExampleBase::windowResize()
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
        vkDestroyFramebuffer(device, frameBuffer, nullptr);
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

void VulkanExampleBase::windowResized() {}

void VulkanExampleBase::handleMouseMove(float x, float y)
{
    float dx = mouseState.position.x - x;
    float dy = mouseState.position.y - y;

    bool handled = false;

    // Check if handled by imgui
    // ...
    mouseMoved(x, y, handled);

    if (handled)
    {
        mouseState.position = glm::vec2{ x, y };
        return;
    }

    if (mouseState.buttons.left)
    {
        camera.rotate(glm::vec3(dy * camera.rotationSpeed, -dx * camera.rotationSpeed, 0.0f));
    }
    if (mouseState.buttons.right)
    {
        camera.translate(glm::vec3{ -0.0f, 0.0f, dy * 0.005f });
    }
    if (mouseState.buttons.middle)
    {
        camera.translate(glm::vec3(-dx * 0.005f, -dy * 0.05f, 0.0f));
    }
    mouseState.position = glm::vec2{ x, y };
}

void VulkanExampleBase::mouseMoved(float x, float y, bool handled) {}

void VulkanExampleBase::nextFrame()
{
    auto tStart = std::chrono::high_resolution_clock::now();
    render();
    frameCounter++;
    auto tEnd = std::chrono::high_resolution_clock::now();

    auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

    frameTimer = (float)tDiff / 1000.0f;
    camera.update(frameTimer);

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

        std::string windowTitle = getWindowTitle();
        SDL_SetWindowTitle(window, windowTitle.c_str());

        frameCounter = 0;
        lastTimestamp = tEnd;
    }
}

std::string VulkanExampleBase::getWindowTitle() const
{
    std::string windowTitle{ title + " - " + deviceProperties.deviceName + " - " + std::to_string(frameCounter) + " fps" };
    return windowTitle;
}

VkResult VulkanExampleBase::createInstance()
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

void VulkanExampleBase::createSurface()
{
    assert(SDL_Vulkan_CreateSurface(window, instance, nullptr, &surface));
    VK_CHECK_RESULT(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &surfaceCaps));
}

void VulkanExampleBase::createCommandPool()
{
    VkCommandPoolCreateInfo cmdPoolInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = queueFamilyIndex
    };
    VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &cmdPool));
}

void VulkanExampleBase::createSwapChain()
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

void VulkanExampleBase::createCommandBuffers()
{
    VkCommandBufferAllocateInfo cmdBufferAI{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = cmdPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = static_cast<uint32_t>(drawCmdBuffers.size())
    };
    VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufferAI, drawCmdBuffers.data()));
}

void VulkanExampleBase::createSynchronizationPrimitives()
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

void VulkanExampleBase::setupDepthStencil()
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

void VulkanExampleBase::createPipelineCache()
{
    VkPipelineCacheCreateInfo pipelineCacheCreateInfo{ .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
    VK_CHECK_RESULT(vkCreatePipelineCache(device, &pipelineCacheCreateInfo, nullptr, &pipelineCache));
}

#pragma once

#include "example_base.h"

#include <SDL3/SDL.h>
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

    Camera camera;

    struct
    {
        struct
        {
            bool left{ false };
            bool right{ false };
            bool middle{ false };
        } buttons;
        glm::vec2 position;
    } mouseState;

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
    VulkanExampleBase();
    virtual ~VulkanExampleBase();
    bool init();
    virtual void prepare();
    void renderLoop();
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
    uint32_t getMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties, VkBool32* memTypeFound = nullptr) const;
    virtual void setupRenderPass();
    virtual void setupFrameBuffer();
    void windowResize();
    virtual void windowResized();
    void handleMouseMove(float x, float y);
    virtual void mouseMoved(float x, float y, bool handled);

private:
    void nextFrame();
    std::string getWindowTitle() const;
    VkResult createInstance();
    void createSurface();
    void createCommandPool();
    void createSwapChain();
    void createCommandBuffers();
    void createSynchronizationPrimitives();
    void setupDepthStencil();
    void createPipelineCache();
};

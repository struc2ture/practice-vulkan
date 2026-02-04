#include <vulkan/vulkan.h>

#include <string>
#include <iostream>

#define VK_CHECK_RESULT(f)																				\
{																										\
	VkResult res = (f);																					\
	if (res != VK_SUCCESS)																				\
	{																									\
		std::cout << "Fatal : VkResult is \"" << vks::tools::errorString(res) << "\" in " << __FILE__ << " at line " << __LINE__ << "\n"; \
		assert(res == VK_SUCCESS);																		\
	}																									\
}

namespace vks
{
	namespace tools
	{
		std::string errorString(VkResult errorCode);
		void exitFatal(const std::string& message, int32_t exitCode);
		void exitFatal(const std::string& message, VkResult resultCode);
	}
}
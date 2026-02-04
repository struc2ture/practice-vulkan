file(GLOB BASE_SRC "*.cpp" "*.hpp" "*.h" "../external/imgui/*.cpp")
file(GLOB BASE_HEADERS "*.hpp" "*.h")

add_library(base STATIC ${BASE_SRC})
target_compile_features(base PRIVATE cxx_std_20)
target_compile_definitions(base PRIVATE VK_NO_PROTOTYPES)
target_compile_definitions(base PRIVATE _CRT_SECURE_NO_WARNINGS)
target_link_libraries(base PRIVATE ${SDL_LIBRARY} ktx ${Slang_LIBRARY})
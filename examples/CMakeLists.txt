set(EXAMPLES
    000_test
)

function (buildExample EXAMPLE_NAME)
    set(EXAMPLE_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_NAME})

    message(STATUS "Generating project file for example in ${EXAMPLE_FOLDER}")

    file(GLOB SOURCE *.cpp ${BASE_HEADERS} ${EXAMPLE_FOLDER}/*.cpp)
    file(GLOB SHADERS ${EXAMPLE_FOLDER}/*.slang)

    set(MAIN_CPP ${EXAMPLE_FOLDER}/${EXAMPLE_NAME}.cpp)
    if(EXISTS ${EXAMPLE_FOLDER}/main.cpp)
        set(MAIN_CPP ${EXAMPLE_FOLDER}/main.cpp)
    endif()

    if(EXISTS ${EXAMPLE_FOLDER}/${EXAMPLE_NAME}.h)
        set(MAIN_HEADER ${EXAMPLE_FOLDER}/${EXAMPLE_NAME}.h)
    endif()

	source_group("Shaders" FILES ${SHADERS})

    add_executable(${EXAMPLE_NAME} ${MAIN_CPP} ${SOURCE} ${MAIN_HEADER} ${SHADERS})
    target_compile_definitions(${EXAMPLE_NAME} PRIVATE VK_NO_PROTOTYPES)
    target_compile_definitions(${EXAMPLE_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)
    set_target_properties(${EXAMPLE_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    target_compile_features(${EXAMPLE_NAME} PRIVATE cxx_std_20)

    target_link_libraries(${EXAMPLE_NAME} PRIVATE base ${SDL_LIBRARY} ktx ${Slang_LIBRARY})
endfunction(buildExample)

function(buildExamples)
    foreach(EXAMPLE ${EXAMPLES})
        buildExample(${EXAMPLE})
    endforeach(EXAMPLE)
endfunction(buildExamples)

buildExamples()
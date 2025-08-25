# cmake/Documentation.cmake
# Documentation generation with Doxygen

if(KPU_BUILD_DOCS)
    find_package(Doxygen REQUIRED dot)
    
    if(DOXYGEN_FOUND)
        # Doxygen configuration
        set(DOXYGEN_PROJECT_NAME "Stillwater KPU Simulator")
        set(DOXYGEN_PROJECT_NUMBER ${PROJECT_VERSION})
        set(DOXYGEN_PROJECT_BRIEF "Knowledge Processing Unit Simulator")
        set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/docs)
        set(DOXYGEN_INPUT "${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/components")
        set(DOXYGEN_FILE_PATTERNS "*.hpp *.cpp *.h *.c *.md")
        set(DOXYGEN_RECURSIVE YES)
        set(DOXYGEN_EXCLUDE_PATTERNS "*/third_party/*" "*/build/*")
        set(DOXYGEN_GENERATE_HTML YES)
        set(DOXYGEN_GENERATE_LATEX NO)
        set(DOXYGEN_HTML_OUTPUT html)
        set(DOXYGEN_USE_MDFILE_AS_MAINPAGE ${CMAKE_SOURCE_DIR}/README.md)
        set(DOXYGEN_EXTRACT_ALL YES)
        set(DOXYGEN_EXTRACT_PRIVATE YES)
        set(DOXYGEN_EXTRACT_STATIC YES)
        set(DOXYGEN_SOURCE_BROWSER YES)
        set(DOXYGEN_INLINE_SOURCES YES)
        set(DOXYGEN_STRIP_CODE_COMMENTS NO)
        set(DOXYGEN_REFERENCED_BY_RELATION YES)
        set(DOXYGEN_REFERENCES_RELATION YES)
        set(DOXYGEN_CALL_GRAPH YES)
        set(DOXYGEN_CALLER_GRAPH YES)
        set(DOXYGEN_HAVE_DOT YES)
        set(DOXYGEN_DOT_NUM_THREADS 0)
        set(DOXYGEN_UML_LOOK YES)
        set(DOXYGEN_TEMPLATE_RELATIONS YES)
        set(DOXYGEN_INCLUDE_GRAPH YES)
        set(DOXYGEN_INCLUDED_BY_GRAPH YES)
        set(DOXYGEN_CLASS_DIAGRAMS YES)
        set(DOXYGEN_COLLABORATION_GRAPH YES)
        set(DOXYGEN_GROUP_GRAPHS YES)
        set(DOXYGEN_DIRECTORY_GRAPH YES)
        
        # Create Doxygen target
        doxygen_add_docs(docs
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/components
            ${CMAKE_SOURCE_DIR}/README.md
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Generating API documentation with Doxygen"
        )
        
        # Install documentation
        install(DIRECTORY ${CMAKE_BINARY_DIR}/docs/html/
            DESTINATION ${CMAKE_INSTALL_DOCDIR}
            OPTIONAL
        )
    endif()
endif()

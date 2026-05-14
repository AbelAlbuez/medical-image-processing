CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

## Project name
PROJECT(ImageRegistration)

## Packages
FIND_PACKAGE(ITK REQUIRED)
INCLUDE(${ITK_USE_FILE})

## Do not allow to build inside the source tree
IF(PROJECT_BINARY_DIR STREQUAL ${PROJECT_SOURCE_DIR})
  MESSAGE(FATAL_ERROR "Building in the source tree is not allowed")
ENDIF(PROJECT_BINARY_DIR STREQUAL ${PROJECT_SOURCE_DIR})

## Where to put executables and libs
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR})
MARK_AS_ADVANCED(
  CMAKE_BACKWARDS_COMPATIBILITY
  EXECUTABLE_OUTPUT_PATH
  LIBRARY_OUTPUT_PATH
  )

## Executables
ADD_EXECUTABLE(ImageRegistration1 ImageRegistration1.cxx )
  TARGET_LINK_LIBRARIES(ImageRegistration1 ${ITK_LIBRARIES})
  
ADD_EXECUTABLE(ImageRegistration2 ImageRegistration2.cxx )
  TARGET_LINK_LIBRARIES(ImageRegistration2 ${ITK_LIBRARIES})
  
ADD_EXECUTABLE(ImageRegistration5 ImageRegistration5.cxx )
  TARGET_LINK_LIBRARIES(ImageRegistration5 ${ITK_LIBRARIES})

ADD_EXECUTABLE(ImageRegistration8 ImageRegistration8.cxx )
  TARGET_LINK_LIBRARIES(ImageRegistration8 ${ITK_LIBRARIES})























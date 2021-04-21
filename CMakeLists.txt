cmake_minimum_required(VERSION 3.13.4)

project(SlicerBS)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/BrainSegment")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Mehran Azimbagirad, Fabrício H Simozo, Antonio CS Senra Filho, Luiz O Murta Junior (CSIM)")
set(EXTENSION_DESCRIPTION "This is a Brain segmentation module")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/BrainSegment.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/BrainSegment/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(BS)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})

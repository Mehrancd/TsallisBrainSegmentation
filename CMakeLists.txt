cmake_minimum_required(VERSION 3.13.4)

project(TsallisBrainSegmentation)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "http://slicer.org/slicerWiki/index.php/Documentation/Nightly/Extensions/TsallisBrainSegmentation")
set(EXTENSION_CATEGORY "Examples")
set(EXTENSION_CONTRIBUTORS "Mehran Azimbagirad (CSIM Lab), Fabricio Simozo (CSIM LAb), A C S S Filho, Luiz O M Jr")
set(EXTENSION_DESCRIPTION "A Slicer module for brain extraction and segmentation")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/TsallisBrainSegmentation.png")
set(EXTENSION_SCREENSHOTURLS "https://github.com/Mehrancd/TsallisBrainSegmentation/blob/main/Screenshot1.jpg")
set(EXTENSION_DEPENDS "NA") # Specified as a space separated string, a list or 'NA' if any

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(TsallisBrainSegmentation)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})

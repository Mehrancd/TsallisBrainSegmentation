cmake_minimum_required(VERSION 3.13.4)

project(SlicerBS)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/Mehrancd/SlicerBS")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Mehran Azimbagirad, Fabrício H Simozo, Antonio CS Senra Filho, Luiz O Murta Junior (CSIM)")
set(EXTENSION_DESCRIPTION "This is a Brain segmentation module")
set(EXTENSION_ICONURL "https://github.com/Mehrancd/SlicerBS/tree/main/BS/Resources/Icons/BS.png")
set(EXTENSION_SCREENSHOTURLS "https://github.com/Mehrancd/SlicerBS/Screenshot1.jpg https://github.com/Mehrancd/SlicerBS/Screenshot2.jpg")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SlicerBS)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})

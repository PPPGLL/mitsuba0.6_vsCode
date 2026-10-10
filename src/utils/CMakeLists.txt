# Utilities
if (NOT MTS_VERSION)
  message(FATAL_ERROR "Use the top level configuration file")
endif()

# macro(add_utility)
#   add_mts_plugin(${ARGN} TYPE utility)
# endmacro()

include_directories(${ILMBASE_INCLUDE_DIRS})

find_package(HDF5 REQUIRED COMPONENTS C CXX HL)

if (HDF5_FOUND)
    message(STATUS "HDF5 found: ${HDF5_INCLUDE_DIRS}, ${HDF5_LIBRARIES}")
else()
    message(FATAL_ERROR "HDF5 not found!")
endif()


# add_utility(addimages      addimages.cpp)
# add_utility(joinrgb        joinrgb.cpp)
# add_utility(cylclip        cylclip.cpp MTS_HW)
# add_utility(kdbench        kdbench.cpp)
# add_utility(tonemap        tonemap.cpp)
# add_utility(show_pdf_BSDF show_pdf_BSDF.cpp)
# add_utility(sample_BSDF sample_BSDF.cpp 
#            LINK_LIBRARIES ${HDF5_LIBRARIES} ${HDF5_CXX_LIBRARIES})
# target_include_directories(sample_BSDF PRIVATE ${HDF5_INCLUDE_DIRS})

# add_utility(sample_microstrcture sample_microstrcture.cpp 
#            LINK_LIBRARIES ${HDF5_LIBRARIES} ${HDF5_CXX_LIBRARIES})
# target_include_directories(sample_microstrcture PRIVATE ${HDF5_INCLUDE_DIRS})


add_library(addimages SHARED "addimages.cpp")
add_library(joinrgb   SHARED "joinrgb.cpp"  )
add_library(cylclip   SHARED "cylclip.cpp"  )
add_library(kdbench   SHARED "kdbench.cpp"  )
add_library(tonemap   SHARED "tonemap.cpp"  )
target_link_libraries(addimages mitsuba-python)
target_link_libraries(joinrgb mitsuba-python)
target_link_libraries(cylclip mitsuba-python)
target_link_libraries(kdbench getopt mitsuba-python)
target_link_libraries(tonemap getopt mitsuba-python)

set_target_properties(addimages PROPERTIES FOLDER "plugins")
set_target_properties(joinrgb PROPERTIES FOLDER "plugins")
set_target_properties(cylclip PROPERTIES FOLDER "plugins")
set_target_properties(kdbench PROPERTIES FOLDER "plugins")
set_target_properties(tonemap PROPERTIES FOLDER "plugins")

#不需要 HDF5
add_library(timecost SHARED "timecost.cpp")
target_link_libraries(timecost mitsuba-python mitsuba-core getopt)
set_target_properties(timecost PROPERTIES FOLDER "plugins")

add_library(reflectance_wetspongecake SHARED "reflectance_wetspongecake.cpp")
target_link_libraries(reflectance_wetspongecake mitsuba-python mitsuba-core getopt)
set_target_properties(reflectance_wetspongecake PROPERTIES FOLDER "plugins")

# show_pdf_BSDF 不需要 HDF5
add_library(show_pdf_BSDF SHARED "show_pdf_BSDF.cpp")
target_link_libraries(show_pdf_BSDF mitsuba-python mitsuba-core getopt)
set_target_properties(show_pdf_BSDF PROPERTIES FOLDER "plugins")

# sample_BSDF 需要 HDF5
add_library(sample_BSDF SHARED "sample_BSDF.cpp")
target_link_libraries(sample_BSDF
    PRIVATE
        mitsuba-python
        mitsuba-core
        getopt
        ${HDF5_LIBRARIES}
        ${HDF5_CXX_LIBRARIES}
)
target_include_directories(sample_BSDF PRIVATE ${HDF5_INCLUDE_DIRS})
set_target_properties(sample_BSDF PROPERTIES FOLDER "plugins")

# sample_microstrcture 也需要 HDF5
add_library(sample_microstrcture SHARED "sample_microstrcture.cpp")
target_link_libraries(sample_microstrcture
    PRIVATE
        mitsuba-python
        mitsuba-core
        getopt
        ${HDF5_LIBRARIES}
        ${HDF5_CXX_LIBRARIES}
)
target_include_directories(sample_microstrcture PRIVATE ${HDF5_INCLUDE_DIRS})
set_target_properties(sample_microstrcture PROPERTIES FOLDER "plugins")

# show_pdf_BSDF 不需要 HDF5
add_library(bsdfprofile SHARED "bsdfprofile.cpp")
target_link_libraries(bsdfprofile
    PRIVATE
        mitsuba-python
        mitsuba-core
        getopt
        ${HDF5_LIBRARIES}
        ${HDF5_CXX_LIBRARIES}
)
target_link_libraries(bsdfprofile PRIVATE ${HDF5_INCLUDE_DIRS})
set_target_properties(bsdfprofile PROPERTIES FOLDER "plugins")

# sample_microstrcture_sss 也需要 HDF5
add_library(sample_microstrcture_sss SHARED "sample_microstrcture_sss.cpp")
target_link_libraries(sample_microstrcture_sss
    PRIVATE
        mitsuba-python
        mitsuba-core
        getopt
        ${HDF5_LIBRARIES}
        ${HDF5_CXX_LIBRARIES}
)
target_include_directories(sample_microstrcture_sss PRIVATE ${HDF5_INCLUDE_DIRS})
set_target_properties(sample_microstrcture_sss PROPERTIES FOLDER "plugins")

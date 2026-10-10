add_library(homogeneous SHARED maxexp.h homogeneous.cpp)
add_library(heterogeneous SHARED heterogeneous.cpp)

target_link_libraries(homogeneous mitsuba-python)
target_link_libraries(heterogeneous mitsuba-python)

set_target_properties(homogeneous PROPERTIES FOLDER "plugins")
set_target_properties(heterogeneous PROPERTIES FOLDER "plugins")


add_library(homogeneous_micro_volume_clamped SHARED homogeneous_micro_volume_clamped.cpp)
target_link_libraries(homogeneous_micro_volume_clamped mitsuba-python)
set_target_properties(homogeneous_micro_volume_clamped PROPERTIES FOLDER "plugins")

add_library(hapke_Ours SHARED hapke_Ours.cpp)
target_link_libraries(hapke_Ours mitsuba-python)
set_target_properties(hapke_Ours PROPERTIES FOLDER "plugins")
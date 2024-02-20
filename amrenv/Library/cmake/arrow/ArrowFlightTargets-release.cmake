#----------------------------------------------------------------
# Generated CMake target import file for configuration "RELEASE".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "arrow_flight_shared" for configuration "RELEASE"
set_property(TARGET arrow_flight_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(arrow_flight_shared PROPERTIES
  IMPORTED_IMPLIB_RELEASE "C:/Users/pavan/Downloads/E2E_DS_Projects/AMR_Prediction/amrenv/Library/arrow_flight.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/arrow_flight.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS arrow_flight_shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_arrow_flight_shared "C:/Users/pavan/Downloads/E2E_DS_Projects/AMR_Prediction/amrenv/Library/arrow_flight.lib" "${_IMPORT_PREFIX}/bin/arrow_flight.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)

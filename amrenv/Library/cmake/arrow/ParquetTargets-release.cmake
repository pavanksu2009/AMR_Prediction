#----------------------------------------------------------------
# Generated CMake target import file for configuration "RELEASE".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "parquet_shared" for configuration "RELEASE"
set_property(TARGET parquet_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(parquet_shared PROPERTIES
  IMPORTED_IMPLIB_RELEASE "C:/Users/pavan/Downloads/E2E_DS_Projects/AMR_Prediction/amrenv/Library/parquet.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/parquet.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS parquet_shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_parquet_shared "C:/Users/pavan/Downloads/E2E_DS_Projects/AMR_Prediction/amrenv/Library/parquet.lib" "${_IMPORT_PREFIX}/bin/parquet.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)

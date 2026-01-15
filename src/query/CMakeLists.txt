# PyFlare Query API Library

add_library(pyflare_query STATIC
    api.cpp
    sql_parser.cpp
)

add_library(PyFlare::Query ALIAS pyflare_query)

target_include_directories(pyflare_query
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/pyflare/query>
)

target_link_libraries(pyflare_query
    PUBLIC
        PyFlare::Common
        PyFlare::Storage
)

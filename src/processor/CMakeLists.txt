# PyFlare Stream Processor Library

add_library(pyflare_processor STATIC
    drift/drift_detector.cpp
    drift/embedding_drift.cpp
    drift/feature_drift.cpp
    eval/evaluator.cpp
    eval/hallucination.cpp
    rca/analyzer.cpp
    cost/tracker.cpp
)

add_library(PyFlare::Processor ALIAS pyflare_processor)

target_include_directories(pyflare_processor
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/pyflare/processor>
)

target_link_libraries(pyflare_processor
    PUBLIC
        PyFlare::Common
        PyFlare::Storage
)

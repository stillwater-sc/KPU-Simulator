# Streamer Tests

kpu_add_component_test(streamer_basic_test "iddo"
    test_streamer_basic.cpp
)

# Streamer tracing tests
kpu_add_component_test(streamer_tracing_test "iddo"
    test_streamer_tracing.cpp
)
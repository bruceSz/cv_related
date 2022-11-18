load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "hello-world",
    srcs = ["hello-world.cc"],
)

cc_test(
    name = "basic_test",
    srcs = ["basic_test.cc"],
    deps = [
        "@com_google_googletest//:gtest_main",
    ]
)

cc_binary(
    name = "img_pyramid",
    srcs = ["img_pyramid.cc"],
    deps = [
        "@opencv//:opencv",
        "@fmt//:fmt",
    ]
)
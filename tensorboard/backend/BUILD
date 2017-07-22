# Description:
# TensorBoard, a dashboard for investigating TensorFlow

package(default_visibility = ["//tensorboard:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "http_util",
    srcs = ["http_util.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":json_util",
        ":compat",
        "@org_pocoo_werkzeug//:werkzeug",
        "@six_archive//:six",
    ],
)

py_test(
    name = "http_util_test",
    size = "small",
    srcs = ["http_util_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":http_util",
        ":compat",
        "@org_pocoo_werkzeug//:werkzeug",
        "@six_archive//:six",
    ],
)

py_library(
    name = "json_util",
    srcs = ["json_util.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":compat",
    ],
)

py_test(
    name = "json_util_test",
    size = "small",
    srcs = ["json_util_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":json_util",
        ":compat",
    ],
)

py_library(
    name = "application",
    srcs = ["application.py"],
    data = ["//tensorboard:webfiles.zip"],
    srcs_version = "PY2AND3",
    deps = [
        ":http_util",
        ":compat",
        "//tensorboard:expect_sqlite3_installed",
        "//tensorboard/backend/event_processing:event_accumulator",
        "//tensorboard/backend/event_processing:event_multiplexer",
        "//tensorboard/plugins/core:core_plugin",
        "@org_pocoo_werkzeug//:werkzeug",
        "@six_archive//:six",
    ],
)

py_test(
    name = "application_test",
    size = "small",
    srcs = ["application_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":application",
        ":compat",
        "//tensorboard",
        "//tensorboard/backend/event_processing:event_multiplexer",
        "//tensorboard/plugins:base_plugin",
        "@org_pocoo_werkzeug//:werkzeug",
    ],
)

py_library(
    name = "process_graph",
    srcs = ["process_graph.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":compat",
    ],
)

py_library(
    name = "flags",
    srcs = ["flags.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
    ],
)

py_library(
    name = "app",
    srcs = ["app.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":flags",
    ],
)

py_library(
    name = "compat",
    srcs = ["compat.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "@six_archive//:six",
        "//tensorboard:expect_numpy_installed",
    ],
)
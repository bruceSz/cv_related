# TODO(xiao) Further clean up to use http archive, after removing the 3rd party source
# we would be able to drop minimal number of bzl files into individual repositories
# and do something like the following and just call the function at top
# load("//third_party/benchmark:repo.bzl", "third_party_repository_benchmark")

## Load the rules_cc_package
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")


TOOLCHAIN_BASE = "/opt/toolchains"
CUDA_TOOLS = TOOLCHAIN_BASE + "/cuda-11.4"
RTI_BASE = TOOLCHAIN_BASE + "/rti_connext_dds-6.0.0"
RTI_6_1_0_BASE = TOOLCHAIN_BASE + "/rti_connext_dds-6.1.0"
LIB_RTI_JVM = RTI_BASE + "/resource/app/jre/x64Linux/lib/amd64/server"
LIB_RTI_JVM_6_1_0 = RTI_6_1_0_BASE + "/resource/app/jre/x64Linux/lib/server"
NVIDIA_BASE_PATH = TOOLCHAIN_BASE + "/drive-linux"
NVIDIA_TOOLCHAIN_PATH = TOOLCHAIN_BASE + "/targetfs-pdk6/usr/lib"
TOOLCHAIN_LIB_PATH = TOOLCHAIN_BASE + "/targetfs-pdk6/usr/lib"
NVIDIA_TARGETFS_PATH = TOOLCHAIN_BASE + "/targetfs-pdk6/"
ARTIFACTORY_URL = "https://artifactory.xiaopeng.us/artifactory/"
ARTIFACTORY_URL_CN = "https://artifactory-gz.xiaopeng.us/artifactory/"

# toolchains
CLANG_BASE_PATH = TOOLCHAIN_BASE + "/clang"
# TODO(FanYe), dummy folder, will be repaced later.
GCC_X86_BASE_PATH = "/usr/"
GCC_AARCH64_BASE_PATH = TOOLCHAIN_BASE + "/aarch64--glibc--stable-2020.08-1"

PATH_NAMES = struct(
    lib_rti_jvm = LIB_RTI_JVM,
    lib_rti_jvm_6_1_0 = LIB_RTI_JVM,
    nddshome = RTI_BASE,
    nddshome_6_1_0 = RTI_6_1_0_BASE,
    nvidia_base_path = NVIDIA_BASE_PATH,
    cuda_tools = CUDA_TOOLS,
    nvidia_toolchain_lib_path = NVIDIA_TOOLCHAIN_PATH,
    toolchain_lib_path = TOOLCHAIN_LIB_PATH,
    gcc_aarch64_base_path = GCC_AARCH64_BASE_PATH,
    nvidia_targetfs_path = NVIDIA_TARGETFS_PATH,
)

# This is the position for those WORKSPACE that included this bzl.
# This why this file name is called xrepo_third_party


THIRD_PARTY_LOCATION = "../third_party"
# TODO change to relative path later
TOOLCHAIN_LOCATION = "/opt/toolchains"
TOOLCHAIN_CLANG_LOCATION = "/opt/toolchains_clang"
TOOLCHAIN_GCC_X86_LOCATION = "/opt/toolchains"

#XDDS_THIRD_PARTY_LOCATION = "../xos/xdds/third_party"
#XDDS_THIRD_PARTY_LOCATION_PRODUCT_LINE_SCHEME = "xos/xdds/third_party"

# The return value is the third party location in relative to
# the WORKSPACE that needed  these 3rd libraries
# for example ../third_party/sqlite
# this is the location in relative to the other WORKSPACE


def THIRD_PARTY_ROOT(repo_location, root = THIRD_PARTY_LOCATION):
    return root + "/" + repo_location

def TOOLCHAIN_CLANG_ROOT(repo_location, root = TOOLCHAIN_CLANG_LOCATION):
    return root + "/" + repo_location

def TOOLCHAIN_ROOT(repo_location, root = TOOLCHAIN_LOCATION):
    return root + "/" + repo_location

def MSG_ROOT_WITH_PRODUCT_LINE_SCHEME(repo_location,REPO_WKSP_REL_TO_REPO_ROOT):
    return REPO_WKSP_REL_TO_REPO_ROOT + repo_location


def THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME(repo_location,REPO_WKSP_REL_TO_REPO_ROOT):
    return REPO_WKSP_REL_TO_REPO_ROOT + "third_party/" + repo_location

#def XDDS_THIRD_PARTY_ROOT(repo_location, root = XDDS_THIRD_PARTY_LOCATION):
#    return root + "/" + repo_location

def XDDS_THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME(repo_location,REPO_WKSP_REL_TO_REPO_ROOT):
    return REPO_WKSP_REL_TO_REPO_ROOT+ "xos/xdds/third_party/" + repo_location


def TOOLCHAIN_GCC_X86_ROOT(repo_location, root = TOOLCHAIN_GCC_X86_LOCATION):
    return root + "/" + repo_location

# we can do as much as we could in bzl file
# but some of the load and execute has to be done in the workspace
# if repo root is /repos/sandbox/
# if repo is      /repos/sandbox/localization
# then REPO_WKSP_REL_TO_REPO_ROOT is ../
# if repo is      /repos/sandbox/xp40/localization
# then REPO_WKSP_REL_TO_REPO_ROOT is ../../
# each repo WORKSPACE will define this and pass it in here
# it has a default value so that SOP's setup can be accommodated wihtout change
def declare_xrepo_third_party_repositories(REPO_WKSP_REL_TO_REPO_ROOT='../'):
    # rules first
    # this only declares boost rule's repo's http
  
    skylib()
    declare_opencv()
    declare_fmt()
    #tcmalloc
    xpilot_absl()
    
    



def xpilot_xtensor(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.new_local_repository,
        name = "xpilot_xtensor",
        build_file = "@third_party//external_repos_bazelization:xtensor.BUILD",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("xtensor",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_sqlite3(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name="xpilot_sqlite3",
	path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("linux/sqlite3",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_sqlite3_latest(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "xpilot_sqlite3_latest",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("sqlite",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def com_google_protobuf(REPO_WKSP_REL_TO_REPO_ROOT):
    # zlib must be call before proto
    xpilot_zlib(REPO_WKSP_REL_TO_REPO_ROOT)
    maybe(
        native.local_repository,
        name = "com_google_protobuf",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("protobuf/3.14.0",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_zlib(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.new_local_repository,
        name = "zlib",
        build_file  = "@third_party//zlib:BUILD",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("zlib",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_gflags(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "com_github_gflags_gflags",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("gflags",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_glog(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "xpilot_glog",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("glog",REPO_WKSP_REL_TO_REPO_ROOT),
    )

#TODO(xiaoxu) ask planning to get rid of this
def xpilot_glog3(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "xpilot_glog3",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("g3log-1.3.2/src",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_yaml_cpp(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "xpilot_yaml_cpp",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("yaml-cpp",REPO_WKSP_REL_TO_REPO_ROOT),
    )

# TODO(xiaoxu) upgrade to match davinci
def xpilot_eigen(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "xpilot_eigen",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("eigen",REPO_WKSP_REL_TO_REPO_ROOT),
    )

    maybe(
        http_archive,
        name = "xpilot_eigen3_4",
        build_file = "@third_party//external_repos_bazelization:eigen3_4.BUILD",
        sha256 = "1882b009e3f118747a59291c94b7d494055100d59f5e8f7d8ba52f0a464f826c",
        strip_prefix = "eigen-a1e1612c287dd68dc8c836e410499711ea92d822",
        urls = [
            "https://gitlab.com/libeigen/eigen/-/archive/a1e1612c287dd68dc8c836e410499711ea92d822/eigen-a1e1612c287dd68dc8c836e410499711ea92d822.tar.gz",
        ],
    )

def declare_fmt():
    maybe(
        native.local_repository,
        name = "fmt",
        path = "../third_party/fmt_bazel"
    )


def declare_opencv():
    maybe(
        native.local_repository,
        name = "opencv",
        path = "../third_party/opencv_bazel" 
    )

def xpilot_jsoncpp(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "xpilot_jsoncpp",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("jsoncpp",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_json(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "xpilot_json",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("json",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_rtklib(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "xpilot_rtklib",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("rtklib",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_rtklib_xngp(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "xpilot_rtklib_xngp",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("rtklib_xngp",REPO_WKSP_REL_TO_REPO_ROOT),
    )
    
def xpilot_com_google_googletest(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "com_google_googletest",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("linux/googletest-src",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_alicloud_oss(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "alicloud_oss",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("oss",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_tinyxml(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "xpilot_tinyxml",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("tinyxml",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_linux_openssl(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "linux_openssl",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("openssl_linux",REPO_WKSP_REL_TO_REPO_ROOT),
    )

#TODO(xiaoxu) Ask map team to sync latest
def xpilot_autonavi(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "xpilot_autonavi",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("autonavi_ehp",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_autonavi_xngp(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "xpilot_autonavi_xngp",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("autonavi_ehp_xngp",REPO_WKSP_REL_TO_REPO_ROOT),
)

def xpilot_position_cryptography(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "xpilot_position_cryptography",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("position_cryptography",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_qpOASES(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "xpilot_qpOASES",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("qpOASES-3.2.1",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_osqp(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "xpilot_osqp",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("bz-osqp-0.4.1",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_variant(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "xpilot_variant",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("variant",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_vsomeip(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "vsomeip",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("vsomeip-2.14.16",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_vsomeip3(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "xpilot_vsomeip3",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("vsomeip-3.1.20.3",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_systemd(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "systemd-dev",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("systemd-dev",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_pcap(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "pcap-dev",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("libpcap",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_g2o(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "xpilot_g2o",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("g2o",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def com_github_nelhage_rules_boost():
    maybe(
        git_repository,
        name = "com_github_nelhage_rules_boost",
        commit = "9f9fb8b2f0213989247c9d5c0e814a8451d18d7f",
        remote = "https://github.com/nelhage/rules_boost",
        shallow_since = "1570056263 -0700",
        patch_args = [ "-p1" ],
        patches = [ "@third_party//com_github_nelhage_rules_boost:boost.patch.1.68" ],
    )

def xpilot_pybind11_bazel(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "pybind11_bazel",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("pybind11_bazel",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_pybind11(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "pybind11",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("pybind11",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_pybind11_json(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "xpilot_pybind11_json",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("pybind11_json",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_vision_interface():
    maybe(
        http_archive,
        name = "xpilot_vision_interface",
        build_file = "@//:xpilot_vision_interface.BUILD",
        sha256 = "a0bbd83a0894cb1569446c0219b7401f76ab33fd45ef54c79d0400d07be7deb4",
        urls = [
            ARTIFACTORY_URL + "perception_release/interface/57d5a09/interface.tar.gz",
            ARTIFACTORY_URL_CN + "perception_release/interface/57d5a09/interface.tar.gz",
        ],
    )

def xpilot_compdb(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "compdb",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("bazel-compilation-database",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def skylib():
    maybe(
        http_archive,
        name = "bazel_skylib",
        urls = [
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.1.1/bazel-skylib-1.1.1.tar.gz",
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.1.1/bazel-skylib-1.1.1.tar.gz",
        ],
        sha256 = "c6966ec828da198c5d9adbaa94c05e3a1c7f21bd012a0b29ba8ddbccb2c93b0d",
    )

def xpilot_xtl(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.new_local_repository,
        name = "xpilot_xtl",
        build_file = "@third_party//external_repos_bazelization:xtl.BUILD",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("xtl",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_xsimd(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.new_local_repository,
        name = "xpilot_xsimd",
        build_file = "@third_party//external_repos_bazelization:xsimd.BUILD",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("xsimd",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_qt(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "qt",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("qt5",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_qwt(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "qwt",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("libqwt",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def com_justbuchanan_rules_qt():
    maybe(
        git_repository,
        name = "com_justbuchanan_rules_qt",
        branch = "master",
        remote = "https://github.com/justbuchanan/bazel_rules_qt.git",
    )

def xpilot_glm():
    maybe(
        new_git_repository,
        name = "glm",
        build_file_content = """
package(default_visibility = ["//visibility:public"])
cc_library(
  name= "glm",
  defines= ["GLM_ENABLE_EXPERIMENTAL", "GLM_FORCE_RADIANS",
  "GLM_FORCE_DEPTH_ZERO_TO_ONE"],
  srcs=glob(["glm/**/*.cpp"]) + ["glm/detail/_fixes.hpp"],
  hdrs=glob(["glm/**/*.hpp"])+glob(["glm/**/*.h"]),
  includes = [".", "glm"],
  textual_hdrs = glob(["glm/**/*.inl"]),
  visibility = ["//visibility:public"],
)
        """,
        remote = "https://github.com/g-truc/glm.git",
        tag = "0.9.9.7",
    )

def xpilot_FreeGLUT(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "FreeGLUT",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("freeglut",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_FreeType2(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "freetype2",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("freetype2",REPO_WKSP_REL_TO_REPO_ROOT),
    )


def xpilot_libde265():
    maybe(
        native.local_repository,
        name = "libde265",
        path = THIRD_PARTY_ROOT("libde265_deploy"),
    )
    
def xpilot_libgl(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "libgl",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("libgl",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_assimp(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "assimp",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("libassimp",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def python_linux():
    maybe(
        native.new_local_repository,
        name = "python_linux",
        path = "/usr",
        build_file_content = """
cc_library(
    name = "python27-lib",
    srcs = ["lib/python2.7/config-x86_64-linux-gnu/libpython2.7.so"],
    hdrs = glob(["include/python2.7/*.h"]),
    includes = ["include/python2.7"],
    visibility = ["//visibility:public"]
)
        """,
    )

def xpilot_concurrentqueue(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "xpilot_concurrentqueue",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("concurrentqueue",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def rules_foreign_cc():
    ## Foreign cc rules
    maybe(
        http_archive,
        name = "rules_foreign_cc",
        # TODO: Get the latest sha256 value from a bazel debug message or the latest
        #       release on the releases page: https://github.com/bazelbuild/rules_foreign_cc/releases
        #
        sha256 = "682fa59997d214d42743d822a1284780fd8fb0db4dd88bcb0725904b423cef20",
        strip_prefix = "rules_foreign_cc-3b72ab3468cc8b101352dbe681353a2f8821a057",
        url = "https://github.com/bazelbuild/rules_foreign_cc/archive/3b72ab3468cc8b101352dbe681353a2f8821a057.tar.gz",
    )

def rules_proto():
    maybe(
        http_archive,
        name = "rules_proto",
        sha256 = "602e7161d9195e50246177e7c55b2f39950a9cf7366f74ed5f22fd45750cd208",
        strip_prefix = "rules_proto-97d8af4dc474595af3900dd85cb3a29ad28cc313",
        urls = [
            "https://apollo-system.cdn.bcebos.com/archive/6.0/97d8af4dc474595af3900dd85cb3a29ad28cc313.tar.gz",
            "https://github.com/bazelbuild/rules_proto/archive/97d8af4dc474595af3900dd85cb3a29ad28cc313.tar.gz",
        ],
    )

def rules_python():
    maybe(
        http_archive,
        name = "rules_python",
        sha256 = "b5668cde8bb6e3515057ef465a35ad712214962f0b3a314e551204266c7be90c",
        strip_prefix = "rules_python-0.0.2",
        urls = [
            "https://apollo-system.cdn.bcebos.com/archive/6.0/rules_python-0.0.2.tar.gz",
            "https://github.com/bazelbuild/rules_python/releases/download/0.0.2/rules_python-0.0.2.tar.gz",
        ],
    )

def rules_fastgen():
    maybe(
        http_archive,
        name = "rules_fastgen",
        sha256 = "dc534c9813c6674be1303c8a2a12f6c7790bc47d286205699652abcdc345cec9",
        urls = [
            ARTIFACTORY_URL + "xpilot_dev/E38/xdds_tool/fastgen/1.0.2/20220531_17_13_18/rules_fastgen.tar.gz",
        ],
    )

# latest tag of 0.5.1 from github is still very buggy
# so this commit is from latest master in Nov.
def rules_pkg():
    # must declare rules_python before rules_pkg's load function
    # otherwise it fails
    rules_python()

    # Pkg rules
    maybe(
        http_archive,
        name = "rules_pkg",
        urls = [
            "https://github.com/bazelbuild/rules_pkg/archive/fa8ddc9f3ab0d5a215b5e6b893e34b216a312a36.tar.gz"
        ],
        sha256 = "ff934cdef05365b51955513f1493e2ba550911023f7d0a7e4a1d70e5f3b0188b",
        strip_prefix = "rules_pkg-fa8ddc9f3ab0d5a215b5e6b893e34b216a312a36",
    )

def xpilot_proj(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "proj",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("proj",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_poco(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "poco",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("poco",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_uuid(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.new_local_repository,
        name = "uuid",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("uuid/2.3.6/",REPO_WKSP_REL_TO_REPO_ROOT),
        build_file = "@third_party//uuid/2.3.6:BUILD",
    )

def xpilot_foonathan_memory(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "foonathan_memory",
        path =  XDDS_THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("foonathan_memory",REPO_WKSP_REL_TO_REPO_ROOT),
   )

def xpilot_fastrtps(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "fastrtps",
        path = XDDS_THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("fastrtps",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_fastcdr(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "fastcdr",
        path = XDDS_THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("fastcdr",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_ncurse(REPO_WKSP_REL_TO_REPO_ROOT):
  maybe(
        native.local_repository,
        name = "ncurses",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("ncurses",REPO_WKSP_REL_TO_REPO_ROOT),
  )

def xpilot_ceres():
    maybe(
        git_repository,
        name = "xpilot_ceres",
        commit = "399cda773035d99eaf1f4a129a666b3c4df9d1b1",
        remote = "https://github.com/ceres-solver/ceres-solver",
        patch_args = [ "-p1" ],
        patches = [ "@third_party//ceres_solver:ceres_solver.patch.2.0.0" ],
        shallow_since = "1603478168 +0100",
    )

def xpilot_linuxptp(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "linuxptp",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("linuxptp",REPO_WKSP_REL_TO_REPO_ROOT),
    )

def xpilot_pcl():
    maybe(
      http_archive,
      name = "pcl",
      urls = [ "https://github.com/PointCloudLibrary/pcl/releases/download/pcl-1.11.1/source.tar.gz" ],
      sha256 = '19d1a0bee2bc153de47c05da54fc6feb23393f306ab2dea2e25419654000336e',
      strip_prefix = 'pcl',
      patch_args = [ "-p1" ],
      patches = [ "@third_party//pcl:pcl.1.11.1.patch" ],
      build_file = "@third_party//external_repos_bazelization:pcl.BUILD",
    )



def xpilot_openmpi():
    maybe(
      http_archive,
      name = "xpilot_openmpi",
      urls = [ "https://artifactory.xiaopeng.us/artifactory/xpilot_dev/buildyard/third_party/omp-4.0.3.tar" ],
      sha256 = '98fd96cc1a7d681224c10be82a9e985e8e4c0746e23d1badc9805a554c812d3d',
      strip_prefix = 'omp-4.0.3',
      build_file  = "@third_party//external_repos_bazelization:omp.BUILD",
    )

def xpilot_flann():
    maybe(
      http_archive,
      name = "xpilot_flann",
      urls = [ "https://github.com/flann-lib/flann/archive/refs/tags/1.9.1.tar.gz" ],
      sha256 = 'b23b5f4e71139faa3bcb39e6bbcc76967fbaf308c4ee9d4f5bfbeceaa76cc5d3',
      strip_prefix = 'flann-1.9.1',
      build_file  = "@third_party//external_repos_bazelization:flann.BUILD",
    )

def xpilot_hdf5():
    maybe(
      http_archive,
      name = "xpilot_hdf5",
      sha256 = 'f426fa3604f8e238fa59ebc71dae6428876130a56744f7c1e41305df76e48848',
      urls = [ "https://artifactory.xiaopeng.us/artifactory/xpilot_dev/buildyard/third_party/hdf5-1.13.1.tar" ],
      strip_prefix = 'hdf5-1.13.1',
      build_file  = "@third_party//external_repos_bazelization:hdf5.BUILD",
    )

def com_github_grpc_grpc(REPO_WKSP_REL_TO_REPO_ROOT):
    maybe(
        native.local_repository,
        name = "com_github_grpc_grpc",
        path = THIRD_PARTY_ROOT_WITH_PRODUCT_LINE_SCHEME("grpc/1.35.0",REPO_WKSP_REL_TO_REPO_ROOT),
    )





def lib_rti_java_libs():
    maybe(
        native.new_local_repository,
        name = "lib_rti_java_libs",
        build_file ="@third_party//external_repos_bazelization:lib_rti_java_libs.BUILD",
        path = PATH_NAMES.lib_rti_jvm,
    )





def nvidia_targetfs():
    maybe(
        native.new_local_repository,
        name = "nvidia_targetfs",
        build_file = "@third_party//external_repos_bazelization:nvidia_targetfs.BUILD",
        path = PATH_NAMES.nvidia_targetfs_path,
    )

def toolchain_libs():
    maybe(
        native.new_local_repository,
        name = "toolchain_libs",
        build_file = "@third_party//external_repos_bazelization:toolchain_libs.BUILD",
        path = PATH_NAMES.toolchain_lib_path,
    )

def cuda():
    maybe(
        native.new_local_repository,
        name = "cuda",
        build_file = "@third_party//external_repos_bazelization:cuda.BUILD",
        path = PATH_NAMES.cuda_tools,
    )



def xpilot_flatbuffer():
    FLAT_BUFFERS_VERSION = "2.0.0"    
    maybe(
        http_archive,
        name = "com_github_google_flatbuffers",
        sha256 = "9ddb9031798f4f8754d00fca2f1a68ecf9d0f83dfac7239af1311e4fd9a565c4",        
        strip_prefix = "flatbuffers-" + FLAT_BUFFERS_VERSION,
        url = "https://github.com/google/flatbuffers/archive/v"+FLAT_BUFFERS_VERSION+".tar.gz",
    )

def xpilot_leveldb():
    maybe(
        http_archive,
        name = "com_github_cschuet_leveldb",        
        strip_prefix = "leveldb-aa785abf30e043110a6258eeefad25ae4d27f677",
        sha256 = "b484e7a4777741c82d265d1192985d9e662b2e0a193638253ebb3060ac62890c",
        urls = [
            "https://github.com/cschuet/leveldb/archive/aa785abf30e043110a6258eeefad25ae4d27f677.tar.gz",
        ],
        patches = ["@third_party//leveldb:leveldb_deps_patch"],
    )
    

        
def xpilot_crc32c():
    maybe(
        http_archive,
        name = "com_github_cschuet_crc32c",
        sha256 = "81adcc96018fe226ae9b32eba74ffea266409891f220ad85fdd043d80edae598",
        strip_prefix = "crc32c-858907d2ed420b0738941df751c85a7c7588a7b6",
        urls = [
            "https://github.com/cschuet/crc32c/archive/858907d2ed420b0738941df751c85a7c7588a7b6.tar.gz",
        ],
        patches = ["@third_party//crc32c:crc32c_deps_patch"],        
    )



def xpilot_absl():
    maybe(
        git_repository,
        name = "com_google_absl",
        remote = "https://github.com/abseil/abseil-cpp",
        tag = "20210324.2",
    )

def xpilot_tcmalloc():
    maybe(
        git_repository,
        name = "com_google_tcmalloc",
        remote = "https://github.com/google/tcmalloc",
        commit = "8e534f50707469baac732559494559db95732e12",
        shallow_since = "1612203684 -0800"
    )



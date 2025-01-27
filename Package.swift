// swift-tools-version:5.5

import PackageDescription

var sources = [
//    "common/common.cpp",
//    "common/log.cpp",
//    "common/console.cpp",
//    "common/grammar-parser.cpp",
//    "common/sampling.cpp",
//    "common-helper.cpp",
//    "src/llama.cpp",
//    "src/llama-vocab.cpp",
//    "src/llama-grammar.cpp",
//    "src/llama-sampling.cpp",
//    "src/unicode.cpp",
//    "src/unicode-data.cpp",
//    "ggml/src/ggml.c",
//    "ggml/src/ggml-aarch64.c",
//    "ggml/src/ggml-alloc.c",
//    "ggml/src/ggml-backend.cpp",
//    "ggml/src/ggml-backend-reg.cpp",
//    "ggml/src/ggml-cpu/ggml-cpu.c",
//    "ggml/src/ggml-cpu/ggml-cpu.cpp",
//    "ggml/src/ggml-cpu/ggml-cpu-aarch64.c",
//    "ggml/src/ggml-cpu/ggml-cpu-quants.c",
//    "ggml/src/ggml-threading.cpp",
//    "ggml/src/ggml-quants.c",
//    "src/llama-arch.cpp",
//    "src/llama-impl.cpp",
    "common/common.cpp",
    "common/log.cpp",
    "common/arg.cpp",
    "common/json-schema-to-grammar.cpp",
    "common/sampling.cpp",
    "common/common-helper.cpp",
    
    "ggml/src/ggml-quants.c", 
    "ggml/src/ggml-alloc.c", 
    "ggml/src/ggml.c", 
    "ggml/src/ggml-cpu/ggml-cpu.c", 
    "ggml/src/ggml-cpu/ggml-cpu-quants.c", 
    
//    "ggml/src/ggml.c",
//    "ggml/src/ggml-quants.c",
//    "ggml/src/ggml-alloc.c",
//    "ggml/src/ggml-backend.cpp",
//    "ggml/src/ggml-threading.cpp",
//    "ggml/src/ggml-backend-reg.cpp",
//    "ggml/src/ggml-metal/ggml-metal.m",
//    "ggml/src/ggml-blas/ggml-blas.cpp",
//    "ggml/src/ggml-aarch64.c",
//    "ggml/src/ggml-cpu/ggml-cpu-aarch64.c",
//    "ggml/src/ggml-cpu/ggml-cpu.c",
//    "ggml/src/ggml-cpu/ggml-cpu.cpp",
//    "ggml/src/ggml-cpu/ggml-cpu-quants.c",
//    "ggml/src/ggml-cpu/ggml-cpu-traits.cpp",
//    "ggml/src/ggml-cpu/llamafile/sgemm.cpp",
    
        "ggml/src/ggml-opt.cpp",
        "ggml/src/ggml-blas/ggml-blas.cpp",
        "ggml/src/ggml-backend.cpp",
        "ggml/src/ggml-cpu/ggml-cpu-aarch64.cpp",
        "ggml/src/ggml-cpu/llamafile/sgemm.cpp",
//        "ggml/src/ggml-cpu/cpu-feats-x86.cpp",
//        "ggml/src/ggml-cpu/amx/mmq.cpp",
//        "ggml/src/ggml-cpu/amx/amx.cpp",
        "ggml/src/ggml-cpu/ggml-cpu-traits.cpp",
//        "ggml/src/ggml-cpu/ggml-cpu-hbm.cpp",
        "ggml/src/ggml-cpu/ggml-cpu.cpp",
//        "ggml/src/ggml-kompute/ggml-kompute.cpp",
//        "ggml/src/ggml-rpc/ggml-rpc.cpp",
//        "ggml/src/ggml-sycl/rope.cpp",
//        "ggml/src/ggml-sycl/tsembd.cpp",
//        "ggml/src/ggml-sycl/common.cpp",
//        "ggml/src/ggml-sycl/im2col.cpp",
//        "ggml/src/ggml-sycl/outprod.cpp",
//        "ggml/src/ggml-sycl/element_wise.cpp",
//        "ggml/src/ggml-sycl/convert.cpp",
//        "ggml/src/ggml-sycl/gla.cpp",
//        "ggml/src/ggml-sycl/norm.cpp",
//        "ggml/src/ggml-sycl/concat.cpp",
//        "ggml/src/ggml-sycl/mmq.cpp",
//        "ggml/src/ggml-sycl/mmvq.cpp",
//        "ggml/src/ggml-sycl/dmmv.cpp",
//        "ggml/src/ggml-sycl/wkv6.cpp",
//        "ggml/src/ggml-sycl/softmax.cpp",
//        "ggml/src/ggml-sycl/conv.cpp",
//        "ggml/src/ggml-sycl/ggml-sycl.cpp",
//        "ggml/src/ggml-opencl/ggml-opencl.cpp",
        "ggml/src/ggml-backend-reg.cpp",
//        "ggml/src/ggml-vulkan/ggml-vulkan.cpp",
//        "ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp",
        "ggml/src/ggml-threading.cpp",
        "ggml/src/gguf.cpp",
    
    
//    "src/llama.cpp",
//    "src/llama-context.cpp",
//    
//    "src/llama-model.cpp",
//    "src/llama-quant.cpp",
//    "src/unicode.cpp",
//    "src/unicode-data.cpp",
//    "src/llama-grammar.cpp",
//    "src/llama-vocab.cpp",
//    "src/llama-sampling.cpp",
    
    
    "src/llama-adapter.cpp",
    "src/llama-arch.cpp",
    "src/llama-batch.cpp",
    "src/llama-chat.cpp",
    "src/llama-context.cpp",
    "src/llama-cparams.cpp",
    "src/llama-grammar.cpp",
    "src/llama-hparams.cpp",
    "src/llama-impl.cpp",
    "src/llama-kv-cache.cpp",
    "src/llama-mmap.cpp",
    "src/llama-model-loader.cpp",
    "src/llama-model.cpp",
    "src/llama-quant.cpp",
    "src/llama-sampling.cpp",
    "src/llama-vocab.cpp",
    "src/llama.cpp",
    "src/unicode-data.cpp",
    "src/unicode.cpp",
    
]

var resources: [Resource] = []
var linkerSettings: [LinkerSetting] = []
var cSettings: [CSetting] =  [
    .define("SWIFT_PACKAGE"),
                    .define("GGML_USE_ACCELERATE"),
                    .define("GGML_BLAS_USE_ACCELERATE"),
                    .define("ACCELERATE_NEW_LAPACK"),
                    .define("ACCELERATE_LAPACK_ILP64"),
                    .define("GGML_USE_BLAS"),
    //                .define("_DARWIN_C_SOURCE"),
                    .define("GGML_USE_LLAMAFILE"),
                    .define("GGML_METAL_NDEBUG"),
                    .define("NDEBUG"),
                    .define("GGML_USE_CPU"),
                    .define("GGML_USE_METAL"),
    
    .unsafeFlags(["-Wno-shorten-64-to-32", "-O3", "-DNDEBUG"]),
    .unsafeFlags(["-fno-objc-arc"]),
    
    .unsafeFlags(["-fno-finite-math-only"], .when(configuration: .release)),
    
    
    
    .headerSearchPath("ggml/include"),
    .headerSearchPath("ggml/src"),
    .headerSearchPath("ggml/src/ggml-cpu"),
    .headerSearchPath("common"),
    .headerSearchPath("include"),
    .headerSearchPath("src"),
    // NOTE: NEW_LAPACK will required iOS version 16.4+
    // We should consider add this in the future when we drop support for iOS 14
    // (ref: ref: https://developer.apple.com/documentation/accelerate/1513264-cblas_sgemm?language=objc)
    // .define("ACCELERATE_NEW_LAPACK"),
    // .define("ACCELERATE_LAPACK_ILP64")
]

#if canImport(Darwin)
sources.append("ggml/src/ggml-common.h")
sources.append("ggml/src/ggml-metal/ggml-metal.m")
resources.append(.process("ggml/src/ggml-metal/ggml-metal.metal"))
linkerSettings.append(.linkedFramework("Accelerate"))

linkerSettings.append(.linkedFramework("Metal"))
linkerSettings.append(.linkedFramework("MetalKit"))
linkerSettings.append(.linkedFramework("MetalPerformanceShaders"))


cSettings.append(
    contentsOf: [
        .define("GGML_USE_ACCELERATE"),
        .define("GGML_USE_METAL"),
        .define("LLAMA_USE_SWIFT")
    ]
)
#endif

#if os(Linux)
    cSettings.append(.define("_GNU_SOURCE"))
#endif

let package = Package(
    name: "llama",
    platforms: [
        .macOS(.v12),
        .iOS(.v14),
        .watchOS(.v4),
        .tvOS(.v14)
    ],
    products: [
        .library(name: "llama", targets: ["llama"]),
    ],
    targets: [
        .target(
            name: "llama",
            path: ".",
            exclude: [
               "build",
               "cmake",
               "examples",
               "scripts",
               "models",
               "tests",
               "CMakeLists.txt",
               "Makefile",
               "ggml/src/ggml-metal-embed.metal"
            ],
            sources: sources,
            resources: resources,
            publicHeadersPath: "spm-headers",
            cSettings: cSettings,
            linkerSettings: linkerSettings
        )
    ],
    cxxLanguageStandard: .cxx17
)

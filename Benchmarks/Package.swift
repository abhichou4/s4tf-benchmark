// swift-tools-version:5.0
import PackageDescription

let package = Package(
    name: "Benchmarks",
    products: [
        .library(name: "Benchmarks", targets: ["Dense-MNIST"]),
    ],
    dependencies: [
        .package(url: "https://github.com/google/swift-benchmark", from: "0.1.0"),
        .package(path: "../Datasets"),
        .package(path: "../Models")
    ],
    targets: [
        .target(
            name: "Dense-MNIST", 
            dependencies: ["Datasets", "Models", "Benchmark"],
            path: "Swift4TF/Dense-MNIST"),
    ]
)
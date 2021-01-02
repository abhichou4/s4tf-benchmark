// swift-tools-version:5.0
import PackageDescription

let package = Package(
    name: "Datasets",
    products: [
        .library(name: "Datasets", targets: ["Datasets"]),
    ],
    dependencies: [],
    targets: [
        .target(name: "Datasets", path: "Sources"),
    ]
)
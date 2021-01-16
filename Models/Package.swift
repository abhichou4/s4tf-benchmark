// swift-tools-version:5.0
import PackageDescription

let package = Package(
    name: "Models",
    products: [
        .library(name: "Models", targets: ["Models"]),
    ],
    dependencies: [],
    targets: [
        .target(name: "Models", path: "Swift4TF"),
    ]
)
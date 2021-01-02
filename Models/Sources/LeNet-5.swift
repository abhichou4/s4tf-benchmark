import TensorFlow

public struct LeNet: Layer {
    public var conv1 = Conv2D<Float>(filterShape: (5, 5, 1, 6), padding: .same, activation: relu)
    public var pool1 = AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    public var conv2 = Conv2D<Float>(filterShape: (5, 5, 6, 16), activation: relu)
    public var pool2 = AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    public var flatten = Flatten<Float>()
    public var fc1 = Dense<Float>(inputSize: 400, outputSize: 120, activation: relu)
    public var fc2 = Dense<Float>(inputSize: 120, outputSize: 84, activation: relu)
    public var fc3 = Dense<Float>(inputSize: 84, outputSize: 10)

    public init() {}

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let convolved = input.sequenced(through: conv1, pool1, conv2, pool2)
        return convolved.sequenced(through: flatten, fc1, fc2, fc3)
    }
}

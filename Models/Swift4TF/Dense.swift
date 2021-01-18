import TensorFlow


public struct DenseModel: Layer {
    var layer1, layer2, layer3: Dense<Float> 

    public init() {
        layer1 = Dense<Float>(inputSize: 784, outputSize: 10, activation: relu)
        layer2 = Dense<Float>(inputSize: 10, outputSize: 10, activation: relu)
        layer3 = Dense<Float>(inputSize: 10, outputSize: 10)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: layer1, layer2, layer3)
    }
}

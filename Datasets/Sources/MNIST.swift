import Foundation
import TensorFlow


public protocol ConvertableFromByte : TensorFlowScalar {
    init(_ byte: UInt8)
}

extension Float : ConvertableFromByte {}
extension Int32 : ConvertableFromByte {}

public func loadMNISTDataset<T: ConvertableFromByte>(
    from pathToFile: String, 
    isTraining train: Bool, isLabel label: Bool, 
    toFlatten flatten: Bool) -> Tensor<T> {
    let samples = train ? 60000 : 10000
    let dropBytes = label ? 8 : 16
    let shape : TensorShape = label ?  [samples] : (flatten ? [samples, 784] : [samples, 28, 28])
    let byteStream = try! Data(contentsOf: URL(fileURLWithPath: pathToFile)).dropFirst(dropBytes)
    return Tensor(shape: shape, scalars: byteStream.map(T.init))
}

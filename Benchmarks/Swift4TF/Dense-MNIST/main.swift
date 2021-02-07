import TensorFlow
import Benchmark
import Datasets
import Models

let device: Device = Device(kind: .GPU, ordinal: 0)

let remoteBaseDirectory = "https://storage.googleapis.com/cvdf-datasets/mnist/"
let localBaseDirectory = "../data/mnist/"

let filenames = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte"
]

for filename in filenames {
    downloadDataset(
        file: filename,
        from: remoteBaseDirectory,
        to: localBaseDirectory)
}


let _: Tensor<Float> = loadMNISTDataset(from: localBaseDirectory + filenames[0], 
                                            isTraining: true, isLabel: false, toFlatten: true, device: device)

let trainingImages: Tensor<Float> = loadMNISTDataset(from: localBaseDirectory + filenames[0], 
                                                    isTraining: true, isLabel: false, toFlatten: true, device: device) / 255.0
let trainingLabels: Tensor<Int32> = loadMNISTDataset(from: localBaseDirectory + filenames[1],
                                                    isTraining: true, isLabel: true, toFlatten: true, device: device)

let batchSize = 32


public struct MNISTBatch {
    let images: Tensor<Float>
    let labels: Tensor<Int32>
}

extension MNISTBatch : Collatable {
    public init<BatchSamples: Collection>(collating samples: BatchSamples) 
        where BatchSamples.Element == Self {
            images = Tensor<Float>(stacking: samples.map{$0.images})
            labels = Tensor<Int32>(stacking: samples.map{$0.labels})
        }
}

let trainingDataset: [MNISTBatch] = zip(trainingImages.unstacked(), trainingLabels.unstacked()).map{MNISTBatch(images: $0.0, labels: $0.1)}
let trainingEpochs: TrainingEpochs = TrainingEpochs(samples: trainingDataset, batchSize: batchSize)

let firstTrainEpoch = trainingEpochs.next()!
let firstTrainBatch = firstTrainEpoch.first!.collated
let firstTrainFeatures = firstTrainBatch.images
let firstTrainLabels = firstTrainBatch.labels

var model = DenseModel()
let optimizer = SGD(for: model, learningRate: 0.01)
let (_, grads) = valueWithGradient(at: model) { model -> Tensor<Float> in
    let logits = model(firstTrainFeatures)
    return softmaxCrossEntropy(logits: logits, labels: firstTrainLabels)
}
    
benchmark("Forward Pass\t") {
    let _ = model(firstTrainFeatures)
}

benchmark("One Update step\t") {
    let (_, _) = valueWithGradient(at: model) { model -> Tensor<Float> in
        let logits = model(firstTrainFeatures)
        return softmaxCrossEntropy(logits: logits, labels: firstTrainLabels)
    }
    optimizer.update(&model, along: grads)
}

let epochCount = 200
benchmark("Total time to train\t", settings: Iterations(1)) {
    for (epochIndex, epoch) in trainingEpochs.prefix(epochCount).enumerated() {
        var epochLoss: Float = 0
        var epochAccuracy: Float = 0
        var batchCount: Int = 0
        for batchSamples in epoch {
            let batch = batchSamples.collated
            let (loss, grad) = valueWithGradient(at: model) { model -> Tensor<Float> in
                let logits = model(batch.images)
                return softmaxCrossEntropy(logits: logits, labels: batch.labels)
            }
            optimizer.update(&model, along: grad)

            let logits = model(batch.images)
            epochLoss += loss.scalarized()
            batchCount += 1
        }
        epochAccuracy /= Float(batchCount)
        epochLoss /= Float(batchCount)
        if epochIndex+1 % 50 == 0 {
            print("Epoch \(epochIndex): Loss: \(epochLoss)")
        }
    }
}

Benchmark.main()

//
//  ARViewProvide.swift
//  Depth Viewer
//
//  Created by Merwan Yeditha on 6/14/21.
//

import SwiftUI
import ARKit
import RealityKit
import VideoToolbox
import Vision

class ARViewProvider: NSObject, ARSessionDelegate, ObservableObject {
    public static var shared = ARViewProvider()
    let arView = ARView(frame: .zero)
    let estimationModel = FastDepth()
    public var img: UIImage?
    // Vision properties
    var request: VNCoreMLRequest?
    var visionModel: VNCoreMLModel?
    //let queue = DispatchQueue(label: "info.queue", attributes: .concurrent)

    private override init() {
        super.init()
        self.arView.session.delegate = self
        self.runModel()
        
    }
    
    func runModel(){
        // Sets up the vision model and passes in the FastDepth mlmodel
        if let visionModel = try? VNCoreMLModel(for: estimationModel.model) {
            self.visionModel = visionModel
            // Uses the VNCoreMLRequest in-built function and call the visionRequestDidComplete function
            // after it has sent the request
            request = VNCoreMLRequest(model: visionModel, completionHandler: visionRequestDidComplete)
            request?.imageCropAndScaleOption = .centerCrop
        } else {
            fatalError()
        }
    }
    
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        // Capture the scene image
        let framee = frame.capturedImage
        self.predict(with: framee)
        
        
    }
    
    func predict(with pixelBuffer: CVPixelBuffer) {
        guard let request = request else { fatalError() }
        
        // vision framework configures the input size of image following our model's input configuration automatically
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform([request])
        
    }

    func visionRequestDidComplete(request: VNRequest, error: Error?) {
        // Runs when the request has been sent
        if let observations = request.results as? [VNCoreMLFeatureValueObservation],
            let array = observations.first?.featureValue.multiArrayValue,
            let map = try? array.reshaped(to: [1,224,224]),
            let image = map.image(min: Double(4), max: 0, channel: nil, axes: nil)
        {
            // Uncomment below to save every frame to camera roll
            //UIImageWriteToSavedPhotosAlbum(image,nil,nil,nil);
            self.img = image
            // Sends signal to update image
            objectWillChange
                .receive(on: DispatchQueue.main)
                .eraseToAnyPublisher()
            objectWillChange.send()
            // UIImageWriteToSavedPhotosAlbum(image,nil,nil,nil);
            let ptr = map.dataPointer.bindMemory(to: Float.self, capacity: map.count)
            let doubleBuffer = UnsafeBufferPointer(start: ptr, count: map.count)
            let output = Array(doubleBuffer)
            let imgArray = convert1DTo2D(linspace: output)
            let midpt = imgArray[112][112]
            print("midpt \(midpt)")
        }
    }
    
    // Converts a 1x50176 array of floats to a 224x224 array
    func convert1DTo2D(linspace: Array<Float>) -> [[Float]] {
        var newArray = [[Float]]()
        var row = [Float]()
        
        // Conversion of 1D to 2D by reading off rows and appending them to new array
        for i in 0...223 {
            for j in 0...223 {
                row.append(linspace[i*224+j])
            }
            newArray.append(row)
            row = []
        }
        
        return newArray
    }
    
}

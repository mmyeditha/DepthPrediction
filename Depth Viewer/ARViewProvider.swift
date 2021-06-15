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
    var img: UIImage?
    // Vision properties
    var request: VNCoreMLRequest?
    var visionModel: VNCoreMLModel?
    
    private override init() {
        super.init()
        arView.session.delegate = self
        runModel()
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
        predict(with: framee)
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
            objectWillChange.send()
            UIImageWriteToSavedPhotosAlbum(image,nil,nil,nil);
            let ptr = map.dataPointer.bindMemory(to: Float.self, capacity: map.count)
            let doubleBuffer = UnsafeBufferPointer(start: ptr, count: map.count)
            let output = Array(doubleBuffer)
            var imgArray = [[Float]]()
            var row = [Float]()
            
            // Converting 1D array to 2D array of pixel values
            for i in 0...223 {
                for j in 0...223 {
                    row.append(output[i*224+j])
                }
                imgArray.append(row)
                row = []
            }
            
            let midpt = imgArray[112][112]
            print("midpt \(midpt)")
        }
    }
        
}

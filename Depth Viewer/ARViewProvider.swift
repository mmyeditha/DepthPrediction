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
import Foundation
import UIKit

class ARViewProvider: NSObject, ARSessionDelegate, ObservableObject {
    public static var shared = ARViewProvider()
    let arView = ARView(frame: .zero)
    let estimationModel = FastDepth()
    public var img: UIImage?
    
    // - MARK: Vision properties
    var request: VNCoreMLRequest?
    var visionModel: VNCoreMLModel?
    let queue = DispatchQueue(label: "info.queue", attributes: .concurrent)
    var isEmpty = true
    var imgArr: [[Float]]?
    var sessionCount = 0
    var buttonPressed: Bool = false
    var sliderValue: Double = 0.5
    
    // - MARK: Haptics variables
    let feedbackGenerator = UIImpactFeedbackGenerator(style: .light)
    var lastImpactTime = Date()
    var desiredInterval: Double?
    var hapticTimer: Timer?

    // - MARK: App configuration
    private override init() {
        super.init()
        let configuration = ARWorldTrackingConfiguration()
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            configuration.frameSemantics = [.smoothedSceneDepth, .sceneDepth]
        }
        self.arView.session.run(configuration)
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
    
    // - MARK: Running app session
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        if isEmpty {
            isEmpty = false
            queue.async {
                // Capture the scene image
                let framee = frame.capturedImage
                self.predict(with: framee)
                // Block of code below only gets passed if the phone has LIDAR
                if let depthMap = frame.sceneDepth?.depthMap, let confMap = frame.sceneDepth?.confidenceMap {
                    // Only capture point cloud if button pressed
                    if self.buttonPressed {
                        let pointCloud = saveSceneDepth(depthMapBuffer: depthMap, confMapBuffer: confMap)
                        let xyz = pointCloud.getFastCloud(intrinsics: frame.camera.intrinsics, strideStep: 1, maxDepth: 1000, throwAwayPadding: 0, rgbWidth: CVPixelBufferGetWidth(framee), rgbHeight: CVPixelBufferGetHeight(framee))
                        var transformedCloud: [simd_float4] = []
                        for p in xyz {
                            transformedCloud.append(simd_float4(simd_normalize(p.0), simd_length(p.0)))
                        }
                        self.write(pointCloud: transformedCloud, fileName: "lidar_\(self.sessionCount).csv")
                        self.buttonPressed = false
                    }
                }
                
                // Add to count
                self.sessionCount += 1
                if let arr = self.imgArr {
                    // If phone is a LIDAR phone, this code gets ignored since button has already been pressed
                    // Otherwise, code is executed
                    if self.buttonPressed{
                        let ptCloud = self.getPointCloud(frame: frame, imgArray: arr)
                        self.write(pointCloud: ptCloud, fileName: "\(NSTimeIntervalSince1970)_mypointcloud\(self.sessionCount).csv")
                        //print(ptCloud)
                        let image = self.convert(cmage: CIImage(cvPixelBuffer: framee))
                        self.writeImg(image: image, session: self.sessionCount)
                        self.buttonPressed = false
                    }
                }
                self.isEmpty = true
            }
        }
    }
    
    func predict(with pixelBuffer: CVPixelBuffer) {
        guard let request = request else { fatalError() }
        // vision framework configures the input size of image following our model's input configuration automatically
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform([request])
    }
    
    func visionRequestDidComplete(request: VNRequest, error: Error?) {
        // Runs when the request has been sent to the Model
        // This if statement checks that we have results for our MLModel request and
        // sets variables to MLMultiArray that corresponds to the image output
        if let observations = request.results as? [VNCoreMLFeatureValueObservation],
            let array = observations.first?.featureValue.multiArrayValue,
            let map = try? array.reshaped(to: [1,224,224]),
            let image = map.image(min: Double(4), max: 0, channel: nil, axes: nil)
        {
            self.img = image
            if self.buttonPressed{
                writeImg(image: image, session: self.sessionCount)
            }
            // Process of converting array to bytearray
            let ptr = map.dataPointer.bindMemory(to: Float.self, capacity: map.count)
            let doubleBuffer = UnsafeBufferPointer(start: ptr, count: map.count)
            let output = Array(doubleBuffer)
            self.imgArr = convert1DTo2D(linspace: output)
            // Prints midpoint, not used now but can be used for haptics
            if let imgArr = self.imgArr {
                let midpt = imgArr[112][112]
                print("midpt \(midpt)")
                DispatchQueue.main.async {
                    // Sends the signal that the variable is changing in the main Dispatch Queue
                    self.objectWillChange.send()
                    // Plays haptics
                    self.desiredInterval = Double(midpt/5)
                    self.haptic(time: NSTimeIntervalSince1970)
                }
            }
        }
    }
    
    // - MARK: Debugging tools
    func buttonPress() {
        self.buttonPressed = true
    }
    
    func updateSliderValue(sliderValue: Double) {
        self.sliderValue = sliderValue
    }
    
    // - MARK: Conversions
    func convert(cmage: CIImage) -> UIImage {
         let context = CIContext(options: nil)
         let cgImage = context.createCGImage(cmage, from: cmage.extent)!
         let image = UIImage(cgImage: cgImage)
         return image
    }
    
    func convert1DTo2D(linspace: Array<Float>) -> [[Float]] {
        // Converts a 1x50176 array of floats to a 224x224 array
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
    
    // - MARK: Creating point cloud
    func getPointCloud(frame: ARFrame, imgArray: [[Float]]) -> [SIMD4<Float>] {
        // Intrinsic matrix, refreshes often to update focal lengths with image stabilization
        let intrinsics = frame.camera.intrinsics
        // ptCloud is a list of 4x1 SIMD Floats
        // elements 0, 1, and 2 represent the x, y, and z components of the unit vector respectively
        // element 4 represents the corresponding depth value for that vector
        var ptCloud: [SIMD4<Float>] = []
            // The intrinsic matrix assumes a 4:3 aspect ratio. The image we have is 1:1, so we have to
            // extrapolate extra pixels that we'll just fill with a 0 depth value
            for i in 0...299 {
                for j in 0...223 {
                    // Remapping original 4:3 resolution (varies by phone) to downscaled 4:3 resolution (299x223)
                    let iRemapped = (Float(i)/299.0)*Float(CVPixelBufferGetWidth(frame.capturedImage))
                    let jRemapped = (Float(j)/223.0)*Float(CVPixelBufferGetHeight(frame.capturedImage))

                    // Convert pixel to vector and normalize
                    let ptVec: SIMD3 = [iRemapped, jRemapped, 1]
                    let vec = simd_normalize(intrinsics.inverse * ptVec)
                    
                    // Sets center of 4:3 image to have actual values, sides of 4:3 images are black
                    if i < 261 && i > 38 {
                        ptCloud.append(simd_float4(vec, imgArray[i-38][j]))
                    } else {
                        ptCloud.append(simd_float4(vec, 0))
                    }
                }
            }
        return ptCloud
    }
    
    // - MARK: Writing data
    // Write point cloud into a file for further review
    func write(pointCloud ptCloud: [SIMD4<Float>], fileName: String) -> Void {
        // Initialize a string where data will be stored line-by-line
        var pointCloudData = ""
        for p in ptCloud {
            pointCloudData += "\(p.x),\(p.y),\(p.z),\(p.w)\n"
        }
        // Save data to a file in AppData
        let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let url = documentsDirectory.appendingPathComponent(fileName)
        if let cloudData = pointCloudData.data(using: .utf8) {
            try? cloudData.write(to: url, options: [.atomic])
        }
        
    }
    
    func writeImg(image: UIImage, session: Int) {
        // Writes image to application data
        let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let fileName = "\(NSTimeIntervalSince1970)image\(session).jpg"
        let fileURL = documentsDirectory.appendingPathComponent(fileName)
        if let data = image.jpegData(compressionQuality:  1.0),
          !FileManager.default.fileExists(atPath: fileURL.path) {
            do {
                // writes the image data to disk
                try data.write(to: fileURL)
                print("file saved")
            } catch {
                print("error saving file:", error)
            }
        }
    }
    
    // - MARK: Haptics and Audio
    func haptic(time: Double) {
        hapticTimer = Timer.scheduledTimer(withTimeInterval: 0.01, repeats: true) { timer in
            if let desiredInterval = self.desiredInterval {
                if -self.lastImpactTime.timeIntervalSinceNow > desiredInterval {
                    self.feedbackGenerator.impactOccurred()
                    self.playSystemSound(id: 1103)
                    self.lastImpactTime = Date()
                }
            }
        }
    }
    
    // Play audio cues
    func playSystemSound(id: Int) {
        AudioServicesPlaySystemSound(SystemSoundID(id));
    }

    
}

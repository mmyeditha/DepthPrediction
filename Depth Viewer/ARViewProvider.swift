//
//  ARViewProvide.swift
//  Depth Viewer
//
//  Created by Neel Dhulipala, Mario Gergis, Merwan Yeditha on 6/14/21.
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
    public var img: UIImage?
    
    let estimationModel: FastDepth = {
        do {
            let config = MLModelConfiguration()
            return try FastDepth(configuration: config)
        } catch {
            print(error)
            fatalError("Could not create FastDepth")
        }
    }()
    
    // - MARK: Vision properties
    var request: VNCoreMLRequest?
    var visionModel: VNCoreMLModel?
    let queue = DispatchQueue(label: "info.queue", attributes: .concurrent)
    var isEmpty = true
    var imgArr: [[Float]]?
    var sessionCount = 0
    var buttonPressed: Bool = false
    var sliderValue: Double = 0.5
    var featureMat: [[Float]] = []
    
    // - MARK: Haptics variables
    let feedbackGenerator = UIImpactFeedbackGenerator(style: .light)
    var lastImpactTime = Date()
    var desiredInterval: Double?
    var hapticTimer: Timer?

    // - MARK: App configuration
    private override init() {
        super.init()
        let configuration = ARWorldTrackingConfiguration()
        // Initializes y-axis parallel to gravity
        configuration.worldAlignment = .gravity
        configuration.planeDetection = .horizontal
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
    func session(_ session: ARSession, didUpdate frame: ARFrame, anchor: ARAnchor) {
        // Show the current position of phone relative to where it was when app started
        //let transform = frame.camera.transform
        //print("Transform: \(transform[3])")
        
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
                        let pointCloud = self.saveSceneDepth(depthMapBuffer: depthMap, confMapBuffer: confMap)
                        let xyz = pointCloud.getFastCloud(intrinsics: frame.camera.intrinsics, strideStep: 1, maxDepth: 1000, throwAwayPadding: 0, rgbWidth: CVPixelBufferGetWidth(framee), rgbHeight: CVPixelBufferGetHeight(framee))
                        var transformedCloud: [simd_float4] = []
                        for p in xyz {
                            transformedCloud.append(simd_float4(simd_normalize(p.0), simd_length(p.0)))
                        }
                        // self.write(pointCloud: transformedCloud, fileName: "lidar_\(self.sessionCount).csv", frame: frame)
                    }
                }
                
                // Add to count
                if let arr = self.imgArr {
                    // Code is executed no matter if phone is a LiDAR or not when button is pressed
                    if self.buttonPressed{
                        // Instantiate ARFrameDataLog type from current frame
                        let dataLog = frame.toLogFrame(type: "mario", trueNorthTransform: nil)
                        // Upload this frame
                        if let dataLog = dataLog {
                            TrialManager.shared.addFrame(frame: dataLog)
                        }
                        let ptCloud = self.getPointCloud(frame: frame, imgArray: arr)
                        self.write(pointCloud: ptCloud, fileName: "\(NSTimeIntervalSince1970)_mypointcloud\(self.sessionCount).csv", frame: frame)
                        print("Wrote the vector data")
                        //print(ptCloud)
                        let image = self.convert(cmage: CIImage(cvPixelBuffer: framee))
                        self.writeImg(image: image, session: self.sessionCount, label: "frame")
                        print("Wrote the frame")
                        if let img = self.img{
                            self.writeImg(image: img, session: self.sessionCount, label: "depth")
                            print("Wrote the depth in session")
                        }
                        // Capture raw feature points into a [[Float]] and write them
                        session.getCurrentWorldMap() {
                            (map, error) in
                            if let map = map {
                                let features = map.rawFeaturePoints.points
                                for feature in features {
                                    self.featureMat.append([feature[0], feature[1], feature[2], 1.0])
                                }
                            }
                        }
                        self.writeFeaturePoints(features: self.featureMat, session: self.sessionCount, frame: frame)
                        // Set buttonPressed to false so that program knows the button is unpressed
                        self.buttonPressed = false
                    }
                }
                // Queue is empty, session can run again
                self.isEmpty = true
                self.sessionCount += 1
                self.featureMat = []
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
//            if self.buttonPressed{
//                print("Wrote the depth data")
//                writeImg(image: image, session: self.sessionCount)
//            }
            // Process of converting array to bytearray
            let ptr = map.dataPointer.bindMemory(to: Float.self, capacity: map.count)
            let doubleBuffer = UnsafeBufferPointer(start: ptr, count: map.count)
            let output = Array(doubleBuffer)
            self.imgArr = convert1DTo2D(linspace: output)
            // Prints midpoint, useful for haptics and grayscale calibration
            /// - TODO:
            /// - The grayscale from the depth map needs to be analyzed further; sometimes it tends to be
            /// - linear when walking more slowly and sometimes the points appear to have a sinusoidal trend.
            /// https://docs.google.com/spreadsheets/d/1jf8F6472G-slNE1-v3t4Ykklwg3-hIV7WwIFJ0UlQcM/edit?usp=sharing
            /// - Link above goes to the graphs in question, where x is the session number and y is the pixel value.
            /// - Figure 1 and Figure 2 are the same paths but at different speeds; however, my laptop was plugged in and
            /// - the wall segment was pretty small.
            /// - Find a clear wall in a room with a lot of space and analyze the graphs.
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
    func saveSceneDepth(depthMapBuffer: CVPixelBuffer, confMapBuffer: CVPixelBuffer, getConfidenceLevels: Bool = true) -> PointCloud {
        let width = CVPixelBufferGetWidth(depthMapBuffer)
        let height = CVPixelBufferGetHeight(depthMapBuffer)
        CVPixelBufferLockBaseAddress(depthMapBuffer, CVPixelBufferLockFlags(rawValue: 0))
        let depthBuffer = unsafeBitCast(CVPixelBufferGetBaseAddress(depthMapBuffer), to: UnsafeMutablePointer<Float32>.self)
        var depthCopy = [Float32](repeating: 0.0, count: width*height)
        memcpy(&depthCopy, depthBuffer, width*height*MemoryLayout<Float32>.size)
        CVPixelBufferUnlockBaseAddress(depthMapBuffer, CVPixelBufferLockFlags(rawValue: 0))
        var confCopy = [ARConfidenceLevel](repeating: .high, count: width*height)
        if getConfidenceLevels {
            // TODO: speed this up using some unsafe C-like operations. Currently we just allow it to be turned off to save time
            CVPixelBufferLockBaseAddress(confMapBuffer, CVPixelBufferLockFlags(rawValue: 0))
            let confBuffer = unsafeBitCast(CVPixelBufferGetBaseAddress(confMapBuffer), to: UnsafeMutablePointer<UInt8>.self)
            for i in 0..<width*height {
                confCopy[i] = ARConfidenceLevel(rawValue: Int(confBuffer[i])) ?? .low
                
            }
            CVPixelBufferUnlockBaseAddress(confMapBuffer, CVPixelBufferLockFlags(rawValue: 0))
            
        }
        return PointCloud(width: width, height: height, depthData: depthCopy)
    }
    
    func getPointCloud(frame: ARFrame, imgArray: [[Float]]) -> [SIMD4<Float>] {
        // Intrinsic matrix, refreshes often to update focal lengths with image stabilization
        let intrinsics = frame.camera.intrinsics
        // Transformation matrix to convert to global frame
        let transform = frame.camera.transform
        // Rotation matrix rotates the point 90 degrees CCW about the z-axis
        let rotation: simd_float4x4 = simd_float4x4(simd_float4(0, 1, 0, 0), simd_float4(-1, 0, 0, 0), simd_float4(0, 0, 1, 0), simd_float4(0, 0, 0, 1))
        // ptCloud is a list of 4x1 SIMD Floats
        // elements 0, 1, and 2 represent the x, y, and z components of the unit vector respectively
        // element 3 represents the corresponding depth value for that vector
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
                    var vec = simd_normalize(intrinsics.inverse * ptVec)
                    // Sets center of 4:3 image to have actual values, sides of 4:3 images are black
                    if i < 261 && i > 38 {
                        vec *= imgArray[i-38][j]
                    } else {
                        vec *= 0
                    }
                    ptCloud.append(rotation*simd_float4(-vec[0], vec[1], -vec[2], 1)*transform.transpose)
                    
                }
            }
        return ptCloud
    }
    
    // - MARK: Writing data
    // Write point cloud into a file for further review
    func write(pointCloud ptCloud: [SIMD4<Float>], fileName: String, frame: ARFrame) -> Void {
        // Initialize a string where data will be stored line-by-line
        var pointCloudData = ""
        for p in ptCloud {
            pointCloudData += "\(p.x),\(p.y),\(p.z),\(p.w)\n"
        }
        // Save data to a file in AppData
//        let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
//        let url = documentsDirectory.appendingPathComponent(fileName)
//        if let cloudData = pointCloudData.data(using: .utf8) {
//            try? cloudData.write(to: url, options: [.atomic])
//        }
    }
    
    func writeImg(image: UIImage, session: Int, label: String) {
        // Writes image to application data
        let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let fileName = "\(NSTimeIntervalSince1970)\(label)image\(session).jpg"
        let fileURL = documentsDirectory.appendingPathComponent(fileName)
        if let data = image.jpegData(compressionQuality:  1.0),
          !FileManager.default.fileExists(atPath: fileURL.path) {
            do {
                // writes the image data to disk
                try data.write(to: fileURL)
                //print("file saved")
            } catch {
                print("error saving file:", error)
            }
        }
    }
    
    // Writes both the raw feature points of a session and the transform matrix
    func writeFeaturePoints(features: [[Float]], session: Int, frame: ARFrame) -> Void {
        var featureString = ""
        for feature in features {
            featureString += "\(feature[0]), \(feature[1]), \(feature[2]), \(feature[3])\n"
        }
        // Writes a csv file to documents directory in application data
        let transform = frame.camera.transform
        let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        // Writes the raw feature points into the documents directory
        let featureURL = documentsDirectory.appendingPathComponent("features_\(session).csv")
        if let featuresData = featureString.data(using: .utf8) {
            try? featuresData.write(to: featureURL, options: [.atomic])
        }
        // Converts transform matrix into a string that can be written into a csv file
        let transformUrl = documentsDirectory.appendingPathComponent("transformMatrix_\(session).csv")
        var transformString = ""
        for i in 0...3 {
            let row = transform[i]
            transformString += "\(row[0]), \(row[1]), \(row[2]), \(row[3])\n"
        }
        // Writes it into documents directory
        if let rowData = transformString.data(using: .utf8) {
            try? rowData.write(to: transformUrl, options: [.atomic])
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

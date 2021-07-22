//
//  ARViewProvide.swift
//  Depth Viewer
//
//  Created by Neel Dhulipala, Mario Gergis, Merwan Yeditha on 6/14/21.
//

import SwiftUI
import ARKit
import RealityKit
import SceneKit
import VideoToolbox
import Vision
import Foundation
import UIKit
import Accelerate
import CoreImage

enum MeshLoggingBehavior {
    case none
    case all
    case updated
}

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
    
    // Captures and uploads every frameCaptureRate'th frame for uploading to Firebase
    let frameCaptureRate: Int = 10
    
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
    var frameCount: Int = 0
    var raycasts: [[Float]] = []
    
    // - MARK: Haptics variables
    let feedbackGenerator = UIImpactFeedbackGenerator(style: .light)
    var lastImpactTime = Date()
    var desiredInterval: Double?
    var hapticTimer: Timer?
    
    // - MARK: Mesh variables
//    var meshNeedsUploading: [UUID] = []
//    var meshRemovalFlagList: [UUID] = []
    var meshNeedsUploading: [UUID: Bool] = [:]
    var meshRemovalFlag: [UUID: Bool] = [:]
    var meshesAreChanging: Bool = false
//    let meshAddQueue = DispatchQueue(label: "mesh.add.queue")
//    let meshUpdateQueue = DispatchQueue(label: "mesh.update.queue")
//    let meshQueue = DispatchQueue(label: "mesh.queue", attributes: .concurrent)

    // - MARK: App configuration
    private override init() {
        super.init()
        let configuration = ARWorldTrackingConfiguration()
        // Initializes y-axis parallel to gravity
        configuration.worldAlignment = .gravity
        configuration.planeDetection = [.horizontal, .vertical]
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            configuration.frameSemantics = [.smoothedSceneDepth, .sceneDepth]
            //configuration.sceneReconstruction = .meshWithClassification
            print("LiDAR phone woohoo")
        }
        self.arView.debugOptions = [.showSceneUnderstanding]
        self.arView.session.run(configuration)
        self.arView.session.delegate = self
        self.runModel()
        // Start the trial
        TrialManager.shared.startTrial()
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
    
    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        var allUpdatedMeshes: [UUID] = []
        for id in anchors.compactMap({$0 as? ARMeshAnchor}).map({$0.identifier}) {
            if !meshesAreChanging {
                meshNeedsUploading[id] = true
                allUpdatedMeshes.append(id)
            }
        }
    }
    
    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        for id in anchors.compactMap({$0 as? ARMeshAnchor}).map({$0.identifier}) {
            if !meshesAreChanging {
                meshNeedsUploading[id] = true
                meshRemovalFlag[id] = false
            }
        }
    }
    
    func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
        for id in anchors.compactMap({$0 as? ARMeshAnchor}).map({$0.identifier}) {
            //print("WARNING: MESH DELETED \(id)")
            meshRemovalFlag[id] = true
        }
    }
    
    // - MARK: Running app session
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        // Show the current position of phone relative to where it was when app started
        //let transform = frame.camera.transform
        //print("Transform: \(transform[3])")
        
        if isEmpty {
            isEmpty = false
            queue.async {
                // Capture the scene image
                let framee = frame.capturedImage
                let image = CIImage(cvPixelBuffer: framee)
                let rotatedImage = image.oriented(.right)
                let context = CIContext()
                var pixelBuffer: CVPixelBuffer?
                let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                             kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
                let width:Int = Int(image.extent.height)
                let height:Int = Int(image.extent.width)
                CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs, &pixelBuffer)
                context.render(rotatedImage, to: pixelBuffer!)
                //self.raycasts = self.raycastMiddlePixel(frame: frame)
                if let frameee = pixelBuffer {
                    self.predict(with: frameee)
                }
                self.frameCount += 1
                
                // Capture every tenth frame and prep it for uploading to firebase
                if self.frameCount % self.frameCaptureRate == 0 {
                    // Instantiate ARFrameDataLog type from current frame
                    let dataLog = frame.toLogFrame(type: "data", trueNorthTransform: nil, meshLoggingBehavior: .updated)
                    // Upload this frame
                    if let dataLog = dataLog {
                        TrialManager.shared.addFrame(frame: dataLog)
                        //print("Trial")
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
            // Process of converting array to bytearray
            let ptr = map.dataPointer.bindMemory(to: Float.self, capacity: map.count)
            let doubleBuffer = UnsafeBufferPointer(start: ptr, count: map.count)
            let output = Array(doubleBuffer)
            self.imgArr = convert1DTo2D(linspace: output)
            // Prints midpoint, useful for haptics and grayscale calibration
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
    
    /// Raycasts from camera to tag and places tag on the nearest mesh if the device supports LiDAR
    func raycastMiddlePixel(frame: ARFrame) -> [[Float]] {
        let startFunctionTime = Date()
        var totalTimeRaycasting = 0.0
        let imageWidth = CVPixelBufferGetWidth(frame.capturedImage)
        let imageHeight = CVPixelBufferGetHeight(frame.capturedImage)
        let cameraTransform = frame.camera.transform
        // get pose of the z-direction relative to the phone
        let pixelSkipFactor = 20
        // array of all raycasts in the frame
        var raycasts: [[Float]] = []
        for i in stride(from: 0, to: imageWidth, by: pixelSkipFactor) {
            for j in stride(from: 0, to: imageHeight, by: pixelSkipFactor) {
                var point: [Float] = []
                let cameraRay = frame.camera.intrinsics.inverse * simd_float3(x: Float(i), y: Float(j), z: 1)
                let cameraRayInDeviceCoordinates = simd_float3(cameraRay.x, -cameraRay.y, -cameraRay.z)
                let cameraRayInWorldCoordinates = cameraTransform*simd_float4(cameraRayInDeviceCoordinates, 0)
                // get pose of the camera from the frame's transform
                let cameraPos = simd_float3(cameraTransform.columns.3.x, cameraTransform.columns.3.y, cameraTransform.columns.3.z)
                let raycastQuery = ARRaycastQuery(origin: cameraPos, direction: simd_float3(cameraRayInWorldCoordinates.x, cameraRayInWorldCoordinates.y, cameraRayInWorldCoordinates.z), allowing: .estimatedPlane, alignment: .any)
                let startRayCast = Date()
                let raycastResult = self.arView.session.raycast(raycastQuery)
                totalTimeRaycasting -= startRayCast.timeIntervalSinceNow
                
                
                if raycastResult.count > 0 {
                    let cameraCoordsOfRaycastResult = frame.camera.transform.inverse*raycastResult[0].worldTransform.columns.3
                    point = [Float(i), Float(j), cameraCoordsOfRaycastResult.x, -cameraCoordsOfRaycastResult.y, -cameraCoordsOfRaycastResult.z]
                    //print(i, j, heatMapValue)
        //            let meshTransform = raycastResult[0].worldTransform
        //            let raycastTagTransform: simd_float4x4 = simd_float4x4(diagonal:simd_float4(1, -1, -1, 1)) * cameraTransform.inverse * meshTransform
        //            return raycastTagTransform
                } else {
                    point = [Float(i), Float(j), 0.0]
                    //print(i, j, 0.0)
                }
                raycasts.append(point)
            }
        }
        print("time to produce cloud", -startFunctionTime.timeIntervalSinceNow, "raycasting time", totalTimeRaycasting)
        return raycasts
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
    
    public func rotate90PixelBuffer(_ srcPixelBuffer: CVPixelBuffer, factor: UInt8) -> CVPixelBuffer? {
      let flags = CVPixelBufferLockFlags(rawValue: 0)
      guard kCVReturnSuccess == CVPixelBufferLockBaseAddress(srcPixelBuffer, flags) else {
        return nil
      }
      defer { CVPixelBufferUnlockBaseAddress(srcPixelBuffer, flags) }

      guard let srcData = CVPixelBufferGetBaseAddress(srcPixelBuffer) else {
        print("Error: could not get pixel buffer base address")
        return nil
      }
      let sourceWidth = CVPixelBufferGetWidth(srcPixelBuffer)
      let sourceHeight = CVPixelBufferGetHeight(srcPixelBuffer)
      var destWidth = sourceHeight
      var destHeight = sourceWidth
      var color = UInt8(0)

      if factor % 2 == 0 {
        destWidth = sourceWidth
        destHeight = sourceHeight
      }

      let srcBytesPerRow = CVPixelBufferGetBytesPerRow(srcPixelBuffer)
      var srcBuffer = vImage_Buffer(data: srcData,
                                    height: vImagePixelCount(sourceHeight),
                                    width: vImagePixelCount(sourceWidth),
                                    rowBytes: srcBytesPerRow)

      let destBytesPerRow = destWidth*4
      guard let destData = malloc(destHeight*destBytesPerRow) else {
        print("Error: out of memory")
        return nil
      }
      var destBuffer = vImage_Buffer(data: destData,
                                     height: vImagePixelCount(destHeight),
                                     width: vImagePixelCount(destWidth),
                                     rowBytes: destBytesPerRow)

      let error = vImageRotate90_ARGB8888(&srcBuffer, &destBuffer, factor, &color, vImage_Flags(0))
      if error != kvImageNoError {
        print("Error:", error)
        free(destData)
        return nil
      }

      let releaseCallback: CVPixelBufferReleaseBytesCallback = { _, ptr in
        if let ptr = ptr {
          free(UnsafeMutableRawPointer(mutating: ptr))
        }
      }

      let pixelFormat = CVPixelBufferGetPixelFormatType(srcPixelBuffer)
      var dstPixelBuffer: CVPixelBuffer?
      let status = CVPixelBufferCreateWithBytes(nil, destWidth, destHeight,
                                                pixelFormat, destData,
                                                destBytesPerRow, releaseCallback,
                                                nil, nil, &dstPixelBuffer)
      if status != kCVReturnSuccess {
        print("Error: could not create new pixel buffer")
        free(destData)
        return nil
      }
      return dstPixelBuffer
    }
}

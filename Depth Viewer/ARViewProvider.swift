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
    
    /// When VoiceOver is not active, we use AVSpeechSynthesizer for speech feedback
    let synth = AVSpeechSynthesizer()
    
    let estimationModel: FCRN = {
        do {
            let config = MLModelConfiguration()
            return try FCRN(configuration: config)
        } catch {
            print(error)
            fatalError("Could not create FCRN")
        }
    }()
    
    // Captures and uploads every frameCaptureRate'th frame for uploading to Firebase
    let frameCaptureRate: Int = 20
    // If there is no need to upload data to Firebase, set this to false
    let areWeUploadingToFirebase = false
    
    // - MARK: Vision properties
    var request: VNCoreMLRequest?
    var visionModel: VNCoreMLModel?
    let queue = DispatchQueue(label: "info.queue", attributes: .concurrent)
    var isEmpty = true
    var imgArr: [[Float]]?
    var confidence: Float?
    var finalConfidence: Float?
    var sessionCount = 0
    var buttonPressed: Bool = false
    var sliderValue: Double = 0.5
    var featureMat: [[Float]] = []
    var frameCount: Int = 0
    var raycasts: [[Float]] = []
    var lastAnnouncementTime = Date()
    static let announcementInterval = 2.0
    var meters = true;
    var useFeaturePoints = false;
    var isAnnouncing = false;
    
    // Point Cloud Variables (debugging)
    // These values be printed out on the ContentView
    // For more info on what these values are, see the getPointCloud function
    var depth: Float = 0
    var zedCameraCoords: Float = 0
    var pixelImage: UIImage?
    var meanDepth: Float = 0
    var meanFeaturePoint: Float = 0
    var slope: Double = 0
    var yIntercept: Double = 0
    var residuals: [Double] = []
    var meanResiduals: Double = 0
    var variance: Float = 0
    var residualSumSquared: Double = 0
    var deviationsFromMean: [Float] = []
    var perceivedFeatureDepth: Double = 0
    
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
        self.arView.debugOptions = [.showSceneUnderstanding, .showFeaturePoints]
        self.arView.session.run(configuration)
        self.arView.session.delegate = self
        self.runModel()
        if areWeUploadingToFirebase {
            // Start the trial
            TrialManager.shared.startTrial()
        }
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
    
    // Session runs everytime an ARAnchor is updated
    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        var allUpdatedMeshes: [UUID] = []
        for id in anchors.compactMap({$0 as? ARMeshAnchor}).map({$0.identifier}) {
            if !meshesAreChanging {
                meshNeedsUploading[id] = true
                allUpdatedMeshes.append(id)
            }
        }
    }
    
    // Session runs everytime an ARAnchor is added
    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        for id in anchors.compactMap({$0 as? ARMeshAnchor}).map({$0.identifier}) {
            if !meshesAreChanging {
                meshNeedsUploading[id] = true
                meshRemovalFlag[id] = false
            }
        }
    }
    
    // Session runs everytime an ARAnchor is removed
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
                // Line below raycasts the middle pixel, be warned that this severely downgrades performance of the app
                //self.raycasts = self.raycastMiddlePixel(frame: frame)
                let uiImage = pixelBuffer!.toUIImage()
                
                // Beginning the process of predicting depth map through FCRN
                if let frameee = pixelBuffer {
                    self.predict(with: frameee)
                }
                self.frameCount += 1
                
                // If we are uploading to Firebase, create a new datalog and add it to trial
                if self.areWeUploadingToFirebase {
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
                }
                
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
                        // Write CSV of pointcloud to the AppData
                        self.write(pointCloud: transformedCloud, fileName: "lidar_\(self.sessionCount).csv")
                    }
                }
                
                // Whether phone has LiDAR or not, collect pointcloud from FCRN data and log it if prompted
                if let arr = self.imgArr {
                    if self.buttonPressed {
                        let ptCloud = self.getPointCloud(frame: frame, imgArray: arr)
                        self.write(pointCloud: ptCloud.0, fileName: "mypointcloud\(self.sessionCount).csv")
                        self.writePixels(points: ptCloud.1, fileName: "pixels\(self.sessionCount).csv")
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
                        if let rawPoints = frame.rawFeaturePoints {
                            for feature in rawPoints.points {
                                self.featureMat.append([feature[0], feature[1], feature[2], 1.0])
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
            let map = try? array.reshaped(to: [1,128,160]),
            let image = map.image(min: Double(4), max: 0, channel: nil, axes: nil)
        {
            self.img = image
            // Process of converting array to bytearray
            let ptr = map.dataPointer.bindMemory(to: Float.self, capacity: map.count)
            let doubleBuffer = UnsafeBufferPointer(start: ptr, count: map.count)
            let output = Array(doubleBuffer)
            self.imgArr = convertTo2DArray(from: map)
            //self.imgArr = convert1DTo2D(linspace: output)
            // Prints midpoint, useful for haptics and grayscale calibration
            if let imgArr = self.imgArr {
                var midpt = imgArr[80][64]
                // If we are using feature points to predict depth instead of FCRN...
                if useFeaturePoints {
                    var minAngle: Float?
                    var distanceAtMinAngle: Float?
                    if let frame = arView.session.currentFrame, let pointCloud = frame.rawFeaturePoints {
                        // Optical axis is vector pointing directly out of center of frame
                        let opticalAxis = -simd_float3(frame.camera.transform.columns.2.x, frame.camera.transform.columns.2.y, frame.camera.transform.columns.2.z)
                        // Camera center tracks pose of the camera
                        let cameraCenter = simd_float3(frame.camera.transform.columns.3.x, frame.camera.transform.columns.3.y, frame.camera.transform.columns.3.z)
                        // For every feature point, find the angle between the vector of that point and the optical axis, determine depth of the center by the z-value of the point with the acutest angle
                        for pt in pointCloud.points {
                            let relativePosition = pt - cameraCenter
                            let angle = acos(simd_dot(relativePosition, opticalAxis)/(simd_length(relativePosition) * simd_length(opticalAxis)))
                            if minAngle == nil || angle < minAngle! {
                                minAngle = angle
                                distanceAtMinAngle = simd_length(relativePosition)
                            }
                        }
                        print("minAngle \(minAngle)")
                    }
                    if let minAngle = minAngle, minAngle < 0.75 {
                        midpt = Float(distanceAtMinAngle!)
                    }
//                    else {
//                        midpt = -1.0
//                    }
                }
//                if self.isAnnouncing && midpt < 0 {
//                    self.announce(announcement: "cannot detect any feature points in this direction")
//                }
                
                // Announce the distances
                if self.isAnnouncing && -self.lastAnnouncementTime.timeIntervalSinceNow > ARViewProvider.announcementInterval {
                    self.lastAnnouncementTime = Date()
                    // This statement was for debugging, check out the getPointCloud function for what is going on here
//                    if let frame = arView.session.currentFrame {
//                        let _ = getPointCloud(frame: frame, imgArray: imgArr)
//                    }
                    if self.meters {
                        self.announce(announcement: String(format: "%.1f", midpt))
                        print("\(midpt) meters")
                    } else {
                        let midptFeet = 3.28 * midpt;
                        self.announce(announcement: String(format: "%.1f", midptFeet))
                        print("\(midptFeet) feet")
                    }
                }
                DispatchQueue.main.async {
                    // Sends the signal that the variable is changing in the main Dispatch Queue
                    self.objectWillChange.send()
                    // Plays haptics if object is within 12 feet
//                    if midpt <= 4.0 {
//                        self.desiredInterval = Double(midpt/5)
//                        self.haptic(time: NSTimeIntervalSince1970)
//                    }
                }
            }
        }
    }
    
    /// Communicates a message to the user via speech.  If VoiceOver is active, then VoiceOver is used to communicate the announcement, otherwise we use the AVSpeechEngine
      ///
      /// - Parameter announcement: the text to read to the user
      func announce(announcement: String) {
          if UIAccessibility.isVoiceOverRunning {
              // use the VoiceOver API instead of text to speech
              UIAccessibility.post(notification: UIAccessibility.Notification.announcement, argument: announcement)
          } else {
              let audioSession = AVAudioSession.sharedInstance()
              do {
                  try audioSession.setCategory(AVAudioSession.Category.playback)
                  try audioSession.setActive(true)
                  let utterance = AVSpeechUtterance(string: announcement)
                  utterance.rate = 0.6
                  synth.speak(utterance)
              } catch {
                  print("Unexpected error announcing something using AVSpeechEngine!")
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
    
    func convertTo2DArray(from heatmaps: MLMultiArray) -> Array<Array<Float>> {
        guard heatmaps.shape.count >= 3 else {
            print("heatmap's shape is invalid. \(heatmaps.shape)")
            return []
        }
        let _/*keypoint_number*/ = heatmaps.shape[0].intValue
        let heatmap_w = heatmaps.shape[1].intValue
        let heatmap_h = heatmaps.shape[2].intValue
        
        var convertedHeatmap: Array<Array<Float>> = Array(repeating: Array(repeating: 0.0, count: heatmap_w), count: heatmap_h)
        
        for i in 0..<heatmap_w {
            for j in 0..<heatmap_h {
                let index = i*(heatmap_h) + j
                self.confidence = heatmaps[index].floatValue
                guard self.confidence! > 0 else { continue }
                convertedHeatmap[j][i] = self.confidence!
            }
        }
        return convertedHeatmap
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
    
    func getPointCloud(frame: ARFrame, imgArray: [[Float]]) -> ([SIMD4<Float>], [SIMD4<Float>]) {
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
            for i in 0...159 {
                for j in 0...127 {
                    // Remapping original 4:3 resolution (varies by phone) to downscaled 5:4 resolution (159x127)
                    let iRemapped = (Float(i)/159.0)*Float(CVPixelBufferGetWidth(frame.capturedImage))
                    let jRemapped = (Float(j)/127.0)*Float(CVPixelBufferGetHeight(frame.capturedImage))

                    // Convert pixel to vector and normalize
                    let ptVec: SIMD3 = [iRemapped, jRemapped, 1]
                    var vec = intrinsics.inverse * ptVec
                    // Sets center of 4:3 image to have actual values, sides of 4:3 images are black
                    // (i,j) -> vec, rotation*simd_float4(vec[0], -vec[1], -vec[2], 1) -> (i,j)
                    if i < 143 && i > 16 {
                        vec *= imgArray[i-16][j]
                    } else {
                        vec *= 0
                    }
                    ptCloud.append(rotation*simd_float4(vec[0], -vec[1], -vec[2], 1)*transform.transpose)
                    
                }
            }
        
        // Code below experiments with the idea of mapping ARFeaturePoints to the FCRN pointcloud and finding the relationships between the two
        // Goal of this experimentation is to see if ARFeaturePoints can be used to modify the FCRN pointclouds and thus the results of the depth predictor
        var featurePixels: [SIMD4<Float>] = []
        
        UIGraphicsBeginImageContextWithOptions(
            CGSize(width: 160, height: 128), true, 1)
        let context = UIGraphicsGetCurrentContext()!
        
        for featurePoint in frame.rawFeaturePoints?.points ?? [] {
            let cameraCoords = frame.camera.transform.inverse * simd_float4(featurePoint, 1)
            let x = cameraCoords.x / cameraCoords.z
            let y = cameraCoords.y / cameraCoords.z
            let z = cameraCoords.z / cameraCoords.z
            let cx = frame.camera.intrinsics.columns.2.x
            let cy = frame.camera.intrinsics.columns.2.y
            let fx = frame.camera.intrinsics.columns.0.x
            let fy = frame.camera.intrinsics.columns.1.y
            // Equation derived from Symbolic Math Toolbox, express relationship between camera pixels (FCRN depth heatmap) and points in space (ARFeaturePoints)
            //[(159*cx + 159*fx*y)/w, (127*cy - 127*fy*x)/h]
            // TODO: x-axis flips when i equation is changed to what it should be (i.e. - is changed to +
            let i = Int((159*cx - 159*fx*y) / Float(CVPixelBufferGetWidth(frame.capturedImage)))
            let j = Int((127*cy - 127*fy*x) / Float(CVPixelBufferGetHeight(frame.capturedImage)))
            
            if i < 143 && i > 16 && j > 0 && j < 127 {
                self.depth = -imgArray[i-16][j]
                self.zedCameraCoords = cameraCoords.z
                print("estimated z", depth, "feature point z", cameraCoords.z)
                featurePixels.append(simd_float4(Float(i), Float(j), depth, cameraCoords.z))
            }
            
            context.setFillColor(UIColor.red.cgColor)
            context.setStrokeColor(UIColor.green.cgColor)
            context.setLineWidth(1)

            // TODO: not sure why i would need to be subtracted from 159/160
            let rectangle = CGRect(x: i, y: j, width: 2, height: 2)
            context.addEllipse(in: rectangle)
            context.drawPath(using: .fillStroke)
            
        }
        
        // Finding means of the depth, best fit lines, etc. are for debugging purposes to see if this result is feasible, needs more testing
        // Advisable to print these values out onto ContentView and screen record it to see results overtime, all of these variables are global
        self.meanDepth = featurePixels.map({$0.z}).reduce(0, {x,y in x + y/Float(featurePixels.count)})
        self.meanFeaturePoint = featurePixels.map({$0.w}).reduce(0, {x,y in x + y/Float(featurePixels.count)})
        var numeratorSum: Float = 0
        var denomSum1: Float = 0
        var denomSum2: Float = 0
        
        for depth in featurePixels {
            numeratorSum += (depth[2] - self.meanDepth)*(depth[3] - self.meanFeaturePoint)
            denomSum1 += (depth[2] - self.meanDepth)*(depth[2] - self.meanDepth)
            denomSum2 += (depth[3] - self.meanFeaturePoint)*(depth[3] - self.meanFeaturePoint)
        }
        
        let corr = Double(numeratorSum) / sqrt(Double(denomSum1*denomSum2))
        print("correlation: ", corr)
        
        // Compute m and b of the best fit line
        let m = Double(numeratorSum) / Double(denomSum1)
        let b = Double(self.meanFeaturePoint) - m*Double(self.meanDepth)
        print("line of best fit: y = \(m)x + \(b)")
        self.slope = m
        self.yIntercept = b
        
        for p in featurePixels {
            print("\(p.z), \(p.w)")
        }
        
        self.residuals = featurePixels.map({Double($0.z)*m + b - Double($0.w)})
        self.meanResiduals = self.residuals.reduce(0.0, {x,y in x + y/Double(residuals.count)})
        self.variance = featurePixels.map({($0.z - self.meanDepth)*($0.z-self.meanDepth)}).reduce(0.0, {x, y in x + y/Float(featurePixels.count)})
        self.residualSumSquared = self.residuals.map({($0*$0) / Double(residuals.count)}).reduce(0.0, {x,y in x+y})
        self.deviationsFromMean = featurePixels.map({$0.z - self.meanDepth})
        print("variance \(self.variance)")
        // Calculate depth of center pixel through equation
        self.perceivedFeatureDepth = m*Double(imgArray[80][64]) + b
        print("middle pixel FCRN depth ", imgArray[80][64])
        print("middle pixel perceived ", self.perceivedFeatureDepth)
        
        let image = UIGraphicsGetImageFromCurrentImageContext()
        self.pixelImage = image
        UIGraphicsEndImageContext()
        
        if meters {
            announce(announcement: String(format: "%.1f", self.perceivedFeatureDepth))
        } else {
            let perceivedFeatureDepthInFeet = 3.28 * self.perceivedFeatureDepth
            announce(announcement: String(format: "%.1f", perceivedFeatureDepthInFeet))
        }
        
        return (ptCloud, featurePixels)
    }
    
    /// Raycasts from camera to objects in frame
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
    
    func writePixels(points: [SIMD4<Float>], fileName: String) {
        // Initialize a string where data will be stored line-by-line
        var featurePixelData = ""
        for p in points {
            featurePixelData += "\(p.x),\(p.y),\(p.z),\(p.w)\n"
        }
        // Save data to a file in AppData
        let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let url = documentsDirectory.appendingPathComponent(fileName)
        if let cloudData = featurePixelData.data(using: .utf8) {
            try? cloudData.write(to: url, options: [.atomic])
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
}

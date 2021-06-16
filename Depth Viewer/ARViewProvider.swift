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
    let queue = DispatchQueue(label: "info.queue", attributes: .concurrent)
    var isEmpty = true
    var imgArr: [[Float]]?
    var sessionCount = 0
    var buttonPressed: Bool = false

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
        
        if isEmpty {
            isEmpty = false
            queue.async {
                // Capture the scene image
                let framee = frame.capturedImage
                self.predict(with: framee)
                // Add to count
                self.sessionCount += 1
                if let arr = self.imgArr {
                    if self.buttonPressed{
                        let ptCloud = self.getPointCloud(frame: frame, imgArray: arr)
                        self.write(pointCloud: ptCloud, fileName: "\(NSTimeIntervalSince1970)_mypointcloud\(self.sessionCount).csv")
                        //print(ptCloud)
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

    func buttonPress() {
        self.buttonPressed = true
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
            // UIImageWriteToSavedPhotosAlbum(image,nil,nil,nil);
            let ptr = map.dataPointer.bindMemory(to: Float.self, capacity: map.count)
            let doubleBuffer = UnsafeBufferPointer(start: ptr, count: map.count)
            let output = Array(doubleBuffer)
            self.imgArr = convert1DTo2D(linspace: output)
            if let imgArr = self.imgArr {
                let midpt = imgArr[112][112]
                print("midpt \(midpt)")
            }
            DispatchQueue.main.async {
                self.objectWillChange.send()
            }
            
            
            /// Paul showing us how to store AppData
            
//            var points : [simd_float3] = []
//            points.append(simd_float3(1,3,3))
//            points.append(simd_float3(3,4,2))
//
//            var pointCloud = ""
//            for p in points {
//                pointCloud += "\(p.x), \(p.y), \(p.z)\n"
//            }
//            let currentFileName = "mycloud.csv"
//            let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
//            let url = documentsDirectory.appendingPathComponent(currentFileName)
//            if let cloudData = pointCloud.data(using: .utf8) {
//                try? cloudData.write(to: url, options: [.atomic])
//            }

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
    
    func getPointCloud(frame: ARFrame, imgArray: [[Float]]) -> [SIMD4<Float>] {
        let intrinsics = frame.camera.intrinsics
            var ptCloud: [SIMD4<Float>] = []
            // Replace with actual ranges in imgArray
            for i in 0...299 {
                for j in 0...223 {
                    let iRemapped = (Float(i)/299.0)*Float(CVPixelBufferGetWidth(frame.capturedImage))
                    let jRemapped = (Float(j)/223.0)*Float(CVPixelBufferGetHeight(frame.capturedImage))

                    
                    let ptVec: SIMD3 = [iRemapped, jRemapped, 1]
                    let vec = simd_normalize(intrinsics.inverse * ptVec)
                    if i < 261 && i > 38 {
                        ptCloud.append(simd_float4(vec, imgArray[i-38][j]))
                    } else {
                        ptCloud.append(simd_float4(vec, 0))
                    }
                }
            }
        return ptCloud
    }
    
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
        let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        // choose a name for your image
        let fileName = "\(NSTimeIntervalSince1970)image\(session).jpg"
        // create the destination file url to save your image
        let fileURL = documentsDirectory.appendingPathComponent(fileName)
        // get your UIImage jpeg data representation and check if the destination file url already exists
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

    
}

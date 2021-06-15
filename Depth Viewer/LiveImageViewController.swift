//
//  LiveImageViewController.swift
//  DepthPrediction-CoreML
//
//  Created by Doyoung Gwak on 20/07/2019.
//  Copyright © 2019 Doyoung Gwak. All rights reserved.
//

import UIKit
import Vision
import AVFoundation
import ARKit
import VideoToolbox
/*
class LiveDepthPrediction: ARSessionDelegate {
    
    // MARK - Core ML model
    // FCRN(iOS11+), FCRNFP16(iOS11+)
    let estimationModel = FastDepth()
    
    // MARK: - Vision Properties
    var request: VNCoreMLRequest?
    var visionModel: VNCoreMLModel?
    
    var depthMax : Float = 4;
    
    // MARK: - Performance Measurement Property
    private let measure = Measure()
    
    // MARK: - Haptics Variables
    let feedbackGenerator = UIImpactFeedbackGenerator(style: .light)
    var lastImpactTime = Date()
    var desiredInterval: Double?
    var hapticTimer: Timer?
    
    // MARK: - View Controller Life Cycle
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // setup ml model
        setUpModel()
        
        // setup camera
        //setUpCamera()
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = [.horizontal, .vertical]
        configuration.isAutoFocusEnabled = false
        arView.session.delegate = self
        arView.session.run(configuration)
        
        // setup delegate for performance measurement
        measure.delegate = self
        
        depthSlider.setValue(depthMax, animated: true)
    }
    
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        print(CVPixelBufferGetWidth(frame.capturedImage))
        predict(with: frame.capturedImage)
        
    }
    
    @IBAction func depthMaxValueChanged(_ sender: UISlider) {
        depthMax = sender.value
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        //self.videoCapture.start()
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        self.videoCapture.stop()
    }
    
    // MARK: - Setup Core ML
    func setUpModel() {
        if let visionModel = try? VNCoreMLModel(for: estimationModel.model) {
            self.visionModel = visionModel
            request = VNCoreMLRequest(model: visionModel, completionHandler: visionRequestDidComplete)
            request?.imageCropAndScaleOption = .centerCrop
        } else {
            fatalError()
        }
    }
    
    // MARK: - Setup camera
    func setUpCamera() {
        videoCapture = VideoCapture()
        videoCapture.delegate = self
        videoCapture.fps = 50
        videoCapture.setUp(sessionPreset: .cif352x288) { success in
            
            if success {
                if let previewLayer = self.videoCapture.previewLayer {
                    self.videoPreview.layer.addSublayer(previewLayer)
                    self.resizePreviewLayer()
                }
                self.videoCapture.start()
            }
        }
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        //resizePreviewLayer()
    }
    
    func playSystemSound(id: Int) {
            AudioServicesPlaySystemSound(SystemSoundID(id))
    }
    
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
    
    func resizePreviewLayer() {
        let bounds = videoPreview.bounds
        videoCapture.previewLayer?.videoGravity = AVLayerVideoGravity.resizeAspectFill
        videoCapture.previewLayer?.bounds = bounds
        videoCapture.previewLayer?.position = CGPoint(x:bounds.midX, y:bounds.midY)
    }
    
    func getArrayOfBytesFromImage(imageData:NSData) -> Array<UInt8>
    {

      // the number of elements:
      let count = imageData.length / MemoryLayout<Int8>.size

      // create array of appropriate length:
      var bytes = [UInt8](repeating: 0, count: count)

      // copy bytes into array
      imageData.getBytes(&bytes, length:count * MemoryLayout<Int8>.size)

      var byteArray:Array = Array<UInt8>()

      for i in 0 ..< count {
        byteArray.append(bytes[i])
      }

      return byteArray
    }

}

// MARK: - VideoCaptureDelegate
extension LiveImageViewController: VideoCaapture, didCaptureVideoFrame pixelBuffer: CVPixelBuffer?/*, timestamp: CMTime*/) {
        
        // the captured image from camera is contained on pixelBuffer
        if let pixelBuffer = pixelBuffer {
            // start of measure
            self.measure.start()
             predict(with: pixelBuffer)
        }
    }
}

// MARK: - Inference
extension LiveImageViewController {
    // prediction
    func predict(with pixelBuffer: CVPixelBuffer) {
        guard let request = request else { fatalError() }
        
        // vision framework configures the input size of image following our model's input configuration automatically
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform([request])
    }
    
    func getDocumentsDirectory() -> URL {
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        print(paths[0])
        return paths[0]
    }
    
    // post-processing
    func visionRequestDidComplete(request: VNRequest, error: Error?) {
        
        self.measure.label(with: "endInference")
        if let observations = request.results as? [VNCoreMLFeatureValueObservation],
            let array = observations.first?.featureValue.multiArrayValue,
            let map = try? array.reshaped(to: [1,224,224]),
            let image = map.image(min: Double(depthMax), max: 0, channel: nil, axes: nil)
        {
            //UIImageWriteToSavedPhotosAlbum(image,nil,nil,nil);

//            let data = image.pngData()
//            let bytes = getArrayOfBytesFromImage(imageData: data! as NSData)
//            let datos: NSData = NSData(bytes: bytes, length: bytes.count)
//            print(datos.length)
            let ptr = map.dataPointer.bindMemory(to: Float.self, capacity: map.count)
            let doubleBuffer = UnsafeBufferPointer(start: ptr, count: map.count)
            let output = Array(doubleBuffer)
            var whee = [[Float]]()
            var something = [Float]()
            
            
            for i in 0...223 {
                for j in 0...223 {
                    something.append(output[i*224+j])
                    //whee[i][j] = 0.8474
                    //print(output[i*224+j])
                }
                whee.append(something)
                something = []
            }
            // let img2 = whee.image
            
            let midpt = whee[112][112]
            print("midpt \(midpt)")
            //print(whee)
            
            DispatchQueue.main.async { [weak self] in
                self?.drawingView.image = image
                self?.desiredInterval = Double(midpt/3)
                print("Running <3 interval is \(self!.desiredInterval!)")
                self?.haptic(time: NSTimeIntervalSince1970)
                // end of measure
                self?.measure.stop()
            }
        } else {
            // end of measure
            self.measure.stop()
        }
    }
}

// MARK: - Performance Measurement Delegate
extension LiveImageViewController: MeasureDelegate {
    func updateMeasure(inferenceTime: Double, executionTime: Double, fps: Int) {
        //print(executionTime, fps)
        self.inferenceLabel.text = "inference: \(Int(inferenceTime*1000.0)) mm"
        self.etimeLabel.text = "execution: \(Int(executionTime*1000.0)) mm"
        self.fpsLabel.text = "fps: \(fps)"
    }
}
*/

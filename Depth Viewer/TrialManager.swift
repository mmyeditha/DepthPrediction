//
//  TrialManager.swift
//  LidarCane
//
//  Created by Paul Ruvolo on 3/5/21.
//

// TODO: need to do a better job connecting together saved routes and the route that is being navigated (hard to align the two right now)
// TODO: Check into why the frame metadata might be missing from some frames (maybe there is an error condition, or maybe just bad Wifi).  It's weird that all of the images were there but none of the metadata for an entire trial
// TODO: voiceover seems to get sluggish over time.  I'm not sure why though.  It might be CPU throttling or something like that.
// TODO: getting a good anchor image independently of having a perpendicular plane would be good (might require state machine)
// TODO: investigate whether the JPEG compression will corrupt our results

import Foundation
import Firebase
import ARKit

struct ARFrameDataLog {
    let timestamp: Double
    let jpegData: Data
    let depthJpeg: Data?
    let planes: [ARPlaneAnchor]
    let pose: simd_float4x4
    let intrinsics: simd_float3x3
    let trueNorth: simd_float4x4?
    let meshes: [[String: [[Float]]]]?
    
    init(timestamp: Double, jpegData: Data, depthJpeg: Data?, intrinsics: simd_float3x3, planes: [ARPlaneAnchor], pose: simd_float4x4, trueNorth: simd_float4x4?, meshes: [[String: [[Float]]]]?) {
        self.timestamp = timestamp
        self.jpegData = jpegData
        self.depthJpeg = depthJpeg
        self.planes = planes
        self.intrinsics = intrinsics
        self.pose = pose
        self.trueNorth = trueNorth
        self.meshes = meshes
    }
    
    func metaDataAsJSON()->Data? {
        let body : [String: Any] = ["timestamp": timestamp, "type": "Hi", "pose": pose.asColumnMajorArray, "intrinsics": intrinsics.asColumnMajorArray, "trueNorth": trueNorth != nil ? trueNorth!.asColumnMajorArray : [], "planes": planes.map({["alignment": $0.alignment == .horizontal ? "horizontal": "vertical", "center": $0.center.asArray, "extent": $0.extent.asArray, "transform": $0.transform.asColumnMajorArray]})]
        if JSONSerialization.isValidJSONObject(body) {
            print("Metadata written into JSON")
            return try? JSONSerialization.data(withJSONObject: body, options: .prettyPrinted)
        } else {
            //NavigationController.shared.logString("Error: JSON is invalid for serialization \(body)")
            return nil
        }
    }
    
    func meshesToProtoBuf()->Data? {
        guard let meshes = meshes else {
            return nil
        }
        var meshesProto = MeshesProto()
        for mesh in meshes {
            var meshProto = MeshProto()
            var columnProtos: [Float4Proto] = []
            for column in mesh["transform"]! {
                var columnProto = Float4Proto()
                columnProto.x = column[0]
                columnProto.y = column[1]
                columnProto.z = column[2]
                columnProto.w = column[3]
                columnProtos.append(columnProto)
            }
            meshProto.transform.c1 = columnProtos[0]
            meshProto.transform.c2 = columnProtos[1]
            meshProto.transform.c3 = columnProtos[2]
            meshProto.transform.c4 = columnProtos[3]

            for (vert, normal) in zip(mesh["vertices"]!, mesh["normals"]!) {
                var vertexProto = VertexProto()
                vertexProto.x = vert[0]
                vertexProto.y = vert[1]
                vertexProto.z = vert[2]
                vertexProto.u = normal[0]
                vertexProto.v = normal[1]
                vertexProto.w = normal[2]
                meshProto.vertices.append(vertexProto)
            }
            meshesProto.meshes.append(meshProto)
        }
        return try? meshesProto.serializedData()
    }
}


class TrialManager {
    public static var shared = TrialManager()
    let uploadManager = UploadManager.shared
    var voiceFeedback: URL?
    var trialID: String?
    var poseLog: [(Double, simd_float4x4)] = []
    var trialLog: [(Double, Any)] = []
    var attributes: [String: Any] = [:]
    var configLog: [String: Bool]?
    var finalizedSet: Set<String> = []
    var lastBodyDetectionTime = Date()
    var baseTrialPath: String = ""
    var frameSequenceNumber: Int = 0
    
    private init() {
    }
    
    
    func addAudioFeedback(audioFileURL: URL) {
        voiceFeedback = audioFileURL
    }
    
    func addFrame(frame: ARFrameDataLog) {
        print("Add frame called")
        // if we saw a body recently, we can't log the data
        if -lastBodyDetectionTime.timeIntervalSinceNow > 1.0 {
            frameSequenceNumber += 1
            DispatchQueue.global(qos: .background).async { [baseTrialPath = self.baseTrialPath, frameSequenceNumber = self.frameSequenceNumber] in
                TrialManager.uploadAFrame(baseTrialPath: baseTrialPath, frameSequenceNumber: frameSequenceNumber, frame: frame)
            }
        }
    }
    
    func logString(logMessage: String) {
        let timestamp: Double
        if let currentFrame = ARViewProvider.shared.arView.session.currentFrame {
            timestamp = currentFrame.timestamp
        } else {
            timestamp = -1
        }
        trialLog.append((timestamp, logMessage))
    }
    
    func logDictionary(logDictionary: [String : Any]) {
        guard JSONSerialization.isValidJSONObject(logDictionary) else {
            return
        }
        let timestamp: Double
        if let currentFrame = ARViewProvider.shared.arView.session.currentFrame {
            timestamp = currentFrame.timestamp
        } else {
            timestamp = -1
        }
        trialLog.append((timestamp, logDictionary))
    }
    
    func logPose(pose: simd_float4x4, at time: Double) {
        poseLog.append((time, pose))
    }
    
    static private func uploadLog(trialLogToUse: [(Double, Any)], baseTrialPath: String) {
        guard let logJSON = try? JSONSerialization.data(withJSONObject: trialLogToUse.map({["timestamp": $0.0, "message": $0.1]}), options: .prettyPrinted) else {
            return
        }
        let logPath = "\(baseTrialPath)/log.json"
        UploadManager.shared.putData(logJSON, contentType: "application/json", fullPath: logPath)
    }
    
    static private func uploadPoses(poseLogToUse: [(Double, simd_float4x4)], baseTrialPath: String) {
        guard let poseJSON = try? JSONSerialization.data(withJSONObject: poseLogToUse.map({["timestamp": $0.0, "pose": $0.1.asColumnMajorArray]}), options: .prettyPrinted) else {
            return
        }
        let posesPath = "\(baseTrialPath)/poses.json"
        UploadManager.shared.putData(poseJSON, contentType: "application/json", fullPath: posesPath)
        print("Uploading poses")
    }
    
    static private func uploadConfig(configLogToUse: [String: Bool]?, attributesToUse: [String: Any], baseTrialPath: String) {
        guard let configLog = configLogToUse else {
            return
        }
        guard let configJSON = try? JSONSerialization.data(withJSONObject: configLog, options: .prettyPrinted) else {
            return
        }
        let configPath = "\(baseTrialPath)/config.json"
        UploadManager.shared.putData(configJSON, contentType: "application/json", fullPath: configPath)
        guard let attributeJSON = try? JSONSerialization.data(withJSONObject: attributesToUse, options: .prettyPrinted) else {
            return
        }
        let attributesPath = "\(baseTrialPath)/attributes.json"
        UploadManager.shared.putData(attributeJSON, contentType: "application/json", fullPath: attributesPath)
        print("Uploading configuration log")
    }
    
    static private func uploadAFrame(baseTrialPath: String, frameSequenceNumber: Int, frame: ARFrameDataLog) {
        let imagePath = "\(baseTrialPath)/\(String(format:"%04d", frameSequenceNumber))/frame.jpg"
        UploadManager.shared.putData(frame.jpegData, contentType: "image/jpeg", fullPath: imagePath)
        guard let frameMetaData = frame.metaDataAsJSON() else {
            //NavigationController.shared.logString("Error: failed to get frame metadata")
            return
        }
        if let depthJpeg = frame.depthJpeg {
            let depthImagePath = "\(baseTrialPath)/\(String(format:"%04d", frameSequenceNumber))/depthmap.jpg"
            UploadManager.shared.putData(depthJpeg, contentType: "image/jpeg", fullPath: depthImagePath)
        }
        let metaDataPath = "\(baseTrialPath)/\(String(format:"%04d", frameSequenceNumber))/framemetadata.json"
        UploadManager.shared.putData(frameMetaData, contentType: "application/json", fullPath: metaDataPath)
        if let meshData = frame.meshesToProtoBuf() {
            let meshDataPath = "\(baseTrialPath)/\(String(format:"%04d", frameSequenceNumber))/meshes.pb"
            // TODO: gzipping gives a 30-40% reduction.  let compressedData: Data = try! meshData.gzipped()
            UploadManager.shared.putData(meshData, contentType: "application/x-protobuf", fullPath: meshDataPath)
        }
        print("Uploading a frame?")
    }
    
    func finalizeTrial() {
        guard let trialID = self.trialID else {
            return
        }
        guard !self.finalizedSet.contains(trialID) else {
            // can't finalize the trial more than once
            return
        }
        finalizedSet.insert(trialID)
        // TODO: we converted the upload interfaces to static to try to fix a bug where the app was crashing.  This might not be an issue anymore, so we should revisit whether we can change back to the old interfaces
        // Upload audio to Firebase
        if let voiceFeedback = voiceFeedback, let data = try? Data(contentsOf: voiceFeedback) {
            let audioFeedbackPath = "\(baseTrialPath)/voiceFeedback.wav"
            UploadManager.shared.putData(data, contentType: "audio/wav", fullPath: audioFeedbackPath)
        }
        print("tpath", baseTrialPath)
        TrialManager.uploadLog(trialLogToUse: trialLog, baseTrialPath: baseTrialPath)
        TrialManager.uploadPoses(poseLogToUse: poseLog, baseTrialPath: baseTrialPath)
        TrialManager.uploadConfig(configLogToUse: configLog, attributesToUse: attributes, baseTrialPath: baseTrialPath)
    }
    
    func startTrial() {
        resetInternalState()
        trialID = UUID().uuidString
        logConfig()

        guard let user = Auth.auth().currentUser, let trialID = self.trialID else {
            print("User is not logged in")
            return
        }
        baseTrialPath = "\(user.uid)/\(trialID)"
        print("Starting trial")
    }
    
    func logConfig() {
        //configLog = CodesignConfiguration.shared.configAsDict()
    }
    
    func logAttribute(key: String, value: Any) {
        if JSONSerialization.isValidJSONObject([key: value]) {
            attributes[key] = value
        } else {
            //NavigationController.shared.logString("Unable to log \(key) as its value cannot be serialized to JSON")
        }
    }
    
    private func resetInternalState() {
        voiceFeedback = nil
        trialID = nil
        trialLog = []
        poseLog = []
        attributes = [:]
        configLog = nil
        frameSequenceNumber = 0
    }
    
    func processNewBodyDetectionStatus(bodyDetected: Bool) {
        if bodyDetected {
            lastBodyDetectionTime = Date()
        }
    }
}


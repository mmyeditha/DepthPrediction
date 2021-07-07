//
//  UploadManager.swift
//  LidarCane
//
//  Created by Paul Ruvolo on 5/10/21.
//
//  This class manages uploading files to Firebase in a persistent fashion.  It will automatically stop uploading data if not connected to the Internet, can retry multiple times, and persists the list of pending uploads if the app enters the background

import Foundation
import FirebaseStorage

fileprivate func doUploadJob(data: Data, contentType: String, fullPath: String, retriesLeft: Int, delayInSeconds: Double = 0) {
    UploadManager.shared.serialQueue.asyncAfter(deadline: .now() + delayInSeconds) {
        if !InternetConnectionUtil.isConnectedToNetwork() {
            if !UploadManager.overrideAllRetries {
                doUploadJob(data: data, contentType: contentType, fullPath: fullPath, retriesLeft: retriesLeft, delayInSeconds: 20)
            }
            return
        }

        let fileType = StorageMetadata()
        fileType.contentType = contentType
        let storageRef = Storage.storage().reference().child(fullPath)
            
        storageRef.putData(data, metadata: fileType) { (metadata, error) in
            if error != nil && retriesLeft > 0 && !UploadManager.overrideAllRetries {
                // Note: this block is usually never executed
                print("Error: \(error)")
                doUploadJob(data: data, contentType: contentType, fullPath: fullPath, retriesLeft: retriesLeft-1, delayInSeconds: 20)
            }
        }
    }
}

fileprivate func getURL(filename: String) -> URL {
    return FileManager().urls(for: .documentDirectory, in: .userDomainMask).first!.appendingPathComponent(filename)
}

/// The upload manager takes care of sending data to Firebase.  Currently we have commented out the section that allows upload jobs to be serialized to local storage: The manager will write the files that should be upload to the phone's local storage if the data cannot be uploaded to Firebase (e.g., if the app enters the background or if the Internet connection drops)
class UploadManager {
    public static let maximumRetryCount = 3
    public static let overrideAllRetries = true
    public static var shared = UploadManager()
    let serialQueue = DispatchQueue(label: "upload.serial.queue", qos: .background)

    private init() {
    }
    
    /// Add an upload job to the manager.  The manager will persist the job across the app entering the background or the internet conenction failing.
    /// - Parameters:
    ///   - uploadData: the data to upload
    ///   - contentType: the MIME content type
    ///   - fullPath: the path to the data on the storage bucket
    func putData(_ uploadData: Data, contentType: String, fullPath: String) {
        doUploadJob(data: uploadData, contentType: contentType, fullPath: fullPath, retriesLeft: UploadManager.maximumRetryCount)
        // Debug statement
        print("Data uploaded \(fullPath)")
    }
}

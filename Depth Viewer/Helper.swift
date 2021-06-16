// Stuff

import SwiftUI
import ARKit
import Foundation



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
  return PointCloud(width: width, height: height, depthData: depthCopy, confData: confCopy)
}

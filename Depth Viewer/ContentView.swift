//
//  ContentView.swift
//  Depth Viewer
//
//  Created by Neel Dhulipala, Mario Gergis, and Merwan Yeditha on 6/14/21.
//

import SwiftUI
import RealityKit
import ARKit
import VideoToolbox
import Vision

// let ARViewProvider = ARViewProvider()

struct ContentView : View {
    // @State var liveDepthPrediction = LiveDepthPrediction()
//    @State var showingPopover = false
    @ObservedObject var viewProvider: ARViewProvider = ARViewProvider.shared
    var body: some View {
        VStack(alignment: /*@START_MENU_TOKEN@*/.center/*@END_MENU_TOKEN@*/, spacing: 0, content: {
            Button("Capture", action: buttonPress)
            ARViewContainer()
            if ARViewProvider.shared.img != nil {
                Image(uiImage: ARViewProvider.shared.img!)
            }
//            Button("Freeze Image") {
//                ARViewProvider.shared.frozenImage = ARViewProvider.shared.arView.session.currentFrame?.capturedImage.toUIImage()
//                print("frozenImage", ARViewProvider.shared.frozenImage)
//                showingPopover = true
//            }
//            .popover(isPresented: $showingPopover) {
//                Image(uiImage: ARViewProvider.shared.frozenImage!).accessibility(hidden: false)
//            }
        })
    }
    
    func buttonPress() {
        ARViewProvider.shared.buttonPressed = true;
    }
    
}


struct ARViewContainer: UIViewRepresentable {
    
    func makeUIView(context: Context) -> ARView {
        // Load the "Box" scene from the "Experience" Reality File
        
        // Add the box anchor to the scene
        
        return ARViewProvider.shared.arView
        
    }
    
    
    func updateUIView(_ uiView: ARView, context: Context) {}
    
}

//#if DEBUG
//struct ContentView_Previews : PreviewProvider {
//    static var previews: some View {
//        ContentView()
//    }
//}
//#endif

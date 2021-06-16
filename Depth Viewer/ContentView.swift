//
//  ContentView.swift
//  Depth Viewer
//
//  Created by Merwan Yeditha on 6/14/21.
//

import SwiftUI
import RealityKit
import ARKit
import VideoToolbox
import Vision

// let ARViewProvider = ARViewProvider()

struct ContentView : View {
    // @State var liveDepthPrediction = LiveDepthPrediction()
    @ObservedObject var viewProvider: ARViewProvider = ARViewProvider.shared
    var body: some View {
        VStack{
            Button(action: {
                ARViewProvider.shared.buttonPress()
            }, label: {
                Text("Generate Cloud")
            })
            ARViewContainer()
                .edgesIgnoringSafeArea(.all)
            
            if let img = viewProvider.img {
                Image(uiImage: img)
                    .rotationEffect(.degrees(90))
            }
        }
        
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

#if DEBUG
struct ContentView_Previews : PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
#endif

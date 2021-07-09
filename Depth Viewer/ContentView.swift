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
    @State var sliderValue = 0.5
    @ObservedObject var viewProvider: ARViewProvider = ARViewProvider.shared
    var body: some View {
        VStack(alignment: /*@START_MENU_TOKEN@*/.center/*@END_MENU_TOKEN@*/, spacing: 0, content: {
            Button(action: {
                ARViewProvider.shared.buttonPress()
            }, label: {
                Text("Generate Cloud")
                    // .offset(x: 0, y: -30.0)
            })
            ARViewContainer()
            
            if let img = viewProvider.img {
                Image(uiImage: img)
            }
            Button(action: {
                ARViewProvider.shared.uploadPress()
            }, label: {
                Text("Upload Data")
            })
            .padding()
//            Slider(value: Binding( get: {
//                self.sliderValue
//            }, set: { (newVal) in
//                self.sliderValue = newVal
//                self.updateSlider()
//            }))
//            .padding(.all)
//            Text("Depth sensitivity: \(sliderValue)")
        })
    }
    
//    // Updates slider value in ARViewProvider
//    func updateSlider() {
//        ARViewProvider.shared.updateSliderValue(sliderValue: sliderValue)
//    }
    
}


struct ARViewContainer: UIViewRepresentable {
    
    func makeUIView(context: Context) -> ARSCNView {
        // Load the "Box" scene from the "Experience" Reality File
        
        // Add the box anchor to the scene
        
        return ARViewProvider.shared.arView
        
    }
    
    
    func updateUIView(_ uiView: ARSCNView, context: Context) {}
    
}

//#if DEBUG
//struct ContentView_Previews : PreviewProvider {
//    static var previews: some View {
//        ContentView()
//    }
//}
//#endif

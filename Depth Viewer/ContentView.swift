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
    @State dynamic var useFeaturePoints = false
    
    var body: some View {
        VStack(alignment: /*@START_MENU_TOKEN@*/.center/*@END_MENU_TOKEN@*/, spacing: 1.0, content: {
            VStack(alignment: .leading, spacing: 0, content: {
                // Uncomment this toggle to allow a toggle between predicting depth with feature points vs. FCRN
//                Toggle("Use Feature Points", isOn: $useFeaturePoints)
//                    .onChange(of: useFeaturePoints, perform: { value in
//                        viewProvider.useFeaturePoints = useFeaturePoints
//                    })
                Button("Meters", action: toggleMeters)
                Button("Feet", action: toggleFeet)
            })
            .buttonStyle(BlueButton())
            
            // Uncomment this line for a button to capture a CSV of the FCRN depth map to be stored in the AppData
//            Button("Capture", action: buttonPress)
            
            ARViewContainer()
            
            // Uncomment this code to show the depth heatmaps produced by FCRN
//            if ARViewProvider.shared.img != nil {
//                Image(uiImage: ARViewProvider.shared.img!)
//            }
        })
    }
    
    func buttonPress() {
        viewProvider.buttonPressed = true;
    }
    
    func toggleMeters() {
        viewProvider.meters = true;
        viewProvider.announce(announcement: "Meters");
        viewProvider.isAnnouncing = true;
    }
    
    func toggleFeet() {
        viewProvider.meters = false;
        viewProvider.announce(announcement: "Feet");
        viewProvider.isAnnouncing = true;
    }
    
}

struct BlueButton: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .padding()
            .background(Color(red: 0, green: 0, blue: 0.5))
            .foregroundColor(.white)
            .clipShape(Capsule())
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

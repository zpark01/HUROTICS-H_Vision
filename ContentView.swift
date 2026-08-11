//
//  ContentView.swift
//  H-Vision — data logger
//
//  Captures forward LiDAR depth + downward ultra-wide RGB simultaneously,
//  along with IMU at 100 Hz and an audio track used for voice labelling.
//
//  Required Info keys (TARGETS > Info):
//    Privacy - Camera Usage Description
//    Privacy - Microphone Usage Description
//    Application supports iTunes file sharing   = YES
//    Supports opening documents in place        = YES
//

import SwiftUI
import AVFoundation
import CoreVideo
import CoreMotion

// MARK: - Session recorder

/// Owns the session folder and all file writes. Everything goes through
/// a serial queue so the capture callbacks never block on disk I/O.
final class SessionRecorder {
    private(set) var isRecording = false
    private(set) var dir: URL?
    private var depthIndex = 0
    private var uwIndex = 0
    private var imuHandle: FileHandle?
    private var metaHandle: FileHandle?
    private let ioQueue = DispatchQueue(label: "recorder.io")

    func startNewSession() -> URL? {
        var result: URL?
        ioQueue.sync {
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let stamp = ISO8601DateFormatter().string(from: Date())
                .replacingOccurrences(of: ":", with: "-")   // colons are illegal in paths
            let d = docs.appendingPathComponent("session_\(stamp)", isDirectory: true)
            try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
            self.dir = d; self.depthIndex = 0; self.uwIndex = 0

            let imuURL = d.appendingPathComponent("imu.csv")
            FileManager.default.createFile(atPath: imuURL.path, contents: nil)
            self.imuHandle = try? FileHandle(forWritingTo: imuURL)
            self.imuHandle?.write("hostTime,ax,ay,az,gx,gy,gz,pitch,roll,yaw,gravx,gravy,gravz\n"
                                    .data(using: .utf8)!)

            let metaURL = d.appendingPathComponent("frames.csv")
            FileManager.default.createFile(atPath: metaURL.path, contents: nil)
            self.metaHandle = try? FileHandle(forWritingTo: metaURL)
            self.metaHandle?.write("kind,filename,hostTime,width,height\n".data(using: .utf8)!)

            self.isRecording = true
            result = d
            print("● recording started: \(d.lastPathComponent)")
        }
        return result
    }

    func stop() {
        ioQueue.sync {
            self.isRecording = false
            try? self.imuHandle?.close(); self.imuHandle = nil
            try? self.metaHandle?.close(); self.metaHandle = nil
            if let d = self.dir { print("■ recording stopped: \(d.path)") }
        }
    }

    /// Written once per session. Needed to unproject depth pixels to 3D.
    func writeCameraInfo(_ text: String) {
        ioQueue.async {
            guard let d = self.dir else { return }
            try? text.write(to: d.appendingPathComponent("camera_info.txt"),
                            atomically: true, encoding: .utf8)
            print("camera_info.txt written\n\(text)")
        }
    }

    /// Raw float32, metres. Not compressed — we need the exact values downstream.
    func writeDepth(_ floats: Data, hostTime: Double, w: Int, h: Int) {
        ioQueue.async {
            guard self.isRecording, let d = self.dir else { return }
            let name = String(format: "depth_%06d.f32", self.depthIndex); self.depthIndex += 1
            try? floats.write(to: d.appendingPathComponent(name))
            self.metaHandle?.write("depth,\(name),\(hostTime),\(w),\(h)\n".data(using: .utf8)!)
        }
    }

    func writeUW(_ jpeg: Data, hostTime: Double, w: Int, h: Int) {
        ioQueue.async {
            guard self.isRecording, let d = self.dir else { return }
            let name = String(format: "uw_%06d.jpg", self.uwIndex); self.uwIndex += 1
            try? jpeg.write(to: d.appendingPathComponent(name))
            self.metaHandle?.write("uw,\(name),\(hostTime),\(w),\(h)\n".data(using: .utf8)!)
        }
    }

    func writeIMU(_ s: MotionManager.Sample) {
        ioQueue.async {
            guard self.isRecording else { return }
            let line = String(format:
                "%.6f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n",
                s.hostTime, s.ax, s.ay, s.az, s.gx, s.gy, s.gz,
                s.pitch, s.roll, s.yaw, s.gravX, s.gravY, s.gravZ)
            self.imuHandle?.write(line.data(using: .utf8)!)
        }
    }
}

// MARK: - Audio

final class AudioRecorder {
    private var recorder: AVAudioRecorder?

    /// Prefers a Bluetooth mic when one is connected. The phone sits on the
    /// waist, so its built-in mic is far from the mouth and transcription
    /// accuracy drops noticeably.
    func start(in dir: URL) {
        let audio = AVAudioSession.sharedInstance()
        try? audio.setCategory(.playAndRecord, mode: .default,
                               options: [.mixWithOthers, .allowBluetooth, .allowBluetoothA2DP])
        if let bt = audio.availableInputs?.first(where: {
            [.bluetoothHFP, .bluetoothLE].contains($0.portType)
        }) {
            try? audio.setPreferredInput(bt)
            print("using Bluetooth mic: \(bt.portName)")
        } else {
            print("using built-in mic")
        }
        try? audio.setActive(true, options: .notifyOthersOnDeactivation)

        let url = dir.appendingPathComponent("audio.m4a")
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
            AVSampleRateKey: 16000,        // speech recognition doesn't need more
            AVNumberOfChannelsKey: 1,
            AVEncoderAudioQualityKey: AVAudioQuality.medium.rawValue
        ]
        do {
            let rec = try AVAudioRecorder(url: url, settings: settings)
            rec.prepareToRecord()
            // Anchor for converting positions inside the audio file back to hostTime.
            let startHost = CMClockGetTime(CMClockGetHostTimeClock()).seconds
            let ok = rec.record()
            self.recorder = rec
            try? String(format: "%.6f", startHost)
                .write(to: dir.appendingPathComponent("audio_start.txt"),
                       atomically: true, encoding: .utf8)
            print(ok ? "audio started, startHost=\(startHost)" : "audio failed to start")
        } catch { print("audio recorder error: \(error)") }
    }

    func stop() { recorder?.stop(); recorder = nil }
}

// MARK: - Capture

/// Forward camera runs without the mirror on purpose: reflected light throws
/// off the LiDAR depth estimate. Only the ultra-wide, which we use for RGB
/// only, goes through the 45° prism.
final class MultiCamManager: NSObject, ObservableObject,
                             AVCaptureDepthDataOutputDelegate,
                             AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureMultiCamSession()
    private let sessionQueue = DispatchQueue(label: "multicam.session")
    private let depthOutput = AVCaptureDepthDataOutput()
    private let uwOutput = AVCaptureVideoDataOutput()
    private let procQueue = DispatchQueue(label: "multicam.proc")

    @Published var isAuthorized = false
    @Published var depthImage: CGImage?
    @Published var uwImage: CGImage?
    @Published var centerDistance: Float = 0
    @Published var validRatio: Float = 0
    @Published var nearRatio: Float = 0        // share of pixels under 1 m
    @Published var focusValue: Float = 0.5
    @Published var focusSupported = true
    @Published var intrinsicsInfo = "waiting"

    private var uwDevice: AVCaptureDevice?
    private var lidarDevice: AVCaptureDevice?
    private var camInfoSaved = false

    let recorder = SessionRecorder()
    let audio = AudioRecorder()
    private let ciContext = CIContext()

    func start() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            DispatchQueue.main.async { self.isAuthorized = true }; configure()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async { self?.isAuthorized = granted }
                if granted { self?.configure() }
            }
        default:
            DispatchQueue.main.async { self.isAuthorized = false }
        }
    }

    func startRecording() {
        camInfoSaved = false
        if let dir = recorder.startNewSession() { audio.start(in: dir) }
    }

    func stopRecording() { audio.stop(); recorder.stop() }

    private func configure() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            guard AVCaptureMultiCamSession.isMultiCamSupported else {
                print("multi-cam not supported on this device"); return
            }
            self.session.beginConfiguration()

            // Forward: LiDAR depth, no mirror.
            guard let lidar = AVCaptureDevice.default(.builtInLiDARDepthCamera,
                                                      for: .video, position: .back),
                  let lidarInput = try? AVCaptureDeviceInput(device: lidar),
                  self.session.canAddInput(lidarInput) else {
                print("could not open LiDAR input"); self.session.commitConfiguration(); return
            }
            self.lidarDevice = lidar
            self.session.addInputWithNoConnections(lidarInput)
            guard self.session.canAddOutput(self.depthOutput) else {
                self.session.commitConfiguration(); return
            }
            self.session.addOutputWithNoConnections(self.depthOutput)
            self.depthOutput.isFilteringEnabled = true
            self.depthOutput.setDelegate(self, callbackQueue: self.procQueue)
            if let p = lidarInput.ports(for: .depthData,
                                        sourceDeviceType: .builtInLiDARDepthCamera,
                                        sourceDevicePosition: .back).first {
                let c = AVCaptureConnection(inputPorts: [p], output: self.depthOutput)
                if self.session.canAddConnection(c) { self.session.addConnection(c) }
            }

            // Downward: ultra-wide through the prism, RGB only.
            guard let uw = AVCaptureDevice.default(.builtInUltraWideCamera,
                                                   for: .video, position: .back),
                  let uwInput = try? AVCaptureDeviceInput(device: uw),
                  self.session.canAddInput(uwInput) else {
                print("could not open ultra-wide input"); self.session.commitConfiguration(); return
            }
            self.session.addInputWithNoConnections(uwInput)
            self.uwDevice = uw
            guard self.session.canAddOutput(self.uwOutput) else {
                self.session.commitConfiguration(); return
            }
            self.session.addOutputWithNoConnections(self.uwOutput)
            self.uwOutput.setSampleBufferDelegate(self, queue: self.procQueue)
            if let p = uwInput.ports(for: .video,
                                     sourceDeviceType: .builtInUltraWideCamera,
                                     sourceDevicePosition: .back).first {
                let c = AVCaptureConnection(inputPorts: [p], output: self.uwOutput)
                if self.session.canAddConnection(c) { self.session.addConnection(c) }
            }

            self.session.commitConfiguration()
            self.session.startRunning()

            let supported = uw.isLockingFocusWithCustomLensPositionSupported
            DispatchQueue.main.async { self.focusSupported = supported }
            self.setUWFocus(self.focusValue)

            let fov = lidar.activeFormat.videoFieldOfView
            DispatchQueue.main.async { self.intrinsicsInfo = String(format: "FOV %.1f°", fov) }
        }
    }

    /// Autofocus tends to lock onto the mirror surface instead of the legs
    /// reflected in it, so the lens position is pinned manually.
    func setUWFocus(_ pos: Float) {
        sessionQueue.async { [weak self] in
            guard let self = self, let dev = self.uwDevice else { return }
            do {
                try dev.lockForConfiguration()
                if dev.isLockingFocusWithCustomLensPositionSupported {
                    dev.setFocusModeLocked(lensPosition: pos, completionHandler: nil)
                }
                dev.unlockForConfiguration()
            } catch { print("focus lock failed: \(error)") }
        }
    }

    func depthDataOutput(_ output: AVCaptureDepthDataOutput,
                         didOutput depthData: AVDepthData,
                         timestamp: CMTime, connection: AVCaptureConnection) {
        let conv = depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        let map = conv.depthDataMap
        let host = timestamp.seconds
        let w = CVPixelBufferGetWidth(map), h = CVPixelBufferGetHeight(map)

        if recorder.isRecording && !camInfoSaved {
            camInfoSaved = true
            var txt = "depth_width \(w)\ndepth_height \(h)\n"
            if let dev = lidarDevice {
                txt += String(format: "hfov_deg %.4f\n", dev.activeFormat.videoFieldOfView)
            }
            // Intrinsics aren't always exposed; FOV is the fallback.
            if let cal = depthData.cameraCalibrationData {
                let m = cal.intrinsicMatrix
                let ref = cal.intrinsicMatrixReferenceDimensions
                txt += String(format: "fx %.4f\nfy %.4f\ncx %.4f\ncy %.4f\n",
                              m.columns.0.x, m.columns.1.y, m.columns.2.x, m.columns.2.y)
                txt += "ref_width \(Int(ref.width))\nref_height \(Int(ref.height))\n"
                txt += String(format: "pixel_size_mm %.6f\n", cal.pixelSize)
            } else {
                txt += "intrinsics unavailable (use hfov_deg)\n"
            }
            recorder.writeCameraInfo(txt)
            DispatchQueue.main.async { self.intrinsicsInfo = "saved \(w)x\(h)" }
        }

        if recorder.isRecording, let data = copyFloatBuffer(map) {
            recorder.writeDepth(data, hostTime: host, w: w, h: h)
        }
        renderDepth(map)
    }

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let host = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        let ci = CIImage(cvPixelBuffer: pb)
        guard let cg = ciContext.createCGImage(ci, from: ci.extent) else { return }
        if recorder.isRecording {
            let ui = UIImage(cgImage: cg, scale: 1, orientation: .right)
            if let jpeg = ui.jpegData(compressionQuality: 0.6) {
                recorder.writeUW(jpeg, hostTime: host, w: cg.width, h: cg.height)
            }
        }
        DispatchQueue.main.async { self.uwImage = cg }
    }

    /// Rows in a CVPixelBuffer are padded, so copy row by row into a packed array.
    private func copyFloatBuffer(_ pb: CVPixelBuffer) -> Data? {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let src = base.assumingMemoryBound(to: Float32.self)
        let row = CVPixelBufferGetBytesPerRow(pb) / MemoryLayout<Float32>.size
        var out = [Float32](repeating: 0, count: w*h)
        for y in 0..<h { for x in 0..<w { out[y*w+x] = src[y*row+x] } }
        return out.withUnsafeBytes { Data($0) }
    }

    /// Grayscale preview plus two coverage numbers. nearRatio is a quick check
    /// that the camera is actually seeing ground and not just the far wall.
    private func renderDepth(_ depthMap: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
        let w = CVPixelBufferGetWidth(depthMap), h = CVPixelBufferGetHeight(depthMap)
        guard let base = CVPixelBufferGetBaseAddress(depthMap) else { return }
        let buf = base.assumingMemoryBound(to: Float32.self)
        let row = CVPixelBufferGetBytesPerRow(depthMap) / MemoryLayout<Float32>.size
        let center = buf[(h/2)*row + (w/2)]

        var validCount = 0, nearCount = 0
        let total = w * h
        var px = [UInt8](repeating: 0, count: total)
        let maxR: Float = 3.0
        for y in 0..<h {
            for x in 0..<w {
                let d = buf[y*row + x]
                if d.isFinite && d > 0 {
                    validCount += 1
                    if d < 1.0 { nearCount += 1 }
                    px[y*w+x] = UInt8(max(0, min(1, 1 - d/maxR)) * 255)
                }
            }
        }
        let vRatio = Float(validCount) / Float(total)
        let nRatio = Float(nearCount) / Float(total)

        let cs = CGColorSpaceCreateDeviceGray()
        guard let ctx = CGContext(data: &px, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w, space: cs, bitmapInfo: 0),
              let cg = ctx.makeImage() else { return }
        DispatchQueue.main.async {
            self.depthImage = cg
            self.centerDistance = center
            self.validRatio = vRatio
            self.nearRatio = nRatio
        }
    }
}

// MARK: - IMU

final class MotionManager: ObservableObject {
    private let mm = CMMotionManager()
    private let queue = OperationQueue()
    @Published var pitchDeg: Double = 0
    @Published var rollDeg: Double = 0
    @Published var sampleCount: Int = 0
    @Published var gravityStr: String = ""

    struct Sample {
        let hostTime: Double
        let ax: Double; let ay: Double; let az: Double
        let gx: Double; let gy: Double; let gz: Double
        let pitch: Double; let roll: Double; let yaw: Double
        // Gravity is logged as well — pitch/roll depend on the reference frame,
        // the gravity vector doesn't, which makes tilt correction unambiguous.
        let gravX: Double; let gravY: Double; let gravZ: Double
    }

    var onSample: ((Sample) -> Void)?

    func start() {
        guard mm.isDeviceMotionAvailable else { print("device motion unavailable"); return }
        mm.deviceMotionUpdateInterval = 1.0/100.0
        mm.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: queue) { [weak self] m, _ in
            guard let self = self, let m = m else { return }
            let host = self.hostSeconds(m.timestamp)
            let s = Sample(hostTime: host,
                           ax: m.userAcceleration.x, ay: m.userAcceleration.y, az: m.userAcceleration.z,
                           gx: m.rotationRate.x, gy: m.rotationRate.y, gz: m.rotationRate.z,
                           pitch: m.attitude.pitch, roll: m.attitude.roll, yaw: m.attitude.yaw,
                           gravX: m.gravity.x, gravY: m.gravity.y, gravZ: m.gravity.z)
            self.onSample?(s)
            let deg = 180.0 / .pi
            DispatchQueue.main.async {
                self.pitchDeg = m.attitude.pitch*deg
                self.rollDeg  = m.attitude.roll*deg
                self.gravityStr = String(format: "g(%.2f,%.2f,%.2f)",
                                         m.gravity.x, m.gravity.y, m.gravity.z)
                self.sampleCount += 1
            }
        }
    }

    /// CoreMotion timestamps are relative to boot. Shift them onto the same
    /// host clock the capture callbacks use so everything lines up later.
    private func hostSeconds(_ ts: TimeInterval) -> Double {
        let boot = ProcessInfo.processInfo.systemUptime
        let nowHost = CMClockGetTime(CMClockGetHostTimeClock()).seconds
        return nowHost - (boot - ts)
    }

    func stop() { mm.stopDeviceMotionUpdates() }
}

// MARK: - UI

struct ContentView: View {
    @StateObject private var cam = MultiCamManager()
    @StateObject private var motion = MotionManager()
    @State private var recording = false

    var body: some View {
        VStack(spacing: 2) {
            ZStack {
                Color.black
                if let d = cam.depthImage {
                    Image(decorative: d, scale: 1, orientation: .right)
                        .resizable().scaledToFit()
                }
                VStack { Spacer()
                    VStack(spacing: 2) {
                        Text(String(format: "forward depth · center %.2f m", cam.centerDistance))
                        Text(String(format: "valid %.0f%% · under 1m %.0f%%",
                                    cam.validRatio*100, cam.nearRatio*100))
                            .foregroundColor(cam.nearRatio < 0.05 ? .red : .green)
                    }
                    .font(.caption).bold()
                    .padding(6).background(.black.opacity(0.5))
                    .cornerRadius(8).padding(.bottom, 6)
                }
            }
            ZStack {
                Color.black
                if let u = cam.uwImage {
                    Image(decorative: u, scale: 1, orientation: .right)
                        .resizable().scaledToFit()
                }
                VStack { Spacer()
                    Text("downward ultra-wide (mirror · legs)")
                        .font(.caption).bold().foregroundColor(.green)
                        .padding(6).background(.black.opacity(0.5))
                        .cornerRadius(8).padding(.bottom, 6)
                }
            }
        }
        .overlay(alignment: .topLeading) {
            VStack(alignment: .leading, spacing: 2) {
                Text(String(format: "pitch %.1f°  roll %.1f°", motion.pitchDeg, motion.rollDeg))
                Text(motion.gravityStr)
                Text("IMU \(motion.sampleCount) · cam \(cam.intrinsicsInfo)")
            }
            .font(.system(.caption2, design: .monospaced)).bold().foregroundColor(.cyan)
            .padding(8).background(.black.opacity(0.55)).cornerRadius(8)
            .padding(.top, 50).padding(.leading, 12)
        }
        .overlay(alignment: .topTrailing) {
            Text(recording ? "REC" : "idle")
                .font(.caption2).bold().foregroundColor(recording ? .red : .white)
                .padding(8).background(.black.opacity(0.55)).cornerRadius(8)
                .padding(.top, 50).padding(.trailing, 12)
        }
        .overlay(alignment: .bottomLeading) {
            VStack(alignment: .leading, spacing: 4) {
                Text(cam.focusSupported
                     ? String(format: "ultra-wide focus: %.2f", cam.focusValue)
                     : "fixed focus")
                    .font(.caption2).bold().foregroundColor(.white)
                if cam.focusSupported {
                    Slider(value: Binding(
                        get: { Double(cam.focusValue) },
                        set: { cam.focusValue = Float($0); cam.setUWFocus(Float($0)) }
                    ), in: 0...1).frame(width: 180)
                }
            }
            .padding(8).background(.black.opacity(0.55)).cornerRadius(8)
            .padding(.leading, 16).padding(.bottom, 96)
        }
        .overlay(alignment: .bottom) {
            Button {
                if recording { cam.stopRecording() } else { cam.startRecording() }
                recording.toggle()
            } label: {
                Text(recording ? "STOP" : "RECORD")
                    .font(.title3).bold().foregroundColor(.white)
                    .padding(.horizontal, 40).padding(.vertical, 14)
                    .background(recording ? Color.red : Color.blue).clipShape(Capsule())
            }.padding(.bottom, 30)
        }
        .ignoresSafeArea()
        .onAppear {
            cam.start()
            motion.onSample = { [weak cam] s in cam?.recorder.writeIMU(s) }
            motion.start()
        }
        .onDisappear { motion.stop() }
    }
}

#Preview { ContentView() }

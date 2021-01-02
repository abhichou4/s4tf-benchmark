import Foundation
import FoundationNetworking


public func downloadDataset(
    file filename: String,
    fileExtension ext: String = "gz", // modify later
    from remoteBaseDirectory: String, 
    to localBaseDirectory: String
) -> Void {
    let localBaseURL = URL(fileURLWithPath: localBaseDirectory)
    if !FileManager.default.fileExists(atPath: localBaseURL.path) {
       do {
            try FileManager.default.createDirectory(atPath: localBaseURL.path, withIntermediateDirectories: true)
        } catch {
            print("Failed to created \(localBaseURL.path)");
        }
        print("\(localBaseURL.path) created.")
    }

    var localFileURL : URL { localBaseURL.appendingPathComponent(filename) }
    if FileManager.default.fileExists(atPath: localFileURL.path) {
        print("\(localFileURL.path) already downloaded.")
        return
    } 
    var archiveFileURL : URL { localFileURL.appendingPathExtension(ext) }
    
    guard let remoteBaseURL = URL(string: remoteBaseDirectory) else {
        fatalError("Failed to create base URL: \(remoteBaseDirectory)")
    }
    var remoteFileURL : URL { remoteBaseURL.appendingPathComponent(filename).appendingPathExtension(ext) }
    
    let downloadedFile : Data?
    do {
        downloadedFile = try Data(contentsOf: remoteFileURL)
        try downloadedFile!.write(to: archiveFileURL)
    } catch {
        fatalError("Failed to save content to \(archiveFileURL.path)")
    }
    print("Downloaded archive \(filename). Extracting...")
    
    let task = Process(); task.executableURL = URL(fileURLWithPath: "/bin/gunzip")
    task.arguments = ["-dk", archiveFileURL.path]
    do {
        try task.run()
        task.waitUntilExit()
    } catch {
        fatalError("Failed to extract achive.")
    }
    print("\(filename) downloaded successfully.")
}
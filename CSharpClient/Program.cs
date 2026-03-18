/*
================================================================================
  Program.cs  –  Demo / Integration Example
================================================================================
  Shows how your C# machine-control code calls the Python wafer-alignment
  server over ZeroMQ using file paths.

  Steps to use in your project
  ────────────────────────────
  1. Install NetMQ via NuGet (works for both x86 and x64):
       PM> Install-Package NetMQ

  2. Copy WaferAlignmentClient.cs into your C# project.

  3. Start the Python server ONCE (before running your machine code):
       python zmq_server.py --port 5555

  4. From C# call LoadTemplate() once at startup, then call Match() for
     every new live image the camera captures.
================================================================================
*/

using System;
using WaferAlignment;

class Program
{
    static void Main(string[] args)
    {
        // ── Server connection parameters ──────────────────────────────────────
        const string host = "localhost";   // change to IP if server is remote
        const int    port = 5555;

        // ── Image paths – the Python server reads these files directly ────────
        // Both paths must be visible to the Python process.
        // If C# and Python run on the same PC, use absolute local paths.
        // If they run on different PCs, use a shared network path (UNC or mapped drive).
        string templatePath = @"C:\Images\wafer_template.png";
        string livePath     = @"C:\Images\live_capture.png";

        Console.WriteLine("=== Wafer Alignment via ZeroMQ ===\n");

        using var client = new WaferAlignmentClient(host, port);

        // ── 1. Sanity check ───────────────────────────────────────────────────
        Console.Write("PING ... ");
        bool ok = client.Ping();
        Console.WriteLine(ok ? "server is alive ✓" : "NO RESPONSE – is zmq_server.py running?");
        if (!ok) return;

        // ── 2. (Optional) Tune matcher parameters ────────────────────────────
        client.SetConfig(numFeatures: 200, threshold: 50.0, angleStep: 5.0);

        // ── 3. Load the reference template (done once, or when recipe changes) ─
        Console.Write($"\nLoading template: {templatePath} ... ");
        bool loaded = client.LoadTemplate(templatePath);
        // With an optional ROI (crop 400×400 pixels at x=100, y=80):
        // bool loaded = client.LoadTemplate(templatePath, roi: new[] { 100, 80, 400, 400 });
        Console.WriteLine(loaded ? "ok ✓" : "FAILED");
        if (!loaded) return;

        // ── 4. Alignment loop – call for every camera frame ──────────────────
        //
        // In your real machine code, replace this loop with your camera
        // acquisition trigger.  Save the image to disk (or a shared path),
        // then call Match() with that path.
        //
        for (int frame = 1; frame <= 3; frame++)
        {
            // ▸ Your machine code saves the latest camera image here:
            string currentLivePath = livePath;   // or e.g. $@"C:\Images\frame_{frame:D4}.png"

            Console.WriteLine($"\n[Frame {frame}] Matching {currentLivePath} ...");
            AlignmentResult result = client.Match(currentLivePath);

            switch (result.Status)
            {
                case "ok":
                    Console.WriteLine($"  ┌─ Alignment Result ──────────────────┐");
                    Console.WriteLine($"  │  X     = {result.X,8:F2} px");
                    Console.WriteLine($"  │  Y     = {result.Y,8:F2} px");
                    Console.WriteLine($"  │  Angle = {result.Angle,8:F3} °");
                    Console.WriteLine($"  │  Score = {result.Score,8:F1} / 100");
                    Console.WriteLine($"  └──────────────────────────────────────┘");

                    // ▸ Pass correction to your motion controller:
                    // motionController.MoveRelative(result.X, result.Y);
                    // motionController.RotateRelative(result.Angle);
                    break;

                case "no_match":
                    Console.WriteLine("  ⚠  No match found – wafer may be outside FOV.");
                    break;

                default:
                    Console.WriteLine($"  ✗  Error: {result.Error}");
                    break;
            }
        }

        // ── 5. (Optional) Shut the Python server down when done ──────────────
        // client.Shutdown();

        Console.WriteLine("\nDone.");
    }
}

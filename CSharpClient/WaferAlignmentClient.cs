/*
================================================================================
  WaferAlignmentClient.cs
================================================================================
  C# ZeroMQ REQ client for the Python wafer-alignment server (zmq_server.py).

  Works with BOTH x86 (32-bit) and x64 (64-bit) builds.
  NuGet dependency:  NetMQ  (install via NuGet Package Manager)
      Install-Package NetMQ
      (NetMQ is pure-managed C# – no native DLL required, so it runs on
       both 32-bit and 64-bit platforms without recompiling.)

  Protocol – all messages are UTF-8 JSON strings:
  ─────────────────────────────────────────────────
    PING
      Request  : { "cmd": "PING" }
      Response : { "status": "pong" }

    SET_CONFIG  (optional – tune the matcher before loading a template)
      Request  : { "cmd": "SET_CONFIG",
                   "num_features": 200,
                   "threshold":    50.0,
                   "angle_step":   5.0  }
      Response : { "status": "ok" }

    LOAD_TEMPLATE
      Request  : { "cmd": "LOAD_TEMPLATE",
                   "template_path": "C:\\images\\template.png",
                   "roi": [x, y, w, h]   // optional
                 }
      Response : { "status": "ok" }
              or { "status": "error", "message": "..." }

    MATCH
      Request  : { "cmd": "MATCH",
                   "search_path": "C:\\images\\search.png" }
      Response : { "status": "ok",
                   "x": 512.0, "y": 310.0,
                   "angle": -2.5, "score": 87.3 }
              or { "status": "no_match" }
              or { "status": "error",  "message": "..." }

    SHUTDOWN
      Request  : { "cmd": "SHUTDOWN" }
      Response : { "status": "ok" }
================================================================================
*/

using System;
using System.Text.Json;          // .NET 5+  or  use Newtonsoft.Json on .NET 4.x
using System.Text.Json.Nodes;
using NetMQ;
using NetMQ.Sockets;

namespace WaferAlignment
{
    // =========================================================================
    //  Result types
    // =========================================================================

    /// <summary>Returned by <see cref="WaferAlignmentClient.Match"/>.</summary>
    public class AlignmentResult
    {
        public bool   Success { get; set; }
        public string Status  { get; set; } = "";   // "ok" | "no_match" | "error"
        public string Error   { get; set; } = "";

        // Only valid when Success == true
        public double X     { get; set; }
        public double Y     { get; set; }
        public double Angle { get; set; }
        public double Score { get; set; }

        public override string ToString() =>
            Success
                ? $"X={X:F2}  Y={Y:F2}  Angle={Angle:F3}°  Score={Score:F1}"
                : $"[{Status}] {Error}";
    }

    /// <summary>Returned by <see cref="WaferAlignmentClient.FindEdge"/>.</summary>
    public class EdgeFindResult
    {
        public bool   Success { get; set; }
        public string Status  { get; set; } = "";   // "ok" | "no_edge" | "error"
        public string Error   { get; set; } = "";

        // Line parameters (ax + by + c = 0)
        public double A { get; set; }
        public double B { get; set; }
        public double C { get; set; }
        public double Vx { get; set; }
        public double Vy { get; set; }
        public double X0 { get; set; }
        public double Y0 { get; set; }

        // Endpoints (depends on orientation)
        public double? XTop { get; set; }
        public double? XBot { get; set; }
        public double? YLeft { get; set; }
        public double? YRight { get; set; }

        public override string ToString() =>
            Success
                ? $"Line [a:{A:F2}, b:{B:F2}, c:{C:F2}]"
                : $"[{Status}] {Error}";
    }

    // =========================================================================
    //  Client
    // =========================================================================

    /// <summary>
    /// Thread-safe ZeroMQ REQ/REP client for the Python wafer-alignment server.
    /// <para>
    /// Usage:
    /// <code>
    ///   using var client = new WaferAlignmentClient("localhost", 5555);
    ///   client.Ping();
    ///   client.LoadTemplate(@"C:\images\template.png");
    ///   var result = client.Match(@"C:\images\live.png");
    ///   Console.WriteLine(result);
    /// </code>
    /// </para>
    /// </summary>
    public sealed class WaferAlignmentClient : IDisposable
    {
        // ── fields ────────────────────────────────────────────────────────────
        private readonly RequestSocket _socket;
        private readonly NetMQPoller   _poller;   // keeps the NetMQ runtime alive
        private readonly string        _address;
        private bool _disposed;

        // ── timeout (milliseconds) ────────────────────────────────────────────
        /// <summary>How long to wait for a server reply (default 10 s).</summary>
        public int TimeoutMs { get; set; } = 10_000;

        // =====================================================================
        //  Constructor / Dispose
        // =====================================================================

        /// <param name="host">Hostname or IP of the Python server (e.g. "localhost").</param>
        /// <param name="port">TCP port the server is bound to (default 5555).</param>
        public WaferAlignmentClient(string host = "localhost", int port = 5555)
        {
            _address = $"tcp://{host}:{port}";
            _socket  = new RequestSocket();
            _socket.Connect(_address);
            Console.WriteLine($"[WaferAlignmentClient] Connected to {_address}");
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            try { _socket.Close(); } catch { /* ignore */ }
            _socket.Dispose();
        }

        // =====================================================================
        //  Public API
        // =====================================================================

        /// <summary>Send PING – returns true if server replies "pong".</summary>
        public bool Ping()
        {
            var resp = SendReceive(new { cmd = "PING" });
            return resp?["status"]?.GetValue<string>() == "pong";
        }

        /// <summary>
        /// Optional: reconfigure the matcher before loading a template.
        /// </summary>
        /// <param name="numFeatures">Number of template features (default 200).</param>
        /// <param name="threshold">Match score threshold 0-100 (default 50).</param>
        /// <param name="angleStep">Degrees between rotations (default 5).</param>
        public bool SetConfig(int numFeatures = 200, double threshold = 50.0, double angleStep = 5.0)
        {
            var resp = SendReceive(new
            {
                cmd          = "SET_CONFIG",
                num_features = numFeatures,
                threshold    = threshold,
                angle_step   = angleStep
            });
            return resp?["status"]?.GetValue<string>() == "ok";
        }

        /// <summary>
        /// Tell the server which template image to use.
        /// The path must be accessible by the Python server process.
        /// </summary>
        /// <param name="templatePath">Absolute path to the template PNG/BMP.</param>
        /// <param name="roi">
        /// Optional region of interest [x, y, width, height] (pixels).
        /// Pass null to use the whole image.
        /// </param>
        /// <returns>True on success.</returns>
        public bool LoadTemplate(string templatePath, int[]? roi = null)
        {
            if (string.IsNullOrWhiteSpace(templatePath))
                throw new ArgumentException("templatePath must not be empty.");

            object payload = roi != null
                ? (object)new { cmd = "LOAD_TEMPLATE", template_path = templatePath, roi }
                : (object)new { cmd = "LOAD_TEMPLATE", template_path = templatePath };

            var resp = SendReceive(payload);
            if (resp == null) return false;

            string status = resp["status"]?.GetValue<string>() ?? "";
            if (status != "ok")
            {
                string msg = resp["message"]?.GetValue<string>() ?? "(no message)";
                Console.Error.WriteLine($"[WaferAlignmentClient] LOAD_TEMPLATE error: {msg}");
                return false;
            }
            return true;
        }

        /// <summary>
        /// Run alignment on a live (search) image.
        /// A template must have been loaded first via <see cref="LoadTemplate"/>.
        /// </summary>
        /// <param name="searchPath">Absolute path to the live camera image.</param>
        /// <returns><see cref="AlignmentResult"/> with X, Y, Angle, Score on success.</returns>
        public AlignmentResult Match(string searchPath)
        {
            if (string.IsNullOrWhiteSpace(searchPath))
                throw new ArgumentException("searchPath must not be empty.");

            var resp = SendReceive(new { cmd = "MATCH", search_path = searchPath });

            if (resp == null)
                return new AlignmentResult { Status = "error", Error = "No response from server." };

            string status = resp["status"]?.GetValue<string>() ?? "error";

            return status switch
            {
                "ok" => new AlignmentResult
                {
                    Success = true,
                    Status  = "ok",
                    X       = resp["x"]?.GetValue<double>()     ?? 0,
                    Y       = resp["y"]?.GetValue<double>()     ?? 0,
                    Angle   = resp["angle"]?.GetValue<double>() ?? 0,
                    Score   = resp["score"]?.GetValue<double>() ?? 0,
                },
                "no_match" => new AlignmentResult { Status = "no_match" },
                _ => new AlignmentResult
                {
                    Status = "error",
                    Error  = resp["message"]?.GetValue<string>() ?? "(unknown error)"
                }
            };
        }

        /// <summary>
        /// Find the wafer edge in the given image.
        /// </summary>
        /// <param name="searchPath">Absolute path to the live camera image.</param>
        /// <param name="scanDirection">Direction to scan (e.g. "LEFT", "RIGHT", "TOP", "BOTTOM").</param>
        /// <returns><see cref="EdgeFindResult"/> with line parameters and endpoints on success.</returns>
        public EdgeFindResult FindEdge(string searchPath, string scanDirection = "LEFT")
        {
            if (string.IsNullOrWhiteSpace(searchPath))
                throw new ArgumentException("searchPath must not be empty.");

            var resp = SendReceive(new { cmd = "FIND_EDGE", search_path = searchPath, scan_direction = scanDirection });

            if (resp == null)
                return new EdgeFindResult { Status = "error", Error = "No response from server." };

            string status = resp["status"]?.GetValue<string>() ?? "error";

            if (status == "ok")
            {
                var result = new EdgeFindResult
                {
                    Success = true,
                    Status = "ok",
                    A = resp["a"]?.GetValue<double>() ?? 0,
                    B = resp["b"]?.GetValue<double>() ?? 0,
                    C = resp["c"]?.GetValue<double>() ?? 0,
                    Vx = resp["vx"]?.GetValue<double>() ?? 0,
                    Vy = resp["vy"]?.GetValue<double>() ?? 0,
                    X0 = resp["x0"]?.GetValue<double>() ?? 0,
                    Y0 = resp["y0"]?.GetValue<double>() ?? 0
                };

                // Extract endpoints if present
                if (resp["x_top"] != null) result.XTop = resp["x_top"]!.GetValue<double>();
                if (resp["x_bot"] != null) result.XBot = resp["x_bot"]!.GetValue<double>();
                if (resp["y_left"] != null) result.YLeft = resp["y_left"]!.GetValue<double>();
                if (resp["y_right"] != null) result.YRight = resp["y_right"]!.GetValue<double>();

                return result;
            }
            else if (status == "no_edge")
            {
                return new EdgeFindResult { Status = "no_edge", Error = resp["reason"]?.GetValue<string>() ?? "" };
            }
            else
            {
                return new EdgeFindResult { Status = "error", Error = resp["message"]?.GetValue<string>() ?? "(unknown error)" };
            }
        }

        /// <summary>Ask the server to shut down cleanly.</summary>
        public void Shutdown()
        {
            try { SendReceive(new { cmd = "SHUTDOWN" }); }
            catch { /* ignore – server may close before replying */ }
        }

        // =====================================================================
        //  Internal helpers
        // =====================================================================

        private JsonNode? SendReceive(object payload)
        {
            string json = JsonSerializer.Serialize(payload);

            bool sent = _socket.TrySendFrame(TimeSpan.FromMilliseconds(TimeoutMs), json);
            if (!sent)
            {
                Console.Error.WriteLine("[WaferAlignmentClient] Send timeout.");
                return null;
            }

            bool received = _socket.TryReceiveFrameString(
                TimeSpan.FromMilliseconds(TimeoutMs), out string? reply);

            if (!received || reply == null)
            {
                Console.Error.WriteLine("[WaferAlignmentClient] Receive timeout.");
                return null;
            }

            try   { return JsonNode.Parse(reply); }
            catch { Console.Error.WriteLine($"[WaferAlignmentClient] Bad JSON reply: {reply}"); return null; }
        }
    }
}

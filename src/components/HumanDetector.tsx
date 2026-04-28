import { useCallback, useEffect, useRef, useState } from "react";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "@tensorflow/tfjs";
import { Button } from "@/components/ui/button";
import { Camera, Image as ImageIcon, Loader2, Square, Upload, Users } from "lucide-react";
import { cn } from "@/lib/utils";

type Mode = "idle" | "image" | "camera";

export const HumanDetector = () => {
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [loading, setLoading] = useState(true);
  const [mode, setMode] = useState<Mode>("idle");
  const [count, setCount] = useState(0);
  const [status, setStatus] = useState("Initializing model...");

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const rafRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    cocoSsd.load({ base: "lite_mobilenet_v2" }).then((m) => {
      setModel(m);
      setLoading(false);
      setStatus("Ready. Choose an image or start the webcam.");
    });
    return () => stopAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const drawDetections = useCallback(
    (
      preds: cocoSsd.DetectedObject[],
      sourceW: number,
      sourceH: number,
      drawSource: CanvasImageSource,
    ) => {
      const canvas = canvasRef.current;
      if (!canvas) return 0;
      canvas.width = sourceW;
      canvas.height = sourceH;
      const ctx = canvas.getContext("2d");
      if (!ctx) return 0;
      ctx.drawImage(drawSource, 0, 0, sourceW, sourceH);

      const persons = preds.filter((p) => p.class === "person");
      ctx.lineWidth = Math.max(2, sourceW / 400);
      ctx.font = `${Math.max(14, sourceW / 50)}px "JetBrains Mono", monospace`;

      persons.forEach((p, i) => {
        const [x, y, w, h] = p.bbox;
        // Bounding box - neon green
        ctx.strokeStyle = "hsl(158, 100%, 52%)";
        ctx.shadowColor = "hsl(158, 100%, 52%)";
        ctx.shadowBlur = 12;
        ctx.strokeRect(x, y, w, h);
        ctx.shadowBlur = 0;

        // Corner brackets for hud feel
        const c = Math.min(20, w / 4, h / 4);
        ctx.strokeStyle = "hsl(195, 100%, 55%)";
        ctx.beginPath();
        ctx.moveTo(x, y + c); ctx.lineTo(x, y); ctx.lineTo(x + c, y);
        ctx.moveTo(x + w - c, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + c);
        ctx.moveTo(x + w, y + h - c); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - c, y + h);
        ctx.moveTo(x + c, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - c);
        ctx.stroke();

        // Label
        const label = `PERSON ${i + 1} · ${Math.round(p.score * 100)}%`;
        const tw = ctx.measureText(label).width + 12;
        const th = Math.max(20, sourceW / 40);
        ctx.fillStyle = "hsl(158, 100%, 52%)";
        ctx.fillRect(x, Math.max(0, y - th), tw, th);
        ctx.fillStyle = "hsl(230, 25%, 6%)";
        ctx.fillText(label, x + 6, Math.max(th - 6, y - 6));
      });

      return persons.length;
    },
    [],
  );

  const stopAll = () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
  };

  const handleStartCamera = async () => {
    if (!model) return;
    stopAll();
    setMode("camera");
    setStatus("Requesting webcam...");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: "user" },
      });
      streamRef.current = stream;
      const v = videoRef.current!;
      v.srcObject = stream;
      await v.play();
      setStatus("Detecting · LIVE");

      const loop = async () => {
        if (!videoRef.current || !model) return;
        const preds = await model.detect(videoRef.current);
        const n = drawDetections(
          preds,
          videoRef.current.videoWidth,
          videoRef.current.videoHeight,
          videoRef.current,
        );
        setCount(n);
        rafRef.current = requestAnimationFrame(loop);
      };
      loop();
    } catch (e) {
      setStatus("Webcam access denied or unavailable.");
      setMode("idle");
    }
  };

  const handleStop = () => {
    stopAll();
    setMode("idle");
    setStatus("Stopped. Ready when you are.");
    setCount(0);
    const c = canvasRef.current;
    if (c) c.getContext("2d")?.clearRect(0, 0, c.width, c.height);
  };

  const handleFile = (file: File) => {
    if (!model) return;
    stopAll();
    setMode("image");
    setStatus("Analyzing image...");
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = async () => {
      const maxW = 1024;
      const scale = img.width > maxW ? maxW / img.width : 1;
      const w = Math.round(img.width * scale);
      const h = Math.round(img.height * scale);
      // Draw scaled into a temp canvas for detection input
      const tmp = document.createElement("canvas");
      tmp.width = w; tmp.height = h;
      tmp.getContext("2d")!.drawImage(img, 0, 0, w, h);
      const preds = await model.detect(tmp);
      const n = drawDetections(preds, w, h, tmp);
      setCount(n);
      setStatus(`Detection complete · ${n} ${n === 1 ? "person" : "people"} found`);
      URL.revokeObjectURL(url);
    };
    img.src = url;
  };

  return (
    <div className="w-full max-w-6xl mx-auto">
      {/* HUD top bar */}
      <div className="grid grid-cols-3 gap-3 mb-4 text-xs uppercase tracking-widest">
        <div className="border border-border bg-card/60 backdrop-blur px-4 py-3 flex items-center gap-2">
          <span className={cn("h-2 w-2 rounded-full", loading ? "bg-muted-foreground" : "bg-primary animate-pulse-ring")} />
          <span className="text-muted-foreground">Model</span>
          <span className="ml-auto text-foreground">{loading ? "LOADING" : "COCO-SSD"}</span>
        </div>
        <div className="border border-border bg-card/60 backdrop-blur px-4 py-3 flex items-center gap-2">
          <span className="text-muted-foreground">Mode</span>
          <span className="ml-auto text-primary text-glow">{mode.toUpperCase()}</span>
        </div>
        <div className="border border-border bg-card/60 backdrop-blur px-4 py-3 flex items-center gap-2">
          <Users className="h-3.5 w-3.5 text-primary" />
          <span className="text-muted-foreground">Detected</span>
          <span className="ml-auto text-2xl font-bold text-primary text-glow leading-none">{count}</span>
        </div>
      </div>

      {/* Viewport */}
      <div className="relative aspect-video w-full border border-border bg-card/40 backdrop-blur overflow-hidden corner-brackets shadow-neon">
        <div className="absolute inset-0 grid-bg opacity-40" />

        {mode === "idle" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 text-center px-6">
            {loading ? (
              <>
                <Loader2 className="h-10 w-10 animate-spin text-primary" />
                <p className="text-sm text-muted-foreground uppercase tracking-widest">Loading neural network...</p>
              </>
            ) : (
              <>
                <div className="h-16 w-16 rounded-full border-2 border-primary flex items-center justify-center animate-pulse-ring">
                  <Users className="h-7 w-7 text-primary" />
                </div>
                <p className="text-sm text-muted-foreground uppercase tracking-widest">Awaiting input signal</p>
                <p className="text-xs text-muted-foreground/70 max-w-md">
                  Detects people in real time using a neural network running entirely in your browser. No data leaves your device.
                </p>
              </>
            )}
          </div>
        )}

        <video ref={videoRef} className="hidden" playsInline muted />
        <img ref={imgRef} className="hidden" alt="" />
        <canvas
          ref={canvasRef}
          className={cn(
            "absolute inset-0 w-full h-full object-contain transition-opacity duration-300",
            mode === "idle" ? "opacity-0" : "opacity-100",
          )}
        />

        {mode === "camera" && (
          <div className="pointer-events-none absolute inset-x-0 top-0 h-1 bg-gradient-to-b from-primary to-transparent animate-scan" />
        )}

        {/* Status strip */}
        <div className="absolute bottom-0 inset-x-0 bg-background/80 backdrop-blur border-t border-border px-4 py-2 flex items-center justify-between text-xs uppercase tracking-widest">
          <span className="flex items-center gap-2">
            <span className={cn("h-1.5 w-1.5 rounded-full", mode === "camera" ? "bg-destructive animate-pulse" : "bg-primary")} />
            <span className="text-muted-foreground">{status}</span>
          </span>
          <span className="text-muted-foreground/60 hidden sm:block">TENSORFLOW.JS · WEBGL</span>
        </div>
      </div>

      {/* Controls */}
      <div className="mt-4 flex flex-wrap gap-3 justify-center">
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleFile(f);
            e.target.value = "";
          }}
        />
        <Button
          size="lg"
          disabled={loading}
          onClick={() => fileInputRef.current?.click()}
          className="gap-2 uppercase tracking-widest"
        >
          <Upload className="h-4 w-4" />
          Upload Image
        </Button>
        <Button
          size="lg"
          disabled={loading || mode === "camera"}
          onClick={handleStartCamera}
          variant="secondary"
          className="gap-2 uppercase tracking-widest border border-primary/40 hover:border-primary"
        >
          <Camera className="h-4 w-4" />
          Start Webcam
        </Button>
        <Button
          size="lg"
          disabled={mode === "idle"}
          onClick={handleStop}
          variant="outline"
          className="gap-2 uppercase tracking-widest"
        >
          <Square className="h-4 w-4" />
          Stop
        </Button>
      </div>

      <p className="mt-6 text-center text-xs text-muted-foreground/60 uppercase tracking-widest flex items-center justify-center gap-2">
        <ImageIcon className="h-3 w-3" />
        Inspired by the OpenCV HOG human detector — reimagined for the browser
      </p>
    </div>
  );
};

import { useCallback, useEffect, useRef, useState } from "react";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import * as tf from "@tensorflow/tfjs";
import { Button } from "@/components/ui/button";
import { Boxes, Camera, Image as ImageIcon, Loader2, Sparkles, Square, Upload, Users } from "lucide-react";
import { cn } from "@/lib/utils";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";

type Mode = "idle" | "image" | "camera";

const ANIMAL_CLASSES = new Set([
  "cat", "dog", "horse", "sheep", "cow",
  "elephant", "bear", "zebra", "giraffe", "bird",
]);

const MAX_BOXES = 80;
const PERSON_ANIMAL_MIN_SCORE = 0.24;
const OBJECT_MIN_SCORE = 0.35;
const IMAGE_MAX_SIDE = 1600;
const CAMERA_MEMORY_MS = 850;

type DetectSource = HTMLCanvasElement | HTMLImageElement | HTMLVideoElement | ImageData;

type Detection = cocoSsd.DetectedObject & { kind: "person" | "animal" };

type SpeciesResult = {
  species: string;
  scientific_name: string | null;
  confidence: number;
  facts: string[];
};

type SceneBox = { x_min: number; y_min: number; x_max: number; y_max: number };

type SceneSubject = {
  name: string;
  category: "animal" | "person" | "object" | "plant" | "vehicle" | "food" | "other";
  scientific_name: string | null;
  count: number;
  confidence: number;
  facts: string[];
  boxes?: SceneBox[];
};

type SceneResult = {
  summary: string;
  subjects: SceneSubject[];
};

const isPersonOrAnimal = (p: cocoSsd.DetectedObject) =>
  p.class === "person" || ANIMAL_CLASSES.has(p.class);

const filterDetections = (preds: cocoSsd.DetectedObject[]) =>
  preds.filter((p) => p.score >= (isPersonOrAnimal(p) ? PERSON_ANIMAL_MIN_SCORE : OBJECT_MIN_SCORE));

const iou = (a: number[], b: number[]) => {
  const [ax, ay, aw, ah] = a;
  const [bx, by, bw, bh] = b;
  const x1 = Math.max(ax, bx);
  const y1 = Math.max(ay, by);
  const x2 = Math.min(ax + aw, bx + bw);
  const y2 = Math.min(ay + ah, by + bh);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = aw * ah + bw * bh - intersection;
  return union > 0 ? intersection / union : 0;
};

const suppressDuplicateDetections = (preds: cocoSsd.DetectedObject[]) => {
  const kept: cocoSsd.DetectedObject[] = [];
  [...preds]
    .sort((a, b) => b.score - a.score)
    .forEach((p) => {
      const duplicate = kept.some((k) => k.class === p.class && iou(k.bbox, p.bbox) > 0.45);
      if (!duplicate) kept.push(p);
    });
  return kept;
};

export const HumanDetector = () => {
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [loading, setLoading] = useState(true);
  const [mode, setMode] = useState<Mode>("idle");
  const [personCount, setPersonCount] = useState(0);
  const [animalCount, setAnimalCount] = useState(0);
  const [objectCount, setObjectCount] = useState(0);
  const [status, setStatus] = useState("Initializing model...");
  const [animals, setAnimals] = useState<Detection[]>([]);
  const [classifying, setClassifying] = useState<number | null>(null);
  const [species, setSpecies] = useState<Record<number, SpeciesResult>>({});
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [sceneLoading, setSceneLoading] = useState(false);
  const [scene, setScene] = useState<SceneResult | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sourceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const rafRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const lastPriorityPredsRef = useRef<cocoSsd.DetectedObject[]>([]);
  const lastPriorityTimeRef = useRef(0);
  const detectingRef = useRef(false);
  const cameraRunRef = useRef(0);

  useEffect(() => {
    const loadModel = async () => {
      try {
        await tf.ready();
        try {
          await tf.setBackend("webgl");
          await tf.ready();
        } catch (backendErr) {
          console.warn("WebGL backend unavailable, using fallback backend:", backendErr);
        }
        // Use full mobilenet_v2 for higher accuracy (vs lite_mobilenet_v2)
        const m = await cocoSsd.load({ base: "mobilenet_v2" });
        setModel(m);
        setLoading(false);
        setStatus("Ready. Upload an image or start the webcam.");
      } catch (err) {
        console.error("Model load failed:", err);
        setStatus("Model failed to load. Please refresh.");
        toast.error("Failed to load detection model");
      }
    };
    loadModel();
    return () => stopAll();
  }, []);

  const detectAccurately = useCallback(
    async (source: DetectSource, useMemory = false) => {
      if (!model) return [] as cocoSsd.DetectedObject[];
      const rawPreds = await model.detect(source, MAX_BOXES, PERSON_ANIMAL_MIN_SCORE);
      let preds = suppressDuplicateDetections(filterDetections(rawPreds));

      if (useMemory) {
        const now = performance.now();
        const priorityPreds = preds.filter(isPersonOrAnimal);
        if (priorityPreds.length > 0) {
          lastPriorityPredsRef.current = priorityPreds;
          lastPriorityTimeRef.current = now;
        } else if (now - lastPriorityTimeRef.current < CAMERA_MEMORY_MS) {
          preds = suppressDuplicateDetections([...lastPriorityPredsRef.current, ...preds]);
        }
      }

      return preds;
    },
    [model],
  );

  const drawDetections = useCallback(
    (preds: cocoSsd.DetectedObject[], sourceW: number, sourceH: number, drawSource: CanvasImageSource) => {
      const canvas = canvasRef.current;
      if (!canvas) return { persons: 0, animals: [] as Detection[], objects: 0 };
      canvas.width = sourceW;
      canvas.height = sourceH;
      const ctx = canvas.getContext("2d");
      if (!ctx) return { persons: 0, animals: [] as Detection[], objects: 0 };
      ctx.drawImage(drawSource, 0, 0, sourceW, sourceH);

      const persons = preds.filter((p) => p.class === "person");
      const animalsList: Detection[] = preds
        .filter((p) => ANIMAL_CLASSES.has(p.class))
        .map((p) => ({ ...p, kind: "animal" as const }));
      const objectsList = preds.filter(
        (p) => p.class !== "person" && !ANIMAL_CLASSES.has(p.class),
      );

      ctx.lineWidth = Math.max(2, sourceW / 400);
      ctx.font = `${Math.max(14, sourceW / 50)}px "JetBrains Mono", monospace`;

      const drawBox = (
        p: cocoSsd.DetectedObject, i: number, color: string, accent: string, label: string,
      ) => {
        const [x, y, w, h] = p.bbox;
        ctx.strokeStyle = color;
        ctx.shadowColor = color;
        ctx.shadowBlur = 12;
        ctx.strokeRect(x, y, w, h);
        ctx.shadowBlur = 0;

        const c = Math.min(20, w / 4, h / 4);
        ctx.strokeStyle = accent;
        ctx.beginPath();
        ctx.moveTo(x, y + c); ctx.lineTo(x, y); ctx.lineTo(x + c, y);
        ctx.moveTo(x + w - c, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + c);
        ctx.moveTo(x + w, y + h - c); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - c);
        ctx.moveTo(x + c, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - c);
        ctx.stroke();

        const tw = ctx.measureText(label).width + 12;
        const th = Math.max(20, sourceW / 40);
        ctx.fillStyle = color;
        ctx.fillRect(x, Math.max(0, y - th), tw, th);
        ctx.fillStyle = "hsl(230, 25%, 6%)";
        ctx.fillText(label, x + 6, Math.max(th - 6, y - 6));
      };

      persons.forEach((p, i) =>
        drawBox(p, i, "hsl(158, 100%, 52%)", "hsl(195, 100%, 55%)",
          `PERSON ${i + 1} · ${Math.round(p.score * 100)}%`),
      );
      animalsList.forEach((p, i) =>
        drawBox(p, i, "hsl(35, 100%, 55%)", "hsl(50, 100%, 60%)",
          `${p.class.toUpperCase()} ${i + 1} · ${Math.round(p.score * 100)}%`),
      );
      objectsList.forEach((p, i) =>
        drawBox(p, i, "hsl(280, 100%, 65%)", "hsl(320, 100%, 70%)",
          `${p.class.toUpperCase()} ${i + 1} · ${Math.round(p.score * 100)}%`),
      );

      return { persons: persons.length, animals: animalsList, objects: objectsList.length };
    },
    [],
  );

  const stopAll = () => {
    cameraRunRef.current += 1;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    detectingRef.current = false;
    lastPriorityPredsRef.current = [];
    lastPriorityTimeRef.current = 0;
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
    setSpecies({});
    setAnimals([]);
    setOriginalImage(null);
    setScene(null);
    setStatus("Requesting webcam...");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "environment",
        },
      });
      streamRef.current = stream;
      const v = videoRef.current!;
      v.srcObject = stream;
      await v.play();
      setStatus("Detecting · LIVE");
      const runId = cameraRunRef.current;

      const loop = async () => {
        if (cameraRunRef.current !== runId) return;
        if (!videoRef.current || !model || detectingRef.current) {
          rafRef.current = requestAnimationFrame(loop);
          return;
        }
        const video = videoRef.current;
        if (!video.videoWidth || !video.videoHeight) {
          rafRef.current = requestAnimationFrame(loop);
          return;
        }
        detectingRef.current = true;
        try {
          const preds = await detectAccurately(video, true);
          const { persons, animals: a, objects } = drawDetections(
            preds, video.videoWidth, video.videoHeight, video,
          );
          setPersonCount(persons);
          setAnimalCount(a.length);
          setObjectCount(objects);
          setAnimals(a);
          // cache the latest video frame for cropping
          const sc = sourceCanvasRef.current ?? document.createElement("canvas");
          sc.width = video.videoWidth;
          sc.height = video.videoHeight;
          sc.getContext("2d")!.drawImage(video, 0, 0);
          sourceCanvasRef.current = sc;
        } catch (err) {
          console.error("Live detection failed:", err);
        } finally {
          detectingRef.current = false;
          if (cameraRunRef.current === runId) {
            rafRef.current = requestAnimationFrame(loop);
          }
        }
      };
      loop();
    } catch {
      setStatus("Webcam access denied or unavailable.");
      setMode("idle");
    }
  };

  const handleStop = () => {
    stopAll();
    setMode("idle");
    setStatus("Stopped. Ready when you are.");
    setPersonCount(0);
    setAnimalCount(0);
    setObjectCount(0);
    setAnimals([]);
    setSpecies({});
    setOriginalImage(null);
    setScene(null);
    const c = canvasRef.current;
    if (c) c.getContext("2d")?.clearRect(0, 0, c.width, c.height);
  };

  const handleFile = (file: File) => {
    if (!model) {
      toast.error("Model not ready yet");
      return;
    }
    stopAll();
    setMode("image");
    setSpecies({});
    setScene(null);
    setAnimals([]);
    setPersonCount(0);
    setAnimalCount(0);
    setObjectCount(0);
    setStatus("Loading image...");

    const url = URL.createObjectURL(file);
    const img = new Image();
    img.decoding = "async";
    img.onerror = () => {
      toast.error("Could not load that image");
      setStatus("Image failed to load.");
      URL.revokeObjectURL(url);
    };
    img.onload = async () => {
      try {
        // Keep more detail for small people/animals while capping very large photos.
        const longestSide = Math.max(img.width, img.height);
        const scale = longestSide > IMAGE_MAX_SIDE ? IMAGE_MAX_SIDE / longestSide : 1;
        const w = Math.max(1, Math.round(img.width * scale));
        const h = Math.max(1, Math.round(img.height * scale));
        const tmp = document.createElement("canvas");
        tmp.width = w; tmp.height = h;
        tmp.getContext("2d")!.drawImage(img, 0, 0, w, h);
        sourceCanvasRef.current = tmp;
        setOriginalImage(tmp.toDataURL("image/jpeg", 0.92));
        setStatus("Analyzing image...");

        // CRITICAL: wait for React to mount the image-mode canvas before drawing
        // (canvasRef is shared between viewport modes — must let DOM update first)
        const waitForCanvas = async () => {
          for (let i = 0; i < 30; i++) {
            if (canvasRef.current) return canvasRef.current;
            await new Promise((r) => requestAnimationFrame(() => r(null)));
          }
          return canvasRef.current;
        };
        await waitForCanvas();
        if (!canvasRef.current) {
          throw new Error("Canvas not available");
        }

        // Run with a lower threshold for people/animals, then filter objects separately.
        const preds = await detectAccurately(tmp);
        const { persons, animals: a, objects } = drawDetections(preds, w, h, tmp);
        setPersonCount(persons);
        setAnimalCount(a.length);
        setObjectCount(objects);
        setAnimals(a);
        setStatus(
          `Detection complete · ${persons} people · ${a.length} animals · ${objects} objects`,
        );
      } catch (err) {
        console.error("Detection failed:", err);
        toast.error("Detection failed. Please try another image.");
        setStatus("Detection failed.");
      } finally {
        URL.revokeObjectURL(url);
      }
    };
    img.src = url;
  };

  const cropToDataUrl = (det: Detection): string | null => {
    const src = sourceCanvasRef.current;
    if (!src) return null;
    const [x, y, w, h] = det.bbox;
    // Add 10% padding for context
    const pad = 0.1;
    const cx = Math.max(0, Math.floor(x - w * pad));
    const cy = Math.max(0, Math.floor(y - h * pad));
    const cw = Math.min(src.width - cx, Math.ceil(w * (1 + 2 * pad)));
    const ch = Math.min(src.height - cy, Math.ceil(h * (1 + 2 * pad)));
    const out = document.createElement("canvas");
    out.width = cw; out.height = ch;
    out.getContext("2d")!.drawImage(src, cx, cy, cw, ch, 0, 0, cw, ch);
    return out.toDataURL("image/jpeg", 0.85);
  };

  const classify = async (idx: number) => {
    const det = animals[idx];
    if (!det) return;
    const image = cropToDataUrl(det);
    if (!image) {
      toast.error("No image frame available to classify.");
      return;
    }
    setClassifying(idx);
    try {
      const { data, error } = await supabase.functions.invoke("classify-animal", {
        body: { image, hint: det.class },
      });
      if (error) throw error;
      if (data?.error) throw new Error(data.error);
      const result = data?.result as SpeciesResult | null;
      if (!result) throw new Error("Empty response");
      setSpecies((prev) => ({ ...prev, [idx]: result }));
      toast.success(`Identified: ${result.species}`);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Classification failed";
      toast.error(msg);
    } finally {
      setClassifying(null);
    }
  };

  const identifyScene = async () => {
    if (!originalImage) {
      toast.error("Upload an image first.");
      return;
    }
    setSceneLoading(true);
    setScene(null);
    try {
      const { data, error } = await supabase.functions.invoke("classify-animal", {
        body: { image: originalImage, mode: "scene" },
      });
      if (error) throw error;
      if (data?.error) throw new Error(data.error);
      const result = data?.result as SceneResult | null;
      if (!result || !Array.isArray(result.subjects)) throw new Error("Empty response");
      setScene(result);
      toast.success(`AI identified ${result.subjects.length} subject(s)`);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "AI identification failed";
      toast.error(msg);
    } finally {
      setSceneLoading(false);
    }
  };

  return (
    <div className="w-full max-w-6xl mx-auto">
      {/* HUD top bar */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mb-4 text-xs uppercase tracking-widest">
        <div className="border border-border bg-card/60 backdrop-blur px-4 py-3 flex items-center gap-2">
          <span className={cn("h-2 w-2 rounded-full", loading ? "bg-muted-foreground" : "bg-primary animate-pulse-ring")} />
          <span className="text-muted-foreground">Model</span>
          <span className="ml-auto text-foreground">{loading ? "LOAD" : "COCO-SSD"}</span>
        </div>
        <div className="border border-border bg-card/60 backdrop-blur px-4 py-3 flex items-center gap-2">
          <span className="text-muted-foreground">Mode</span>
          <span className="ml-auto text-primary text-glow">{mode.toUpperCase()}</span>
        </div>
        <div className="border border-border bg-card/60 backdrop-blur px-4 py-3 flex items-center gap-2">
          <Users className="h-3.5 w-3.5 text-primary" />
          <span className="text-muted-foreground">People</span>
          <span className="ml-auto text-2xl font-bold text-primary text-glow leading-none">{personCount}</span>
        </div>
        <div className="border border-border bg-card/60 backdrop-blur px-4 py-3 flex items-center gap-2">
          <Sparkles className="h-3.5 w-3.5" style={{ color: "hsl(35 100% 55%)" }} />
          <span className="text-muted-foreground">Animals</span>
          <span className="ml-auto text-2xl font-bold leading-none" style={{ color: "hsl(35 100% 55%)" }}>{animalCount}</span>
        </div>
        <div className="border border-border bg-card/60 backdrop-blur px-4 py-3 flex items-center gap-2">
          <Boxes className="h-3.5 w-3.5" style={{ color: "hsl(280 100% 70%)" }} />
          <span className="text-muted-foreground">Objects</span>
          <span className="ml-auto text-2xl font-bold leading-none" style={{ color: "hsl(280 100% 70%)" }}>{objectCount}</span>
        </div>
      </div>

      {/* Side-by-side comparison for image uploads */}
      {mode === "image" && originalImage && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <figure className="border border-border bg-card/40 backdrop-blur overflow-hidden corner-brackets">
            <div className="aspect-video w-full bg-background/40 flex items-center justify-center overflow-hidden">
              <img src={originalImage} alt="Original upload" className="w-full h-full object-contain" />
            </div>
            <figcaption className="border-t border-border px-4 py-2 text-xs uppercase tracking-widest text-muted-foreground flex items-center gap-2">
              <ImageIcon className="h-3 w-3 text-primary" />
              Your Image
            </figcaption>
          </figure>
          <figure className="border border-border bg-card/40 backdrop-blur overflow-hidden corner-brackets shadow-neon">
            <div className="aspect-video w-full bg-background/40 flex items-center justify-center overflow-hidden">
              <canvas ref={canvasRef} className="w-full h-full object-contain" />
            </div>
            <figcaption className="border-t border-border px-4 py-2 text-xs uppercase tracking-widest flex items-center justify-between">
              <span className="flex items-center gap-2 text-primary text-glow">
                <Sparkles className="h-3 w-3" />
                Model Predicted
              </span>
              <span className="text-muted-foreground">{personCount}P · {animalCount}A · {objectCount}O</span>
            </figcaption>
          </figure>
        </div>
      )}

      {/* Viewport (idle + camera) */}
      {mode !== "image" && (
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
                    Detects people and animals in real time. Tap "Identify species" to classify any detected animal via the cloud AI.
                  </p>
                </>
              )}
            </div>
          )}

          <video ref={videoRef} className="hidden" playsInline muted />
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

          <div className="absolute bottom-0 inset-x-0 bg-background/80 backdrop-blur border-t border-border px-4 py-2 flex items-center justify-between text-xs uppercase tracking-widest">
            <span className="flex items-center gap-2">
              <span className={cn("h-1.5 w-1.5 rounded-full", mode === "camera" ? "bg-destructive animate-pulse" : "bg-primary")} />
              <span className="text-muted-foreground">{status}</span>
            </span>
            <span className="text-muted-foreground/60 hidden sm:block">TF.JS · LOVABLE AI</span>
          </div>
        </div>
      )}

      {mode === "image" && (
        <div className="border border-border bg-background/60 backdrop-blur px-4 py-2 flex items-center justify-between text-xs uppercase tracking-widest">
          <span className="flex items-center gap-2">
            <span className="h-1.5 w-1.5 rounded-full bg-primary" />
            <span className="text-muted-foreground">{status}</span>
          </span>
          <span className="text-muted-foreground/60 hidden sm:block">TF.JS · LOVABLE AI</span>
        </div>
      )}

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
        <Button size="lg" disabled={loading} onClick={() => fileInputRef.current?.click()} className="gap-2 uppercase tracking-widest">
          <Upload className="h-4 w-4" /> Upload Image
        </Button>
        <Button size="lg" disabled={loading || mode === "camera"} onClick={handleStartCamera} variant="secondary"
          className="gap-2 uppercase tracking-widest border border-primary/40 hover:border-primary">
          <Camera className="h-4 w-4" /> Start Webcam
        </Button>
        <Button
          size="lg"
          disabled={mode === "idle"}
          onClick={handleStop}
          variant={mode !== "idle" ? "destructive" : "outline"}
          className={cn(
            "gap-2 uppercase tracking-widest transition-all",
            mode !== "idle" &&
              "bg-destructive text-destructive-foreground border border-destructive shadow-[0_0_24px_hsl(var(--destructive)/0.6)] hover:bg-destructive/90 animate-pulse-ring",
          )}
        >
          <Square className="h-4 w-4" /> Stop
        </Button>
        {mode === "image" && originalImage && (
          <Button
            size="lg"
            onClick={identifyScene}
            disabled={sceneLoading}
            className="gap-2 uppercase tracking-widest bg-gradient-to-r from-primary to-accent text-primary-foreground border border-primary shadow-neon hover:opacity-90"
          >
            {sceneLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
            Identify Everything with AI
          </Button>
        )}
      </div>

      {/* Full-scene AI identification (any animal/object, including those COCO-SSD can't see) */}
      {scene && (
        <div className="mt-6 border border-primary/40 bg-card/60 backdrop-blur p-4 shadow-neon">
          <h3 className="text-xs uppercase tracking-widest text-primary text-glow mb-2 flex items-center gap-2">
            <Sparkles className="h-3.5 w-3.5" />
            AI Scene Analysis
          </h3>
          {scene.summary && (
            <p className="text-sm text-muted-foreground mb-4 italic">"{scene.summary}"</p>
          )}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {scene.subjects.map((s, i) => (
              <div key={i} className="border border-border bg-background/40 p-3 flex flex-col gap-1.5">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-sm font-bold text-foreground">
                    {s.name}
                    {s.count > 1 && <span className="text-muted-foreground"> ×{s.count}</span>}
                  </span>
                  <span className="text-[10px] uppercase tracking-widest px-2 py-0.5 border border-border text-muted-foreground">
                    {s.category} · {s.confidence}%
                  </span>
                </div>
                {s.scientific_name && (
                  <div className="text-xs text-muted-foreground italic">{s.scientific_name}</div>
                )}
                {s.facts?.length > 0 && (
                  <ul className="list-disc list-inside text-xs text-muted-foreground/80 space-y-0.5 pt-1">
                    {s.facts.slice(0, 2).map((f, j) => <li key={j}>{f}</li>)}
                  </ul>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Animals panel */}
      {animals.length > 0 && (
        <div className="mt-6 border border-border bg-card/60 backdrop-blur p-4">
          <h3 className="text-xs uppercase tracking-widest text-muted-foreground mb-3 flex items-center gap-2">
            <Sparkles className="h-3.5 w-3.5" style={{ color: "hsl(35 100% 55%)" }} />
            Animals detected — identify species
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {animals.map((a, i) => {
              const sp = species[i];
              return (
                <div key={i} className="border border-border bg-background/40 p-3 flex flex-col gap-2">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-sm uppercase tracking-widest" style={{ color: "hsl(35 100% 55%)" }}>
                      {a.class} #{i + 1} · {Math.round(a.score * 100)}%
                    </span>
                    <Button
                      size="sm"
                      onClick={() => classify(i)}
                      disabled={classifying !== null}
                      className="gap-2 uppercase tracking-widest text-xs"
                    >
                      {classifying === i ? <Loader2 className="h-3 w-3 animate-spin" /> : <Sparkles className="h-3 w-3" />}
                      {sp ? "Re-identify" : "Identify species"}
                    </Button>
                  </div>
                  {sp && (
                    <div className="text-xs space-y-1">
                      <div className="text-foreground">
                        <span className="text-muted-foreground">Species: </span>
                        <span className="font-bold text-primary text-glow">{sp.species}</span>
                        <span className="text-muted-foreground"> · {sp.confidence}%</span>
                      </div>
                      {sp.scientific_name && (
                        <div className="text-muted-foreground italic">{sp.scientific_name}</div>
                      )}
                      {sp.facts?.length > 0 && (
                        <ul className="list-disc list-inside text-muted-foreground/80 space-y-0.5 pt-1">
                          {sp.facts.slice(0, 3).map((f, j) => <li key={j}>{f}</li>)}
                        </ul>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      <p className="mt-6 text-center text-xs text-muted-foreground/60 uppercase tracking-widest flex items-center justify-center gap-2">
        <ImageIcon className="h-3 w-3" />
        Browser detection + cloud species classifier (Gemini vision)
      </p>
    </div>
  );
};

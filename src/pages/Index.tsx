import { HumanDetector } from "@/components/HumanDetector";

const Index = () => {
  return (
    <main className="min-h-screen w-full px-4 py-10 sm:py-16">
      <header className="max-w-6xl mx-auto mb-10 text-center">
        <div className="inline-flex items-center gap-2 px-3 py-1 border border-primary/40 bg-primary/5 text-primary text-xs uppercase tracking-[0.3em] mb-6">
          <span className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />
          Live Vision System
        </div>
        <h1 className="text-4xl sm:text-6xl font-bold text-foreground">
          Human <span className="text-primary text-glow">Detector</span>
        </h1>
        <p className="mt-4 text-sm sm:text-base text-muted-foreground max-w-xl mx-auto">
          Real-time person detection running 100% in your browser. Powered by TensorFlow.js and the COCO-SSD neural network.
        </p>
      </header>

      <HumanDetector />
    </main>
  );
};

export default Index;

import { Button } from "@/components/ui/button";
import Link from "next/link";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24 bg-gradient-to-b from-slate-900 to-slate-800 text-white">
      <div className="z-10 max-w-5xl w-full items-center justify-center font-mono text-sm flex flex-col">
        <h1 className="text-4xl font-bold mb-6">LLM Graph Builder</h1>
        <p className="text-xl mb-8 text-center max-w-2xl">
          Visually construct LLM training components and generate PyTorch code
        </p>
        
        <div className="flex gap-4">
          <Button asChild className="bg-blue-600 hover:bg-blue-700">
            <Link href="/builder">
              Start Building
            </Link>
          </Button>
          
          <Button asChild variant="outline" className="border-blue-600 text-blue-400 hover:bg-blue-950">
            <Link href="https://github.com/altyni86/llm-graph-trainer" target="_blank">
              View on GitHub
            </Link>
          </Button>
        </div>
        
        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8 w-full">
          <FeatureCard 
            title="Visual Component Builder" 
            description="Drag and drop LLM components to create your architecture"
            icon="🧩"
          />
          <FeatureCard 
            title="PyTorch Code Generation" 
            description="Generate ready-to-use PyTorch code from your visual design"
            icon="📝"
          />
          <FeatureCard 
            title="Component Library" 
            description="Access embeddings, positional encodings, QKV blocks, and more"
            icon="📚"
          />
        </div>
      </div>
    </main>
  );
}

function FeatureCard({ title, description, icon }: { title: string; description: string; icon: string }) {
  return (
    <div className="border border-slate-700 rounded-lg p-6 bg-slate-800/50 hover:bg-slate-800 transition-colors">
      <div className="text-4xl mb-4">{icon}</div>
      <h3 className="text-xl font-semibold mb-2">{title}</h3>
      <p className="text-slate-300">{description}</p>
    </div>
  );
}

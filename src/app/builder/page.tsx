"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import Link from "next/link";
import { FlowEditor } from "@/components/flow-editor/FlowEditor";
import { CodePreview } from "@/components/code-preview/CodePreview";
import { OptimizationPanel, OptimizationSettings, defaultOptimizationSettings } from "@/components/optimization-panel/OptimizationPanel";

export default function BuilderPage() {
  const [activeTab, setActiveTab] = useState<string>("editor");
  const [generatedCode, setGeneratedCode] = useState<string>("");
  const [optimizationSettings, setOptimizationSettings] = useState<OptimizationSettings>(defaultOptimizationSettings);

  const handleGenerateCode = (code: string) => {
    setGeneratedCode(code);
    setActiveTab("code");
  };

  return (
    <div className="flex flex-col h-screen bg-slate-900 text-white">
      <header className="border-b border-slate-700 p-4 flex justify-between items-center">
        <div className="flex items-center gap-4">
          <Link href="/" className="text-xl font-bold">
            LLM Graph Builder
          </Link>
          <div className="hidden md:flex gap-2">
            <Button variant="ghost" size="sm">
              Save
            </Button>
            <Button variant="ghost" size="sm">
              Load
            </Button>
            <Button variant="ghost" size="sm">
              Export
            </Button>
          </div>
        </div>
        <div className="flex gap-2">
          <Button 
            variant="outline" 
            size="sm"
            onClick={() => setActiveTab(activeTab === "editor" ? "code" : "editor")}
          >
            {activeTab === "editor" ? "View Code" : "View Editor"}
          </Button>
          <Button 
            variant="default" 
            size="sm"
            className="bg-blue-600 hover:bg-blue-700"
            onClick={() => setActiveTab("code")}
          >
            Generate Code
          </Button>
        </div>
      </header>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col">
        <div className="border-b border-slate-700 px-4">
          <TabsList className="bg-transparent border-b-0">
            <TabsTrigger value="editor">Visual Editor</TabsTrigger>
            <TabsTrigger value="code">Generated Code</TabsTrigger>
            <TabsTrigger value="optimizations">Optimizations</TabsTrigger>
          </TabsList>
        </div>
        
        <TabsContent value="editor" className="flex-1 p-0 m-0">
          <FlowEditor 
            onGenerateCode={handleGenerateCode} 
            optimizationSettings={optimizationSettings}
          />
        </TabsContent>
        
        <TabsContent value="code" className="flex-1 p-0 m-0">
          <CodePreview code={generatedCode} />
        </TabsContent>
        
        <TabsContent value="optimizations" className="flex-1 p-0 m-0">
          <div className="p-4 h-full">
            <OptimizationPanel 
              settings={optimizationSettings} 
              onSettingsChange={setOptimizationSettings} 
            />
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
} 
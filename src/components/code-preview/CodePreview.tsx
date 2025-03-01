"use client";

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Editor } from '@monaco-editor/react';
import { Card } from '@/components/ui/card';

interface CodePreviewProps {
  code: string;
}

export function CodePreview({ code }: CodePreviewProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  const defaultCode = `# Generated PyTorch code will appear here
# Build your model in the visual editor and click "Generate Code"

import torch
import torch.nn as nn

class LLMModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Your model components will be defined here
        
    def forward(self, x):
        # Your model forward pass will be defined here
        return x
`;

  return (
    <div className="h-full flex flex-col">
      <div className="p-4 border-b border-slate-700 flex justify-between items-center">
        <h2 className="text-lg font-semibold">Generated PyTorch Code</h2>
        <div className="flex gap-2">
          <Button 
            variant="outline" 
            size="sm" 
            onClick={handleCopy}
          >
            {copied ? "Copied!" : "Copy Code"}
          </Button>
          <Button 
            variant="default" 
            size="sm"
            className="bg-blue-600 hover:bg-blue-700"
          >
            Download
          </Button>
        </div>
      </div>
      
      <div className="flex-1 p-4">
        <Card className="h-full overflow-hidden border-slate-700 bg-slate-800">
          <Editor
            height="100%"
            defaultLanguage="python"
            theme="vs-dark"
            value={code || defaultCode}
            options={{
              readOnly: true,
              minimap: { enabled: true },
              scrollBeyondLastLine: false,
              fontSize: 14,
              tabSize: 4,
            }}
          />
        </Card>
      </div>
    </div>
  );
} 
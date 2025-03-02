"use client";

import { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { LLMNodeData } from '@/lib/types';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

export const EmbeddingNode = memo(({ data, isConnectable }: NodeProps<LLMNodeData>) => {
  const hasErrors = data.connectionErrors && data.connectionErrors.length > 0;
  
  return (
    <div className="bg-slate-800 p-4 rounded-md border-2 shadow-md w-64">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <div className="text-2xl">📊</div>
          <div className="font-semibold">{data.label}</div>
        </div>
        
        {hasErrors && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="text-red-500 text-xl">⚠️</div>
              </TooltipTrigger>
              <TooltipContent className="max-w-xs bg-red-900 border-red-700">
                <ul className="list-disc pl-4">
                  {data.connectionErrors?.map((error: string, index: number) => (
                    <li key={index} className="text-xs">{error}</li>
                  ))}
                </ul>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
      </div>
      
      <div className="text-xs text-slate-400 mb-2">
        Converts token IDs to embeddings
      </div>
      
      <div className="text-xs">
        <div className="flex justify-between mb-1">
          <span>Vocab Size:</span>
          <span>{String(data.params.vocabSize)}</span>
        </div>
        <div className="flex justify-between mb-1">
          <span>Embedding Dim:</span>
          <span>{String(data.params.embeddingDim)}</span>
        </div>
      </div>
      
      {/* Input handle */}
      <Handle
        type="target"
        position={Position.Left}
        isConnectable={isConnectable}
        className="w-3 h-3 bg-blue-500"
      />
      
      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Right}
        isConnectable={isConnectable}
        className="w-3 h-3 bg-green-500"
      />
    </div>
  );
});

EmbeddingNode.displayName = 'EmbeddingNode'; 
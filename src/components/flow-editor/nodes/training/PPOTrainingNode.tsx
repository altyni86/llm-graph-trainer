"use client";

import { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { LLMNodeData } from '@/lib/types';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

export const PPOTrainingNode = memo(({ data, isConnectable }: NodeProps<LLMNodeData>) => {
  const hasErrors = data.connectionErrors && data.connectionErrors.length > 0;
  
  return (
    <div className="bg-emerald-900 p-4 rounded-md border-2 shadow-md w-64">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <div className="text-2xl">🤖</div>
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
      
      <div className="text-xs text-slate-300 mb-2">
        Proximal Policy Optimization (PPO) Training
      </div>
      
      <div className="text-xs">
        <div className="flex justify-between mb-1">
          <span>Learning Rate:</span>
          <span>{String(data.params.learningRate)}</span>
        </div>
        <div className="flex justify-between mb-1">
          <span>Batch Size:</span>
          <span>{String(data.params.batchSize)}</span>
        </div>
        <div className="flex justify-between mb-1">
          <span>Epochs:</span>
          <span>{String(data.params.numEpochs)}</span>
        </div>
        <div className="flex justify-between mb-1">
          <span>Clip Epsilon:</span>
          <span>{String(data.params.clipEpsilon)}</span>
        </div>
        <div className="flex justify-between mb-1">
          <span>Value Coef:</span>
          <span>{String(data.params.valueCoefficient)}</span>
        </div>
        <div className="flex justify-between mb-1">
          <span>Entropy Coef:</span>
          <span>{String(data.params.entropyCoefficient)}</span>
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

PPOTrainingNode.displayName = 'PPOTrainingNode'; 
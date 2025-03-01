"use client";

import { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { FFNNodeData } from '@/lib/types';

const FFNNodeComponent = ({ data, isConnectable }: NodeProps<FFNNodeData>) => {
  return (
    <div className="px-4 py-2 shadow-md rounded-md bg-white border-2 border-orange-500 min-w-[180px]">
      <div className="flex items-center">
        <div className="rounded-full w-10 h-10 flex items-center justify-center bg-orange-100 text-orange-800 text-xl">
          🔄
        </div>
        <div className="ml-2">
          <div className="text-lg font-bold text-slate-800">{data.label}</div>
          <div className="text-xs text-slate-500">Feed Forward Network</div>
        </div>
      </div>

      <div className="mt-2 text-xs text-slate-600">
        <div className="flex justify-between">
          <span>Input Dim:</span>
          <span className="font-mono">{data.params.inputDim}</span>
        </div>
        <div className="flex justify-between">
          <span>Hidden Dim:</span>
          <span className="font-mono">{data.params.hiddenDim}</span>
        </div>
        <div className="flex justify-between">
          <span>Output Dim:</span>
          <span className="font-mono">{data.params.outputDim}</span>
        </div>
        <div className="flex justify-between">
          <span>Activation:</span>
          <span className="font-mono">{data.params.activation || 'relu'}</span>
        </div>
      </div>

      {/* Input handle */}
      <Handle
        type="target"
        position={Position.Left}
        id="in"
        isConnectable={isConnectable}
        className="w-3 h-3 bg-orange-500"
      />

      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Right}
        id="out"
        isConnectable={isConnectable}
        className="w-3 h-3 bg-orange-500"
      />
    </div>
  );
};

FFNNodeComponent.displayName = 'FFNNode';

export const FFNNode = memo(FFNNodeComponent); 
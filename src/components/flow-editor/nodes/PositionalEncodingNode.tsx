"use client";

import { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { PositionalEncodingNodeData } from '@/lib/types';

const PositionalEncodingNodeComponent = ({ data, isConnectable }: NodeProps<PositionalEncodingNodeData>) => {
  return (
    <div className="px-4 py-2 shadow-md rounded-md bg-white border-2 border-green-500 min-w-[180px]">
      <div className="flex items-center">
        <div className="rounded-full w-10 h-10 flex items-center justify-center bg-green-100 text-green-800 text-xl">
          📍
        </div>
        <div className="ml-2">
          <div className="text-lg font-bold text-slate-800">{data.label}</div>
          <div className="text-xs text-slate-500">Positional Encoding</div>
        </div>
      </div>

      <div className="mt-2 text-xs text-slate-600">
        <div className="flex justify-between">
          <span>Type:</span>
          <span className="font-mono">{data.params.encodingType}</span>
        </div>
        <div className="flex justify-between">
          <span>Embedding Dim:</span>
          <span className="font-mono">{data.params.embeddingDim}</span>
        </div>
        <div className="flex justify-between">
          <span>Max Seq Length:</span>
          <span className="font-mono">{data.params.maxSeqLength}</span>
        </div>
      </div>

      {/* Input handle */}
      <Handle
        type="target"
        position={Position.Left}
        id="in"
        isConnectable={isConnectable}
        className="w-3 h-3 bg-green-500"
      />

      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Right}
        id="out"
        isConnectable={isConnectable}
        className="w-3 h-3 bg-green-500"
      />
    </div>
  );
};

PositionalEncodingNodeComponent.displayName = 'PositionalEncodingNode';

export const PositionalEncodingNode = memo(PositionalEncodingNodeComponent); 
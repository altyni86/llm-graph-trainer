"use client";

import { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { EmbeddingNodeData } from '@/lib/types';

const EmbeddingNodeComponent = ({ data, isConnectable }: NodeProps<EmbeddingNodeData>) => {
  return (
    <div className="px-4 py-2 shadow-md rounded-md bg-white border-2 border-blue-500 min-w-[180px]">
      <div className="flex items-center">
        <div className="rounded-full w-10 h-10 flex items-center justify-center bg-blue-100 text-blue-800 text-xl">
          📊
        </div>
        <div className="ml-2">
          <div className="text-lg font-bold text-slate-800">{data.label}</div>
          <div className="text-xs text-slate-500">Embedding Layer</div>
        </div>
      </div>

      <div className="mt-2 text-xs text-slate-600">
        <div className="flex justify-between">
          <span>Vocab Size:</span>
          <span className="font-mono">{data.params.vocabSize}</span>
        </div>
        <div className="flex justify-between">
          <span>Embedding Dim:</span>
          <span className="font-mono">{data.params.embeddingDim}</span>
        </div>
      </div>

      {/* Input handle */}
      <Handle
        type="target"
        position={Position.Left}
        id="in"
        isConnectable={isConnectable}
        className="w-3 h-3 bg-blue-500"
      />

      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Right}
        id="out"
        isConnectable={isConnectable}
        className="w-3 h-3 bg-blue-500"
      />
    </div>
  );
};

EmbeddingNodeComponent.displayName = 'EmbeddingNode';

export const EmbeddingNode = memo(EmbeddingNodeComponent); 
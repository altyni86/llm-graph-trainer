"use client";

import { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { QKVAttentionNodeData } from '@/lib/types';

const QKVAttentionNodeComponent = ({ data, isConnectable }: NodeProps<QKVAttentionNodeData>) => {
  return (
    <div className="px-4 py-2 shadow-md rounded-md bg-white border-2 border-purple-500 min-w-[180px]">
      <div className="flex items-center">
        <div className="rounded-full w-10 h-10 flex items-center justify-center bg-purple-100 text-purple-800 text-xl">
          🔍
        </div>
        <div className="ml-2">
          <div className="text-lg font-bold text-slate-800">{data.label}</div>
          <div className="text-xs text-slate-500">Multi-Head Attention</div>
        </div>
      </div>

      <div className="mt-2 text-xs text-slate-600">
        <div className="flex justify-between">
          <span>Embedding Dim:</span>
          <span className="font-mono">{data.params.embeddingDim}</span>
        </div>
        <div className="flex justify-between">
          <span>Num Heads:</span>
          <span className="font-mono">{data.params.numHeads}</span>
        </div>
        <div className="flex justify-between">
          <span>Attention Type:</span>
          <span className="font-mono">{data.params.attentionType || 'scaled_dot_product'}</span>
        </div>
        {data.params.causal && (
          <div className="flex justify-between">
            <span>Causal:</span>
            <span className="font-mono">True</span>
          </div>
        )}
      </div>

      {/* Input handle */}
      <Handle
        type="target"
        position={Position.Left}
        id="in"
        isConnectable={isConnectable}
        className="w-3 h-3 bg-purple-500"
      />

      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Right}
        id="out"
        isConnectable={isConnectable}
        className="w-3 h-3 bg-purple-500"
      />
    </div>
  );
};

QKVAttentionNodeComponent.displayName = 'QKVAttentionNode';

export const QKVAttentionNode = memo(QKVAttentionNodeComponent); 
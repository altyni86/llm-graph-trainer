"use client";

import { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { LayerNormNodeData } from '@/lib/types';

const LayerNormNodeComponent = ({ data, isConnectable }: NodeProps<LayerNormNodeData>) => {
  return (
    <div className="px-4 py-2 shadow-md rounded-md bg-white border-2 border-purple-500 min-w-[180px]">
      <div className="flex items-center">
        <div className="rounded-full w-10 h-10 flex items-center justify-center bg-purple-100 text-purple-800 text-xl">
          📏
        </div>
        <div className="ml-2">
          <div className="text-lg font-bold text-slate-800">{data.label}</div>
          <div className="text-xs text-slate-500">Layer Normalization</div>
        </div>
      </div>

      <div className="mt-2 text-xs text-slate-600">
        <div className="flex justify-between">
          <span>Normalized Shape:</span>
          <span className="font-mono">{data.params.normalizedShape}</span>
        </div>
        <div className="flex justify-between">
          <span>Epsilon:</span>
          <span className="font-mono">{data.params.eps || '1e-5'}</span>
        </div>
        <div className="flex justify-between">
          <span>Elementwise Affine:</span>
          <span className="font-mono">{data.params.elementwiseAffine === false ? 'false' : 'true'}</span>
        </div>
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

LayerNormNodeComponent.displayName = 'LayerNormNode';

export const LayerNormNode = memo(LayerNormNodeComponent); 
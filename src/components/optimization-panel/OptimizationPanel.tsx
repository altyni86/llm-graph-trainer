"use client";

import { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { InfoIcon } from 'lucide-react';

export interface OptimizationSettings {
  // Parallelism options
  fsdp: {
    enabled: boolean;
    shardingStrategy: 'FULL_SHARD' | 'SHARD_GRAD_OP' | 'NO_SHARD';
    autoWrap: boolean;
    minNumParams: number;
  };
  deepSpeed: {
    enabled: boolean;
    stageThree: boolean;
    offloadOptimizer: boolean;
    offloadParams: boolean;
  };
  // Mixture of Experts options
  moe: {
    enabled: boolean;
    numExperts: number;
    topK: number;
    capacityFactorTrain: number;
    capacityFactorEval: number;
    expertParallelism: boolean;
    expertDropout: number;
  };
  // Training hyperparameters
  hyperparameters: {
    batchSize: number;
    blockSize: number; // maximum context length for predictions
    maxIters: number;
    evalInterval: number;
    learningRate: number;
    evalIters: number;
    nEmbd: number;
    nHead: number;
    nLayer: number;
    dropout: number;
  };
  // Attention optimizations
  flashAttention: boolean;
  xformers: boolean;
  // Debugging options
  debug?: boolean;
  gradientCheckpointing: boolean;
  mixedPrecision: 'none' | 'fp16' | 'bf16';
  // Compilation
  torchCompile: boolean;
  torchCompileMode: 'default' | 'reduce-overhead' | 'max-autotune';
  // Device detection
  deviceDetection: {
    enabled: boolean;
    preferMps: boolean; // For Mac Metal (M1/M2/M3)
  };
  // Experiment settings
  experiment: {
    enabled: boolean;
    batchSize: number;
    epochs: number;
    trackMetrics: boolean;
    saveCheckpoints: boolean;
    generateSyntheticData: boolean;
    datasetSize: number;
    sequenceLength: number;
  };
}

export const defaultOptimizationSettings: OptimizationSettings = {
  fsdp: {
    enabled: false,
    shardingStrategy: 'FULL_SHARD',
    autoWrap: true,
    minNumParams: 1e8,
  },
  deepSpeed: {
    enabled: false,
    stageThree: false,
    offloadOptimizer: false,
    offloadParams: false,
  },
  moe: {
    enabled: false,
    numExperts: 8,
    topK: 2,
    capacityFactorTrain: 1.25,
    capacityFactorEval: 2.0,
    expertParallelism: false,
    expertDropout: 0.1,
  },
  hyperparameters: {
    batchSize: 64,
    blockSize: 256,
    maxIters: 5000,
    evalInterval: 500,
    learningRate: 0.0003, // 3e-4
    evalIters: 200,
    nEmbd: 384,
    nHead: 6,
    nLayer: 6,
    dropout: 0.2,
  },
  flashAttention: false,
  xformers: false,
  gradientCheckpointing: false,
  mixedPrecision: 'none',
  torchCompile: false,
  torchCompileMode: 'default',
  deviceDetection: {
    enabled: true,
    preferMps: true,
  },
  debug: false,
  experiment: {
    enabled: false,
    batchSize: 16,
    epochs: 3,
    trackMetrics: true,
    saveCheckpoints: true,
    generateSyntheticData: true,
    datasetSize: 1000,
    sequenceLength: 128,
  },
};

interface OptimizationPanelProps {
  settings: OptimizationSettings;
  onSettingsChange: (settings: OptimizationSettings) => void;
}

export function OptimizationPanel({ settings, onSettingsChange }: OptimizationPanelProps) {
  const [activeTab, setActiveTab] = useState<string>('parallelism');

  const updateSettings = (partialSettings: Partial<OptimizationSettings>) => {
    onSettingsChange({ ...settings, ...partialSettings });
  };

  const updateFSDPSettings = (partialSettings: Partial<OptimizationSettings['fsdp']>) => {
    onSettingsChange({
      ...settings,
      fsdp: { ...settings.fsdp, ...partialSettings },
    });
  };

  const updateDeepSpeedSettings = (partialSettings: Partial<OptimizationSettings['deepSpeed']>) => {
    onSettingsChange({
      ...settings,
      deepSpeed: { ...settings.deepSpeed, ...partialSettings },
    });
  };

  const updateMoESettings = (partialSettings: Partial<OptimizationSettings['moe']>) => {
    onSettingsChange({
      ...settings,
      moe: { ...settings.moe, ...partialSettings },
    });
  };

  const updateHyperparameters = (partialSettings: Partial<OptimizationSettings['hyperparameters']>) => {
    onSettingsChange({
      ...settings,
      hyperparameters: { ...settings.hyperparameters, ...partialSettings },
    });
  };

  const updateDeviceSettings = (partialSettings: Partial<OptimizationSettings['deviceDetection']>) => {
    onSettingsChange({
      ...settings,
      deviceDetection: { ...settings.deviceDetection, ...partialSettings },
    });
  };

  const updateExperimentSettings = (partialSettings: Partial<OptimizationSettings['experiment']>) => {
    onSettingsChange({
      ...settings,
      experiment: { ...settings.experiment, ...partialSettings },
    });
  };

  return (
    <div className="flex flex-col h-full">
      <h3 className="text-lg font-semibold mb-4">Optimization Settings</h3>
      
      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1">
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="parallelism">Parallelism</TabsTrigger>
          <TabsTrigger value="moe">MoE</TabsTrigger>
          <TabsTrigger value="hyperparameters">Hyperparameters</TabsTrigger>
          <TabsTrigger value="attention">Attention</TabsTrigger>
          <TabsTrigger value="memory">Memory</TabsTrigger>
          <TabsTrigger value="experiment">Experiment</TabsTrigger>
        </TabsList>
        
        {/* Parallelism Tab */}
        <TabsContent value="parallelism" className="flex-1 overflow-y-auto space-y-4">
          <Card className="p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <Checkbox 
                  id="fsdp-enabled" 
                  checked={settings.fsdp.enabled}
                  onCheckedChange={(checked: boolean | "indeterminate") => {
                    if (checked === true) {
                      // Disable DeepSpeed if FSDP is enabled
                      updateSettings({
                        fsdp: { ...settings.fsdp, enabled: true },
                        deepSpeed: { ...settings.deepSpeed, enabled: false },
                      });
                    } else {
                      updateFSDPSettings({ enabled: false });
                    }
                  }}
                />
                <Label htmlFor="fsdp-enabled" className="font-medium">Fully Sharded Data Parallel (FSDP)</Label>
              </div>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger>
                    <InfoIcon className="h-4 w-4 text-slate-400" />
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs">FSDP shards model parameters, gradients, and optimizer states across data parallel workers to reduce memory usage.</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
            
            {settings.fsdp.enabled && (
              <div className="pl-6 space-y-3 mt-3">
                <div className="space-y-1">
                  <Label htmlFor="fsdp-sharding">Sharding Strategy</Label>
                  <Select 
                    value={settings.fsdp.shardingStrategy} 
                    onValueChange={(value: 'FULL_SHARD' | 'SHARD_GRAD_OP' | 'NO_SHARD') => updateFSDPSettings({ shardingStrategy: value })}
                  >
                    <SelectTrigger id="fsdp-sharding">
                      <SelectValue placeholder="Select sharding strategy" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="FULL_SHARD">Full Sharding</SelectItem>
                      <SelectItem value="SHARD_GRAD_OP">Gradient & Optimizer Sharding</SelectItem>
                      <SelectItem value="NO_SHARD">No Sharding</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Checkbox 
                    id="fsdp-autowrap" 
                    checked={settings.fsdp.autoWrap}
                    onCheckedChange={(checked: boolean | "indeterminate") => updateFSDPSettings({ autoWrap: checked === true })}
                  />
                  <Label htmlFor="fsdp-autowrap">Auto Wrap</Label>
                </div>
              </div>
            )}
          </Card>
          
          <Card className="p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <Checkbox 
                  id="deepspeed-enabled" 
                  checked={settings.deepSpeed.enabled}
                  onCheckedChange={(checked: boolean | "indeterminate") => {
                    if (checked === true) {
                      // Disable FSDP if DeepSpeed is enabled
                      updateSettings({
                        deepSpeed: { ...settings.deepSpeed, enabled: true },
                        fsdp: { ...settings.fsdp, enabled: false },
                      });
                    } else {
                      updateDeepSpeedSettings({ enabled: false });
                    }
                  }}
                />
                <Label htmlFor="deepspeed-enabled" className="font-medium">DeepSpeed</Label>
              </div>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger>
                    <InfoIcon className="h-4 w-4 text-slate-400" />
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs">DeepSpeed provides optimization techniques for distributed training of large models.</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
            
            {settings.deepSpeed.enabled && (
              <div className="pl-6 space-y-3 mt-3">
                <div className="flex items-center space-x-2">
                  <Checkbox 
                    id="deepspeed-stage3" 
                    checked={settings.deepSpeed.stageThree}
                    onCheckedChange={(checked: boolean | "indeterminate") => updateDeepSpeedSettings({ stageThree: checked === true })}
                  />
                  <Label htmlFor="deepspeed-stage3">ZeRO Stage 3</Label>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Checkbox 
                    id="deepspeed-offload-optimizer" 
                    checked={settings.deepSpeed.offloadOptimizer}
                    onCheckedChange={(checked: boolean | "indeterminate") => updateDeepSpeedSettings({ offloadOptimizer: checked === true })}
                  />
                  <Label htmlFor="deepspeed-offload-optimizer">Offload Optimizer</Label>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Checkbox 
                    id="deepspeed-offload-params" 
                    checked={settings.deepSpeed.offloadParams}
                    onCheckedChange={(checked: boolean | "indeterminate") => updateDeepSpeedSettings({ offloadParams: checked === true })}
                  />
                  <Label htmlFor="deepspeed-offload-params">Offload Parameters</Label>
                </div>
              </div>
            )}
          </Card>
        </TabsContent>
        
        {/* MoE Tab */}
        <TabsContent value="moe" className="flex-1 overflow-y-auto space-y-4">
          <Card className="p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <Checkbox 
                  id="moe-enabled" 
                  checked={settings.moe.enabled}
                  onCheckedChange={(checked: boolean | "indeterminate") => updateMoESettings({ enabled: checked === true })}
                />
                <Label htmlFor="moe-enabled" className="font-medium">Enable Mixture of Experts (MoE)</Label>
              </div>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger>
                    <InfoIcon className="h-4 w-4 text-slate-400" />
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs">Mixture of Experts (MoE) increases model capacity without increasing computation by routing tokens to specialized expert networks.</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
            
            {settings.moe.enabled && (
              <div className="pl-6 space-y-3 mt-3">
                <div className="space-y-1">
                  <Label htmlFor="num-experts">Number of Experts</Label>
                  <Select 
                    value={settings.moe.numExperts.toString()} 
                    onValueChange={(value: string) => updateMoESettings({ numExperts: parseInt(value) })}
                  >
                    <SelectTrigger id="num-experts">
                      <SelectValue placeholder="Select number of experts" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="4">4 experts</SelectItem>
                      <SelectItem value="8">8 experts</SelectItem>
                      <SelectItem value="16">16 experts</SelectItem>
                      <SelectItem value="32">32 experts</SelectItem>
                      <SelectItem value="64">64 experts</SelectItem>
                      <SelectItem value="128">128 experts</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-1">
                  <Label htmlFor="top-k">Top-K Experts</Label>
                  <Select 
                    value={settings.moe.topK.toString()} 
                    onValueChange={(value: string) => updateMoESettings({ topK: parseInt(value) })}
                  >
                    <SelectTrigger id="top-k">
                      <SelectValue placeholder="Select top-k experts" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1">1 expert (Switch Transformers)</SelectItem>
                      <SelectItem value="2">2 experts (Standard MoE)</SelectItem>
                      <SelectItem value="4">4 experts</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-slate-500 mt-1">Number of experts each token is routed to</p>
                </div>
                
                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="capacity-factor-train">Capacity Factor (Training)</Label>
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger>
                          <InfoIcon className="h-4 w-4 text-slate-400" />
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="max-w-xs">Determines how many tokens each expert can process. Higher values reduce dropped tokens but increase memory usage.</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </div>
                  <Select 
                    value={settings.moe.capacityFactorTrain.toString()} 
                    onValueChange={(value: string) => updateMoESettings({ capacityFactorTrain: parseFloat(value) })}
                  >
                    <SelectTrigger id="capacity-factor-train">
                      <SelectValue placeholder="Select capacity factor" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1">1.0</SelectItem>
                      <SelectItem value="1.25">1.25</SelectItem>
                      <SelectItem value="1.5">1.5</SelectItem>
                      <SelectItem value="2">2.0</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-1">
                  <Label htmlFor="capacity-factor-eval">Capacity Factor (Evaluation)</Label>
                  <Select 
                    value={settings.moe.capacityFactorEval.toString()} 
                    onValueChange={(value: string) => updateMoESettings({ capacityFactorEval: parseFloat(value) })}
                  >
                    <SelectTrigger id="capacity-factor-eval">
                      <SelectValue placeholder="Select capacity factor" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1">1.0</SelectItem>
                      <SelectItem value="1.5">1.5</SelectItem>
                      <SelectItem value="2">2.0</SelectItem>
                      <SelectItem value="2.5">2.5</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Checkbox 
                    id="expert-parallelism" 
                    checked={settings.moe.expertParallelism}
                    onCheckedChange={(checked: boolean | "indeterminate") => updateMoESettings({ expertParallelism: checked === true })}
                  />
                  <Label htmlFor="expert-parallelism">Expert Parallelism</Label>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <InfoIcon className="h-4 w-4 text-slate-400" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs">Distribute experts across multiple devices for parallel computation.</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                
                <div className="space-y-1">
                  <Label htmlFor="expert-dropout">Expert Dropout</Label>
                  <Select 
                    value={settings.moe.expertDropout.toString()} 
                    onValueChange={(value: string) => updateMoESettings({ expertDropout: parseFloat(value) })}
                  >
                    <SelectTrigger id="expert-dropout">
                      <SelectValue placeholder="Select expert dropout" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="0">0.0</SelectItem>
                      <SelectItem value="0.1">0.1</SelectItem>
                      <SelectItem value="0.2">0.2</SelectItem>
                      <SelectItem value="0.3">0.3</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-slate-500 mt-1">Randomly drop experts during training to improve robustness</p>
                </div>
              </div>
            )}
          </Card>
        </TabsContent>
        
        {/* Hyperparameters Tab */}
        <TabsContent value="hyperparameters" className="flex-1 overflow-y-auto space-y-4">
          <Card className="p-4">
            <h3 className="text-sm font-medium mb-3">Training Hyperparameters</h3>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <Label htmlFor="batch-size-param" className="text-xs">Batch Size</Label>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <InfoIcon className="h-3 w-3 text-slate-400" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs">Number of samples processed in each training step.</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <input
                  id="batch-size-param"
                  type="number"
                  className="w-full h-8 rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm"
                  value={settings.hyperparameters.batchSize}
                  onChange={(e) => updateHyperparameters({ batchSize: parseInt(e.target.value) || 64 })}
                  min="1"
                />
              </div>
              
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <Label htmlFor="block-size-param" className="text-xs">Block Size</Label>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <InfoIcon className="h-3 w-3 text-slate-400" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs">Maximum context length for predictions (sequence length).</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <input
                  id="block-size-param"
                  type="number"
                  className="w-full h-8 rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm"
                  value={settings.hyperparameters.blockSize}
                  onChange={(e) => updateHyperparameters({ blockSize: parseInt(e.target.value) || 256 })}
                  min="1"
                />
              </div>
              
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <Label htmlFor="max-iters-param" className="text-xs">Max Iterations</Label>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <InfoIcon className="h-3 w-3 text-slate-400" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs">Total number of training iterations.</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <input
                  id="max-iters-param"
                  type="number"
                  className="w-full h-8 rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm"
                  value={settings.hyperparameters.maxIters}
                  onChange={(e) => updateHyperparameters({ maxIters: parseInt(e.target.value) || 5000 })}
                  min="1"
                />
              </div>
              
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <Label htmlFor="eval-interval-param" className="text-xs">Eval Interval</Label>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <InfoIcon className="h-3 w-3 text-slate-400" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs">How often to evaluate the model during training.</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <input
                  id="eval-interval-param"
                  type="number"
                  className="w-full h-8 rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm"
                  value={settings.hyperparameters.evalInterval}
                  onChange={(e) => updateHyperparameters({ evalInterval: parseInt(e.target.value) || 500 })}
                  min="1"
                />
              </div>
              
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <Label htmlFor="learning-rate-param" className="text-xs">Learning Rate</Label>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <InfoIcon className="h-3 w-3 text-slate-400" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs">Step size for gradient descent optimization.</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <input
                  id="learning-rate-param"
                  type="number"
                  className="w-full h-8 rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm"
                  value={settings.hyperparameters.learningRate}
                  onChange={(e) => updateHyperparameters({ learningRate: parseFloat(e.target.value) || 0.0003 })}
                  min="0"
                  step="0.0001"
                />
              </div>
              
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <Label htmlFor="eval-iters-param" className="text-xs">Eval Iterations</Label>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <InfoIcon className="h-3 w-3 text-slate-400" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs">Number of iterations to use for evaluation.</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <input
                  id="eval-iters-param"
                  type="number"
                  className="w-full h-8 rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm"
                  value={settings.hyperparameters.evalIters}
                  onChange={(e) => updateHyperparameters({ evalIters: parseInt(e.target.value) || 200 })}
                  min="1"
                />
              </div>
            </div>
          </Card>
          
          <Card className="p-4">
            <h3 className="text-sm font-medium mb-3">Model Architecture Parameters</h3>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <Label htmlFor="n-embd-param" className="text-xs">Embedding Dimension</Label>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <InfoIcon className="h-3 w-3 text-slate-400" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs">Size of the embedding vectors (hidden dimension).</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <input
                  id="n-embd-param"
                  type="number"
                  className="w-full h-8 rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm"
                  value={settings.hyperparameters.nEmbd}
                  onChange={(e) => updateHyperparameters({ nEmbd: parseInt(e.target.value) || 384 })}
                  min="1"
                  step="64"
                />
              </div>
              
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <Label htmlFor="n-head-param" className="text-xs">Number of Heads</Label>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <InfoIcon className="h-3 w-3 text-slate-400" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs">Number of attention heads in each attention layer.</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <input
                  id="n-head-param"
                  type="number"
                  className="w-full h-8 rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm"
                  value={settings.hyperparameters.nHead}
                  onChange={(e) => updateHyperparameters({ nHead: parseInt(e.target.value) || 6 })}
                  min="1"
                />
              </div>
              
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <Label htmlFor="n-layer-param" className="text-xs">Number of Layers</Label>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <InfoIcon className="h-3 w-3 text-slate-400" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs">Number of transformer blocks in the model.</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <input
                  id="n-layer-param"
                  type="number"
                  className="w-full h-8 rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm"
                  value={settings.hyperparameters.nLayer}
                  onChange={(e) => updateHyperparameters({ nLayer: parseInt(e.target.value) || 6 })}
                  min="1"
                />
              </div>
              
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <Label htmlFor="dropout-param" className="text-xs">Dropout Rate</Label>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <InfoIcon className="h-3 w-3 text-slate-400" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs">Probability of dropping connections during training (regularization).</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <input
                  id="dropout-param"
                  type="number"
                  className="w-full h-8 rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm"
                  value={settings.hyperparameters.dropout}
                  onChange={(e) => updateHyperparameters({ dropout: parseFloat(e.target.value) || 0.2 })}
                  min="0"
                  max="0.9"
                  step="0.1"
                />
              </div>
            </div>
          </Card>
        </TabsContent>
        
        {/* Attention Optimizations Tab */}
        <TabsContent value="attention" className="flex-1 overflow-y-auto space-y-4">
          <Card className="p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <Checkbox 
                  id="flash-attention" 
                  checked={settings.flashAttention}
                  onCheckedChange={(checked: boolean | "indeterminate") => {
                    if (checked === true) {
                      // Disable xformers if Flash Attention is enabled
                      updateSettings({
                        flashAttention: true,
                        xformers: false,
                      });
                    } else {
                      updateSettings({ flashAttention: false });
                    }
                  }}
                />
                <Label htmlFor="flash-attention" className="font-medium">Flash Attention</Label>
              </div>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger>
                    <InfoIcon className="h-4 w-4 text-slate-400" />
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs">Flash Attention is a faster, more memory-efficient attention algorithm that reduces memory usage and increases training speed.</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          </Card>
          
          <Card className="p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <Checkbox 
                  id="xformers" 
                  checked={settings.xformers}
                  onCheckedChange={(checked: boolean | "indeterminate") => {
                    if (checked === true) {
                      // Disable Flash Attention if xformers is enabled
                      updateSettings({
                        xformers: true,
                        flashAttention: false,
                      });
                    } else {
                      updateSettings({ xformers: false });
                    }
                  }}
                />
                <Label htmlFor="xformers" className="font-medium">xFormers Memory-Efficient Attention</Label>
              </div>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger>
                    <InfoIcon className="h-4 w-4 text-slate-400" />
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs">xFormers provides memory-efficient attention mechanisms that can speed up training and reduce memory usage.</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          </Card>
        </TabsContent>
        
        {/* Memory Optimizations Tab */}
        <TabsContent value="memory" className="flex-1 overflow-y-auto space-y-4">
          <Card className="p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <Checkbox 
                  id="gradient-checkpointing" 
                  checked={settings.gradientCheckpointing}
                  onCheckedChange={(checked: boolean | "indeterminate") => updateSettings({ gradientCheckpointing: checked === true })}
                />
                <Label htmlFor="gradient-checkpointing" className="font-medium">Gradient Checkpointing</Label>
              </div>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger>
                    <InfoIcon className="h-4 w-4 text-slate-400" />
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs">Gradient checkpointing trades compute for memory by recomputing intermediate activations during the backward pass instead of storing them.</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
            
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <Checkbox 
                  id="debug-mode" 
                  checked={settings.debug}
                  onCheckedChange={(checked: boolean | "indeterminate") => updateSettings({ debug: checked === true })}
                />
                <Label htmlFor="debug-mode" className="font-medium">Debug Mode</Label>
              </div>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger>
                    <InfoIcon className="h-4 w-4 text-slate-400" />
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs">Adds print statements to show tensor shapes during forward pass for debugging.</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          </Card>
          
          <Card className="p-4">
            <div className="space-y-1">
              <div className="flex items-center justify-between">
                <Label htmlFor="mixed-precision" className="font-medium">Mixed Precision Training</Label>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger>
                      <InfoIcon className="h-4 w-4 text-slate-400" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="max-w-xs">Mixed precision training uses lower precision formats (FP16 or BF16) to reduce memory usage and increase training speed.</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              <Select 
                value={settings.mixedPrecision} 
                onValueChange={(value: 'none' | 'fp16' | 'bf16') => updateSettings({ mixedPrecision: value })}
              >
                <SelectTrigger id="mixed-precision">
                  <SelectValue placeholder="Select precision" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">None (FP32)</SelectItem>
                  <SelectItem value="fp16">FP16</SelectItem>
                  <SelectItem value="bf16">BF16</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </Card>
          
          <Card className="p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <Checkbox 
                  id="torch-compile" 
                  checked={settings.torchCompile}
                  onCheckedChange={(checked: boolean | "indeterminate") => updateSettings({ torchCompile: checked === true })}
                />
                <Label htmlFor="torch-compile" className="font-medium">torch.compile()</Label>
              </div>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger>
                    <InfoIcon className="h-4 w-4 text-slate-400" />
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs">torch.compile() uses PyTorch 2.0&apos;s compiler to optimize model execution for faster training and inference.</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
            
            {settings.torchCompile && (
              <div className="pl-6 space-y-1 mt-3">
                <Label htmlFor="torch-compile-mode">Compilation Mode</Label>
                <Select 
                  value={settings.torchCompileMode} 
                  onValueChange={(value: 'default' | 'reduce-overhead' | 'max-autotune') => updateSettings({ torchCompileMode: value })}
                >
                  <SelectTrigger id="torch-compile-mode">
                    <SelectValue placeholder="Select compilation mode" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="default">Default</SelectItem>
                    <SelectItem value="reduce-overhead">Reduce Overhead</SelectItem>
                    <SelectItem value="max-autotune">Max Autotune</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            )}
          </Card>
        </TabsContent>
        
        {/* Experiment Settings Tab */}
        <TabsContent value="experiment" className="flex-1 overflow-y-auto space-y-4">
          <Card className="p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <Checkbox 
                  id="device-detection" 
                  checked={settings.deviceDetection.enabled}
                  onCheckedChange={(checked: boolean | "indeterminate") => updateDeviceSettings({ enabled: checked === true })}
                />
                <Label htmlFor="device-detection" className="font-medium">Auto Device Detection</Label>
              </div>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger>
                    <InfoIcon className="h-4 w-4 text-slate-400" />
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs">Automatically detect and use the best available device (CUDA, MPS, CPU).</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
            
            {settings.deviceDetection.enabled && (
              <div className="pl-6 space-y-3 mt-3">
                <div className="flex items-center space-x-2">
                  <Checkbox 
                    id="prefer-mps" 
                    checked={settings.deviceDetection.preferMps}
                    onCheckedChange={(checked: boolean | "indeterminate") => updateDeviceSettings({ preferMps: checked === true })}
                  />
                  <Label htmlFor="prefer-mps">Prefer MPS on Mac (M1/M2/M3)</Label>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <InfoIcon className="h-4 w-4 text-slate-400" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs">Use Metal Performance Shaders (MPS) on Apple Silicon Macs for GPU acceleration.</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
              </div>
            )}
          </Card>
          
          <Card className="p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <Checkbox 
                  id="experiment-enabled" 
                  checked={settings.experiment.enabled}
                  onCheckedChange={(checked: boolean | "indeterminate") => updateExperimentSettings({ enabled: checked === true })}
                />
                <Label htmlFor="experiment-enabled" className="font-medium">Enable Small-Scale Experiment</Label>
              </div>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger>
                    <InfoIcon className="h-4 w-4 text-slate-400" />
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs">Generate code to run a small-scale experiment with synthetic data to test the model.</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
            
            {settings.experiment.enabled && (
              <div className="pl-6 space-y-3 mt-3">
                <div className="space-y-1">
                  <Label htmlFor="batch-size">Batch Size</Label>
                  <Select 
                    value={settings.experiment.batchSize.toString()} 
                    onValueChange={(value: string) => updateExperimentSettings({ batchSize: parseInt(value) })}
                  >
                    <SelectTrigger id="batch-size">
                      <SelectValue placeholder="Select batch size" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="4">4</SelectItem>
                      <SelectItem value="8">8</SelectItem>
                      <SelectItem value="16">16</SelectItem>
                      <SelectItem value="32">32</SelectItem>
                      <SelectItem value="64">64</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-1">
                  <Label htmlFor="epochs">Epochs</Label>
                  <Select 
                    value={settings.experiment.epochs.toString()} 
                    onValueChange={(value: string) => updateExperimentSettings({ epochs: parseInt(value) })}
                  >
                    <SelectTrigger id="epochs">
                      <SelectValue placeholder="Select number of epochs" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1">1</SelectItem>
                      <SelectItem value="3">3</SelectItem>
                      <SelectItem value="5">5</SelectItem>
                      <SelectItem value="10">10</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-1">
                  <Label htmlFor="dataset-size">Synthetic Dataset Size</Label>
                  <Select 
                    value={settings.experiment.datasetSize.toString()} 
                    onValueChange={(value: string) => updateExperimentSettings({ datasetSize: parseInt(value) })}
                  >
                    <SelectTrigger id="dataset-size">
                      <SelectValue placeholder="Select dataset size" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="100">100 samples</SelectItem>
                      <SelectItem value="500">500 samples</SelectItem>
                      <SelectItem value="1000">1,000 samples</SelectItem>
                      <SelectItem value="5000">5,000 samples</SelectItem>
                      <SelectItem value="10000">10,000 samples</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-1">
                  <Label htmlFor="sequence-length">Sequence Length</Label>
                  <Select 
                    value={settings.experiment.sequenceLength.toString()} 
                    onValueChange={(value: string) => updateExperimentSettings({ sequenceLength: parseInt(value) })}
                  >
                    <SelectTrigger id="sequence-length">
                      <SelectValue placeholder="Select sequence length" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="32">32 tokens</SelectItem>
                      <SelectItem value="64">64 tokens</SelectItem>
                      <SelectItem value="128">128 tokens</SelectItem>
                      <SelectItem value="256">256 tokens</SelectItem>
                      <SelectItem value="512">512 tokens</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Checkbox 
                    id="track-metrics" 
                    checked={settings.experiment.trackMetrics}
                    onCheckedChange={(checked: boolean | "indeterminate") => updateExperimentSettings({ trackMetrics: checked === true })}
                  />
                  <Label htmlFor="track-metrics">Track Training Metrics</Label>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Checkbox 
                    id="save-checkpoints" 
                    checked={settings.experiment.saveCheckpoints}
                    onCheckedChange={(checked: boolean | "indeterminate") => updateExperimentSettings({ saveCheckpoints: checked === true })}
                  />
                  <Label htmlFor="save-checkpoints">Save Model Checkpoints</Label>
                </div>
              </div>
            )}
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
} 
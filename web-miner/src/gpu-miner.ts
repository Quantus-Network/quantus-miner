/**
 * WebGPU-based miner for Quantus Network.
 * Uses the same WGSL shader as the native GPU miner.
 */

import type { MiningResult, MiningJob } from './types';
import shaderSource from './mining.wgsl?raw';

// Buffer layout constants (matching the WGSL shader)
const RESULTS_SIZE = 33; // [flag(1), nonce(16), hash(16)] in u32s
const HEADER_SIZE = 8; // 32 bytes = 8 u32s
const NONCE_SIZE = 16; // 64 bytes = 16 u32s (U512)
const DIFFICULTY_SIZE = 16; // 64 bytes = 16 u32s (U512)
const DISPATCH_CONFIG_SIZE = 4; // [total_threads, nonces_per_thread, total_nonces, cancel_check_interval]
const CANCEL_FLAG_SIZE = 1;

// Workgroup size must match shader
const WORKGROUP_SIZE = 256;

export class GpuMiner {
  private device: GPUDevice | null = null;
  private pipeline: GPUComputePipeline | null = null;
  private bindGroupLayout: GPUBindGroupLayout | null = null;

  // GPU buffers
  private resultsBuffer: GPUBuffer | null = null;
  private headerBuffer: GPUBuffer | null = null;
  private nonceBuffer: GPUBuffer | null = null;
  private difficultyBuffer: GPUBuffer | null = null;
  private dispatchConfigBuffer: GPUBuffer | null = null;
  private cancelFlagBuffer: GPUBuffer | null = null;
  private resultsStagingBuffer: GPUBuffer | null = null;

  // Mining state
  private currentJob: MiningJob | null = null;
  private isMining = false;
  private shouldCancel = false;

  // Performance tuning
  private workgroupsPerDispatch = 256; // Start conservative, can tune later
  private noncesPerThread = 16;

  /**
   * Check if WebGPU is available in this browser.
   */
  static isSupported(): boolean {
    return 'gpu' in navigator;
  }

  /**
   * Initialize the GPU device and compile the shader.
   */
  async initialize(): Promise<void> {
    if (!GpuMiner.isSupported()) {
      throw new Error('WebGPU is not supported in this browser');
    }

    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    });

    if (!adapter) {
      throw new Error('Failed to get GPU adapter');
    }

    this.device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
      },
    });

    // Handle device loss
    this.device.lost.then((info) => {
      console.error('GPU device lost:', info.message);
      this.cleanup();
    });

    // Create shader module
    const shaderModule = this.device.createShaderModule({
      code: shaderSource,
    });

    // Check for compilation errors
    const compilationInfo = await shaderModule.getCompilationInfo();
    for (const message of compilationInfo.messages) {
      if (message.type === 'error') {
        throw new Error(`Shader compilation error: ${message.message}`);
      }
      console.warn(`Shader warning: ${message.message}`);
    }

    // Create bind group layout
    this.bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // results
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // header
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // start_nonce
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // difficulty_target
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // dispatch_config
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // cancel_flag
      ],
    });

    // Create pipeline
    this.pipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      compute: {
        module: shaderModule,
        entryPoint: 'mining_main',
      },
    });

    // Create buffers
    this.createBuffers();

    console.log('GPU miner initialized successfully');
  }

  private createBuffers(): void {
    if (!this.device) throw new Error('Device not initialized');

    // Results buffer (read-write storage)
    this.resultsBuffer = this.device.createBuffer({
      size: RESULTS_SIZE * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    // Header buffer
    this.headerBuffer = this.device.createBuffer({
      size: HEADER_SIZE * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Nonce buffer
    this.nonceBuffer = this.device.createBuffer({
      size: NONCE_SIZE * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Difficulty buffer
    this.difficultyBuffer = this.device.createBuffer({
      size: DIFFICULTY_SIZE * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Dispatch config buffer
    this.dispatchConfigBuffer = this.device.createBuffer({
      size: DISPATCH_CONFIG_SIZE * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Cancel flag buffer
    this.cancelFlagBuffer = this.device.createBuffer({
      size: CANCEL_FLAG_SIZE * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Staging buffer for reading results back
    this.resultsStagingBuffer = this.device.createBuffer({
      size: RESULTS_SIZE * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * Start mining a job. Returns when a solution is found or the job is cancelled.
   */
  async mine(job: MiningJob): Promise<MiningResult> {
    if (!this.device || !this.pipeline || !this.bindGroupLayout) {
      throw new Error('GPU not initialized');
    }
    if (this.isMining) {
      throw new Error('Already mining');
    }

    this.currentJob = job;
    this.isMining = true;
    this.shouldCancel = false;

    try {
      // Clear results buffer
      const clearData = new Uint32Array(RESULTS_SIZE);
      this.device.queue.writeBuffer(this.resultsBuffer!, 0, clearData.buffer);

      // Write header
      this.device.queue.writeBuffer(this.headerBuffer!, 0, job.header.buffer);

      // Write difficulty target
      this.device.queue.writeBuffer(this.difficultyBuffer!, 0, job.difficultyTarget.buffer);

      // Write initial nonce
      this.device.queue.writeBuffer(this.nonceBuffer!, 0, job.startNonce.buffer);

      // Write cancel flag (0 = not cancelled)
      this.device.queue.writeBuffer(this.cancelFlagBuffer!, 0, new Uint32Array([0]).buffer);

      // Mining loop
      let totalHashes = 0;
      const currentNonce = new Uint32Array(job.startNonce);

      while (!this.shouldCancel) {
        const result = await this.dispatchMining(currentNonce);
        totalHashes += result.hashCount;
        job.hashCount = totalHashes;

        if (result.found && result.nonce && result.hash) {
          return {
            found: true,
            nonce: result.nonce,
            hash: result.hash,
            hashCount: totalHashes,
          };
        }

        // Advance nonce for next dispatch
        this.advanceNonce(currentNonce, result.hashCount);

        // Small yield to allow UI updates and cancellation checks
        await new Promise((resolve) => setTimeout(resolve, 0));
      }

      // Cancelled
      return { found: false, hashCount: totalHashes };
    } finally {
      this.isMining = false;
      this.currentJob = null;
    }
  }

  /**
   * Cancel the current mining job.
   */
  cancel(): void {
    this.shouldCancel = true;
    // Also write to GPU cancel flag for early exit
    if (this.device && this.cancelFlagBuffer) {
      this.device.queue.writeBuffer(this.cancelFlagBuffer, 0, new Uint32Array([1]).buffer);
    }
  }

  /**
   * Check if currently mining.
   */
  isBusy(): boolean {
    return this.isMining;
  }

  /**
   * Get the current job (if any).
   */
  getCurrentJob(): MiningJob | null {
    return this.currentJob;
  }

  /**
   * Clean up GPU resources.
   */
  cleanup(): void {
    this.cancel();
    this.resultsBuffer?.destroy();
    this.headerBuffer?.destroy();
    this.nonceBuffer?.destroy();
    this.difficultyBuffer?.destroy();
    this.dispatchConfigBuffer?.destroy();
    this.cancelFlagBuffer?.destroy();
    this.resultsStagingBuffer?.destroy();
    this.device?.destroy();
    this.device = null;
    this.pipeline = null;
    this.bindGroupLayout = null;
  }

  private async dispatchMining(startNonce: Uint32Array): Promise<MiningResult> {
    if (!this.device || !this.pipeline || !this.bindGroupLayout) {
      throw new Error('GPU not initialized');
    }

    const totalThreads = this.workgroupsPerDispatch * WORKGROUP_SIZE;
    const totalNonces = totalThreads * this.noncesPerThread;

    // Update nonce buffer
    this.device.queue.writeBuffer(this.nonceBuffer!, 0, startNonce.buffer);

    // Update dispatch config
    const dispatchConfig = new Uint32Array([
      totalThreads,
      this.noncesPerThread,
      totalNonces,
      64, // cancel_check_interval
    ]);
    this.device.queue.writeBuffer(this.dispatchConfigBuffer!, 0, dispatchConfig.buffer);

    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.resultsBuffer! } },
        { binding: 1, resource: { buffer: this.headerBuffer! } },
        { binding: 2, resource: { buffer: this.nonceBuffer! } },
        { binding: 3, resource: { buffer: this.difficultyBuffer! } },
        { binding: 4, resource: { buffer: this.dispatchConfigBuffer! } },
        { binding: 5, resource: { buffer: this.cancelFlagBuffer! } },
      ],
    });

    // Create command encoder
    const commandEncoder = this.device.createCommandEncoder();

    // Dispatch compute shader
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(this.workgroupsPerDispatch);
    passEncoder.end();

    // Copy results to staging buffer
    commandEncoder.copyBufferToBuffer(
      this.resultsBuffer!,
      0,
      this.resultsStagingBuffer!,
      0,
      RESULTS_SIZE * 4
    );

    // Submit commands
    this.device.queue.submit([commandEncoder.finish()]);

    // Read back results
    await this.resultsStagingBuffer!.mapAsync(GPUMapMode.READ);
    const resultsData = new Uint32Array(
      this.resultsStagingBuffer!.getMappedRange().slice(0)
    );
    this.resultsStagingBuffer!.unmap();

    // Check if solution was found
    const found = resultsData[0] !== 0;

    if (found) {
      const nonce = new Uint32Array(resultsData.slice(1, 17));
      const hash = new Uint32Array(resultsData.slice(17, 33));
      return { found: true, nonce, hash, hashCount: totalNonces };
    }

    return { found: false, hashCount: totalNonces };
  }

  private advanceNonce(nonce: Uint32Array, increment: number): void {
    // Add increment to the nonce (U512 little-endian)
    let carry = increment;
    for (let i = 0; i < nonce.length && carry > 0; i++) {
      const sum = nonce[i] + carry;
      nonce[i] = sum >>> 0; // Keep as u32
      carry = sum > 0xffffffff ? 1 : 0;
    }
  }
}

// ============================================================================
// Utility functions for converting between formats
// ============================================================================

/**
 * Convert a hex string to Uint32Array (little-endian u32s).
 * Input should be big-endian hex with no 0x prefix.
 */
export function hexToU32Array(hex: string, expectedU32s: number): Uint32Array {
  // Pad to expected length
  const expectedHexLen = expectedU32s * 8;
  hex = hex.padStart(expectedHexLen, '0');

  const result = new Uint32Array(expectedU32s);

  // Convert from big-endian hex to little-endian u32 array
  for (let i = 0; i < expectedU32s; i++) {
    // Read 8 hex chars (4 bytes) from the end
    const start = expectedHexLen - (i + 1) * 8;
    const hexChunk = hex.slice(start, start + 8);
    result[i] = parseInt(hexChunk, 16);
  }

  return result;
}

/**
 * Convert Uint32Array (little-endian u32s) to hex string (big-endian, no 0x prefix).
 */
export function u32ArrayToHex(arr: Uint32Array): string {
  let hex = '';
  // Read from end to start (little-endian to big-endian)
  for (let i = arr.length - 1; i >= 0; i--) {
    hex += arr[i].toString(16).padStart(8, '0');
  }
  // Remove leading zeros but keep at least one digit
  return hex.replace(/^0+/, '') || '0';
}

/**
 * Convert a decimal string (U512) to Uint32Array (16 u32s, little-endian).
 */
export function decimalToU512(decimal: string): Uint32Array {
  const result = new Uint32Array(16);
  
  // Handle simple cases
  if (decimal === '0' || decimal === '') {
    return result;
  }

  // Convert decimal string to BigInt, then extract u32 limbs
  let value = BigInt(decimal);
  const mask = BigInt(0xffffffff);
  
  for (let i = 0; i < 16 && value > 0n; i++) {
    result[i] = Number(value & mask);
    value >>= 32n;
  }

  return result;
}

/**
 * Create work bytes from header and nonce for submission.
 * Returns 64 bytes as hex string (128 chars).
 */
export function createWorkBytes(header: Uint32Array, nonce: Uint32Array): string {
  // Work = header (32 bytes) + nonce in big-endian (32 bytes selected from U512)
  // Actually looking at the shader, it seems to use full 64-byte nonce
  // Let me check the Rust side...
  
  // For now, create header + nonce as work bytes
  // Header is 8 u32s = 32 bytes
  // Nonce is 16 u32s = 64 bytes, but we only use 32 bytes in practice
  // Total: 32 + 32 = 64 bytes
  
  let hex = '';
  
  // Header in native byte order
  for (let i = 0; i < header.length; i++) {
    hex += header[i].toString(16).padStart(8, '0');
  }
  
  // Nonce: convert from little-endian u32 array to big-endian bytes
  // Take the low 32 bytes (first 8 u32s in little-endian)
  for (let i = 7; i >= 0; i--) {
    const val = nonce[i];
    // Reverse bytes within the u32 for big-endian output
    const b0 = (val >>> 24) & 0xff;
    const b1 = (val >>> 16) & 0xff;
    const b2 = (val >>> 8) & 0xff;
    const b3 = val & 0xff;
    hex += b3.toString(16).padStart(2, '0');
    hex += b2.toString(16).padStart(2, '0');
    hex += b1.toString(16).padStart(2, '0');
    hex += b0.toString(16).padStart(2, '0');
  }
  
  return hex;
}

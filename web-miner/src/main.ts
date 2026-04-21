/**
 * Quantus Web Miner - Main Entry Point
 * 
 * Connects to a mining pool via WebSocket and mines using WebGPU.
 */

import { PoolClient } from './pool-client';
import { GpuMiner, hexToU32Array, u32ArrayToHex, decimalToU512, createWorkBytes } from './gpu-miner';
import type { MinerState, MinerStats, NewJobMessage, MiningJob } from './types';

// ============================================================================
// Global State
// ============================================================================

let poolClient: PoolClient | null = null;
let gpuMiner: GpuMiner | null = null;
let currentJob: MiningJob | null = null;
let stats: MinerStats = {
  hashRate: 0,
  totalHashes: 0,
  blocksWon: 0,
  jobsCompleted: 0,
  uptime: 0,
};
let startTime: number | null = null;
let hashRateInterval: number | null = null;
let lastHashCount = 0;
let lastHashTime = 0;

// ============================================================================
// UI Updates
// ============================================================================

function $(id: string): HTMLElement | null {
  return document.getElementById(id);
}

function updateStatus(status: string): void {
  const el = $('status');
  if (el) el.textContent = status;
}

function updateState(state: MinerState): void {
  const el = $('state');
  if (el) {
    el.textContent = state;
    el.className = `state-${state}`;
  }

  // Update button states
  const connectBtn = $('connect-btn') as HTMLButtonElement | null;
  const disconnectBtn = $('disconnect-btn') as HTMLButtonElement | null;
  const addressInput = $('address') as HTMLInputElement | null;

  if (connectBtn) {
    connectBtn.disabled = state !== 'disconnected';
  }
  if (disconnectBtn) {
    disconnectBtn.disabled = state === 'disconnected';
  }
  if (addressInput) {
    addressInput.disabled = state !== 'disconnected';
  }
}

function updateStats(): void {
  const hashRateEl = $('hash-rate');
  const totalHashesEl = $('total-hashes');
  const blocksWonEl = $('blocks-won');
  const jobsCompletedEl = $('jobs-completed');
  const uptimeEl = $('uptime');

  if (hashRateEl) hashRateEl.textContent = formatHashRate(stats.hashRate);
  if (totalHashesEl) totalHashesEl.textContent = formatNumber(stats.totalHashes);
  if (blocksWonEl) blocksWonEl.textContent = stats.blocksWon.toString();
  if (jobsCompletedEl) jobsCompletedEl.textContent = stats.jobsCompleted.toString();
  if (uptimeEl) uptimeEl.textContent = formatUptime(stats.uptime);
}

function updateCurrentJob(job: MiningJob | null): void {
  const jobIdEl = $('job-id');
  const jobHashesEl = $('job-hashes');

  if (jobIdEl) jobIdEl.textContent = job?.jobId ?? '-';
  if (jobHashesEl) jobHashesEl.textContent = job ? formatNumber(job.hashCount) : '-';
}

function addLog(message: string, type: 'info' | 'success' | 'error' | 'warning' = 'info'): void {
  const logEl = $('log');
  if (!logEl) return;

  const entry = document.createElement('div');
  entry.className = `log-entry log-${type}`;
  entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
  logEl.appendChild(entry);
  logEl.scrollTop = logEl.scrollHeight;

  // Keep only last 100 entries
  while (logEl.children.length > 100) {
    logEl.removeChild(logEl.firstChild!);
  }
}

// ============================================================================
// Formatting Helpers
// ============================================================================

function formatHashRate(hashRate: number): string {
  if (hashRate >= 1e9) return `${(hashRate / 1e9).toFixed(2)} GH/s`;
  if (hashRate >= 1e6) return `${(hashRate / 1e6).toFixed(2)} MH/s`;
  if (hashRate >= 1e3) return `${(hashRate / 1e3).toFixed(2)} KH/s`;
  return `${hashRate.toFixed(2)} H/s`;
}

function formatNumber(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(2)}K`;
  return n.toString();
}

function formatUptime(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

// ============================================================================
// Mining Logic
// ============================================================================

async function startMiningJob(jobMsg: NewJobMessage): Promise<void> {
  if (!gpuMiner) {
    addLog('GPU miner not initialized', 'error');
    return;
  }

  // Cancel any existing job
  if (gpuMiner.isBusy()) {
    gpuMiner.cancel();
    // Wait for cancellation
    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  // Parse job parameters
  const header = hexToU32Array(jobMsg.mining_hash, 8);
  const difficultyTarget = decimalToU512(jobMsg.difficulty);

  // Generate random starting nonce
  const startNonce = new Uint32Array(16);
  crypto.getRandomValues(startNonce);

  currentJob = {
    jobId: jobMsg.job_id,
    header,
    difficultyTarget,
    startNonce,
    hashCount: 0,
    startTime: Date.now(),
  };

  addLog(`Starting job ${jobMsg.job_id}`, 'info');
  updateCurrentJob(currentJob);

  // Start mining in background
  mineJob(currentJob).catch((err) => {
    addLog(`Mining error: ${err.message}`, 'error');
  });
}

async function mineJob(job: MiningJob): Promise<void> {
  if (!gpuMiner || !poolClient) return;

  const result = await gpuMiner.mine(job);

  // Update stats
  stats.totalHashes += result.hashCount;

  if (result.found && result.nonce && result.hash) {
    addLog(`Solution found for job ${job.jobId}!`, 'success');

    // Create work bytes for submission
    const nonceHex = u32ArrayToHex(result.nonce);
    const workHex = createWorkBytes(job.header, result.nonce);
    const elapsedTime = (Date.now() - job.startTime) / 1000;

    // Submit to pool
    poolClient.submitResult(
      job.jobId,
      nonceHex,
      workHex,
      result.hashCount,
      elapsedTime
    );

    addLog(`Submitted solution: nonce=${nonceHex.slice(0, 16)}...`, 'info');
  }

  updateCurrentJob(null);
  currentJob = null;
}

function cancelCurrentJob(): void {
  if (gpuMiner?.isBusy()) {
    gpuMiner.cancel();
    addLog('Job cancelled', 'warning');
  }
  currentJob = null;
  updateCurrentJob(null);
}

// ============================================================================
// Hash Rate Calculation
// ============================================================================

function startHashRateTracking(): void {
  lastHashCount = stats.totalHashes;
  lastHashTime = Date.now();

  hashRateInterval = window.setInterval(() => {
    const now = Date.now();
    const elapsed = (now - lastHashTime) / 1000;

    if (elapsed > 0) {
      const hashes = stats.totalHashes - lastHashCount;
      stats.hashRate = hashes / elapsed;

      // Also add current job hashes if mining
      if (currentJob && gpuMiner?.isBusy()) {
        const jobHashes = currentJob.hashCount;
        updateCurrentJob(currentJob);
      }
    }

    lastHashCount = stats.totalHashes;
    lastHashTime = now;

    // Update uptime
    if (startTime) {
      stats.uptime = (Date.now() - startTime) / 1000;
    }

    updateStats();
  }, 1000);
}

function stopHashRateTracking(): void {
  if (hashRateInterval !== null) {
    clearInterval(hashRateInterval);
    hashRateInterval = null;
  }
}

// ============================================================================
// Connection Management
// ============================================================================

async function connect(): Promise<void> {
  const addressInput = $('address') as HTMLInputElement | null;
  const poolUrlInput = $('pool-url') as HTMLInputElement | null;

  const address = addressInput?.value.trim();
  const poolUrl = poolUrlInput?.value.trim() || 'wss://localhost:9834';

  if (!address) {
    addLog('Please enter a rewards address', 'error');
    return;
  }

  // Validate address format (basic check for SS58)
  if (address.length < 40 || address.length > 50) {
    addLog('Invalid address format (expected SS58)', 'error');
    return;
  }

  updateStatus('Initializing GPU...');
  addLog('Initializing WebGPU...', 'info');

  try {
    // Initialize GPU miner if needed
    if (!gpuMiner) {
      if (!GpuMiner.isSupported()) {
        throw new Error('WebGPU is not supported in this browser. Try Chrome/Edge 113+ or enable WebGPU in flags.');
      }
      gpuMiner = new GpuMiner();
      await gpuMiner.initialize();
      addLog('WebGPU initialized successfully', 'success');
    }

    // Connect to pool
    updateStatus('Connecting to pool...');
    addLog(`Connecting to ${poolUrl}...`, 'info');

    poolClient = new PoolClient(poolUrl);
    poolClient.setHandlers({
      onStateChange: (state) => {
        updateState(state);
        addLog(`State: ${state}`, 'info');
      },
      onRegistered: (minerId) => {
        addLog(`Registered with miner ID: ${minerId}`, 'success');
        startTime = Date.now();
        startHashRateTracking();
      },
      onNewJob: (job) => {
        addLog(`New job: ${job.job_id}`, 'info');
        startMiningJob(job);
      },
      onBlockWon: (blockNumber, jobId) => {
        stats.blocksWon++;
        addLog(`Block ${blockNumber} won! (job: ${jobId})`, 'success');
        updateStats();
      },
      onJobCompleted: (jobId, youWon) => {
        stats.jobsCompleted++;
        if (!youWon) {
          cancelCurrentJob();
          addLog(`Job ${jobId} completed by another miner`, 'warning');
        }
        updateStats();
      },
      onError: (message) => {
        addLog(`Pool error: ${message}`, 'error');
      },
    });

    await poolClient.connect(address);
    updateStatus('Connected and mining');
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    addLog(`Connection failed: ${message}`, 'error');
    updateStatus('Connection failed');
    updateState('disconnected');
  }
}

function disconnect(): void {
  cancelCurrentJob();
  stopHashRateTracking();

  if (poolClient) {
    poolClient.disconnect();
    poolClient = null;
  }

  updateStatus('Disconnected');
  updateState('disconnected');
  addLog('Disconnected from pool', 'info');

  // Reset stats
  stats = {
    hashRate: 0,
    totalHashes: 0,
    blocksWon: 0,
    jobsCompleted: 0,
    uptime: 0,
  };
  startTime = null;
  updateStats();
}

// ============================================================================
// Initialization
// ============================================================================

function init(): void {
  // Check WebGPU support
  if (!GpuMiner.isSupported()) {
    updateStatus('WebGPU not supported');
    addLog('WebGPU is not supported in this browser', 'error');
    addLog('Try Chrome/Edge 113+ or enable WebGPU in browser flags', 'info');

    // Disable connect button
    const connectBtn = $('connect-btn') as HTMLButtonElement | null;
    if (connectBtn) connectBtn.disabled = true;
    return;
  }

  addLog('WebGPU is supported', 'success');
  updateStatus('Ready to connect');
  updateState('disconnected');
  updateStats();

  // Set up event listeners
  const connectBtn = $('connect-btn');
  const disconnectBtn = $('disconnect-btn');

  if (connectBtn) {
    connectBtn.addEventListener('click', () => {
      connect().catch((err) => {
        console.error('Connect error:', err);
      });
    });
  }

  if (disconnectBtn) {
    disconnectBtn.addEventListener('click', disconnect);
  }

  // Allow Enter key to connect
  const addressInput = $('address');
  if (addressInput) {
    addressInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        connect().catch((err) => {
          console.error('Connect error:', err);
        });
      }
    });
  }
}

// Start when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

/**
 * WebSocket client for connecting to the Quantus mining pool.
 */

import type {
  MinerToPool,
  PoolToMiner,
  MinerState,
  NewJobMessage,
} from './types';

export type PoolEventHandler = {
  onStateChange?: (state: MinerState) => void;
  onNewJob?: (job: NewJobMessage) => void;
  onBlockWon?: (blockNumber: number, jobId: string) => void;
  onJobCompleted?: (jobId: string, youWon: boolean) => void;
  onError?: (message: string) => void;
  onRegistered?: (minerId: number) => void;
};

export class PoolClient {
  private ws: WebSocket | null = null;
  private state: MinerState = 'disconnected';
  private handlers: PoolEventHandler = {};
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private minerId: number | null = null;
  private pendingAddress: string | null = null;

  constructor(private poolUrl: string) {}

  /**
   * Set event handlers for pool events.
   */
  setHandlers(handlers: PoolEventHandler): void {
    this.handlers = handlers;
  }

  /**
   * Get the current connection state.
   */
  getState(): MinerState {
    return this.state;
  }

  /**
   * Get the assigned miner ID (null if not registered).
   */
  getMinerId(): number | null {
    return this.minerId;
  }

  /**
   * Connect to the pool and register with the given address.
   */
  async connect(address: string): Promise<void> {
    if (this.state !== 'disconnected') {
      throw new Error(`Cannot connect: already ${this.state}`);
    }

    this.pendingAddress = address;
    this.setState('connecting');

    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.poolUrl);

        this.ws.onopen = () => {
          this.reconnectAttempts = 0;
          this.setState('registering');
          this.send({ type: 'register', address });
          resolve();
        };

        this.ws.onclose = (event) => {
          console.log(`WebSocket closed: code=${event.code}, reason=${event.reason}`);
          this.handleDisconnect();
        };

        this.ws.onerror = (event) => {
          console.error('WebSocket error:', event);
          if (this.state === 'connecting') {
            reject(new Error('Failed to connect to pool'));
          }
        };

        this.ws.onmessage = (event) => {
          this.handleMessage(event.data);
        };
      } catch (err) {
        this.setState('disconnected');
        reject(err);
      }
    });
  }

  /**
   * Disconnect from the pool.
   */
  disconnect(): void {
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    this.minerId = null;
    this.setState('disconnected');
  }

  /**
   * Signal readiness to receive jobs.
   */
  sendReady(): void {
    if (this.state !== 'ready' && this.state !== 'mining') {
      // Allow sending ready after registration
      if (this.minerId === null) {
        throw new Error('Cannot send ready: not registered');
      }
    }
    this.send({ type: 'ready' });
  }

  /**
   * Submit a job result to the pool.
   */
  submitResult(
    jobId: string,
    nonce: string,
    work: string,
    hashCount: number,
    elapsedTime: number
  ): void {
    this.send({
      type: 'job_result',
      job_id: jobId,
      nonce,
      work,
      hash_count: hashCount,
      elapsed_time: elapsedTime,
    });
  }

  private setState(state: MinerState): void {
    if (this.state !== state) {
      this.state = state;
      this.handlers.onStateChange?.(state);
    }
  }

  private send(msg: MinerToPool): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.error('Cannot send: WebSocket not connected');
      return;
    }
    this.ws.send(JSON.stringify(msg));
  }

  private handleMessage(data: string): void {
    let msg: PoolToMiner;
    try {
      msg = JSON.parse(data);
    } catch (err) {
      console.error('Failed to parse pool message:', err);
      return;
    }

    switch (msg.type) {
      case 'registered':
        this.minerId = msg.miner_id;
        this.setState('ready');
        this.handlers.onRegistered?.(msg.miner_id);
        // Automatically signal ready after registration
        this.sendReady();
        break;

      case 'new_job':
        this.setState('mining');
        this.handlers.onNewJob?.(msg);
        break;

      case 'block_won':
        this.handlers.onBlockWon?.(msg.block_number, msg.job_id);
        break;

      case 'job_completed':
        this.handlers.onJobCompleted?.(msg.job_id, msg.you_won);
        // Go back to ready state, wait for next job
        this.setState('ready');
        break;

      case 'error':
        console.error('Pool error:', msg.message);
        this.handlers.onError?.(msg.message);
        break;

      default:
        console.warn('Unknown message type:', (msg as PoolToMiner));
    }
  }

  private handleDisconnect(): void {
    this.ws = null;
    const wasConnected = this.state !== 'disconnected' && this.state !== 'connecting';
    this.setState('disconnected');

    // Attempt reconnection if we were previously connected
    if (wasConnected && this.pendingAddress && this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
      console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      
      setTimeout(() => {
        if (this.pendingAddress) {
          this.connect(this.pendingAddress).catch((err) => {
            console.error('Reconnection failed:', err);
          });
        }
      }, delay);
    }
  }
}

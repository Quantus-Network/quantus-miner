/**
 * Protocol types for Quantus mining pool communication.
 * These types match the Rust pool-api crate definitions.
 */

// ============================================================================
// Miner -> Pool Messages
// ============================================================================

export interface RegisterMessage {
  type: 'register';
  address: string;
}

export interface ReadyMessage {
  type: 'ready';
}

export interface JobResultMessage {
  type: 'job_result';
  job_id: string;
  /** Nonce as hex string (U512, no 0x prefix) */
  nonce: string;
  /** Work bytes as hex string (64 bytes = 128 chars, no 0x prefix) */
  work: string;
  /** Number of hashes computed */
  hash_count: number;
  /** Time spent mining this job (seconds) */
  elapsed_time: number;
}

export type MinerToPool = RegisterMessage | ReadyMessage | JobResultMessage;

// ============================================================================
// Pool -> Miner Messages
// ============================================================================

export interface RegisteredMessage {
  type: 'registered';
  miner_id: number;
}

export interface NewJobMessage {
  type: 'new_job';
  job_id: string;
  /** Header hash to mine (hex-encoded, 64 chars, no 0x prefix) */
  mining_hash: string;
  /** Difficulty threshold (U512 as decimal string) */
  difficulty: string;
}

export interface BlockWonMessage {
  type: 'block_won';
  block_number: number;
  job_id: string;
}

export interface JobCompletedMessage {
  type: 'job_completed';
  job_id: string;
  you_won: boolean;
}

export interface ErrorMessage {
  type: 'error';
  message: string;
}

export type PoolToMiner =
  | RegisteredMessage
  | NewJobMessage
  | BlockWonMessage
  | JobCompletedMessage
  | ErrorMessage;

// ============================================================================
// Mining Job State
// ============================================================================

export interface MiningJob {
  jobId: string;
  /** 32-byte header as Uint32Array (8 elements) */
  header: Uint32Array;
  /** U512 difficulty target as Uint32Array (16 elements, little-endian) */
  difficultyTarget: Uint32Array;
  /** Current nonce position (U512 as Uint32Array, 16 elements) */
  startNonce: Uint32Array;
  /** Total hashes computed for this job */
  hashCount: number;
  /** Job start timestamp */
  startTime: number;
}

// ============================================================================
// Miner State
// ============================================================================

export type MinerState =
  | 'disconnected'
  | 'connecting'
  | 'registering'
  | 'ready'
  | 'mining';

export interface MinerStats {
  hashRate: number;
  totalHashes: number;
  blocksWon: number;
  jobsCompleted: number;
  uptime: number;
}

// ============================================================================
// GPU Mining Result
// ============================================================================

export interface MiningResult {
  found: boolean;
  /** Winning nonce (if found) as Uint32Array (16 elements, U512 little-endian) */
  nonce?: Uint32Array;
  /** Hash result (if found) as Uint32Array (16 elements, U512 little-endian) */
  hash?: Uint32Array;
  /** Number of hashes computed in this dispatch */
  hashCount: number;
}

#!/usr/bin/env python3
"""Compare unchanged WGSL, unchanged CUDA, and candidate CUDA on one GPU host."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def gpu_info():
    fields = 'name,uuid,driver_version,memory.total,power.limit,power.draw,clocks.sm,clocks.mem,temperature.gpu'
    result = subprocess.run(
        ['nvidia-smi', f'--query-gpu={fields}', '--format=csv,noheader'],
        check=True, capture_output=True, text=True, timeout=15,
    )
    rows = result.stdout.strip().splitlines()
    if len(rows) != 1:
        raise RuntimeError(f'This comparison requires exactly one visible GPU: {rows}')
    return dict(zip(fields.split(','), (x.strip() for x in rows[0].split(',')), strict=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=int, default=15)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--batches', type=int, nargs='+', default=[1_000_000, 4_000_000])
    args = parser.parse_args()
    if args.duration < 1 or args.warmup < 1 or any(b < 1 or b > 0xffffffff for b in args.batches):
        parser.error('Durations and batch sizes must be positive; batches must fit u32')
    baseline, candidate = args.baseline.resolve(), args.candidate.resolve()
    digests = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [baseline, candidate]}
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, RUST_LOG='warn')
    engines = {
        'wgsl': (baseline, []),
        'cuda_base': (baseline, ['--cuda-gpu']),
        'cuda_candidate': (candidate, ['--cuda-gpu']),
    }
    expected_uuid = gpu_info()['uuid']

    def run(name, batch, duration, tag):
        binary, flags = engines[name]
        command = [str(binary), 'benchmark', *flags, '--gpu-devices', '1', '--cpu-workers', '0',
                   '--duration', str(duration), '--gpu-batch-size', str(batch)]
        before = gpu_info()
        result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=duration + 120)
        log_path = args.output / f'{tag}-{name}-{batch}.log'
        log_path.write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {log_path}\n{result.stdout[-4000:]}')
        match = re.search(r'Average rate: ([0-9.]+)([kMGT]?) H/s', result.stdout)
        if not match:
            raise RuntimeError(f'Missing benchmark rate; see {log_path}')
        after = gpu_info()
        if before['uuid'] != expected_uuid or after['uuid'] != expected_uuid:
            raise RuntimeError('GPU changed during the comparison')
        scale = {'': 1, 'k': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12}[match.group(2)]
        record = {
            'tag': tag, 'engine': name, 'batch': batch, 'duration': duration,
            'mh_s': float(match.group(1)) * scale / 1e6,
            'gpu_before': before, 'gpu_after': after,
            'binary_sha256': digests[str(binary)], 'command': command,
            'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        }
        with (args.output / 'comparison.jsonl').open('a') as log:
            log.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)

    orders = [
        ['wgsl', 'cuda_base', 'cuda_candidate'],
        ['cuda_candidate', 'cuda_base', 'wgsl'],
        ['cuda_base', 'wgsl', 'cuda_candidate'],
    ]
    for batch in args.batches:
        for name in engines:
            run(name, batch, args.warmup, 'warmup')
        for i, order in enumerate(orders, 1):
            for name in order:
                run(name, batch, args.duration, f'round{i}')


if __name__ == '__main__':
    main()

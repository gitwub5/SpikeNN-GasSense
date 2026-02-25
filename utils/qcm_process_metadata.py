import os
import re
import glob
import pandas as pd

# =============================================================================
# Configuration (Run from project root)
# =============================================================================
RAW_DATA_DIR = 'data/qcm/csv'
OUTPUT_DIR = 'data/qcm'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'qcm_metadata.csv')

# =============================================================================
# Token Dictionaries (case-insensitive matching via precomputed sets)
# =============================================================================
SENSORS = {
    'TEP', 'TEPDDM',
    'POSS',
    'POM', 'POF', 'POFB', 'POA',
    'PPD',
    'PECH',
    'EMIM',
    'PONN',
    'DEP', 'DER',
    'PSS',
    'TEAX', 'TEAO',
    'TRITONX',
}

# 분류 라벨로 확실히 쓰려는 것만
TARGETS = {'cees', 'DMMP', 'DIMP'}

SENSORS_UPPER = {s.upper() for s in SENSORS}
TARGET_LOWER  = {t.lower() for t in TARGETS}

# =============================================================================
# Regex Patterns
# =============================================================================
DATE_REGEX = re.compile(r'^\d{6}$')          # 220602
PPM_REGEX  = re.compile(r'^(\d+(?:\.\d+)?)\s*ppm$', re.IGNORECASE)  # 2.5ppm, 60ppm
NUM_REGEX  = re.compile(r'^\d+(?:\.\d+)?$')  # 25, 75, 340, 20, ...

def _to_float(s: str):
    try:
        return float(s)
    except Exception:
        return None

def parse_filename(filename: str) -> dict:
    name_body = os.path.splitext(filename)[0]
    tokens = re.split(r'[-_\(\)\s]+', name_body)
    tokens = [t.strip() for t in tokens if t.strip()]

    meta = {
        'file_name': filename,
        'experiment_date': None,
        'sensor_type': None,
        'target_gas': None,

        # NEW (may be None)
        'target_concentration_ppm': None,
        'target_flow_sccm': None,
        'n2_flow_sccm': None,
    }

    # 1) Date: first token if 6 digits
    if tokens and DATE_REGEX.match(tokens[0]):
        meta['experiment_date'] = tokens[0]

    # 2) sensor_type: longest-match 우선 + 표준형(대문자) 저장
    sensors_sorted = sorted(SENSORS_UPPER, key=len, reverse=True)
    for token in tokens:
        tu = token.upper()
        if tu in sensors_sorted:
            meta['sensor_type'] = tu
            break

    # 3) target_gas: 표준 라벨로 저장
    target_map = {'cees': 'CEES', 'dmmp': 'DMMP', 'dimp': 'DIMP'}
    for token in tokens:
        tl = token.lower()
        if tl in TARGET_LOWER:
            meta['target_gas'] = target_map.get(tl, token.upper())
            break

    # 4) target_concentration_ppm: find token like "2.5ppm" or "60ppm"
    ppm_idx = None
    for i, token in enumerate(tokens):
        m = PPM_REGEX.match(token)
        if m:
            meta['target_concentration_ppm'] = _to_float(m.group(1))
            ppm_idx = i
            break

    # 5) flows: try to parse two numeric tokens AFTER ppm token
    #    - For CEES examples: ...-2.5ppm-25-75 => target=25, n2=75 (sum=100)
    #    - For DMMP examples: ...-60ppm-340-20 => n2=340, target=20 (sum=360)
    if ppm_idx is not None:
        numeric_after_ppm = []
        for t in tokens[ppm_idx + 1:]:
            if NUM_REGEX.match(t):
                numeric_after_ppm.append(_to_float(t))
            # 숫자 2개만 확보되면 종료(뒤에 run index 같은 게 더 있어도 일단 2개만 사용)
            if len(numeric_after_ppm) >= 2:
                break

        if len(numeric_after_ppm) >= 2:
            a, b = numeric_after_ppm[0], numeric_after_ppm[1]
            if a is not None and b is not None:
                s = a + b

                # 확신 가능한 합(총유량)일 때만 매핑
                # (float 오차 대비)
                if abs(s - 100.0) < 1e-6:
                    meta['target_flow_sccm'] = a
                    meta['n2_flow_sccm'] = b
                elif abs(s - 360.0) < 1e-6:
                    meta['n2_flow_sccm'] = a
                    meta['target_flow_sccm'] = b
                else:
                    # 합이 100/360이 아니면 확신 어려우니 비움
                    pass

    return meta

def read_csv_header_fields(filepath: str) -> dict:
    info = {
        'sample_interval_sec': None,
        'qcm_init_freq_hz': None,
        'run_time_sec': None,
    }

    # 1) Try: columns-based (fast path)
    try:
        df = pd.read_csv(filepath, nrows=5)
        cols_lower = {c.lower(): c for c in df.columns}

        for key, candidates in [
            ('sample_interval_sec', ['sample_interval_sec', 'sample interval (s)', 'sample interval', 'sample_interval']),
            ('qcm_init_freq_hz', ['qcm_init_freq_hz', 'qcm init freq (hz)', 'qcm init freq', 'init freq', 'init_freq_hz']),
            ('run_time_sec', ['run_time_sec', 'run time (sec)', 'run time', 'runtime_sec', 'runtime']),
        ]:
            for cand in candidates:
                if cand in cols_lower:
                    col = cols_lower[cand]
                    val = df[col].iloc[0]
                    try:
                        info[key] = float(val)
                    except Exception:
                        pass
                    break

        if any(v is not None for v in info.values()):
            return info
    except Exception:
        pass

    # 2) Fallback: text scan first ~60 lines
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in range(60):
                line = f.readline()
                if not line:
                    break
                line = line.strip()

                if line.lower().startswith('time') or 'time/sec' in line.lower():
                    break

                if 'Sample Interval' in line:
                    parts = line.split('=')
                    if len(parts) > 1:
                        try:
                            info['sample_interval_sec'] = float(parts[1].strip())
                        except Exception:
                            pass

                if 'QCM Init Freq' in line:
                    parts = line.split('=')
                    if len(parts) > 1:
                        try:
                            info['qcm_init_freq_hz'] = float(parts[1].strip())
                        except Exception:
                            pass

                if 'Run Time' in line:
                    parts = line.split('=')
                    if len(parts) > 1:
                        val = parts[1].strip()
                        try:
                            info['run_time_sec'] = float(val)
                        except Exception:
                            pass
    except Exception as e:
        print(f"[WARN] Error reading header fields of {filepath}: {e}")

    return info

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_files = glob.glob(os.path.join(RAW_DATA_DIR, '*.csv'))
    print(f"Found {len(all_files)} files in {RAW_DATA_DIR}")

    rows = []
    for filepath in all_files:
        filename = os.path.basename(filepath)

        meta = parse_filename(filename)
        header_info = read_csv_header_fields(filepath)

        row = {
            'file_name': meta['file_name'],
            'experiment_date': meta['experiment_date'],
            'sensor_type': meta['sensor_type'],
            'target_gas': meta['target_gas'],

            # NEW
            'target_concentration_ppm': meta['target_concentration_ppm'],
            'target_flow_sccm': meta['target_flow_sccm'],
            'n2_flow_sccm': meta['n2_flow_sccm'],

            'sample_interval_sec': header_info['sample_interval_sec'],
            'qcm_init_freq_hz': header_info['qcm_init_freq_hz'],
            'run_time_sec': header_info['run_time_sec'],
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    cols = [
        'file_name',
        'experiment_date',
        'sensor_type',
        'target_gas',

        'target_concentration_ppm',
        'target_flow_sccm',
        'n2_flow_sccm',

        'sample_interval_sec',
        'qcm_init_freq_hz',
        'run_time_sec',
    ]
    df = df[cols]

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved metadata CSV to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
import os
import pandas as pd
import glob
import io
import re

# Configuration
RAW_DATA_DIR = '/Users/gwshin/Dev/Nanolatis/qcm/raw_data'
OUTPUT_DIR = '/Users/gwshin/Dev/Nanolatis/qcm/data/qcm/csv'

def parse_header_metadata(lines):
    metadata = {
        'Instrument Model': None,
        'Sample Interval (s)': None,
        'Run Time (sec)': None,
        'QCM Init Freq (Hz)': None
    }
    
    # Regex patterns
    # "Instrument Model:  CHI400C"
    inst_model_pat = re.compile(r'Instrument Model:\s*(.*)', re.I)
    # "Sample Interval (s) = 0.1"
    sample_int_pat = re.compile(r'Sample Interval\s*\(s\)\s*=\s*([\d\.]+)', re.I)
    # "Run Time (sec) = 1e+5" or "1000"
    run_time_pat = re.compile(r'Run Time\s*\(sec\)\s*=\s*([\d\.e\+\-]+)', re.I)
    # "QCM Init Freq (Hz) = 7996429"
    init_freq_pat = re.compile(r'QCM Init Freq\s*\(Hz\)\s*=\s*([\d\.]+)', re.I)
    
    for line in lines:
        line = line.strip()
        
        m = inst_model_pat.search(line)
        if m:
            metadata['Instrument Model'] = m.group(1).strip()
            
        m = sample_int_pat.search(line)
        if m:
            metadata['Sample Interval (s)'] = float(m.group(1))
            
        m = run_time_pat.search(line)
        if m:
            metadata['Run Time (sec)'] = float(m.group(1))
            
        m = init_freq_pat.search(line)
        if m:
            metadata['QCM Init Freq (Hz)'] = float(m.group(1))
            
    return metadata

def convert_file(filepath):
    filename = os.path.basename(filepath)
    output_filename = os.path.splitext(filename)[0] + '.csv'
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    try:
        with open(filepath, 'r', encoding='latin-1') as f:
            lines = f.readlines()
            
        # Parse metadata
        metadata = parse_header_metadata(lines)
            
        # Find the header line index where data starts
        header_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('Time/sec'):
                header_idx = i
                break
        
        if header_idx != -1:
            # Extract data content
            data_content = "".join(lines[header_idx:])
            
            # Parse data into DataFrame
            df = pd.read_csv(io.StringIO(data_content), sep=',', skipinitialspace=True)
            
            # Clean up column names
            df.columns = [c.strip() for c in df.columns]
            
            # Add metadata columns
            for key, value in metadata.items():
                df[key] = value
            
            # Save to CSV
            df.to_csv(output_path, index=False)
            print(f"Converted {filename} -> {output_filename}")
            return True
        else:
            print(f"Skipping {filename}: Data header not found.")
            return False
            
    except Exception as e:
        print(f"Error converting {filename}: {e}")
        return False

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    all_files = glob.glob(os.path.join(RAW_DATA_DIR, '*.txt'))
    print(f"Found {len(all_files)} files to convert.")
    
    count = 0
    for filepath in all_files:
        if convert_file(filepath):
            count += 1
            
    print(f"Completed conversion. {count}/{len(all_files)} files processed.")

if __name__ == "__main__":
    main()

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = '/Users/gwshin/Dev/Nanolatis/qcm/data/qcm/csv'
PLOTS_DIR = '/Users/gwshin/Dev/Nanolatis/qcm/data/qcm/plots'

# csv 파일 그래프로 시각화
def visualize_file(filepath):
    filename = os.path.basename(filepath)
    
    # Skip metadata file
    if filename == 'qcm_metadata.csv':
        return False
        
    try:
        df = pd.read_csv(filepath)
        
        # Check if required columns exist
        required_cols = ['Time/sec', 'dF/Hz', 'dR/ohm']
        if not all(col in df.columns for col in required_cols):
            print(f"Skipping {filename}: Missing columns. Found: {df.columns.tolist()}")
            return False
            
        # Create plot
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot dF (Frequency) on left axis
        color = 'tab:blue'
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Frequency Change (dF/Hz)', color=color)
        ax1.plot(df['Time/sec'], df['dF/Hz'], color=color, label='dF/Hz')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Create second y-axis for dR (Resistance)
        ax2 = ax1.twinx()  
        color = 'tab:orange'
        ax2.set_ylabel('Resistance Change (dR/ohm)', color=color)
        ax2.plot(df['Time/sec'], df['dR/ohm'], color=color, label='dR/ohm')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Title and layout
        plt.title(f'QCM Data: {os.path.splitext(filename)[0]}')
        fig.tight_layout()  
        
        # Save
        output_filename = os.path.splitext(filename)[0] + '.png'
        output_path = os.path.join(PLOTS_DIR, output_filename)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        
        print(f"Generated plot: {output_filename}")
        return True
        
    except Exception as e:
        print(f"Error visualizing {filename}: {e}")
        return False

def main():
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
        
    csv_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    print(f"Found {len(csv_files)} CSV files.")
    
    count = 0
    for filepath in csv_files:
        if visualize_file(filepath):
            count += 1
            
    print(f"Visualization complete. Generated {count} plots.")

if __name__ == "__main__":
    main()

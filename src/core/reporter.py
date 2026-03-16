import os
import datetime
import shutil
import glob
import statistics
import matplotlib.pyplot as plt

def generate_single_chart(raw_data: dict, stats: dict, mode: str, output_path: str):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    fig.suptitle(f'System Metric Overview: Spectre Terminal ({mode})', fontsize=16, fontweight='bold')
    
    ax1.plot(raw_data['time'], raw_data['cpu'], label='CPU Usage (%)', color='tab:blue', linewidth=2)
    ax1.fill_between(raw_data['time'], raw_data['cpu'], color='tab:blue', alpha=0.15)
    ax1.set_ylabel('CPU (%)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    ax2.plot(raw_data['time'], raw_data['gpu'], label='GPU Load (%)', color='tab:green', linewidth=2)
    ax2.fill_between(raw_data['time'], raw_data['gpu'], color='tab:green', alpha=0.15)
    ax2.set_ylabel('GPU (%)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    ax3.plot(raw_data['time'], raw_data['ram'], label='RAM Committed (MB)', color='tab:red', linewidth=2)
    ax3.fill_between(raw_data['time'], raw_data['ram'], color='tab:red', alpha=0.15)
    ax3.set_xlabel('Время (секунды)', fontweight='bold')
    ax3.set_ylabel('RAM (MB)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')
    
    summary_text = (
        f"RESOURCE USAGE RANGES:\n"
        f"CPU usage: from {stats['cpu']['min']}% to {stats['cpu']['max']}%\n"
        f"RAM usage: from {stats['ram']['min']} MB to {stats['ram']['max']} MB\n"
        f"GPU usage: from {stats['gpu']['min']}% to {stats['gpu']['max']}%"
    )
    
    fig.text(0.5, 0.03, summary_text, ha='center', va='center', fontsize=12, fontweight='bold',
             bbox=dict(facecolor='#f0f0f0', edgecolor='black', boxstyle='round,pad=0.5'))

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(output_path)
    plt.close()
    return output_path

def generate_comparison_chart(dev_raw: dict, feature_raw: dict, diff_stats: dict, dev_stats: dict, feature_stats: dict, output_path: str):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    fig.suptitle('СРАВНЕНИЕ: DEV (пунктир) vs FEATURE (сплошная)', fontsize=16, fontweight='bold')
    
    ax1.plot(dev_raw['time'], dev_raw['cpu'], label='[DEV] CPU (%)', color='tab:blue', linestyle='--', alpha=0.6, linewidth=2)
    ax1.plot(feature_raw['time'], feature_raw['cpu'], label='[FEATURE] CPU (%)', color='tab:blue', linewidth=2.5)
    ax1.set_ylabel('CPU (%)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    ax2.plot(dev_raw['time'], dev_raw['gpu'], label='[DEV] GPU (%)', color='tab:green', linestyle='--', alpha=0.6, linewidth=2)
    ax2.plot(feature_raw['time'], feature_raw['gpu'], label='[FEATURE] GPU (%)', color='tab:green', linewidth=2.5)
    ax2.set_ylabel('GPU (%)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    ax3.plot(dev_raw['time'], dev_raw['ram'], label='[DEV] RAM (MB)', color='tab:red', linestyle='--', alpha=0.6, linewidth=2)
    ax3.plot(feature_raw['time'], feature_raw['ram'], label='[FEATURE] RAM (MB)', color='tab:red', linewidth=2.5)
    ax3.set_xlabel('Время (секунды)', fontweight='bold')
    ax3.set_ylabel('RAM (MB)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')
    
    def get_conclusion(name, dev_stat, feature_stat, unit):
        dev_range = f"from {dev_stat['min']}{unit} to {dev_stat['max']}{unit}"
        feat_range = f"from {feature_stat['min']}{unit} to {feature_stat['max']}{unit}"
        return f"{name} usage: DEV ({dev_range}) vs FEATURE ({feat_range})"

    cpu_text = get_conclusion("CPU", dev_stats['cpu'], feature_stats['cpu'], "%")
    ram_text = get_conclusion("RAM", dev_stats['ram'], feature_stats['ram'], " MB")
    gpu_text = get_conclusion("GPU", dev_stats['gpu'], feature_stats['gpu'], "%")
    
    summary_text = f"RESOURCE USAGE RANGES:\n{cpu_text}\n{ram_text}\n{gpu_text}"
    
    bg_color = '#ffe6e6' if (diff_stats['cpu_diff'] > 0 or diff_stats['ram_diff'] > 0 or diff_stats['gpu_diff'] > 0) else '#e6ffe6'
    
    fig.text(0.5, 0.04, summary_text, ha='center', va='center', fontsize=12, fontweight='bold',
             bbox=dict(facecolor=bg_color, edgecolor='black', boxstyle='round,pad=0.7'))

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(output_path)
    plt.close()
    return output_path

def safe_calc_stats(samples):
    valid = [s for s in samples if s > 0]
    if not valid:
        valid = [0.0]
    return {
        "avg": round(sum(valid) / len(valid), 2),
        "max": round(max(valid), 2),
        "min": round(min(valid), 2),
        "median": round(statistics.median(valid), 2)
    }

def archive_run(latest_dir: str, history_dir: str, history_file_path: str):
    today = datetime.datetime.now()
    day = today.strftime("%d").lstrip('0') or "0"
    month = today.strftime("%B")
    year = today.strftime("%Y")
    base_prefix = f"{day} {month} {year}"
    
    existing_dirs = [d for d in os.listdir(history_dir) if os.path.isdir(os.path.join(history_dir, d)) and d.startswith(base_prefix)]
    run_number = len(existing_dirs) + 1
    
    ordinals = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 
                6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth"}
    run_str = ordinals.get(run_number, f"{run_number}th")
    
    run_folder_name = f"{base_prefix} - {run_str} run"
    run_folder_path = os.path.join(history_dir, run_folder_name)
    os.makedirs(run_folder_path, exist_ok=True)
    
    for png_file in glob.glob(os.path.join(latest_dir, "*.png")):
        shutil.copy2(png_file, run_folder_path)
    shutil.copy2(history_file_path, run_folder_path)

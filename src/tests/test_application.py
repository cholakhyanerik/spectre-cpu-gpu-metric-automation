import os
import sys
import time
import json
import subprocess
import psutil
import pytest
import GPUtil
import matplotlib.pyplot as plt
import statistics
import datetime
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# КОНФИГУРАЦИЯ И ПУТИ
# ==========================================
DEV_BUILD_PATH = os.getenv("DEV_BUILD_PATH")
FUTURE_BUILD_PATH = os.getenv("FUTURE_BUILD_PATH")

# Настройка директорий для отчетов
REPORTS_DIR = "reports"
LATEST_DIR = os.path.join(REPORTS_DIR, "latest")
HISTORY_DIR = os.path.join(REPORTS_DIR, "history")

def init_directories():
    os.makedirs(LATEST_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)

# Лимиты деградации
CPU_TOLERANCE_PERCENT = float(os.getenv("CPU_TOLERANCE_PERCENT", "0.1"))
RAM_TOLERANCE_MB = float(os.getenv("RAM_TOLERANCE_MB", "1.0"))
GPU_TOLERANCE_PERCENT = float(os.getenv("GPU_TOLERANCE_PERCENT", "0.1"))

# ==========================================
# ФУНКЦИИ МОНИТОРИНГА
# ==========================================
def get_process_tree_metrics(parent_process: psutil.Process, process_cache: dict):
    total_cpu = 0.0
    total_ram = 0.0
    try:
        children = parent_process.children(recursive=True)
        current_pids = {p.pid for p in children}
        current_pids.add(parent_process.pid)
        
        # Initialize new processes in cache
        for p in [parent_process] + children:
            if p.pid not in process_cache:
                process_cache[p.pid] = p
                try:
                    p.cpu_percent(interval=None) # Initialize CPU counter
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                    
        # Clean up dead processes from cache
        keys_to_remove = [pid for pid in process_cache if pid not in current_pids]
        for pid in keys_to_remove:
            del process_cache[pid]
            
        # Collect metrics using cached objects
        for p in process_cache.values():
            try:
                cpu = p.cpu_percent(interval=None) / psutil.cpu_count()
                ram = p.memory_info().rss / (1024 * 1024)
                total_cpu += cpu
                total_ram += ram
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return total_cpu, total_ram

LAST_GPU_CHECK_TIME = 0.0
CACHED_GPU_VAL = 0.0

def get_gpu_metric():
    global LAST_GPU_CHECK_TIME, CACHED_GPU_VAL
    try:
        gpus = GPUtil.getGPUs()
        if gpus and gpus[0].load > 0:
            return gpus[0].load * 100 
    except Exception:
        pass

    # Use WMI/Powershell fallback but throttle to every 3 seconds
    current_time = time.time()
    if current_time - LAST_GPU_CHECK_TIME < 3.0:
        return CACHED_GPU_VAL
        
    LAST_GPU_CHECK_TIME = current_time

    if sys.platform != "win32":
        return CACHED_GPU_VAL

    try:
        cmd = r'(Get-Counter "\GPU Engine(*engtype_3D)\Utilization Percentage" -ErrorAction SilentlyContinue).CounterSamples | Measure-Object -Property CookedValue -Maximum | Select-Object -ExpandProperty Maximum'
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd], 
            capture_output=True, 
            text=True, 
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        stdout = result.stdout.strip()
        if stdout:
            # handle potential commas in float formatting from powershell
            val = float(stdout.replace(',', '.'))
            CACHED_GPU_VAL = val if val <= 100.0 else 100.0
            return CACHED_GPU_VAL
    except Exception:
        pass
        
    return CACHED_GPU_VAL

# ==========================================
# ФУНКЦИИ ГЕНЕРАЦИИ ГРАФИКОВ (С ТЕКСТОМ)
# ==========================================
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
    
    # Добавляем текстовый блок со средними значениями вниз картинки
    summary_text = (
        f"СРЕДНИЕ ЗНАЧЕНИЯ ЗА СЕССИЮ:\n"
        f"CPU: {stats['cpu']['avg']}%  |  "
        f"RAM: {stats['ram']['avg']} MB  |  "
        f"GPU: {stats['gpu']['avg']}%"
    )
    
    fig.text(0.5, 0.03, summary_text, ha='center', va='center', fontsize=12, fontweight='bold',
             bbox=dict(facecolor='#f0f0f0', edgecolor='black', boxstyle='round,pad=0.5'))

    plt.tight_layout(rect=[0, 0.08, 1, 1]) # Оставляем 8% снизу для текста
    plt.savefig(output_path)
    plt.close()
    return output_path

def generate_comparison_chart(dev_raw: dict, feature_raw: dict, diff_stats: dict, output_path: str):
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
    
    # --- ЛОГИКА ТЕКСТОВОГО ЗАКЛЮЧЕНИЯ ---
    def get_conclusion(name, diff, unit):
        if diff > 0:
            return f"[!] {name}: УХУДШЕНИЕ на {abs(diff)}{unit} (DEV работает лучше)"
        elif diff < 0:
            return f"[+] {name}: УЛУЧШЕНИЕ на {abs(diff)}{unit} (FEATURE работает лучше)"
        else:
            return f"[=] {name}: БЕЗ ИЗМЕНЕНИЙ (0.0{unit})"

    cpu_text = get_conclusion("CPU", diff_stats['cpu_diff'], "%")
    ram_text = get_conclusion("RAM", diff_stats['ram_diff'], " MB")
    gpu_text = get_conclusion("GPU", diff_stats['gpu_diff'], "%")
    
    summary_text = f"ВЫВОДЫ ПО ИТОГАМ ТЕСТА:\n{cpu_text}\n{ram_text}\n{gpu_text}"
    
    # Цвет фона рамки в зависимости от наличия деградации
    bg_color = '#ffe6e6' if (diff_stats['cpu_diff'] > 0 or diff_stats['ram_diff'] > 0 or diff_stats['gpu_diff'] > 0) else '#e6ffe6'
    
    fig.text(0.5, 0.04, summary_text, ha='center', va='center', fontsize=12, fontweight='bold',
             bbox=dict(facecolor=bg_color, edgecolor='black', boxstyle='round,pad=0.7'))

    plt.tight_layout(rect=[0, 0.12, 1, 1]) # Оставляем 12% снизу для большого блока с выводами
    plt.savefig(output_path)
    plt.close()
    return output_path

# ==========================================
# ОСНОВНОЙ ТЕСТ
# ==========================================
def test_manual_qa_monitoring_parallel():
    init_directories()
    print("\n[АВТОМАТИЗАЦИЯ] Режим: PARALLEL (DEV & FUTURE)")
    print(f"[СТРОГИЕ ЛИМИТЫ] CPU: {CPU_TOLERANCE_PERCENT}%, RAM: {RAM_TOLERANCE_MB}MB, GPU: {GPU_TOLERANCE_PERCENT}%")
    
    if not DEV_BUILD_PATH or not os.path.exists(DEV_BUILD_PATH):
        pytest.skip(f"Билд DEV не найден: {DEV_BUILD_PATH}")
    if not FUTURE_BUILD_PATH or not os.path.exists(FUTURE_BUILD_PATH):
        pytest.skip(f"Билд FUTURE не найден: {FUTURE_BUILD_PATH}")

    # Launch both processes
    dev_process = subprocess.Popen([DEV_BUILD_PATH])
    future_process = subprocess.Popen([FUTURE_BUILD_PATH])
    
    try:
        dev_ps = psutil.Process(dev_process.pid)
    except psutil.NoSuchProcess:
        future_process.terminate()
        pytest.fail("DEV приложение сразу закрылось.")
        
    try:
        future_ps = psutil.Process(future_process.pid)
    except psutil.NoSuchProcess:
        dev_process.terminate()
        pytest.fail("FUTURE приложение сразу закрылось.")

    print("\n" + "="*50)
    print("🟢 ОБА ПРИЛОЖЕНИЯ ЗАПУЩЕНЫ! (DEV и FUTURE)")
    print("👉 Тестируй мануально ОБА приложения. Закрой их оба для получения отчета.")
    print("="*50 + "\n")

    time_sec = []
    dev_cpu_samples, dev_ram_samples, dev_gpu_samples = [], [], []
    future_cpu_samples, future_ram_samples, future_gpu_samples = [], [], []
    
    dev_cache, future_cache = {}, {}
    
    get_process_tree_metrics(dev_ps, dev_cache)
    get_process_tree_metrics(future_ps, future_cache)
    time.sleep(1)

    seconds_passed = 0
    dev_running = True
    future_running = True
    
    try:
        while dev_running or future_running:
            # Check DEV state
            if dev_running:
                if dev_process.poll() is not None:
                    dev_running = False
                else:
                    try:
                        if not dev_ps.is_running():
                            dev_running = False
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        dev_running = False
            
            # Check FUTURE state
            if future_running:
                if future_process.poll() is not None:
                    future_running = False
                else:
                    try:
                        if not future_ps.is_running():
                            future_running = False
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        future_running = False

            if not dev_running and not future_running:
                break
                
            # Read metrics
            if dev_running:
                d_cpu, d_ram = get_process_tree_metrics(dev_ps, dev_cache)
                dev_cpu_samples.append(d_cpu)
                dev_ram_samples.append(d_ram)
            else:
                dev_cpu_samples.append(0.0)
                dev_ram_samples.append(0.0)
                
            if future_running:
                f_cpu, f_ram = get_process_tree_metrics(future_ps, future_cache)
                future_cpu_samples.append(f_cpu)
                future_ram_samples.append(f_ram)
            else:
                future_cpu_samples.append(0.0)
                future_ram_samples.append(0.0)
                
            gpu_load = get_gpu_metric()
            if dev_running:
                dev_gpu_samples.append(gpu_load)
            else:
                dev_gpu_samples.append(0.0)
            if future_running:
                future_gpu_samples.append(gpu_load)
            else:
                future_gpu_samples.append(0.0)

            time_sec.append(seconds_passed)
            seconds_passed += 1
            time.sleep(0.5)
            
    finally:
        for p in [dev_process, future_process]:
            if p.poll() is None:
                try:
                    p.terminate()
                    p.wait(timeout=3)
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass

    print("\n[АВТОМАТИЗАЦИЯ] Оба приложения закрыты. Генерация отчетов...")

    def safe_calc_stats(samples):
        valid = [s for s in samples if s > 0]
        if not valid:
            valid = [0.0]
        return {
            "avg": round(sum(valid) / len(valid), 2),
            "max": round(max(valid), 2),
            "median": round(statistics.median(valid), 2)
        }
        
    dev_metrics = {
        "duration_seconds": len(dev_cpu_samples),
        "cpu": safe_calc_stats(dev_cpu_samples),
        "ram": safe_calc_stats(dev_ram_samples),
        "gpu": safe_calc_stats(dev_gpu_samples),
        "raw_data": {"time": time_sec, "cpu": dev_cpu_samples, "ram": dev_ram_samples, "gpu": dev_gpu_samples}
    }
    
    future_metrics = {
        "duration_seconds": len(future_cpu_samples),
        "cpu": safe_calc_stats(future_cpu_samples),
        "ram": safe_calc_stats(future_ram_samples),
        "gpu": safe_calc_stats(future_gpu_samples),
        "raw_data": {"time": time_sec, "cpu": future_cpu_samples, "ram": future_ram_samples, "gpu": future_gpu_samples}
    }
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # DEV files
    dev_chart_path = os.path.join(LATEST_DIR, "dev_metrics_chart.png")
    generate_single_chart(dev_metrics["raw_data"], dev_metrics, "DEV", dev_chart_path)
    with open(os.path.join(LATEST_DIR, "dev_baseline_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(dev_metrics, f, indent=4)

    # FUTURE files
    feature_chart_path = os.path.join(LATEST_DIR, "feature_metrics_chart.png")
    generate_single_chart(future_metrics["raw_data"], future_metrics, "FEATURE", feature_chart_path)
    
    # Comparison
    cpu_diff = round(future_metrics["cpu"]["avg"] - dev_metrics["cpu"]["avg"], 2)
    ram_diff = round(future_metrics["ram"]["avg"] - dev_metrics["ram"]["avg"], 2)
    gpu_diff = round(future_metrics["gpu"]["avg"] - dev_metrics["gpu"]["avg"], 2)

    report = {
        "baseline_dev_stats": dev_metrics["cpu"], 
        "feature_test_stats": future_metrics["cpu"],
        "difference": {
            "cpu_diff": cpu_diff,
            "ram_diff": ram_diff,
            "gpu_diff": gpu_diff
        }
    }
    
    comparison_chart = os.path.join(LATEST_DIR, "comparison_chart.png")
    generate_comparison_chart(dev_metrics["raw_data"], future_metrics["raw_data"], report["difference"], comparison_chart)

    full_report = {"summary": report, "dev_data": dev_metrics, "feature_data": future_metrics}
    history_file = os.path.join(HISTORY_DIR, f"parallel_run_{timestamp}.json")
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=4)

    print("\n📊 ИТОГОВЫЙ ОТЧЕТ СО СРАВНЕНИЕМ:")
    print(json.dumps(report, indent=4))
    print(f"\n📁 Все актуальные графики сохранены в папке: {LATEST_DIR}")

    assert cpu_diff <= CPU_TOLERANCE_PERCENT, \
        f"[БЛОКЕР] Деградация ЦП! Feature использует на {cpu_diff}% больше CPU."
        
    assert ram_diff <= RAM_TOLERANCE_MB, \
        f"Утечка ОЗУ! Feature использует на {ram_diff} MB больше памяти."
        
    assert gpu_diff <= GPU_TOLERANCE_PERCENT, \
        f"[БЛОКЕР] Деградация ГП! Feature использует на {gpu_diff}% больше GPU."
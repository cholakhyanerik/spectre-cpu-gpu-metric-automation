import os
import time
import subprocess
import psutil
import pytest
import datetime
import json
import sys

# Добавляем корневую директорию проекта в sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.config import (
    DEV_BUILD_PATH, FUTURE_BUILD_PATH,
    LATEST_DIR, HISTORY_DIR,
    CPU_TOLERANCE_PERCENT, RAM_TOLERANCE_MB, GPU_TOLERANCE_PERCENT,
    init_directories
)
from src.core.monitor import ProcessMonitor, GPUMonitor
from src.core.reporter import (
    generate_single_chart, generate_comparison_chart, 
    safe_calc_stats, archive_run
)

def test_manual_qa_monitoring_parallel():
    init_directories()
    print("\n[АВТОМАТИЗАЦИЯ] Режим: PARALLEL (DEV & FUTURE)")
    print(f"[СТРОГИЕ ЛИМИТЫ] CPU: {CPU_TOLERANCE_PERCENT}%, RAM: {RAM_TOLERANCE_MB}MB, GPU: {GPU_TOLERANCE_PERCENT}%")
    
    if not DEV_BUILD_PATH or not os.path.exists(DEV_BUILD_PATH):
        pytest.skip(f"Билд DEV не найден: {DEV_BUILD_PATH}")
    if not FUTURE_BUILD_PATH or not os.path.exists(FUTURE_BUILD_PATH):
        pytest.skip(f"Билд FUTURE не найден: {FUTURE_BUILD_PATH}")

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
    
    dev_monitor = ProcessMonitor(dev_ps)
    future_monitor = ProcessMonitor(future_ps)
    gpu_monitor = GPUMonitor()
    
    dev_monitor.get_metrics()
    future_monitor.get_metrics()
    time.sleep(1)

    seconds_passed = 0
    dev_running = True
    future_running = True
    
    try:
        while dev_running or future_running:
            if dev_running:
                if dev_process.poll() is not None:
                    dev_running = False
                else:
                    try:
                        if not dev_ps.is_running():
                            dev_running = False
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        dev_running = False
            
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
                
            if dev_running:
                d_cpu, d_ram = dev_monitor.get_metrics()
                dev_cpu_samples.append(d_cpu)
                dev_ram_samples.append(d_ram)
            else:
                dev_cpu_samples.append(0.0)
                dev_ram_samples.append(0.0)
                
            if future_running:
                f_cpu, f_ram = future_monitor.get_metrics()
                future_cpu_samples.append(f_cpu)
                future_ram_samples.append(f_ram)
            else:
                future_cpu_samples.append(0.0)
                future_ram_samples.append(0.0)
                
            gpu_load = gpu_monitor.get_metric()
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

    dev_chart_path = os.path.join(LATEST_DIR, "dev_metrics_chart.png")
    generate_single_chart(dev_metrics["raw_data"], dev_metrics, "DEV", dev_chart_path)
    with open(os.path.join(LATEST_DIR, "dev_baseline_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(dev_metrics, f, indent=4)

    feature_chart_path = os.path.join(LATEST_DIR, "feature_metrics_chart.png")
    generate_single_chart(future_metrics["raw_data"], future_metrics, "FEATURE", feature_chart_path)
    
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
    generate_comparison_chart(dev_metrics["raw_data"], future_metrics["raw_data"], report["difference"], dev_metrics, future_metrics, comparison_chart)

    full_report = {"summary": report, "dev_data": dev_metrics, "feature_data": future_metrics}
    history_file = os.path.join(HISTORY_DIR, f"parallel_run_{timestamp}.json")
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=4)
        
    archive_run(LATEST_DIR, HISTORY_DIR, history_file)

    print("\n📊 ИТОГОВЫЙ ОТЧЕТ СО СРАВНЕНИЕМ:")
    print(json.dumps(report, indent=4))
    print(f"\n📁 Все актуальные графики сохранены в папке: {LATEST_DIR}")

    assert cpu_diff <= CPU_TOLERANCE_PERCENT, \
        f"[БЛОКЕР] Деградация ЦП! Feature использует на {cpu_diff}% больше CPU."
        
    assert ram_diff <= RAM_TOLERANCE_MB, \
        f"Утечка ОЗУ! Feature использует на {ram_diff} MB больше памяти."
        
    assert gpu_diff <= GPU_TOLERANCE_PERCENT, \
        f"[БЛОКЕР] Деградация ГП! Feature использует на {gpu_diff}% больше GPU."
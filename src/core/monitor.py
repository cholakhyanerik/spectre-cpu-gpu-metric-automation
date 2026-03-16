import sys
import time
import subprocess
import psutil
import GPUtil

class ProcessMonitor:
    def __init__(self, process: psutil.Process):
        self.parent_process = process
        self.process_cache = {}

    def get_metrics(self):
        total_cpu = 0.0
        total_ram = 0.0
        try:
            children = self.parent_process.children(recursive=True)
            current_pids = {p.pid for p in children}
            current_pids.add(self.parent_process.pid)
            
            for p in [self.parent_process] + children:
                if p.pid not in self.process_cache:
                    self.process_cache[p.pid] = p
                    try:
                        p.cpu_percent(interval=None) 
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                        
            keys_to_remove = [pid for pid in self.process_cache if pid not in current_pids]
            for pid in keys_to_remove:
                del self.process_cache[pid]
                
            for p in self.process_cache.values():
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


class GPUMonitor:
    def __init__(self):
        self.last_check_time = 0.0
        self.cached_val = 0.0

    def get_metric(self):
        try:
            gpus = GPUtil.getGPUs()
            if gpus and gpus[0].load > 0:
                return gpus[0].load * 100 
        except Exception:
            pass

        current_time = time.time()
        if current_time - self.last_check_time < 3.0:
            return self.cached_val
            
        self.last_check_time = current_time

        if sys.platform != "win32":
            return self.cached_val

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
                val = float(stdout.replace(',', '.'))
                self.cached_val = val if val <= 100.0 else 100.0
                return self.cached_val
        except Exception:
            pass
            
        return self.cached_val

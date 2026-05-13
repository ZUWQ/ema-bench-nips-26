import psutil
import os

def print_memory_usage(ue_process_name_hint='UnrealEditor'):
    """
    打印当前系统内存、GPU显存、当前Python进程以及Unreal Engine进程的内存占用。

    Args:
        ue_process_name_hint (str, optional): UE项目可执行文件名的部分或完整名称，
                                            用于在进程列表中查找UE进程。
                                            例如，如果你的游戏是 "MyGame.exe"，可以传入 "MyGame"。
                                            默认为 None，将只查找通用的UE编辑器进程。
    """
    # 确保必要的库已导入
    print(f"\n{'='*60}")
    
    # 1. 打印系统内存信息
    mem = psutil.virtual_memory()
    print(f"[System Memory] Total: {mem.total / (1024**3):.2f} GB, "
          f"Available: {mem.available / (1024**3):.2f} GB, "
          f"Used: {mem.used / (1024**3):.2f} GB, "
          f"Percent: {mem.percent:.1f}%")
    
    # 2. 打印当前Python进程的内存占用
    process = psutil.Process(os.getpid())
    process_mem = process.memory_info()
    print(f"[Current Python Process] PID: {os.getpid()}, "
          f"RSS: {process_mem.rss / (1024**3):.2f} GB, "
          f"VMS: {process_mem.vms / (1024**3):.2f} GB, "
          f"Percent: {process.memory_percent():.2f}%")
    
    # 3. 打印子进程的内存占用（如果有多进程）
    children = process.children(recursive=True)
    if children:
        total_children_mem = sum([child.memory_info().rss for child in children])
        print(f"[Child Processes] Count: {len(children)}, "
              f"Total RSS: {total_children_mem / (1024**3):.2f} GB")
        # 只显示内存占用最高的前5个子进程
        sorted_children = sorted(children, key=lambda p: p.memory_info().rss, reverse=True)
        for i, child in enumerate(sorted_children[:5]):
            try:
                child_mem = child.memory_info()
                print(f"  - Child {i+1} (PID: {child.pid}, Name: {child.name()}): "
                      f"RSS: {child_mem.rss / (1024**3):.2f} GB")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue # 如果子进程在获取信息时已退出，则跳过
        if len(children) > 5:
            print(f"  ... and {len(children) - 5} more child processes")

    # 4. 新增：查找并打印Unreal Engine进程的内存占用
    # 定义要查找的UE进程名称列表（不区分大小写）
    target_names = ["unrealeditor", "ue4editor"] # 通用编辑器名称
    if ue_process_name_hint:
        # 如果用户提供了项目名，也加入到查找列表
        target_names.append(ue_process_name_hint.lower())

    found_ue_processes = []
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            proc_name_lower = proc.info['name'].lower()
            # 使用 startswith 来匹配，例如 "MyGame" 可以匹配 "MyGame.exe" 和 "MyGame-Win64-Shipping.exe"
            if any(proc_name_lower.startswith(name) for name in target_names):
                found_ue_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # 某些系统进程可能无法访问，或者进程在迭代时已消失，忽略即可
            continue
            
    if found_ue_processes:
        print("[Unreal Engine Processes]")
        for ue_proc in found_ue_processes:
            try:
                ue_mem = ue_proc.memory_info()
                print(f"  - Process: {ue_proc.name()} (PID: {ue_proc.pid}), "
                      f"RSS: {ue_mem.rss / (1024**3):.2f} GB, "
                      f"VMS: {ue_mem.vms / (1024**3):.2f} GB")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # 如果进程在获取内存信息时已退出
                print(f"  - Process: {ue_proc.name()} (PID: {ue_proc.pid}) - Could not retrieve info (process may have exited).")
    else:
        print("[Unreal Engine Processes] Not found. (Hint: pass ue_process_name_hint='YourGameName')")

    # # 5. 打印GPU显存信息
    # if torch.cuda.is_available():
    #     for i in range(torch.cuda.device_count()):
    #         allocated = torch.cuda.memory_allocated(i) / (1024**3)
    #         reserved = torch.cuda.memory_reserved(i) / (1024**3)
    #         total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
    #         print(f"[GPU {i}] Total: {total:.2f} GB, "
    #               f"Allocated: {allocated:.2f} GB, "
    #               f"Reserved: {reserved:.2f} GB, "
    #               f"Free (within reserved): {(reserved - allocated):.2f} GB, "
    #               f"Free (total): {total - reserved:.2f} GB")
    # else:
    #     print("[GPU] No CUDA device available")
        
    print(f"{'='*60}\n")
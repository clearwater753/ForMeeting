from multiprocessing import Process
import time

def worker():
    print(f"子进程开始: {time.time()}")
    time.sleep(2)  # 模拟子进程工作 2 秒
    print(f"子进程结束: {time.time()}")

if __name__ == '__main__':
    print(f"主进程开始: {time.time()}")

    # 启动子进程并等待完成
    p = Process(target=worker)
    p.start()
    # p.join()  # 等待子进程完成

    print(f"主进程结束: {time.time()}")
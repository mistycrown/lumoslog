import mss

with mss.mss() as sct:
    print("Monitors detected:")
    for i, monitor in enumerate(sct.monitors):
        if i == 0: continue # Skip the "all in one" monitor
        print(f"Index {i}: {monitor}")

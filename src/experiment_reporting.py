def print_run_start(task_name, args):
    print("=" * 80, flush=True)
    print(f"Task: {task_name}", flush=True)
    print(f"Model: {args.net}", flush=True)
    print(f"Dataset: {args.dataset}", flush=True)
    print(f"Method: {args.method}", flush=True)
    print(f"Seed: {args.seed}", flush=True)
    if hasattr(args, "forget_class"):
        print(f"Forget class: {args.forget_class}", flush=True)
    if hasattr(args, "forget_perc"):
        print(f"Forget percentage: {args.forget_perc}", flush=True)
    print("=" * 80, flush=True)


def print_run_result(testacc, retainacc, zrf, mia, d_f, time_elapsed):
    print("-" * 80, flush=True)
    print("Result", flush=True)
    print(f"TestAcc: {testacc}", flush=True)
    print(f"RetainTestAcc: {retainacc}", flush=True)
    print(f"ZRF: {zrf}", flush=True)
    print(f"MIA: {mia}", flush=True)
    print(f"Df: {d_f}", flush=True)
    print(f"MethodTime: {time_elapsed}", flush=True)
    print("-" * 80, flush=True)

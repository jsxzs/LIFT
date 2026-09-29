"""LR schedule construction shared by the three camera-control trainers.

Adds one scheduler diffusers does not provide: warmup, then cosine decay to a NON-ZERO floor
("1e-4 -> 1e-5"). diffusers' get_scheduler("cosine") always anneals to 0 and takes no eta_min,
so it cannot express that. Every other name is delegated to get_scheduler unchanged.

Two details are easy to get wrong here, and both are silent -- the run trains, the LR is just
on the wrong curve:

1. Horizons are multiplied by num_processes. accelerate wraps the scheduler in
   AcceleratedScheduler, whose step() calls the wrapped scheduler's step() num_processes times per
   optimizer step (accelerate/scheduler.py; it returns early when sync_gradients is False, so
   gradient accumulation does NOT add steps). The get_scheduler() call this replaces already used
   this convention.

2. The cosine spans lr_num_training_steps, not max_train_steps. A chained run passes a different
   max_train_steps per segment (2000 / 4000 / 6000 / 8000 for a 4-segment 8000-step run) because
   that argument is the stop point. If the cosine used it, each segment would compress the decay
   into its own endpoint and restart from the peak after every resume. lr_num_training_steps is
   the whole run's length and is passed identically to every segment; accelerate's load_state
   restores last_epoch, so the curve continues where the previous segment stopped.
"""
import math

from diffusers.optimization import get_scheduler
from torch.optim.lr_scheduler import LambdaLR

COSINE_MIN_LR = "cosine_with_min_lr"


def add_lr_schedule_args(parser):
    """CLI args for the floor and the decoupled horizon. Defaults keep old behaviour."""
    parser.add_argument(
        "--lr_min", type=float, default=0.0,
        help=f"Floor the LR decays to under --lr_scheduler={COSINE_MIN_LR} (e.g. 1e-5).",
    )
    parser.add_argument(
        "--lr_num_training_steps", type=int, default=None,
        help="Steps the LR schedule spans. Defaults to --max_train_steps. Set this to the FULL "
             "run length when chaining segments, so each segment continues one curve instead of "
             "restarting its own.",
    )
    return parser


def build_lr_scheduler(args, optimizer, num_processes):
    horizon = args.lr_num_training_steps or args.max_train_steps
    warmup_p = args.lr_warmup_steps * num_processes
    total_p = horizon * num_processes

    if args.lr_scheduler != COSINE_MIN_LR:
        return get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=warmup_p,
            num_training_steps=total_p,
        )

    base, floor = args.learning_rate, args.lr_min
    if floor > base:
        raise ValueError(f"--lr_min ({floor}) must not exceed --learning_rate ({base})")

    # LambdaLR multiplies base_lr by this, so return the ratio, not the LR itself.
    def lr_lambda(step: int) -> float:
        if step < warmup_p:
            return float(step) / max(1, warmup_p)
        p = min(1.0, float(step - warmup_p) / max(1, total_p - warmup_p))
        cos = 0.5 * (1.0 + math.cos(math.pi * p))          # 1 at p=0 -> 0 at p=1
        return (floor + (base - floor) * cos) / base       # base -> floor

    return LambdaLR(optimizer, lr_lambda)

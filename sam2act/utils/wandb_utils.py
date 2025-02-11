import wandb
from omegaconf import DictConfig, OmegaConf

def wandb_init(args, mode, key, save_dir=None):
    if mode != "offline" and key:
        wandb.login(key=key)
    
    wandb.init(
        project=args.exp_id,
        name=args.exp_name,
        config=args,
        save_code=False
    )
import os


def set_output_dir(cfg):
    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=False)

    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())

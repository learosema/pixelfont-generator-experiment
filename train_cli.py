import os
import time
import argparse
from train import (
    get_models_and_optimizers,
    load_checkpoint,
    train_step,
    save_prediction,
)
from loader import load_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--data_dir", type=str, default="training_pairs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="generated")
    parser.add_argument("--preview_interval", type=int, default=50)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    generator, discriminator, gen_optimizer, disc_optimizer = get_models_and_optimizers()
    checkpoint, manager = load_checkpoint(generator, discriminator, gen_optimizer, disc_optimizer, args.checkpoint_dir)
    dataset = load_dataset(args.data_dir, batch_size=args.batch_size)

    step = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        start = time.time()

        for input_image, target in dataset:
            gen_loss, disc_loss = train_step(generator, discriminator, input_image, target,
                                             gen_optimizer, disc_optimizer)
            step += 1

            if step % args.preview_interval == 0:
                print(f"Step {step}: Gen loss {gen_loss:.4f} | Disc loss {disc_loss:.4f}")
                output = generator(input_image, training=False)
                save_prediction(input_image, output, step, args.output_dir)

        manager.save()
        print(f"Epoch {epoch+1} done in {time.time() - start:.2f}s")

if __name__ == "__main__":
    main()
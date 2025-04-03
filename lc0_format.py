input_file = "chess960_training_data.txt"
output_file = "training.sft"

with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        parts = line.strip().split()
        if len(parts) >= 2:
            fen, best_move = " ".join(parts[:-1]), parts[-1]
            f_out.write(f"1 {fen} {best_move}\n")

print(f"Converted data saved to {output_file}")

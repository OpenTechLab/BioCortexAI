import random

with open("CZ_QA_MID.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

random.shuffle(lines)

with open("CZ_QA_MID.txt", "w", encoding="utf-8") as f:
    f.writelines(lines)

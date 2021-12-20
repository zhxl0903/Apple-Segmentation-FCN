# This scripts outputs lines in train_results_path with train mean IoU scores and val mean IoU scores given
# train_results_path

train_results_path = './/train_results.txt'

with open(train_results_path, 'r') as f:
    lines = f.readlines()

# Gets lines that start with Mean IoU
iou_lines = [str(l).rstrip().lstrip('Mean IoU: ') for l in lines if l.startswith('Mean IoU: ')]

train_lines = []
val_lines = []

for i in range(len(iou_lines)):
    if i % 2 == 0:
        train_lines.append(iou_lines[i])
    else:
        val_lines.append(iou_lines[i])

# Prints IoU scores for each epoch for train and val set
print('Train Mean IoU Scores')
for line in train_lines:
    print(line)

print('Val Mean IoU Scores')
for line in val_lines:
    print(line)
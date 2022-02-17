import os

# This file was used to extract details about the dataset

file = "..."
i = 0
for root, dirs, files in os.walk(file):
    for file in files:
        if file.endswith(".flac"):
            print(os.path.join(root, file))
            i += 1
print(i)


def check_grad(model):
    grads = []
    i = 1
    for param in model.parameters():
        if not param.grad is None:
            grads.append(param.grad)
            i += 1
            if i >= 5:
                break

    return grads

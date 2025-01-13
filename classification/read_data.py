import os

def get_datasets(split):
    exclusions = ['mnist_784.pt', 'CIFAR_10.pt', 'Devnagari-Script.pt', 'Fashion-MNIST.pt']
    
    data_dir = f"../datasets/classification/OpenML-CC18-{split}"
    filenames = os.listdir(data_dir)
    filenames = [f for f in filenames if f.endswith(".pt") and (f not in exclusions)]
    
    # Create a list of tuples containing file name and size
    file_sizes = [(f, os.path.getsize(os.path.join(data_dir, f))) for f in filenames]

    # Sort the list by file size (descending)
    file_sizes.sort(key=lambda x: x[1], reverse=False)
    
    return data_dir, [f_name for f_name, _ in file_sizes]
    
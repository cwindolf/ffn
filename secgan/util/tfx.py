# import tensorflow as tf


def get_tensor_dependencies(tensor, placeholders_only=True):
    '''For debugging.
    stackoverflow.com/q/46657891/3979938
    '''
    # If a tensor is passed in, get its op
    try:
        tensor_op = tensor.op
    except AttributeError:
        tensor_op = tensor

    # Recursively analyze inputs
    dependencies = []
    for inp in tensor_op.inputs:
        new_d = get_tensor_dependencies(inp)
        non_repeated = [d for d in new_d if d not in dependencies]
        dependencies = [*dependencies, *non_repeated]

    # If we've reached the "end", return the op's name
    if len(tensor_op.inputs) == 0:
        dependencies = [tensor_op.name]
        if placeholders_only and tensor_op.type != 'Placeholder':
            dependencies = []

    # Return a list of tensor op names
    return dependencies


def center_crop_vol(x, crop):
    return x[:, crop:-crop, crop:-crop, crop:-crop, :]

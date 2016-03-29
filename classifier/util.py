import theano


def wrap_shared_var(numpy_var, name, borrow=True):
    return theano.shared(
        value=numpy_var,
        name=name,
        borrow=borrow
    )
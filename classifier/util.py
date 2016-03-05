import theano


def wrap_shared_var(numpy_var, name, borrow):
    return theano.shared(
        value=numpy_var,
        name=name,
        borrow=True
    )
import lasagne


class ListIndexLayer(lasagne.layers.Layer):
    """
    If a layer outputs a list we use this layer to fetch a specif index
    in the list.
    In general you should not expect this to work because it violates some
    of the assumptions lasagne currently makes.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
    The layer feeding into this layer, or the expected input shape

    index :  int
        The list index to be selected.
    """
    def __init__(self, incoming, index, **kwargs):
        super(ListIndexLayer, self).__init__(incoming, **kwargs)
        self.index = index

    def get_output_for(self, input, **kwargs):
        return input[self.index]
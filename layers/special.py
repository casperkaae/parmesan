import lasagne

class ListIndexLayer(lasagne.layers.Layer):
    """
     If a layer outputs a list we use this layer to fetch a specif index
     in the list.
     In general you should not expect this to work because it violates some
     of the assumptions lasagne currently makes.
    """
    def __init__(self, incoming, index, **kwargs):
        super(ListIndexLayer, self).__init__(incoming, **kwargs)
        self.index = index

    def get_output_for(self, input, **kwargs):
        return input[self.index]
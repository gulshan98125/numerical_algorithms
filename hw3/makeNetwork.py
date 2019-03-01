import random

def makeNetwork(name='wheatstone', m=4):
    """Generate an example resistor network.

    Returned value is a pair (m, tuples) where
    - m = number of nodes,
    - tuples = [(i, j, Rij), ...] is list of resistors Rij connecting nodes i, j

    Try makeNetwork('wheatstone') first, which gives an unbalanced Wheatstone
    bridge on 4 nodes. The computed voltages should be [0.8, 0.4, 0.5, 0.2]."""

    if name == 'wheatstone':
        # Unbalanced Wheatstone bridge
        m = 4
        tuples = [(0,1, 8.), (0,2, 2.), (1,2, 2.), (1,3, 2.), (2,3, 3.)]
    elif name == 'random1':
        # Random connected network, equal resistances
        tuples = []
        for i in range(1,m):
            tuples.append((i-1,i, 1.))
        for k in range(m):
            i = random.randrange(m)
            j = random.randrange(m)
            i, j = sorted((i, j))
            if i != j:
                tuples.append((i,j, 1.))
    elif name == 'random2':
        # Random connected network, varying resistances
        tuples = []
        for i in range(1,m):
            tuples.append((i-1,i, min(i,m-i)))
        for k in range(m):
            i = random.randrange(m)
            j = random.randrange(m)
            i, j = sorted((i, j))
            if i != j:
                tuples.append((i,j, m))
    else:
        raise ValueError("Unknown name {}".format(name))
    return m, tuples
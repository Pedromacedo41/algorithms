from collections import deque


class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def serialize(root):
    if root is None:
        return '#'
    return '{} {} {}'.format(root.val, serialize(root.left), serialize(root.right))

def deserialize(data):
    def helper():
        val = next(vals)
        if val == '#':
            return None
        node = Node(str(val))
        node.left = helper()
        node.right = helper()
        return node
    vals = iter(data.split())
    return helper()



if __name__== "__main__":
    node = Node('root', Node('left', Node('left.left')), Node('right'))
    print(serialize(node))
    stack = deque()
    stack.append('a')
    stack.append('b')
    stack.append('c')
    
    assert deserialize(serialize(node)).left.left.val == 'left.left'


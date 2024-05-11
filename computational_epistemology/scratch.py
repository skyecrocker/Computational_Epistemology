from bandit_utils import slot_machines

machine = slot_machines.NormalMachine(0, 1)

print(machine.pull())

class A():
    def __init__(self, a):
        self.a = a
        self.z = 100

class B(A):
    def __init__(self, b):
        self.b = b


b = B(1)
print(b.z)
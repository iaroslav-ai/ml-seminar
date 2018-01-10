# complete import with alias
import S6_classes as s6

cls = s6.MyClass('import')
print(cls.myfunction('done'))


# parital import
from S6_classes import MyClass

cls = MyClass('import')
print(cls.myfunction('done'))


# built in import
import random
print(random.randint(1,100))
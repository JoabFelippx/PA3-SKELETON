from google.protobuf.struct_pb2 import Struct
from google.protobuf import json_format


struct_msg = Struct()


struct_msg.update({
    "name": "Alice",
    "age": 30,
    "isStudent": False,
    "grades": [90, 85, 92],
    "address": {
        "street": "123 Main St",
        "city": "Anytown"
    }
})

print(struct_msg['address'])
dict_representation = json_format.MessageToDict(struct_msg)
print(dict_representation)

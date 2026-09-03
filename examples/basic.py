from tiny_tensor_compiler import GraphBuilder, execute_cpu, lower_to_cpu, verify

builder = GraphBuilder()
x = builder.tensor([1, 2, 3])
z = (x * 2 + 1).relu()
module = builder.finish(z)

verify(module)
print("Tensor IR:")
print(module.dump())
print("\nLowered CPU IR:")
program = lower_to_cpu(module)
print(program.dump())
print("\nResult:", execute_cpu(program))

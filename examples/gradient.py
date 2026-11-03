def f(x):     return (x - 3) ** 2
def grad(x):  return 2 * (x - 3)

x, lr = 0.0, 0.1
for step in range(25):
    x = x - lr * grad(x)
    if step % 5 == 0:
        print(f"step {step:2d}  x={x:.4f}  loss={f(x):.6f}")
print("final x:", round(x, 4))   # → ~3.0
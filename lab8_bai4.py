# Ham mat mat L(w) = w^2 - 4w + 4 = (w - 2)^2, dao ham L'(w) = 2w - 4
def loss(w):
    return w ** 2 - 4 * w + 4


def gradient(w):
    return 2 * w - 4


def gradient_descent_1d(initial_w, learning_rate, epochs):
    w = initial_w
    loss_history = [loss(w)]

    for _ in range(epochs):
        w = w - learning_rate * gradient(w)   # Buoc cap nhat nguoc chieu gradient
        loss_history.append(loss(w))

    return w, loss_history


if __name__ == "__main__":
    w_star, history = gradient_descent_1d(initial_w=10.0, learning_rate=0.1, epochs=20)
    for epoch, l in enumerate(history):
        print(f"Epoch {epoch:2d}: Loss = {l:.6f}")
    print(f"\nw* = {w_star:.4f} (ly thuyet: 2.0), Loss cuoi = {history[-1]:.6f}")

    # So sanh toc do hoc
    for lr in (0.01, 0.5, 1.1):
        w, h = gradient_descent_1d(10.0, lr, 20)
        print(f"lr = {lr:<4}: w sau 20 epoch = {w:.4f}, Loss = {h[-1]:.4f}")

# Nhan xet: w - 2 bi nhan voi (1 - 2*lr) moi buoc.
# - lr = 0.1 -> he so 0.8: hoi tu deu ve 2.0 (sau 20 epoch sai so 8 * 0.8^20 ~ 0.0922).
# - lr = 0.01 -> he so 0.98: hoi tu rat cham.
# - lr = 0.5 -> he so 0: nhay thang den nghiem sau 1 buoc (rieng voi ham bac 2 nay).
# - lr = 1.1 -> |he so| = 1.2 > 1: dao dong va phan ky -> chon learning rate qua lon se hong.

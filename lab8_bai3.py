# Primal: max c^T x, A x <= b, x >= 0  <->  Dual: min b^T y, A^T y >= c, y >= 0
def transpose(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def generate_dual_problem(c_primal, A_primal, b_primal):
    c_dual = list(b_primal)        # He so ham muc tieu doi ngau
    A_dual = transpose(A_primal)   # Ma tran rang buoc doi ngau
    b_dual = list(c_primal)        # Ve phai doi ngau
    return c_dual, A_dual, b_dual


def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


if __name__ == "__main__":
    c_primal = [40, 30]
    A_primal = [[2, 1], [1, 1]]
    b_primal = [100, 80]

    c_dual, A_dual, b_dual = generate_dual_problem(c_primal, A_primal, b_primal)
    print("Bai toan doi ngau: min", " + ".join(f"{c}*y{i + 1}" for i, c in enumerate(c_dual)))
    for i, row in enumerate(A_dual):
        print("  ", " + ".join(f"{a}*y{j + 1}" for j, a in enumerate(row)), ">=", b_dual[i])
    print("   y1, y2 >= 0")

    # Kiem chung dinh ly doi ngau manh
    x_star = [20, 60]
    y_star = [10, 20]

    # y* phai kha thi cho bai doi ngau: A^T y* >= c
    feasible = all(dot(A_dual[i], y_star) >= b_dual[i] for i in range(len(b_dual)))
    print(f"\ny* = {y_star} kha thi cho Dual: {feasible}")

    primal_val = dot(c_primal, x_star)
    dual_val = dot(b_primal, y_star)
    print(f"c^T x* = 40*20 + 30*60 = {primal_val}")
    print(f"b^T y* = 100*10 + 80*20 = {dual_val}")
    print(f"Doi ngau manh c^T x* == b^T y*: {primal_val == dual_val}")  # True, = 2600

# Y nghia: y* = (10, 20) la "gia bong" (shadow price): them 1 gio GPU tang loi nhuan 10 USD,
# them 1 don vi VRAM tang 20 USD. Doi ngau yeu cho moi y kha thi la chan tren cua Primal; khi
# hai gia tri bang nhau thi ca x* va y* deu toi uu. SVM giai bai toan doi ngau (theo alpha)
# thay vi bai goc, vi thanh phan doi ngau chi phu thuoc tich vo huong -> dung duoc kernel trick.

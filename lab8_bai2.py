# Dua LP: max c^T x, A_ub x <= b_ub, x >= 0 ve dang chinh tac:
# min c_std^T [x, s], A_eq [x, s] = b_eq, x, s >= 0 (s la bien bu)
def convert_to_standard_form(c, A_ub, b_ub):
    m = len(A_ub)  # So rang buoc
    n = len(c)     # So bien goc

    # 1. Ham muc tieu chuan: min(-c^T * x + 0*s)
    c_std = [-val for val in c] + [0.0] * m

    # 2. Ma tran A_eq mo rong voi ma tran don vi I_m
    A_eq = []
    for i in range(m):
        slack_row = [1.0 if j == i else 0.0 for j in range(m)]
        A_eq.append(A_ub[i] + slack_row)

    # 3. Ve phai giu nguyen
    b_eq = list(b_ub)
    return c_std, A_eq, b_eq


if __name__ == "__main__":
    c = [40, 30]
    A_ub = [[2, 1], [1, 1]]
    b_ub = [100, 80]

    c_std, A_eq, b_eq = convert_to_standard_form(c, A_ub, b_ub)
    print("Ham muc tieu c_std:", c_std)  # [-40, -30, 0.0, 0.0]
    print("Ma tran A_eq:")
    for row in A_eq:
        print(row)                       # [2, 1, 1.0, 0.0] / [1, 1, 0.0, 1.0]
    print("Vector b_eq:", b_eq)          # [100, 80]

    # Kiem tra tai nghiem toi uu x* = (20, 60): bien bu s = b - A x = (0, 0) -> ca 2 rang buoc chat
    x_opt = [20, 60]
    slack = [b_ub[i] - sum(A_ub[i][j] * x_opt[j] for j in range(len(c))) for i in range(len(b_ub))]
    print(f"Tai x* = {x_opt}: bien bu s = {slack} -> het sach gio GPU va VRAM")

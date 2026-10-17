# Giai LP 2D bang phuong phap hinh hoc: cuc tri cua ham tuyen tinh tren mien da dien loi
# luon dat tai mot dinh -> chi can duyet cac dinh.
def solve_lp_geometric(vertices, c1, c2):
    best_val = -float('inf')
    best_vertex = None

    for x1, x2 in vertices:
        # Tinh gia tri ham muc tieu f = c1*x1 + c2*x2
        current_val = c1 * x1 + c2 * x2
        print(f"Dinh ({x1}, {x2}) -> Gia tri f = {current_val}")

        if current_val > best_val:
            best_val = current_val
            best_vertex = (x1, x2)

    return best_vertex, best_val


if __name__ == "__main__":
    # max f = 40*x1 + 30*x2; 2*x1 + x2 <= 100; x1 + x2 <= 80; x1, x2 >= 0
    vertices = [(0, 0), (0, 80), (20, 60), (50, 0)]
    c1, c2 = 40, 30

    optimal_point, max_profit = solve_lp_geometric(vertices, c1, c2)
    print(f"\nNghiem toi uu: {optimal_point}")       # (20, 60)
    print(f"Loi nhuan toi da: {max_profit} USD")    # 2600

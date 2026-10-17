import math


def mse_loss(w1, w2, center=(3, 3)):
    return (w1 - center[0]) ** 2 + (w2 - center[1]) ** 2


# Rang buoc L1 |w1| + |w2| <= 2: tam (3, 3) nam ngoai hinh thoi nen nghiem nam tren bien.
# Duyet cac diem tren 4 canh cua hinh thoi (dinh (2,0), (0,2), (-2,0), (0,-2)).
def solve_constrained_l1(radius=2.0, center=(3, 3), steps=4000):
    corners = [(radius, 0), (0, radius), (-radius, 0), (0, -radius)]
    best_point, best_loss = None, float('inf')
    for k in range(4):
        (ax, ay), (bx, by) = corners[k], corners[(k + 1) % 4]
        for i in range(steps + 1):
            t = i / steps
            w1, w2 = ax + t * (bx - ax), ay + t * (by - ay)
            l = mse_loss(w1, w2, center)
            if l < best_loss:
                best_point, best_loss = (w1, w2), l
    return best_point, best_loss


# Rang buoc L2 w1^2 + w2^2 <= 4: duyet cac diem tren duong tron ban kinh 2
def solve_constrained_l2(radius=2.0, center=(3, 3), steps=36000):
    best_point, best_loss = None, float('inf')
    for i in range(steps):
        theta = 2 * math.pi * i / steps
        w1, w2 = radius * math.cos(theta), radius * math.sin(theta)
        l = mse_loss(w1, w2, center)
        if l < best_loss:
            best_point, best_loss = (w1, w2), l
    return best_point, best_loss


def fmt(p):
    return f"({p[0]:.3f}, {p[1]:.3f})"


if __name__ == "__main__":
    print("=== Tam sai so (3, 3) theo de bai ===")
    p1, l1 = solve_constrained_l1()
    p2, l2 = solve_constrained_l2()
    print(f"L1 (hinh thoi): w* = {fmt(p1)}, Loss = {l1:.4f}")
    print(f"   so sanh tai cac dinh: Loss(2, 0) = {mse_loss(2, 0):.1f}, Loss(0, 2) = {mse_loss(0, 2):.1f}")
    print(f"L2 (hinh tron): w* = {fmt(p2)}, Loss = {l2:.4f}  "
          f"(sqrt(2) = {math.sqrt(2):.3f})")

    print("\n=== Tam sai so lech (3, 1): mot dac trung quan trong hon ===")
    p1b, l1b = solve_constrained_l1(center=(3, 1))
    p2b, l2b = solve_constrained_l2(center=(3, 1))
    print(f"L1 (hinh thoi): w* = {fmt(p1b)}, Loss = {l1b:.4f}  -> w2 = 0 (thua)")
    print(f"L2 (hinh tron): w* = {fmt(p2b)}, Loss = {l2b:.4f}  -> ca 2 trong so khac 0")

# Phan tich ket qua:
# - Voi tam (3, 3) DOI XUNG qua duong w1 = w2, duong dong muc cua Loss la cac duong tron tam (3, 3).
#   Duong tron nho nhat cham canh w1 + w2 = 2 cua hinh thoi tai chan duong vuong goc ha tu (3, 3),
#   tuc trung diem (1, 1) voi Loss = 8 < Loss(2, 0) = Loss(0, 2) = 10. Vay o dung de bai nay
#   nghiem L1 KHONG roi vao dinh (2, 0)/(0, 2) nhu de goi y; code tinh ra (1, 1) la chinh xac.
#   Ly do: gradient cua Loss tai (1, 1) la (-4, -4), song song voi phap tuyen (1, 1) cua canh.
# - Tinh thua xuat hien khi tam lech: voi tam (3, 1), chan vuong goc len duong w1 + w2 = 2 la
#   (2, 0) -> nghiem L1 roi dung vao dinh tren truc, w2 = 0 (loai bo dac trung 2). Tong quat, voi
#   tam (c1, c2) o goc phan tu I, nghiem L1 roi vao dinh khi |c1 - c2| >= 2: mot vung rat rong cua
#   khong gian tham so "do" vao cac dinh nhon nam tren truc. Day la co che Feature Selection cua
#   Lasso: dac trung yeu bi ep ve dung 0.
# - Hinh tron L2 tron, khong co dinh nhon: nghiem la giao cua tia tu goc toi tam voi duong tron,
#   w* = 2 * c / ||c||. Voi (3, 3) -> (sqrt(2), sqrt(2)) ~ (1.414, 1.414); voi (3, 1) ->
#   (1.897, 0.632). Toa do chi bang 0 khi tam nam dung tren truc, nen Ridge chi co rut (shrink)
#   deu cac trong so theo cung ty le, khong triet tieu ve 0.
#
# Tong ket mon hoc - mach lien ket 8 bai:
# - Lab 1-2 (Logic, Tap hop, Ham, Quan he): ngon ngu hinh thuc de mo ta du lieu, nhan, ham kich hoat.
# - Lab 3-4 (Dai so tuyen tinh, To hop): du lieu la vector/ma tran, moi lop mang neuron la phep
#   bien doi tuyen tinh W x + b; tri rieng/vector rieng (PCA) nen du lieu; to hop dem khong gian
#   tim kiem (grid search, so cau hinh).
# - Lab 5 (Xac suat, Thong tin): mo hinh AI xuat ra phan phoi xac suat; Entropy/Information Gain
#   cho cay quyet dinh, Cross-Entropy la ham mat mat chuan cua phan loai, Naive Bayes la vi du
#   suy dien Bayes.
# - Lab 6-7 (Do thi, Luong, Markov): du lieu quan he (mang xa hoi, knowledge graph, GNN), BFS/DFS,
#   phan cum lien thong, PageRank = vector rieng lambda = 1 cua chuoi Markov, ghep cap = max flow.
# - Lab 8 (Toi uu hoa): "huan luyen" mot mo hinh chinh la giai bai toan toi uu - LP/doi ngau
#   (nen tang SVM), Gradient Descent (Deep Learning), Regularization L1/L2 la toi uu co rang buoc.
# => Mot mo hinh AI hien dai = Dai so tuyen tinh (bieu dien) + Xac suat (mo hinh hoa bat dinh)
#    + Do thi (cau truc) + Toi uu hoa (hoc tham so).

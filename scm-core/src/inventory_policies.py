
import math

class EOQPolicy:
    """Economic Order Quantity (EOQ) policy implementation."""

    def __init__(self, annual_demand: int, order_cost: float, holding_cost: float):
        self.D = annual_demand
        self.S = order_cost
        self.H = holding_cost
        self.eoq = self.calculate_eoq()

    def calculate_eoq(self) -> float:
        if self.H == 0:
            raise ValueError("Holding cost cannot be zero.")
        return math.sqrt((2 * self.D * self.S) / self.H)

    def reorder_point(self, daily_demand: int, lead_time_days: int) -> int:
        return daily_demand * lead_time_days

    def order_quantity(self, current_inventory: int, daily_demand: int, lead_time_days: int) -> int:
        rop = self.reorder_point(daily_demand, lead_time_days)
        return int(self.eoq) if current_inventory <= rop else 0


class SSPolicy:
    """(s, S) inventory control policy implementation."""

    def __init__(self, s: int, S: int):
        self.s = s  # reorder point
        self.S = S  # target inventory level

    def order_quantity(self, current_inventory: int) -> int:
        return self.S - current_inventory if current_inventory <= self.s else 0


# ===== 사용 예시 (참고용) =====
# 실제 실행 시 main.py 사용
# ...def main():
#    eoq = EOQPolicy(annual_demand=12000, order_cost=50, holding_cost=2)
#    ss = SSPolicy(s=2000, S=8000)
#
#    print("EOQ:", eoq.eoq)
#    print("SS Order Quantity:", ss.order_quantity(150))

#if __name__ == "__main__":
#    main()
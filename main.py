from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class Order:  # Our own placed order
    timestamp: float
    order_id: int
    side: str
    size: float
    price: float
    execut: bool


@dataclass
class AnonTrade:  # Market trade
    timestamp: float
    side: str
    size: float
    price: float


@dataclass
class OwnTrade:  # Execution of own placed order
    timestamp: float
    trade_id: int
    order_id: int
    side: str
    size: float
    price: float


@dataclass
class OrderbookSnapshotUpdate:  # Orderbook tick snapshot
    timestamp: float
    asks: list[tuple[float, float]]  # tuple[price, size]
    bids: list[tuple[float, float]]


@dataclass
class MdUpdate:  # Data of a tick
    orderbook: Optional[OrderbookSnapshotUpdate] = None
    trades: Optional[list[AnonTrade]] = None
    timestamp: int = 0


class Strategy:
    def __init__(self, max_position: float) -> None:
        pass

    def run(self, sim: "Sim"):
        while True:
            try:
                md_update = sim.tick()
                # call sim.place_order and sim.cancel_order here
            except StopIteration:
                break


def load_md_from_file(path: str) -> list[MdUpdate]:
    trades = pd.read_csv(path + "trades.csv")
    lobs = pd.read_csv(path + "lobs.csv")
    for name in lobs:
        if 'btcusdt:Binance:LinearPerpetual' in name:
            lobs.rename(columns={name: "_".join(name.split("_")[1:])}, inplace=True)
        if name == " exchange_ts":
            lobs.rename(columns={name: "exchange_ts"}, inplace=True)
    lobs['is_lobs'] = 1
    trades_lobs = pd.concat([lobs[:1000], trades[:1000]], ignore_index=True)
    print(len(trades_lobs), len(lobs) + len(trades))
    sorted_df = trades_lobs.sort_values(by='receive_ts')
    ans_array = []

    cur_md = MdUpdate()
    cur_md.timestamp = sorted_df['receive_ts'][0]
    cur_ts = 0
    sorted_df = sorted_df.reset_index()
    print(len(sorted_df))
    for ind, row in sorted_df.head().iterrows():
        if cur_ts != row['exchange_ts'] and ind != 0:
            ans_array.append(cur_md)
            cur_md = MdUpdate()
            cur_md.timestamp = row['receive_ts']
        if not np.isnan(row['is_lobs']):
            cur_md.orderbook = OrderbookSnapshotUpdate(row['receive_ts'], [], [])
            for i in range(10):
                cur_md.orderbook.asks.append((row['ask_price_' + str(i)], row['ask_vol_' + str(i)]))
                cur_md.orderbook.bids.append((row['bid_price_' + str(i)], row['bid_vol_' + str(i)]))
        else:
            if cur_md.trades is None:
                cur_md.trades = []
            cur_md.trades.append(AnonTrade(row['receive_ts'], row['aggro_side'], row['size'], row['price']))
    ans_array.append(cur_md)

    return ans_array


class Sim:
    def __init__(self, execution_latency: float, md_latency: float) -> None:
        self.md_latency = md_latency
        self.execution_latency = execution_latency
        self.md = iter(load_md_from_file("md/btcusdt_Binance_LinearPerpetual/"))
        self.curr_md: Optional[MdUpdate] = None
        self.curr_ts: int = 0
        self.pending_orders: list[Order] = []
        self.pending_orders_for_cansel: list[tuple[int, int]] = []
        self.active_orders: list[Order] = []
        self.order_id: int = 0
        self.executed_orders: list[Order] = []

    def tick(self) -> MdUpdate:
        self.curr_md = next(self.md)
        self.curr_ts = self.curr_md.timestamp

        self.execute_orders()
        self.prepare_orders()

        return next(self.md)

    def prepare_orders(self):
        while len(self.pending_orders) > 0 and self.pending_orders[0].timestamp + self.md_latency < self.curr_ts:
            self.active_orders.append(self.pending_orders.pop(0))

    def execute_orders(self):
        p_trade_min: float = np.inf
        p_trade_max: float = 0
        trades = self.curr_md.trades
        for trade in trades:
            p_trade_max = max(p_trade_max, trade.price)
            p_trade_min = min(p_trade_min, trade.price)
        if self.curr_md.orderbook:
            p_trade_min = min(p_trade_min, self.curr_md.orderbook.bids[0][0])
            p_trade_max = max(p_trade_max, self.curr_md.orderbook.asks[0][0])

        for order_cons in self.active_orders:
            if not order_cons.execut:
                if order_cons.side == 'bid' and order_cons.price > p_trade_max:
                    order_cons.execut = True
                    self.executed_orders.append(order_cons)
                if order_cons.side == 'ask' and order_cons.price > p_trade_min:
                    order_cons.execut = True
                    self.executed_orders.append(order_cons)
        pass

    def place_order(self, side: str, size: int, price: int):
        self.order_id += 1
        order = Order(self.curr_ts,
                      self.order_id,
                      side, size, price, False)
        self.pending_orders.append(order)
        return self.order_id

    def cancel_order(self, id: int):
        self.pending_orders_for_cansel.append((id, self.curr_ts))


'''
if __name__ == "__main__":
    strategy = Strategy(10)
    sim = Sim(10, 10)
    strategy.run(sim)
'''
data = load_md_from_file('md/btcusdt_Binance_LinearPerpetual/')
for el in data[:5]:
    print(el)
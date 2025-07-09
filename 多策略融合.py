# 克隆自聚宽文章：https://www.joinquant.com/post/57343
# 标题：多策略10.02社区学习版（更适合量化交易）
# 作者：O_iX

# 导入函数库
from jqdata import *
from jqfactor import get_factor_values
import datetime
import math
from scipy.optimize import minimize


# -------------------- 运行调度函数 --------------------
def initialize(context):

    set_option("avoid_future_data", True)  # 打开防未来函数
    set_option("use_real_price", True)  # 开启动态复权模式(真实价格)
    log.info("初始函数开始运行且全局只运行一次")  # 输出内容到日志 log.info()
    log.set_level("order", "error")  # 过滤掉order系列API产生的比error级别低的log

    # 全局变量
    g.strategys = {}
    # g.portfolio_value_proportion = [1, 0, 0, 0, 0]  # 测试版
    g.portfolio_value_proportion = [0.35, 0.35, 0.1, 0.1, 0.1]  # 标准版
    # g.portfolio_value_proportion = [0.6, 0, 0.2, 0, 0.2]  # 极端版
    g.positions = {i: {} for i in range(len(g.portfolio_value_proportion))}  # 记录每个子策略的持仓股票

    # 子策略执行计划
    if g.portfolio_value_proportion[0] > 0:
        run_weekly(jsg_adjust, 1, "11:00")
        run_daily(jsg_check, "14:30")
    if g.portfolio_value_proportion[1] > 0:
        run_weekly(all_day_adjust, 1, "11:00")
    if g.portfolio_value_proportion[2] > 0:
        run_monthly(simple_roa_adjust, 1, "11:00")
        run_daily(simple_roa_check, "14:30")
    if g.portfolio_value_proportion[3] > 0:
        run_weekly(weak_cyc_adjust, 1, "11:00")
    if g.portfolio_value_proportion[4] > 0:
        run_daily(etf_rotation_adjust, "11:00")
    # 每日剩余资金购买货币ETF
    run_daily(end_trade, "14:59")


def process_initialize(context):
    print("重启程序")
    g.strategys = {
        name: cls(context, index=idx, name=name)
        for name, cls, idx in [
            ("搅屎棍策略", JSG_Strategy, 0),
            ("全天候策略", All_Day_Strategy, 1),
            ("简单ROA策略", Simple_ROA_Strategy, 2),
            ("弱周期价投策略", Weak_Cyc_Strategy, 3),
            ("核心资产轮动策略", Etf_Rotation_Strategy, 4),
        ]
    }


# 尾盘处理
def end_trade(context):

    marked = {s for d in g.positions.values() for s in d}
    for stock in context.portfolio.positions:
        if stock not in marked and order_target_value(stock, 0):
            log.info(f"卖出{stock}因送股未记录在持仓中")


def jsg_check(context):
    g.strategys["搅屎棍策略"].check()


def jsg_adjust(context):
    g.strategys["搅屎棍策略"].adjust()


def all_day_adjust(context):
    g.strategys["全天候策略"].adjust()


def simple_roa_adjust(context):
    g.strategys["简单ROA策略"].adjust()


def simple_roa_check(context):
    g.strategys["简单ROA策略"].check()


def weak_cyc_adjust(context):
    g.strategys["弱周期价投策略"].adjust()


def etf_rotation_adjust(context):
    g.strategys["核心资产轮动策略"].adjust()


# -------------------- 策略基类 --------------------
class Strategy:

    def __init__(self, context, index, name):
        self.context = context
        self.index = index
        self.name = name
        self.stock_sum = 1
        self.hold_list = []
        self.min_money = 10000  # 最小交易额(限制手续费)
        self.pass_months = [1, 4]
        self.def_stocks = ["511260.XSHG", "518880.XSHG", "512800.XSHG"]  # 债券ETF、黄金ETF、银行ETF

    # 获取持仓市值
    def get_total_value(self):
        if not g.positions[self.index]:
            return 0
        return sum(self.context.portfolio.positions[key].price * value for key, value in g.positions[self.index].items())

    # 调仓(targets为字典，key为股票代码，value为目标市值)
    def _adjust(self, targets):

        current_data = get_current_data()

        # 输出信息
        strategy_data = {
            "strategy_name": self.name,
            "strategy_id": self.index,
            "stocks": [{"stock_name": current_data[stock].name, "stock_code": stock, "weight": weight} for stock, weight in targets.items()],
        }

        # log.info(strategy_data)

        # 获取已持有列表
        self.hold_list = list(g.positions[self.index].keys())
        portfolio = self.context.portfolio

        # 获取目标策略市值
        target_value = self.context.portfolio.total_value * g.portfolio_value_proportion[self.index]

        # 清仓被调出的
        for stock in self.hold_list:
            if stock not in targets:
                self.order_target_value_(stock, 0)

        # 先卖出
        for stock, weight in targets.items():
            target = target_value * weight
            price = current_data[stock].last_price
            value = g.positions[self.index].get(stock, 0) * price
            if value - target > max(self.min_money, price * 100):
                self.order_target_value_(stock, target)

        # 后买入
        for stock, weight in targets.items():
            target = target_value * weight
            price = current_data[stock].last_price
            value = g.positions[self.index].get(stock, 0) * price
            if min(target - value, portfolio.available_cash) > max(self.min_money, price * 100):
                self.order_target_value_(stock, target)

    # 自定义下单(涨跌停不交易)
    def order_target_value_(self, security, value):
        current_data = get_current_data()

        # 检查标的是否停牌、涨停、跌停
        if current_data[security].paused:
            log.info(f"{security}: 今日停牌")
            return False

        if current_data[security].last_price == current_data[security].high_limit:
            log.info(f"{security}: 当前涨停")
            return False

        if current_data[security].last_price == current_data[security].low_limit:
            log.info(f"{security}: 当前跌停")
            return False

        # 获取当前标的的价格
        price = current_data[security].last_price

        # 获取当前策略的持仓数量
        current_position = g.positions[self.index].get(security, 0)

        # 计算目标持仓数量
        target_position = (int(value / price) // 100) * 100 if price != 0 else 0

        # 计算需要调整的数量
        adjustment = target_position - current_position

        # 检查是否当天买入卖出
        closeable_amount = self.context.portfolio.positions[security].closeable_amount if security in self.context.portfolio.positions else 0
        if adjustment < 0 and closeable_amount == 0:
            log.info(f"{security}: 当天买入不可卖出")
            return False

        # 下单并更新持仓
        if adjustment != 0:
            o = order(security, adjustment)
            if o:
                # 更新持仓数量
                filled = o.filled if o.is_buy else -o.filled
                g.positions[self.index][security] = filled + current_position
                # 如果当前持仓为零，移除该证券
                if g.positions[self.index][security] == 0:
                    g.positions[self.index].pop(security, None)
                # 更新持有列表
                self.hold_list = list(g.positions[self.index].keys())
                return True
        return False

    # 基础过滤(过滤科创北交、ST、停牌、次新股)
    def filter_basic_stock(self, stock_list):

        current_data = get_current_data()
        return [
            stock
            for stock in stock_list
            if not current_data[stock].paused
            and not current_data[stock].is_st
            and "ST" not in current_data[stock].name
            and "*" not in current_data[stock].name
            and "退" not in current_data[stock].name
            and not (stock[0] == "4" or stock[0] == "8" or stock[:2] == "68")
            and not self.context.previous_date - get_security_info(stock).start_date < datetime.timedelta(375)
        ]

    # 过滤当前时间涨跌停的股票
    def filter_limitup_limitdown_stock(self, stock_list):
        current_data = get_current_data()
        return [
            stock
            for stock in stock_list
            if current_data[stock].last_price < current_data[stock].high_limit and current_data[stock].last_price > current_data[stock].low_limit
        ]

    # 过滤近几日涨停过的股票
    def filter_limitup_stock(self, stock_list, days):
        df = get_price(
            stock_list,
            end_date=self.context.previous_date,
            frequency="daily",
            fields=["close", "high_limit"],
            count=days,
            panel=False,
        )
        df = df[df["close"] == df["high_limit"]]
        filterd_stocks = df.code.drop_duplicates().tolist()
        return [stock for stock in stock_list if stock not in filterd_stocks]


    # 检查持仓中曾经涨停但当前未涨停的股票
    def _check(self):
        hold = list(g.positions[self.index].keys())
        if not hold:
            return []
        current_data = get_current_data()
        filtered = self.filter_limitup_stock(hold, 3)
        return [s for s in hold if s not in filtered and current_data[s].last_price < current_data[s].high_limit]

    # 识别无法交易的股票（停牌、涨跌停）
    def filter_untradeable_stock(self, stocks):
        current_data = get_current_data()
        return [
            stock
            for stock in stocks
            if current_data[stock].paused or current_data[stock].last_price in (current_data[stock].high_limit, current_data[stock].low_limit)
        ]

    # 根据调仓逻辑计算最终保留的股票列表
    def get_adjusted_stocks(self, selected, sell):
        fixed = self.filter_untradeable_stock(list(g.positions[self.index].keys()))
        sum = len(self.def_stocks) if selected == self.def_stocks else self.stock_sum - len(fixed)
        return fixed + [s for s in selected if s not in fixed and s not in sell][:sum]


# 搅屎棍策略
class JSG_Strategy(Strategy):

    def __init__(self, context, index, name):
        super().__init__(context, index, name)

        self.stock_sum = 6
        # 判断买卖点的行业数量
        self.num = 1
        # 空仓的月份
        self.pass_months = [1, 4]

    def getStockIndustry(self, stocks):
        industry = get_industry(stocks)
        return pd.Series({stock: info["sw_l1"]["industry_name"] for stock, info in industry.items() if "sw_l1" in info})

    # 获取市场宽度
    def get_market_breadth(self):
        # 指定日期防止未来数据
        yesterday = self.context.previous_date
        # 获取初始列表
        stocks = get_index_stocks("000985.XSHG")
        count = 1
        h = get_price(
            stocks,
            end_date=yesterday,
            frequency="1d",
            fields=["close"],
            count=count + 20,
            panel=False,
        )
        h["date"] = pd.DatetimeIndex(h.time).date
        df_close = h.pivot(index="code", columns="date", values="close").dropna(axis=0)
        # 计算20日均线
        df_ma20 = df_close.rolling(window=20, axis=1).mean().iloc[:, -count:]
        # 计算偏离程度
        df_bias = df_close.iloc[:, -count:] > df_ma20
        df_bias["industry_name"] = self.getStockIndustry(stocks)
        # 计算行业偏离比例
        df_ratio = ((df_bias.groupby("industry_name").sum() * 100.0) / df_bias.groupby("industry_name").count()).round()
        # 获取偏离程度最高的行业
        top_values = df_ratio.loc[:, yesterday].nlargest(self.num)
        I = top_values.index.tolist()
        return I

    # 过滤股票
    def filter(self):
        stocks = get_index_stocks("399101.XSHE")
        stocks = self.filter_basic_stock(stocks)
        stocks = self.filter_limitup_stock(stocks, 5)
        stocks = get_fundamentals(
            query(
                valuation.code,
            )
            .filter(
                valuation.code.in_(stocks),
                indicator.adjusted_profit > 0,
            )
            .order_by(valuation.market_cap.asc())
        ).code
        stocks = self.filter_limitup_limitdown_stock(stocks)
        return stocks

    # 择时
    def select(self):
        I = self.get_market_breadth()
        industries = {"银行I", "煤炭I", "采掘I", "钢铁I"}
        if not industries.intersection(I) and not self.context.current_dt.month in self.pass_months:
            return self.filter()[: self.stock_sum * 2]
        return self.def_stocks

    # 调仓
    def adjust(self):
        stocks = self.get_adjusted_stocks(self.select(), [])
        self._adjust({stock: round(1 / len(stocks), 3) for stock in stocks})

    # 检查昨日涨停票
    def check(self):
        banner = self._check()
        if banner:
            stocks = self.get_adjusted_stocks(self.select(), banner)
            self._adjust({s: round(1 / len(stocks), 3) for s in stocks})


# 全天候ETF策略
class All_Day_Strategy(Strategy):

    def __init__(self, context, index, name):
        super().__init__(context, index, name)

        # 全天候ETF组合参数
        self.etf_pool = [
            "511260.XSHG",  # 十年国债ETF
            "518880.XSHG",  # 黄金ETF
            "513100.XSHG",  # 纳指100
        ]
        # 标的仓位占比
        self.rates = [0.43, 0.285, 0.285]

    # 调仓
    def adjust(self):
        self._adjust({etf: round(rate, 3) for etf, rate in zip(self.etf_pool, self.rates)})


# 简单ROA策略
class Simple_ROA_Strategy(Strategy):
    def __init__(self, context, index, name):
        super().__init__(context, index, name)

        self.stock_sum = 1

    def filter(self):
        stocks = get_all_securities("stock", date=self.context.previous_date).index.tolist()
        stocks = self.filter_basic_stock(stocks)
        stocks = self.filter_limitup_stock(stocks, 5)
        stocks = list(
            get_fundamentals(
                query(valuation.code, indicator.roa).filter(
                    valuation.code.in_(stocks),
                    valuation.pb_ratio > 0,
                    valuation.pb_ratio < 1,
                    indicator.adjusted_profit > 0,
                )
            )
            .sort_values(by="roa", ascending=False)
            .code
        )
        stocks = self.filter_limitup_limitdown_stock(stocks)
        return stocks

    # 调仓
    def adjust(self):
        stocks = self.get_adjusted_stocks(self.filter(), [])
        self._adjust({stock: round(1 / len(stocks), 3) for stock in stocks})

    # 检查昨日涨停票
    def check(self):
        banner = self._check()
        if banner:
            stocks = self.get_adjusted_stocks(self.filter(), banner)
            self._adjust({s: round(1 / len(stocks), 3) for s in stocks})


# 弱周期价投策略
class Weak_Cyc_Strategy(Strategy):
    def __init__(self, context, index, name):
        super().__init__(context, index, name)

        # 行业比例（公用事业、交通运输）
        self.industry_ratio = [0.5, 0.5]
        # 个股比例
        self.stock_ratio = [1]

    # 分红过滤(近几年的分红总和满足股息率与股利支付率)
    def filter_dividend(self, stocks, year, div_yield, payout_rate):

        if not stocks:
            return []

        time1 = self.context.previous_date
        time0 = time1 - timedelta(days=(year + 0.1) * 365)
        f = finance.STK_XR_XD
        q = query(f.code, f.bonus_amount_rmb).filter(
            f.code.in_(stocks),
            f.a_registration_date >= time0,
            f.a_registration_date <= time1,
            # f.board_plan_pub_date >= time0,
            # f.board_plan_pub_date <= time1,
        )
        df = finance.run_query(q).fillna(0).set_index("code").groupby("code").sum()

        # 获取市值相关数据
        q = query(valuation.code, valuation.market_cap, valuation.pe_ratio).filter(valuation.code.in_(list(df.index)))
        cap = get_fundamentals(q, date=time1).set_index("code")

        # 计算股息率, 股利支付率
        df = pd.concat([df, cap], axis=1, sort=False)
        df["div_yield"] = df["bonus_amount_rmb"] / (df["market_cap"] * 10000) / year
        df["payout_rate"] = df["bonus_amount_rmb"] / ((df["market_cap"] * 10000) / df["pe_ratio"]) / year
        df = df.query("div_yield > @div_yield and payout_rate > @payout_rate")
        return list(df.index)

    # 利润过滤(近几年的营业收入、净利率、毛利率)
    def filter_profit(self, stocks, year, net_profit_margin, gross_profit_margin):

        if not stocks:
            return []

        df = get_history_fundamentals(
            stocks,
            fields=[income.operating_revenue, indicator.net_profit_margin, indicator.gross_profit_margin],
            watch_date=self.context.previous_date,
            count=4 * year,
            interval="1q",
            stat_by_year=False,
        )

        def agg_func(group):
            revenue = group["operating_revenue"].sum()
            return pd.Series(
                {
                    "operating_revenue": revenue,
                    "weighted_net_profit_margin": (group["operating_revenue"] * group["net_profit_margin"]).sum() / revenue,
                    "weighted_gross_profit_margin": (group["operating_revenue"] * group["gross_profit_margin"]).sum() / revenue,
                }
            )

        df = df.groupby("code").apply(agg_func).reset_index()
        df = df.query("weighted_net_profit_margin > @net_profit_margin and weighted_gross_profit_margin > @gross_profit_margin")
        return list(df.code)

    # 获取净利润波动率(近几年的净利润标准差)
    def get_profit_vol(self, stocks, year):

        df = get_history_fundamentals(
            stocks,
            fields=[income.net_profit],
            watch_date=self.context.previous_date,
            count=4 * year,
            interval="1q",
            stat_by_year=False,
        )
        df["rolling_profit"] = df.groupby("code")["net_profit"].rolling(4).sum().reset_index(level=0, drop=True)
        df["growth_rate"] = df.groupby("code")["rolling_profit"].pct_change()
        return df.groupby("code")["growth_rate"].std().reset_index(name="volatility")

    # 过滤 801160: 公用事业I
    def filter1(self):
        stocks = get_industry_stocks("801160")
        # 基本面过滤
        stocks = self.filter_basic_stock(stocks)
        df = get_fundamentals(
            query(valuation.code, valuation.pe_ratio).filter(
                valuation.code.in_(stocks),
                # 现金流
                cash_flow.net_operate_cash_flow > 0,
                cash_flow.subtotal_operate_cash_inflow / indicator.adjusted_profit > 1.0,
                # 资产
                balance.total_liability / balance.total_assets < 0.8,
                # 市值
                valuation.market_cap > 200,
            )
        )

        stocks = self.filter_dividend(list(df.code), 3, 0.02, 0.4)
        stocks = self.filter_profit(stocks, 1, 20, 30)

        if not stocks:
            return {}

        # 业绩排序
        vol = self.get_profit_vol(stocks, 3)
        df = vol.merge(df[["code", "pe_ratio"]], on="code", how="left")
        df["score"] = 1 / df["pe_ratio"] * (1 - 2 * df["volatility"])
        df = df.sort_values(by="score", ascending=False).reset_index(drop=True)

        return {df.code[i]: round(ratio * self.industry_ratio[0], 3) for i, ratio in enumerate(self.stock_ratio[: len(df)])}

    # 过滤 801170: 交通运输I
    def filter2(self):
        stocks = get_industry_stocks("801170")
        # 基本面过滤
        stocks = self.filter_basic_stock(stocks)
        df = get_fundamentals(
            query(valuation.code, valuation.pe_ratio, indicator.roa).filter(
                valuation.code.in_(stocks),
                # 现金流
                cash_flow.net_operate_cash_flow > 0,
                cash_flow.subtotal_operate_cash_inflow / indicator.adjusted_profit > 1.0,
                # 资产
                balance.total_liability / balance.total_assets < 0.6,
                # 市值
                valuation.market_cap > 200,
            )
        )
        stocks = self.filter_dividend(list(df.code), 3, 0.02, 0.3)
        stocks = self.filter_profit(stocks, 1, 20, 30)

        if not stocks:
            return {}

        # 业绩排序
        vol = self.get_profit_vol(stocks, 3)
        df = vol.merge(df[["code", "pe_ratio"]], on="code", how="left")
        df["score"] = 1 / df["pe_ratio"] * (1 - 2 * df["volatility"])
        df = df.sort_values(by="score", ascending=False).reset_index(drop=True)

        return {df.code[i]: round(ratio * self.industry_ratio[1], 3) for i, ratio in enumerate(self.stock_ratio[: len(df)])}

    # 调仓
    def adjust(self):
        targets = {}
        targets.update(self.filter1())
        targets.update(self.filter2())
        self._adjust(targets)


# 核心资产轮动策略
class Etf_Rotation_Strategy(Strategy):
    def __init__(self, context, index, name):
        super().__init__(context, index, name)

        self.stock_sum = 1
        self.etf_pool = [
            # 境外
            "513100.XSHG",  # 纳指ETF
            "513520.XSHG",  # 日经ETF
            "513030.XSHG",  # 德国ETF
            # 商品
            "518880.XSHG",  # 黄金ETF
            "159980.XSHE",  # 有色ETF
            "159985.XSHE",  # 豆粕ETF
            "501018.XSHG",  # 南方原油
            # 债券
            "511090.XSHG",  # 30年国债ETF
            # 国内
            "513130.XSHG",  # 恒生科技
        ]
        self.m_days = 25  # 动量参考天数

    def filter(self):
        data = pd.DataFrame(index=self.etf_pool, columns=["annualized_returns", "r2", "score"])
        current_data = get_current_data()
        for etf in self.etf_pool:
            # 获取数据
            df = attribute_history(etf, self.m_days, "1d", ["close", "high"])
            prices = np.append(df["close"].values, current_data[etf].last_price)

            # 设置参数
            y = np.log(prices)
            x = np.arange(len(y))
            weights = np.linspace(1, 2, len(y))

            # 计算年化收益率
            slope, intercept = np.polyfit(x, y, 1, w=weights)
            data.loc[etf, "annualized_returns"] = math.exp(slope * 250) - 1

            # 计算R²
            ss_res = np.sum(weights * (y - (slope * x + intercept)) ** 2)
            ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
            data.loc[etf, "r2"] = 1 - ss_res / ss_tot if ss_tot else 0

            # 计算得分
            data.loc[etf, "score"] = data.loc[etf, "annualized_returns"] * data.loc[etf, "r2"]

            # 过滤近3日跌幅超过5%的ETF
            if min(prices[-1] / prices[-2], prices[-2] / prices[-3], prices[-3] / prices[-4]) < 0.95:
                data.loc[etf, "score"] = 0

        # 过滤ETF，并按得分降序排列
        data = data.query("0 < score < 6").sort_values(by="score", ascending=False)

        return data.index.tolist()

    # 调仓
    def adjust(self):
        targets = self.filter()[: self.stock_sum]
        self._adjust({etf: round(1 / len(targets), 3) for etf in targets})
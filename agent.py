import numpy as np
import utils

class Agent:
	# 에이전트 상태가 구성하는 값 개수
	STATE_DIM = 2

	# 거래 수수료
	TRADING_CHARGE = 0.00015
	TRADING_TAX = 0.0025

	ACTION_BUY = 0
	ACTION_SELL = 1
	ACTION_HOLD = 2

	ACTIONS = [ACTION_BUY, ACTION_SELL]
	NUM_ACTIONS = len(ACTIONS)

	def __init__(self, environment, min_trading_unit = 1, max_trading_unit = 2, delayed_reward_threshold = 0.05):
		self.environment = environment

		self.min_trading_unit = min_trading_unit
		self.max_trading_unit = max_trading_unit
		self.delayed_reward_threshold = delayed_reward_threshold

		self.initial_balance = 0
		self.balance = 0
		self.num_stocks = 0
		# PV = balance + num_stocks * (현재주식가격)
		self.portfolio_value = 0
		self.base_portfolio_value = 0
		self.num_buy = 0
		self.num_sell = 0
		self.num_hold = 0
		self.immediate_reward = 0
		self.profitloss = 0
		self.base_profitloss = 0
		self.exploration_base = 0
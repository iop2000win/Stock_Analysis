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
		self.immediate_reward = 0 # 즉시 보상
		self.profitloss = 0 # 현재 손익
		self.base_profitloss = 0 # 직전 지연 보상 이후 손익
		self.exploration_base = 0 # 탐험 행동 결정 기준

		# Agent 클래스의 상태
		self.ratio_hold = 0 # 주식 보유 비율
		self.ratio_portfolio_value = 0 # 포트폴리오 가치 비율


	def reset(self):
		'''
		Agent 클래스의 속성들을 초기화하는 함수
		학습 단계에서 한 에포크마다 속성값들을 초기화한 후에, 다시 학습을 진행하도록 하기 위함
		'''
		self.balance = 0
		self.num_stocks = 0
		self.portfolio_value = self.initial_balance
		self.base_portfolio_value = self.initial_balance
		self.num_buy = 0
		self.num_sell = 0
		self.num_hold = 0
		self.immediate_reward = 0
		self.ratio_hold = 0
		self.ratio_portfolio_value = 0


	def reset_exploration(self):
		'''
		50% 매수 탐험 확률을 미리 설정
		'''
		self.exploration_base = 0.5 + np.random.rand() / 2


	def set_balance(self, balance):
		'''
		Agent의 초기 자본금 설정
		'''
		self.initial_balance = balance


	def get_states(self):
		'''
		주식 보유 비율(ratio_hold) = 보유 주식 수 / (포트폴리오 가치/현재 주가)
		포트폴리오 가치 비율(ratio_portfolio_value) = 포트폴리오
		'''
		self.ratio_hold = self.num_stocks / int(self.portfolio_value / self.environment.get_price())
		self.ratio_portfolio_value = (self.portfolio_value / self.base_portfolio_value)

		return (self.ratio_hold, self.ratio_portfolio_value)


	def decide_action(self, pred_value, pred_policy, epsilon):
		'''
		행동을 결정하고 결정한 행동의 유효성을 검사하는 함수
		epsilon이 입력될 경우, 무작위로 행동 결정
		'''
		confidence = 0

		pred = pred_policy
		if pred is None:
			pred = pred_value

		if pred is None:
			# 예측 값이 없을 경우 탐험
			epsilon = 1
		else:
			# 값이 모두 같은 경우 탐험
			maxpred = np.max(pred)
			if (pred == maxpred).all():
				epsilon = 1

		# 탐험 결정
		if np.random.rand() < epsilon: # np.random.rand() >>> [0~1) 균일 분포에서 값 추출
			exploration = True
			if np.random.rand() < self.exploration_base:
				action = self.ACTION_BUY
			else:
				action = np.random.randint(self.NUM_ACTIONS -1) + 1

		else:
			exploration = False
			action = np.argmax(pred)

		confidence = 0.5
		if pred_policy is not None:
			confidence = pred[action]
		elif pred_value is not None:
			confidence = utils.sigmoid(pred[action])

		return action, confidence, exploration


	def validate_action(self, action):
		if action == Agent.ACTION_BUY:
			# 주식을 살 돈이 있는지 확인
			if self.balance < self.environment.get_price() * (1 + self.TRADING_CHARGE) * self.min_trading_unit:
				return False

		elif action == Agent.ACTION_SELL:
			# 주식 잔고가 있는지 확인
			if self.num_stocks <= 0:
				return False

		return True


	def decide_trading_unit(self, confidence):
		if np.isnan(confidence):
			return self.min_trading_unit

		added_trading = max(
							min(int(confidence * (self.max_trading_unit - self.min_trading_unit))), 0
			)

		return self.min_trading_unit + added_trading


	def act(self, action, confidence):
		if not self.validate_action(action):
			action = Agent.ACTION_HOLD

		# environment에서 현재 가격 얻기
		curr_price = self.environment.get_price()

		# 즉시 보상 초기화
		self.immediate_reward = 0

		# 매수
		if action == Agent.ACTION_BUY:
			# 매수할 단위를 판단
			trading_unit = self.decide_trading_unit(confidence)
			balance = (self.balance - curr_price * (1 + self.TRADING_CHARGE) * trading_unit)

			if balance < 0:
				trading_unit = max(
									min(int(self.balance / (curr_price * (1 + self.TRADING_CHARGE))), self.max_trading_unit),
									self.min_trading_unit
					)

			# 수수료를 적용해 총 매수 금액 산정
			invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
			if invest_amount > 0:
				self.balance -= invest_amount
				self.num_stocks += trading_unit
				self.num_buy += 1

		# 매도
		elif action == Agent.ACTION_SELL:
			# 매도할 단위를 판단
			trading_unit = self.decide_trading_unit(confidence)
			# 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
			trading_unit = min(trading_unit, self.num_stocks)
			# 매도
			invest_amount = curr_price * (1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit
			if invest_amount > 0:
				self.balance += invest_amount
				self.num_stocks -= trading_unit
				self.num_sell += 1

		# 관망
		elif action == Agent.ACTION_HOLD:
			self.num_hold += 1

		# 포트폴리오 가치 갱신
		


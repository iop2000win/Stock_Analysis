class Environment:
	PRICE_IDX = 4 # 종가의 위치 (입력 데이터에 대해서 컬럼 인덱스)

	def __init__(self, chart_data = None):
		self.chart_data = chart_data
		self.observation = None
		self.idx = -1

	def reset(self): # 초기 값을 다시 복구하는 것
		self.observation = None
		self.idx = -1

	def observe(self):
		# row 값을 부여
		if len(self.chart_data) > self.idx + 1:
			self.idx += 1
			self.observation = self.chart_data.iloc[self.idx]

			return self.observation

		return None

	def get_price(self):
		if self.observation is not None:
			return self.observation[self.PRICE_IDX]

		return None

	def set_chart_data(self, chart_data):
		self.chart_data = chart_data
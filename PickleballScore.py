from enum import Enum

class ServerSide(Enum):
	NR = 1
	NL = 2
	FR = 3
	FL = 4

class PickleballScore:
	def __init__(self):
		# score1 is first server score, score2 is second server score
		self.score1 = [] 	 # Near side, Far side. First serve doesnt matter here - maybe change later.
		self.score2 = [0, 0] # sometimes score is ambiguous because we only get NR, NL, FR, FL, we dont know who served
		self.serverSide = ServerSide.NR # serverSide can have NR, NL, FR, FL. near right, near left, far right, far left
		self.error = False
		self.start = True
		self.debug = False

	def check_errors(self, newServerSide, enum1, enum2):
		if newServerSide not in (enum1, enum2):
			print("1 Invalid new server side " + str(newServerSide) + str(self.score1) + str(self.score2) + str(self.serverSide))
			self.error = True
		if not self.score2 and newServerSide != enum1:
			print("2 Invalid new server side " + str(newServerSide) + str(self.score1) + str(self.score2) + str(self.serverSide))
			self.error = True

	def update_score_sameside(self, inx):
		if self.score1 and self.score2:
			self.score1[inx] += 1
			self.score2[inx] += 1
		elif self.score1:
			self.score2 = self.score1.copy()
			self.score1[inx] += 1
		elif self.score2:
			self.score2[inx] += 1
		else:
			print("update_score_sameside: scores are empty")
			self.error = True

	def update_state(self, newServerSide):
		if self.start == True:
			self.serverSide = newServerSide
			self.start = False
			if self.debug:
				print(self.serverSide, self.score1, self.score2, self.error)
			return

		if   self.serverSide == ServerSide.NR: #can only go to NL or FR
			# Error conditions
			self.check_errors(newServerSide, ServerSide.NL, ServerSide.FR)
			# Update state
			if   newServerSide == ServerSide.NL:
				self.update_score_sameside(0)
			elif newServerSide == ServerSide.FR: # side out
				self.score1 = self.score2.copy()
				self.score2 = []
			else:
				print("4 unkown error " + str(newServerSide) + str(self.score1) + str(self.score2) + str(self.serverSide))
				self.error = True
			
		elif self.serverSide == ServerSide.NL: #can only go to NR or FR
			# Error conditions
			self.check_errors(newServerSide, ServerSide.NR, ServerSide.FR)
			# Update state
			if   newServerSide == ServerSide.NR:
				self.update_score_sameside(0)
			elif newServerSide == ServerSide.FR: # side out
				self.score1 = self.score2.copy()
				self.score2 = []
			else:
				print("8 unkown error " + str(newServerSide) + str(self.score1) + str(self.score2) + str(self.serverSide))
				self.error = True
		elif self.serverSide == ServerSide.FR: # can only go to FL or NR
			# Error conditions
			self.check_errors(newServerSide, ServerSide.FL, ServerSide.NR)
			# Update state
			if   newServerSide == ServerSide.FL:
				self.update_score_sameside(1)
			elif newServerSide == ServerSide.NR: # side out
				self.score1 = self.score2.copy()
				self.score2 = []
			else:
				print("4 unkown error " + str(newServerSide) + str(self.score1) + str(self.score2) + str(self.serverSide))
				self.error = True
		elif self.serverSide == ServerSide.FL: # can only go to FR or NR
			# Error conditions
			self.check_errors(newServerSide, ServerSide.FR, ServerSide.NR)
			# Update state
			if   newServerSide == ServerSide.FR:
				self.update_score_sameside(1)
			elif newServerSide == ServerSide.NR: # side out
				self.score1 = self.score2.copy()
				self.score2 = []
			else:
				print("4 unkown error " + str(newServerSide) + str(self.score1) + str(self.score2) + str(self.serverSide))
				self.error = True
		else:
			print("Invalid server state" + str(self.serverSide))
			self.error = True
		# update self.serverSide
		self.serverSide = newServerSide
		if self.debug:
			print(self.serverSide, self.score1, self.score2, self.error)





